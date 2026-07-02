#!/usr/bin/env python3
"""Comparison plot: bb144 [[144,12]] d=12 LER vs p, three decoders at their best
(tv=5 for the p-bit decoders):

  1. p-bit FILTERED (memory_Z, basis Z)   -- ours, tuned; parsed from campaign RESULT.txt
  2. RelayBP-300sets (memory_Z, basis Z)  -- reference (same filtered problem)
  3. p-bit CORRELATED (full DEM, memory_X) -- ours; parsed from RESULT_FINAL.txt

Outputs PNG + standalone TikZ + a tidy CSV into the plots/ dir.

Usage: python make_compare_plot.py [campaign_RESULT.txt] [outdir]
"""
import sys, os, re, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = sys.argv[1] if len(sys.argv) > 1 else "/tmp/zcamp/RESULT.txt"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "plots")
OUTDIR = os.path.abspath(OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)
RESULT_FINAL = os.path.join(HERE, "..", "RESULT_FINAL.txt")

# RelayBP-300sets reference, filtered memory_Z basis Z (errors, shots).
# Source: relay/bicycle_bivariate_144_12_12_native_bpgd_cluster_Z_basis_summary.csv
RELAYBP = {0.002: (21, 868028), 0.003: (22, 46081), 0.004: (23, 5360), 0.005: (21, 439)}

def wilson(e, n):
    """LER point estimate, stderr, and a 95% upper bound (for zero-error too)."""
    if not n:
        return 0.0, 0.0, 0.0
    ler = e / n
    se = math.sqrt(e) / n if e else 0.0
    ul = (e + 1.64 * math.sqrt(e)) / n if e else 3.0 / n
    return ler, se, ul

def parse_result(path, tv=5):
    """Parse a RESULT.txt / RESULT_FINAL.txt -> {p: (errors, shots)} for given tv."""
    out = {}
    if not os.path.exists(path):
        return out
    cur_p = None
    for line in open(path):
        m = re.search(r'p=([0-9.]+)\s+N=(\d+)', line)
        if m:
            cur_p = float(m.group(1))
        mt = re.search(r'tv=%d\b.*errors=(\d+)/(\d+)' % tv, line)
        if mt:
            p = float(re.search(r'p=([0-9.]+)', line).group(1)) if 'p=' in line else cur_p
            out[p] = (int(mt.group(1)), int(mt.group(2)))
    return out

def series(d):
    ps = sorted(d)
    xs, ys, los, his = [], [], [], []
    for p in ps:
        e, n = d[p]
        ler, se, ul = wilson(e, n)
        xs.append(p)
        ys.append(ler if ler > 0 else ul)   # plot UL for zero-error points
        los.append(se if ler > 0 else 0.0)
        his.append(se if ler > 0 else 0.0)
    return xs, ys, los, his

ours_filt = parse_result(CAMP, 5)
ours_corr = parse_result(RESULT_FINAL, 5)
relay = {p: v for p, v in RELAYBP.items()}

# ---- plot ----
fig, ax = plt.subplots(figsize=(7.0, 5.2))
defs = [
    ("p-bit filtered (ours, tuned)", ours_filt, "#1f77b4", "o", "-"),
    ("RelayBP-300sets (ref)",        relay,     "#d62728", "s", "--"),
    ("p-bit correlated / full DEM (ours)", ours_corr, "#2ca02c", "^", "-"),
]
for label, d, color, mk, ls in defs:
    if not d:
        continue
    xs, ys, lo, hi = series(d)
    ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=mk, ls=ls,
                lw=1.8, ms=7, capsize=3, label=label)

ax.set_yscale("log")
ax.set_xlabel("physical error rate $p$")
ax.set_ylabel("logical error rate per shot (12 rounds)")
ax.set_title("bb144 [[144,12]] d=12 memory: p-bit vs RelayBP")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
png = os.path.join(OUTDIR, "compare_pbit_relaybp.png")
fig.savefig(png, dpi=160)
print("wrote", png)

# ---- CSV ----
csv = os.path.join(OUTDIR, "compare_pbit_relaybp.csv")
with open(csv, "w") as fh:
    fh.write("decoder,p,errors,shots,ler,ler_stderr,ul95\n")
    for label, d, *_ in defs:
        for p in sorted(d):
            e, n = d[p]
            ler, se, ul = wilson(e, n)
            fh.write(f"{label},{p},{e},{n},{ler:.4e},{se:.2e},{ul:.4e}\n")
print("wrote", csv)

# ---- standalone TikZ (pgfplots) ----
def tikz_coords(d):
    return " ".join(f"({p},{wilson(*d[p])[0] if wilson(*d[p])[0] > 0 else wilson(*d[p])[2]:.4e})"
                    for p in sorted(d))
tikz = os.path.join(OUTDIR, "compare_pbit_relaybp.tex")
with open(tikz, "w") as fh:
    fh.write(r"""\documentclass[tikz]{standalone}
\usepackage{pgfplots}\pgfplotsset{compat=1.17}
\begin{document}
\begin{tikzpicture}
\begin{semilogyaxis}[
  width=11cm,height=8cm,xlabel={physical error rate $p$},
  ylabel={logical error rate per shot},
  title={bb144 [[144,12]] $d{=}12$ memory: p-bit vs RelayBP},
  grid=both,legend pos=south east,legend cell align=left]
""")
    names = {0: "p-bit filtered (ours)", 1: "RelayBP-300sets", 2: "p-bit correlated (ours)"}
    for i, (label, d, *_ ) in enumerate(defs):
        if not d:
            continue
        fh.write(f"\\addplot+[thick,mark=*] coordinates {{{tikz_coords(d)}}};\n")
        fh.write(f"\\addlegendentry{{{label}}}\n")
    fh.write(r"""\end{semilogyaxis}
\end{tikzpicture}
\end{document}
""")
print("wrote", tikz)
