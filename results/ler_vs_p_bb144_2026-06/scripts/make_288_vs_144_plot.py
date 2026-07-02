#!/usr/bin/env python3
"""Overlay LER-vs-p plot: bivariate-bicycle FILTERED memory_Z (basis Z), p-bit
ColoredBlockGibbsDecoder at t_v=5, two code distances:

  * bb288 [[288,12,18]] d=18  -- /tmp/zcamp288/RESULT.txt  (tuned rho=0.30, sweeps=1000, beta 0.5->6)
  * bb144 [[144,12,12]] d=12  -- /tmp/zcamp/RESULT.txt     (tuned rho=0.20, sweeps=750,  beta 0.5->8)

Both are the SAME filtered problem family (memory_Z, Z-basis detector filter, 12
observables, uniform circuit noise), so this is a clean distance-scaling overlay.
Outputs PNG + standalone TikZ (pgfplots) + a tidy CSV into the plots/ dir.

Usage: python make_288_vs_144_plot.py [RESULT_288] [RESULT_144] [outdir]
"""
import sys
import os
import re
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R288 = sys.argv[1] if len(sys.argv) > 1 else "/tmp/zcamp288/RESULT.txt"
R144 = sys.argv[2] if len(sys.argv) > 2 else "/tmp/zcamp/RESULT.txt"
OUTDIR = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "plots"))
os.makedirs(OUTDIR, exist_ok=True)


def wilson(e, n):
    """LER point estimate, stderr, and a 95% upper bound (handles zero-error)."""
    if not n:
        return 0.0, 0.0, 0.0
    ler = e / n
    se = math.sqrt(e) / n if e else 0.0
    ul = (e + 1.64 * math.sqrt(e)) / n if e else 3.0 / n
    return ler, se, ul


def parse_result(path, tv=5):
    """Parse a campaign RESULT.txt -> {p: (errors, shots)} for the given t_v."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        mt = re.search(r'tv=%d\b.*errors=(\d+)/(\d+)' % tv, line)
        mp = re.search(r'p=([0-9.]+)', line)
        if mt and mp:
            out[float(mp.group(1))] = (int(mt.group(1)), int(mt.group(2)))
    return out


def series(d):
    """(xs, ys, yerr) with zero-error points plotted at their 95% upper bound."""
    xs, ys, err = [], [], []
    for p in sorted(d):
        e, n = d[p]
        ler, se, ul = wilson(e, n)
        xs.append(p)
        ys.append(ler if ler > 0 else ul)
        err.append(se)
    return xs, ys, err


d288 = parse_result(R288, 5)
d144 = parse_result(R144, 5)
if not d288 or not d144:
    raise SystemExit(f"missing data: 288={bool(d288)} ({R288}) 144={bool(d144)} ({R144})")

DEFS = [
    ("bb288 $[[288,12,18]]$  $d{=}18$ (18 rounds)", d288, "#1f77b4", "o", "-"),
    ("bb144 $[[144,12,12]]$  $d{=}12$ (12 rounds)", d144, "#d62728", "s", "--"),
]

# ---- PNG ----
fig, ax = plt.subplots(figsize=(7.2, 5.4))
for label, d, color, mk, ls in DEFS:
    xs, ys, err = series(d)
    ax.errorbar(xs, ys, yerr=err, color=color, marker=mk, ls=ls, lw=1.9, ms=7,
                capsize=3, markerfacecolor="white", markeredgewidth=1.5, label=label)

# Annotate the d=18 suppression factor at each shared p.
for p in sorted(set(d288) & set(d144)):
    l288 = wilson(*d288[p])[0] or wilson(*d288[p])[2]
    l144 = wilson(*d144[p])[0] or wilson(*d144[p])[2]
    if l288 > 0:
        ax.annotate(rf"${l144 / l288:.1f}\times$", (p, l288), textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color="#1f77b4")

ax.set_yscale("log")
ax.set_xlabel("physical error rate $p$")
ax.set_ylabel("logical error rate per shot ($t_v=5$)")
ax.set_title("Bivariate-bicycle FILTERED memory$_Z$: distance scaling (p-bit decoder)")
ax.set_xticks(sorted(set(d288) | set(d144)))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
ax.grid(True, which="both", color="0.85", lw=0.6)
ax.legend(loc="lower right", frameon=True, fontsize=9)
fig.tight_layout()
png = os.path.join(OUTDIR, "ler_vs_p_288_vs_144_filtered_tv5.png")
fig.savefig(png, dpi=200)
plt.close(fig)
print("wrote", png)

# ---- CSV ----
csv = os.path.join(OUTDIR, "ler_vs_p_288_vs_144_filtered_tv5.csv")
with open(csv, "w") as fh:
    fh.write("code,distance,p,errors,shots,ler,ler_stderr,ul95\n")
    for label, d, *_ in DEFS:
        dist = "18" if "288" in label else "12"
        code = "bb288" if "288" in label else "bb144"
        for p in sorted(d):
            e, n = d[p]
            ler, se, ul = wilson(e, n)
            fh.write(f"{code},{dist},{p},{e},{n},{ler:.4e},{se:.2e},{ul:.4e}\n")
    fh.write("# d=18 advantage (LER_144 / LER_288) per p:\n")
    for p in sorted(set(d288) & set(d144)):
        l288 = wilson(*d288[p])[0] or wilson(*d288[p])[2]
        l144 = wilson(*d144[p])[0] or wilson(*d144[p])[2]
        fh.write(f"# p={p}: {l144 / l288:.2f}x\n")
print("wrote", csv)


# ---- standalone TikZ (pgfplots) ----
def coords(d):
    return " ".join(
        f"({p:.3f}, {(wilson(*d[p])[0] or wilson(*d[p])[2]):.4e}) +- (0, {wilson(*d[p])[1]:.4e})"
        for p in sorted(d))


tex = os.path.join(OUTDIR, "ler_vs_p_288_vs_144_filtered_tv5.tex")
lines = [
    "% Standalone pgfplots figure -- compile with pdflatex/lualatex,",
    "% or \\input{} into a doc that loads \\usepackage{pgfplots}.",
    "% Bivariate-bicycle FILTERED memory_Z (basis Z), p-bit decoder, t_v=5.",
    "\\documentclass[tikz,border=3mm]{standalone}",
    "\\usepackage{pgfplots}",
    "\\pgfplotsset{compat=1.17}",
    "\\begin{document}",
    "\\begin{tikzpicture}",
    "\\begin{semilogyaxis}[",
    "    width=12cm, height=9cm,",
    "    xlabel={physical error rate $p$},",
    "    ylabel={logical error rate per shot ($t_v=5$)},",
    "    title={Bivariate-bicycle filtered memory$_Z$: distance scaling},",
    "    xtick={0.002,0.003,0.004,0.005},",
    "    scaled x ticks=false, x tick label style={/pgf/number format/fixed},",
    "    grid=both, grid style={gray!25},",
    "    legend pos=south east, legend cell align=left,",
    "    mark options={solid}, every axis plot/.append style={thick},",
    "]",
    f"\\addplot+[color=blue, mark=*, error bars/.cd, y dir=both, y explicit]",
    f"    coordinates {{{coords(d288)}}};",
    "\\addlegendentry{bb288 $[[288,12,18]]$ $d{=}18$}",
    f"\\addplot+[color=red, mark=square*, error bars/.cd, y dir=both, y explicit]",
    f"    coordinates {{{coords(d144)}}};",
    "\\addlegendentry{bb144 $[[144,12,12]]$ $d{=}12$}",
    "\\end{semilogyaxis}",
    "\\end{tikzpicture}",
    "\\end{document}",
]
with open(tex, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("wrote", tex)
