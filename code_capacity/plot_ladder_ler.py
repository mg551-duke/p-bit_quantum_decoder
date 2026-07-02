#!/usr/bin/env python3
"""Plot B1 code-capacity content-LER vs p for the L1-L4 decoder ladder.

Two data sources are merged (high-statistics pooled points win when present):
  * fixed-budget sweep grid : results/b1_psweep/p<p>/<step>/summary.csv
  * pooled low-p campaigns   : results/b1_lowp/DRIVER.log  (>=10 content errors;
                               L3/L4 at p=0.04, 0.05)

content LER = (logic + nonconv_logic) / runs  (true logical error rate).
x-axis linear (p), y-axis log (content LER). Cells with zero observed content
errors are drawn as rule-of-three (3/N) upper bounds with a downward arrow.

Usage: python code_capacity/plot_ladder_ler.py
"""
import csv
import glob
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PSWEEP = os.path.join(HERE, "results", "b1_psweep")
LOWP_LOGS = [
    os.path.join(HERE, "results", "b1_lowp", "DRIVER.log"),         # L3/L4 pool
    os.path.join(HERE, "results", "b1_lowp_l1l2", "DRIVER.log"),    # L1/L2 sharpen
]
OUT_PNG = os.path.join(HERE, "results", "b1_ladder_ler_vs_p.png")
OUT_CSV = os.path.join(HERE, "results", "b1_ladder_ler_vs_p.csv")
OUT_TEX = os.path.join(HERE, "results", "b1_ladder_ler_vs_p.tex")

STEPS = ["L1_s300_l1", "L2_s800_l4", "L3_s1500_l10", "L4_s3000_l24"]
LABELS = {
    "L1_s300_l1": "L1  sw300  legs1",
    "L2_s800_l4": "L2  sw800  legs4",
    "L3_s1500_l10": "L3  sw1500 legs10",
    "L4_s3000_l24": "L4  sw3000 legs24",
}
COLORS = {
    "L1_s300_l1": "#1f77b4",
    "L2_s800_l4": "#2ca02c",
    "L3_s1500_l10": "#ff7f0e",
    "L4_s3000_l24": "#d62728",
}
MARKERS = {"L1_s300_l1": "o", "L2_s800_l4": "s",
           "L3_s1500_l10": "^", "L4_s3000_l24": "D"}
TEX_MARKERS = {"L1_s300_l1": "*", "L2_s800_l4": "square*",
               "L3_s1500_l10": "triangle*", "L4_s3000_l24": "diamond*"}


def _f(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def load_psweep():
    """{step: {p: {runs, err, ler, se}}} from the fixed-budget sweep grid."""
    out = {}
    for path in glob.glob(os.path.join(PSWEEP, "p*", "*", "summary.csv")):
        parts = path.split(os.sep)
        m = re.match(r"p([0-9.]+)", parts[-3])
        step = parts[-2]
        if not m:
            continue
        p = float(m.group(1))
        try:
            row = next(iter(csv.DictReader(open(path))))
        except StopIteration:
            continue
        runs = _f(row, "runs")
        err = int(_f(row, "logic")) + int(_f(row, "nonconv_logic"))
        out.setdefault(step, {})[p] = {
            "runs": runs,
            "err": err,
            "ler": _f(row, "ler_content"),
            "se": _f(row, "ler_content_se"),
            "src": "psweep",
        }
    return out


def load_lowp(log_path):
    """{step: {p: {runs, err, ler, se, done}}} parsed from the pool DRIVER.log."""
    out = {}
    cur = None
    re_start = re.compile(r"step=(\S+)")
    re_done = re.compile(
        r"p=([0-9.]+) DONE pooled runs=(\d+) content=(\d+) "
        r"LER\(content\)=([0-9.eE+-]+) \+- ([0-9.eE+-]+)")
    re_prog = re.compile(r"p=([0-9.]+) pooled runs=(\d+) content=(\d+)/\d+")
    if not os.path.exists(log_path):
        return out
    for line in open(log_path):
        if "START" in line and "step=" in line:
            m = re_start.search(line)
            if m:
                cur = m.group(1)
            continue
        md = re_done.search(line)
        if md and cur:
            p = float(md.group(1))
            out.setdefault(cur, {})[p] = {
                "runs": float(md.group(2)), "err": int(md.group(3)),
                "ler": float(md.group(4)), "se": float(md.group(5)),
                "done": True, "src": "lowp"}
            continue
        mp = re_prog.search(line)
        if mp and cur:
            p = float(mp.group(1))
            runs = float(mp.group(2))
            err = int(mp.group(3))
            prev = out.setdefault(cur, {}).get(p)
            if prev is None or not prev.get("done"):
                ler = err / runs if (err and runs) else 0.0
                se = math.sqrt(err) / runs if (err and runs) else 0.0
                out.setdefault(cur, {})[p] = {
                    "runs": runs, "err": err, "ler": ler, "se": se,
                    "done": False, "src": "lowp"}
    return out


def load_lowp_many(log_paths):
    """Merge pooled results from several DRIVER.logs (later ones win on conflict)."""
    merged = {}
    for path in log_paths:
        for step, by_p in load_lowp(path).items():
            merged.setdefault(step, {}).update(by_p)
    return merged


def merge(psweep, lowp):
    """Prefer the high-statistics pooled point whenever one exists."""
    combined = {}
    for step in STEPS:
        combined[step] = dict(psweep.get(step, {}))
        for p, rec in lowp.get(step, {}).items():
            combined[step][p] = rec
    return combined


def _hex_to_rgb255(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def write_tex(data):
    """Emit a standalone-compilable, \\input-able pgfplots figure of the ladder."""
    L = []
    a = L.append
    a("% B1 code-capacity content-LER vs p -- decoder ladder L1-L4.")
    a("% Auto-generated by plot_ladder_ler.py -- do not edit by hand.")
    a("%")
    a("% In your document preamble:")
    a("%     \\usepackage{pgfplots}")
    a("%     \\pgfplotsset{compat=1.18}")
    a("% then in the body:")
    a("%     \\input{b1_ladder_ler_vs_p.tex}")
    a("%")
    a("% Or compile this file on its own with:")
    a("%     \\documentclass{standalone}")
    a("%     \\usepackage{pgfplots}\\pgfplotsset{compat=1.18}")
    a("%     \\begin{document}\\input{b1_ladder_ler_vs_p.tex}\\end{document}")
    a("\\begin{tikzpicture}")
    a("\\begin{semilogyaxis}[")
    a("    width=12cm, height=9cm,")
    a("    xlabel={physical error rate $p$},")
    a("    ylabel={logical error rate (content)},")
    a("    title={B1 code-capacity LER vs $p$ -- decoder ladder L1--L4},")
    a("    xmin=0.035, xmax=0.105,")
    a("    xtick={0.04,0.05,0.06,0.07,0.08,0.09,0.1},")
    a("    grid=both, grid style={gray!25},")
    a("    legend pos=south east, legend cell align={left},")
    a("    legend style={font=\\small},")
    a("    every axis plot/.append style={thick, mark options={solid}},")
    a("]")
    for step in STEPS:
        pts = sorted(data.get(step, {}).items())
        if not pts:
            continue
        r, g, b = _hex_to_rgb255(COLORS[step])
        col = "{rgb,255:red,%d;green,%d;blue,%d}" % (r, g, b)
        meas, ub = [], []
        for p, rec in pts:
            runs = rec["runs"]
            if rec["err"] > 0:
                v = rec["ler"]
                meas.append((p, v, min(rec["se"], v * 0.999)))
            elif runs:
                ub.append((p, 3.0 / runs))
        if meas:
            a("\\addplot[color=%s, mark=%s, mark size=2.5pt,"
              % (col, TEX_MARKERS[step]))
            a("    error bars/.cd, y dir=both, y explicit]")
            a("  coordinates {")
            for p, v, se in meas:
                a("    (%g, %.4e) +- (0, %.4e)" % (p, v, se))
            a("  };")
            a("\\addlegendentry{%s}" % LABELS[step])
        if ub:
            a("\\addplot[color=%s, only marks, mark=triangle*," % col)
            a("    mark options={rotate=180}, mark size=3pt, forget plot]")
            a("  coordinates {")
            for p, v in ub:
                a("    (%g, %.4e)" % (p, v))
            a("  };")
    a("\\draw[dashed, gray] (axis cs:0.035,1e-3) -- (axis cs:0.105,1e-3);")
    a("\\node[gray, font=\\small, anchor=south east] at (axis cs:0.105,1e-3)")
    a("    {LER $=10^{-3}$};")
    a("\\end{semilogyaxis}")
    a("\\end{tikzpicture}")
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    data = merge(load_psweep(), load_lowp_many(LOWP_LOGS))

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    rows_out = []
    for step in STEPS:
        pts = sorted(data.get(step, {}).items())
        if not pts:
            continue
        mx, my, mlo, mhi = [], [], [], []
        ubx, uby = [], []
        for p, rec in pts:
            runs = rec["runs"]
            if rec["err"] > 0:
                v = rec["ler"]
                se = rec["se"]
                mx.append(p)
                my.append(v)
                mlo.append(min(se, v * 0.999))   # clip so log-axis stays valid
                mhi.append(se)
            else:
                ub = 3.0 / runs if runs else float("nan")
                ubx.append(p)
                uby.append(ub)
            rows_out.append((step, p, rec["src"], rec["err"], int(runs),
                             rec["ler"], rec["se"]))
        c = COLORS[step]
        if mx:
            ax.errorbar(mx, my, yerr=[mlo, mhi], color=c, marker=MARKERS[step],
                        ms=6, lw=1.8, capsize=3, label=LABELS[step], zorder=3)
        if ubx:
            ax.errorbar(ubx, uby, yerr=[[u * 0.4 for u in uby], [0] * len(uby)],
                        uplims=True, color=c, marker="v", ms=7, ls="none",
                        zorder=3)

    ax.axhline(1e-3, color="0.5", ls="--", lw=1.0, zorder=1)
    ax.text(0.0995, 1.15e-3, "LER = 1e-3", color="0.4", fontsize=8,
            ha="right", va="bottom")

    ax.set_yscale("log")
    ax.set_xlim(0.035, 0.105)
    ax.set_xlabel("physical error rate  p")
    ax.set_ylabel("logical error rate  (content)")
    ax.set_title("B1 code-capacity LER vs p  —  decoder ladder L1-L4")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(title="decoder (downward arrows = upper bounds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "p", "source", "content_err", "runs",
                    "ler_content", "ler_content_se"])
        for r in sorted(rows_out, key=lambda t: (t[0], t[1])):
            w.writerow(r)

    write_tex(data)

    print(f"# wrote {OUT_PNG}")
    print(f"# wrote {OUT_CSV}")
    print(f"# wrote {OUT_TEX}")
    print(f"{'step':14s}{'p':>6s}{'src':>8s}{'err':>5s}{'runs':>12s}"
          f"{'ler_content':>14s}{'se':>12s}")
    for r in sorted(rows_out, key=lambda t: (t[0], t[1])):
        step, p, src, err, runs, ler, se = r
        shown = f"{ler:.3e}" if err > 0 else f"<{3.0 / runs:.1e}"
        print(f"{step:14s}{p:>6g}{src:>8s}{err:>5d}{runs:>12d}"
              f"{shown:>14s}{se:>12.2e}")


if __name__ == "__main__":
    main()
