#!/usr/bin/env python3
"""LER-vs-p plots for the rotated SURFACE CODE d=11 memory_X, filtered-X p-bit decode.

Parses the campaign RESULT.txt (tv=1..5 per p) and emits PNG + standalone pgfplots .tex
+ CSV. y-axis = logical error rate PER CYCLE (LER_per_shot / ROUNDS, ROUNDS=11); the
per-shot values are kept in the CSV.

Decoder: p-bit ColoredBlockGibbsDecoder, --basis-filter X (720 det / 3631 mech),
tuned config rho=0.20, beta 0.5->12, sweeps=1000, chains=128, max-legs=24.
t_v = capture target (content errors at trajectory budget v); t_v=5 is the headline.

Outputs (../plots):
  ler_vs_p_filtered_tv5.{png,tex}     headline t_v=5
  ler_vs_p_filtered_tv1-5.{png,tex}   full t_v ladder
  ler_vs_p_filtered.csv               all data (per-shot + per-cycle)
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", os.path.join(HERE, "..", "data", "RESULT.txt"))
OUT = os.path.join(HERE, "..", "plots")
ROUNDS = 11  # d=11 memory cycles; LER-per-cycle = LER_per_shot / ROUNDS (<0.1% off exact)

TV_MPL = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728", 5: "#9467bd"}
TV_COLORS = {1: "blue", 2: "orange", 3: "green!60!black", 4: "red", 5: "violet"}
TV_MARK = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
TV_TEXMARK = {1: "*", 2: "square*", 3: "triangle*", 4: "diamond*", 5: "pentagon*"}


def parse(path):
    """Return (ps, data[tv]=[(p,ler,se),...], rows[(p,tv)]=dict, meta[p]=dict)."""
    tv_re = re.compile(
        r"p=([\d.]+)\s+tv=(\d+)\s+LER=([\d.eE+-]+)\s+\+-([\d.eE+-]+)\s+"
        r"ul95=([\d.eE+-]+)\s+errors=(\d+)/(\d+)\s+invalid=(\d+)")
    fin_re = re.compile(r"p=([\d.]+)\s+FINAL\s+N=(\d+)\s+ms_shot=([\d.]+|nan)")
    rows, meta = {}, {}
    with open(path) as fh:
        for line in fh:
            m = fin_re.search(line)
            if m:
                meta[float(m.group(1))] = {"N": int(m.group(2)), "ms_shot": m.group(3)}
                continue
            m = tv_re.search(line)
            if m:
                p, tv = float(m.group(1)), int(m.group(2))
                rows[(p, tv)] = {"ler": float(m.group(3)), "se": float(m.group(4)),
                                 "ul95": float(m.group(5)), "errors": int(m.group(6)),
                                 "N": int(m.group(7)), "invalid": int(m.group(8))}
    ps = sorted({p for (p, _) in rows})
    data = {tv: [(p, rows[(p, tv)]["ler"], rows[(p, tv)]["se"])
                 for p in ps if (p, tv) in rows] for tv in range(1, 6)}
    return ps, data, rows, meta


def make_png(path, title, series_list):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for series, label, color, marker in series_list:
        ps = [p for (p, _, _) in series]
        ler = [l / ROUNDS for (_, l, _) in series]
        err = [e / ROUNDS for (_, _, e) in series]
        ax.errorbar(ps, ler, yerr=err, label=label, color=color, marker=marker,
                    markersize=6, linewidth=1.8, capsize=3, markerfacecolor="white",
                    markeredgewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate per cycle")
    ax.set_title(title)
    ax.set_xticks([0.001, 0.002, 0.003, 0.004, 0.005])
    ax.grid(True, which="both", color="0.85", linewidth=0.6)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def coords(series):
    return " ".join(f"({p:.3f}, {l / ROUNDS:.4e}) +- (0, {e / ROUNDS:.4e})"
                    for (p, l, e) in series)


def make_tex(path, title, series_list, legendpos="north west"):
    lines = [
        "% Standalone pgfplots figure -- compile with pdflatex/lualatex, or",
        "% \\input{} into a document that loads \\usepackage{pgfplots}.",
        "% Surface code d=11 memory_X, filtered-X p-bit decode. LER per cycle (=LER_shot/11).",
        "\\documentclass[tikz,border=3mm]{standalone}",
        "\\usepackage{pgfplots}",
        "\\pgfplotsset{compat=1.17}",
        "\\begin{document}",
        "\\begin{tikzpicture}",
        "\\begin{semilogyaxis}[",
        "    width=12cm, height=9cm,",
        "    xlabel={physical error rate $p$},",
        "    ylabel={logical error rate per cycle},",
        f"    title={{{title}}},",
        "    xtick={0.001,0.002,0.003,0.004,0.005},",
        "    scaled x ticks=false, x tick label style={/pgf/number format/fixed},",
        "    grid=both, grid style={gray!25},",
        f"    legend pos={legendpos}, legend cell align=left,",
        "    mark options={solid}, every axis plot/.append style={thick},",
        "]",
    ]
    for series, legend, color, mark in series_list:
        lines.append(f"\\addplot+[color={color}, mark={mark}, error bars/.cd, y dir=both, y explicit]")
        lines.append(f"    coordinates {{{coords(series)}}};")
        lines.append(f"\\addlegendentry{{{legend}}}")
    lines += ["\\end{semilogyaxis}", "\\end{tikzpicture}", "\\end{document}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)


def make_csv(path, ps, rows, meta):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["p", "tv", "ler_per_shot", "se_per_shot", "ul95_per_shot",
                    "ler_per_cycle", "se_per_cycle", "errors", "shots", "invalid", "ms_shot"])
        for p in ps:
            for tv in range(1, 6):
                r = rows.get((p, tv))
                if not r:
                    continue
                w.writerow([f"{p:.4f}", tv, f"{r['ler']:.4e}", f"{r['se']:.4e}",
                            f"{r['ul95']:.4e}", f"{r['ler'] / ROUNDS:.4e}",
                            f"{r['se'] / ROUNDS:.4e}", r["errors"], r["N"], r["invalid"],
                            meta.get(p, {}).get("ms_shot", "")])
    print("wrote", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    ps, data, rows, meta = parse(DATA)
    print(f"parsed {len(ps)} p-points: {ps}")

    title5 = r"Surface $d=11$ memory$_X$, filtered ($X$-basis): LER vs $p$ ($t_v=5$)"
    make_png(os.path.join(OUT, "ler_vs_p_filtered_tv5.png"), title5,
             [(data[5], r"filtered ($X$-basis), $t_v=5$", TV_MPL[5], "v")])
    make_tex(os.path.join(OUT, "ler_vs_p_filtered_tv5.tex"), title5,
             [(data[5], r"filtered ($X$-basis), $t_v=5$", "violet", "pentagon*")])

    titleL = r"Surface $d=11$ memory$_X$, filtered ($X$-basis): LER vs $p$"
    make_png(os.path.join(OUT, "ler_vs_p_filtered_tv1-5.png"), titleL,
             [(data[tv], f"$t_v={tv}$", TV_MPL[tv], TV_MARK[tv]) for tv in range(1, 6)])
    make_tex(os.path.join(OUT, "ler_vs_p_filtered_tv1-5.tex"), titleL,
             [(data[tv], f"$t_v={tv}$", TV_COLORS[tv], TV_TEXMARK[tv]) for tv in range(1, 6)])

    make_csv(os.path.join(OUT, "ler_vs_p_filtered.csv"), ps, rows, meta)


if __name__ == "__main__":
    main()
