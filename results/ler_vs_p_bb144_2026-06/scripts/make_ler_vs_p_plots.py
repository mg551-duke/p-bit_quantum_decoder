#!/usr/bin/env python3
"""Generate LER-vs-p plots (p = 0.001 .. 0.005) as PNG + standalone pgfplots .tex.

bb144 [[144,12,12]] d=12 circuit-level noise, p-bit ColoredBlockGibbsDecoder.

Two decoding problems:
  * filtered   : memory_Z, Z-basis detector filter (936 det / 8784 mech), rho=0.20, sweeps=750
  * correlated : memory_X, full DEM            (1728 det / 67824 mech), rho=0.30, sweeps=500

Data sources:
  p=0.002..0.005  finalized  -> FILTERED_RESULT.txt (filtered), RESULT_FINAL.txt (correlated)
  p=0.001         live campaign snapshot (logs/p001) -- PRELIMINARY (campaign ongoing 2026-06-25)

Outputs (in ../plots):
  ler_vs_p_compare_tv5.{png,tex}        filtered vs correlated at t_v=5 (headline)
  ler_vs_p_filtered_tv1-5.{png,tex}     filtered decoder, full t_v ladder
  ler_vs_p_correlated_tv1-5.{png,tex}   correlated decoder, full t_v ladder
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "plots")
P = [0.001, 0.002, 0.003, 0.004, 0.005]
ROUNDS = 12  # memory experiment cycles; LER-per-cycle = LER / ROUNDS
P001_NOTE = "$p=0.001$ preliminary (campaign ongoing 2026-06-25)"

# (LER, stderr) per t_v, aligned with P.  p=0.001 row is the live-campaign snapshot.
FILTERED = {
    1: [(7.813e-7, 2.357e-7), (1.807e-5, 3.0e-6), (4.740e-4, 3.5e-5), (6.391e-3, 3.2e-4), (4.281e-2, 2.6e-3)],
    2: [(7.813e-7, 2.357e-7), (1.660e-5, 2.8e-6), (3.906e-4, 3.2e-5), (5.484e-3, 2.9e-4), (3.828e-2, 2.4e-3)],
    3: [(6.392e-7, 2.131e-7), (1.514e-5, 2.7e-6), (3.620e-4, 3.1e-5), (5.156e-3, 2.8e-4), (3.516e-2, 2.3e-3)],
    4: [(6.392e-7, 2.131e-7), (1.318e-5, 2.5e-6), (3.516e-4, 3.0e-5), (5.000e-3, 2.8e-4), (3.375e-2, 2.3e-3)],
    5: [(6.392e-7, 2.131e-7), (1.123e-5, 2.3e-6), (3.203e-4, 2.9e-5), (4.672e-3, 2.7e-4), (3.219e-2, 2.2e-3)],
}
CORRELATED = {
    1: [(7.429e-7, 3.714e-7), (2.7443e-6, 4.2859e-7), (5.3333e-5, 9.4281e-6), (1.0521e-3, 1.0469e-4), (1.0000e-2, 7.4536e-4)],
    2: [(7.429e-7, 3.714e-7), (2.4766e-6, 4.0715e-7), (3.1667e-5, 7.2648e-6), (7.5000e-4, 8.8388e-5), (9.0000e-3, 7.0711e-4)],
    3: [(7.429e-7, 3.714e-7), (2.2758e-6, 3.9029e-7), (3.3333e-5, 7.4536e-6), (6.3542e-4, 8.1357e-5), (8.6667e-3, 6.9389e-4)],
    4: [(7.429e-7, 3.714e-7), (2.2088e-6, 3.8451e-7), (2.8333e-5, 6.8718e-6), (5.7292e-4, 7.7252e-5), (8.3889e-3, 6.8268e-4)],
    5: [(7.429e-7, 3.714e-7), (2.2088e-6, 3.8451e-7), (2.5000e-5, 6.4550e-6), (5.5208e-4, 7.5834e-5), (7.7222e-3, 6.5499e-4)],
}

TV_COLORS = {1: "blue", 2: "orange", 3: "green!60!black", 4: "red", 5: "violet"}
TV_MPL = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728", 5: "#9467bd"}
TV_MARK = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}


# ----------------------------------------------------------------------------- PNG
def png_series(ax, series, label, color, marker):
    ler = [v[0] / ROUNDS for v in series]
    err = [v[1] / ROUNDS for v in series]
    ax.errorbar(P, ler, yerr=err, label=label, color=color, marker=marker,
                markersize=6, linewidth=1.8, capsize=3, markerfacecolor="white",
                markeredgewidth=1.4)
    # solid marker overlay for finalized points (p>=0.002); p=0.001 left hollow
    ax.plot(P[1:], ler[1:], color=color, marker=marker, markersize=6, linestyle="none")


def make_png(path, title, plots):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for series, label, color, marker in plots:
        png_series(ax, series, label, color, marker)
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate per cycle")
    ax.set_title(title)
    ax.set_xticks(P)
    ax.grid(True, which="both", color="0.85", linewidth=0.6)
    ax.axvline(0.001, color="0.7", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(0.02, 0.02, P001_NOTE.replace("$", ""), transform=ax.transAxes,
            fontsize=7, color="0.4", ha="left", va="bottom")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------------- TeX
def coords(series):
    return " ".join(f"({p:.3f}, {l / ROUNDS:.4e}) +- (0, {e / ROUNDS:.4e})"
                    for p, (l, e) in zip(P, series))


def make_tex(path, title, addplots, legendpos="north west"):
    lines = [
        "% Standalone pgfplots figure -- compile directly with pdflatex/lualatex,",
        "% or \\input{<this file>} into a document that loads \\usepackage{pgfplots}.",
        "% bb144 d=12 circuit-level LER vs p (p-bit ColoredBlockGibbsDecoder).",
        f"% NOTE: {P001_NOTE.replace('$','')}",
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
    for series, legend, color, mark in addplots:
        lines.append(
            f"\\addplot+[color={color}, mark={mark}, error bars/.cd, y dir=both, y explicit]")
        lines.append(f"    coordinates {{{coords(series)}}};")
        lines.append(f"\\addlegendentry{{{legend}}}")
    lines += [
        "\\node[anchor=south west, font=\\scriptsize, text=gray]",
        f"    at (rel axis cs:0.02,0.02) {{{P001_NOTE}}};",
        "\\end{semilogyaxis}",
        "\\end{tikzpicture}",
        "\\end{document}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1) headline comparison at t_v=5
    title = "bb144 $[[144,12,12]]$ $d=12$ circuit-level: LER vs $p$ ($t_v=5$)"
    make_png(os.path.join(OUT, "ler_vs_p_compare_tv5.png"), title, [
        (FILTERED[5], "filtered (mem. Z, $Z$-basis)", TV_MPL[1], "o"),
        (CORRELATED[5], "correlated (mem. X, full DEM)", TV_MPL[4], "s"),
    ])
    make_tex(os.path.join(OUT, "ler_vs_p_compare_tv5.tex"), title, [
        (FILTERED[5], "filtered (mem.\\,Z, $Z$-basis)", "blue", "*"),
        (CORRELATED[5], "correlated (mem.\\,X, full DEM)", "red", "square*"),
    ])

    # 2) filtered t_v ladder
    title = "bb144 $d=12$ filtered (mem. Z, $Z$-basis): LER vs $p$"
    make_png(os.path.join(OUT, "ler_vs_p_filtered_tv1-5.png"), title,
             [(FILTERED[tv], f"$t_v={tv}$", TV_MPL[tv], TV_MARK[tv]) for tv in range(1, 6)])
    make_tex(os.path.join(OUT, "ler_vs_p_filtered_tv1-5.tex"), title,
             [(FILTERED[tv], f"$t_v={tv}$", TV_COLORS[tv], "*") for tv in range(1, 6)])

    # 3) correlated t_v ladder
    title = "bb144 $d=12$ correlated (mem. X, full DEM): LER vs $p$"
    make_png(os.path.join(OUT, "ler_vs_p_correlated_tv1-5.png"), title,
             [(CORRELATED[tv], f"$t_v={tv}$", TV_MPL[tv], TV_MARK[tv]) for tv in range(1, 6)])
    make_tex(os.path.join(OUT, "ler_vs_p_correlated_tv1-5.tex"), title,
             [(CORRELATED[tv], f"$t_v={tv}$", TV_COLORS[tv], "*") for tv in range(1, 6)])


if __name__ == "__main__":
    main()
