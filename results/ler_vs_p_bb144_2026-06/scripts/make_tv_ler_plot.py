#!/usr/bin/env python3
"""Finalize the tv-sweep campaign and render the LER-vs-p figure.

Pools the bb144 d=12 circuit-level p-bit campaign shard logs in /tmp/tvsweep for
p in {0.002, 0.003, 0.004, 0.005} and target_valid (tv) = 1..5, then emits:

  * ler_vs_p_tv1-5.png   -- 5 curves (one per tv), x = p, y = LER (log scale)
  * ler_vs_p_tv1-5.tex   -- standalone pgfplots/TikZ version of the same figure
  * ler_vs_p_tv.csv      -- finalized per-(p, tv) table (N, errors, invalid, strict)
  * RESULT_FINAL.txt     -- human-readable finalized table

LER is the standard sinter/RelayBP convention (predicted != actual observables).
Error bars are 1-sigma Poisson (sqrt(E)/N). p=0.002 is the stopped high-res
extension (~15M shots); p=0.005 is the production-config fill-in run.
"""
from __future__ import annotations

import glob
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RE_PAIR = re.compile(r"\((\d+)/(\d+)\)")
RE_TV = re.compile(r"\btv=(\d+)")
RE_INV = re.compile(r"\binvalid=(\d+)")

# Physical error rate -> glob of shard logs to pool for that point.
SOURCES = {
    0.002: "/tmp/tvsweep/p0.002_r*_g*.log",
    0.003: "/tmp/tvsweep/p0.003_r*_g*.log",
    0.004: "/tmp/tvsweep/p0.004_r*_g*.log",
    0.005: "/tmp/tvsweep/p0.005_hires_g*.log",
}

OUTDIR = "/home/mert/p-bit_quantum_decoder/results/ler_vs_p_bb144_2026-06/plots"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["o", "s", "^", "D", "v"]


def pool(pattern: str) -> dict[int, list[int]]:
    """Return {tv: [errors, shots, invalid, errors_strict]} pooled over shards."""
    agg = {k: [0, 0, 0, 0] for k in range(1, 6)}
    for f in glob.glob(pattern):
        for line in open(f):
            if " tv=" not in line or "LER=" not in line:
                continue
            mtv = RE_TV.search(line)
            pairs = RE_PAIR.findall(line)
            if not mtv or not pairs:
                continue
            tv = int(mtv.group(1))
            if tv not in agg:
                continue
            e, n = int(pairs[0][0]), int(pairs[0][1])
            inv = int(RE_INV.search(line).group(1)) if RE_INV.search(line) else 0
            es = int(pairs[1][0]) if ("LER_strict" in line and len(pairs) >= 2) else e
            a = agg[tv]
            a[0] += e
            a[1] += n
            a[2] += inv
            a[3] += es
    return agg


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    data: dict[float, dict[int, dict]] = {}
    for p, pat in SOURCES.items():
        agg = pool(pat)
        data[p] = {}
        for tv in range(1, 6):
            E, N, INV, ES = agg[tv]
            ler = E / N if N else float("nan")
            se = math.sqrt(E) / N if (E and N) else 0.0
            data[p][tv] = dict(
                N=N, E=E, INV=INV, ES=ES, ler=ler, se=se,
                lerstrict=ES / N if N else float("nan"),
            )
    ps = sorted(data)

    # ---- finalized CSV + txt table ----
    with open(os.path.join(OUTDIR, "ler_vs_p_tv.csv"), "w") as fh:
        fh.write("p,target_valid,N,errors,invalid,errors_strict,LER,LER_se,LER_strict\n")
        for p in ps:
            for tv in range(1, 6):
                d = data[p][tv]
                fh.write(
                    f"{p},{tv},{d['N']},{d['E']},{d['INV']},{d['ES']},"
                    f"{d['ler']:.6e},{d['se']:.6e},{d['lerstrict']:.6e}\n"
                )
    result_txt = "/tmp/tvsweep/RESULT_FINAL.txt"
    with open(result_txt, "w") as fh:
        fh.write("bb144 d=12 circuit-level p-bit decoder -- FINALIZED LER vs p (tv=1..5)\n")
        fh.write("LER = standard (predicted != actual observables); strict fails all invalid.\n\n")
        for p in ps:
            n5 = data[p][5]["N"]
            fh.write(f"--- p={p}  N={n5} ---\n")
            for tv in range(1, 6):
                d = data[p][tv]
                fh.write(
                    f"  tv={tv}  LER={d['ler']:.3e} +-{d['se']:.1e}  "
                    f"errors={d['E']}/{d['N']}  invalid={d['INV']}  "
                    f"LER_strict={d['lerstrict']:.3e}\n"
                )
            fh.write("\n")

    # ---- PNG ----
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for i, tv in enumerate(range(1, 6)):
        xs = [p for p in ps if data[p][tv]["N"] > 0 and data[p][tv]["E"] > 0]
        ys = [data[p][tv]["ler"] for p in xs]
        es = [data[p][tv]["se"] for p in xs]
        ax.errorbar(
            xs, ys, yerr=es, color=COLORS[i], marker=MARKERS[i], ms=6,
            lw=1.6, capsize=3, label=f"$t_v={tv}$",
        )
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate")
    ax.set_title("bb144 [[144,12]] d=12 circuit-level: LER vs p")
    ax.set_xticks(ps)
    ax.ticklabel_format(axis="x", style="plain")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(title="solutions collected", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "ler_vs_p_tv1-5.png"), dpi=200)

    # ---- TikZ / pgfplots (standalone) ----
    lines = [
        r"\documentclass[tikz,border=3mm]{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.17}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{semilogyaxis}[",
        r"    width=12cm, height=9cm,",
        r"    xlabel={physical error rate $p$},",
        r"    ylabel={logical error rate},",
        r"    title={bb144 [[144,12]] $d=12$ circuit-level: LER vs $p$},",
        r"    xtick={0.002,0.003,0.004,0.005},",
        r"    scaled x ticks=false, x tick label style={/pgf/number format/fixed},",
        r"    grid=both, grid style={gray!25},",
        r"    legend pos=south east, legend cell align=left,",
        r"    mark options={solid}, every axis plot/.append style={thick},",
        r"]",
    ]
    tikz_colors = ["blue", "orange", "green!60!black", "red", "violet"]
    for i, tv in enumerate(range(1, 6)):
        coords = []
        for p in ps:
            d = data[p][tv]
            if d["N"] > 0 and d["E"] > 0:
                coords.append(f"({p:.3f}, {d['ler']:.4e}) +- (0, {d['se']:.4e})")
        if not coords:
            continue
        lines.append(
            f"\\addplot+[color={tikz_colors[i]}, mark=*, "
            r"error bars/.cd, y dir=both, y explicit]"
        )
        lines.append("    coordinates {" + " ".join(coords) + "};")
        lines.append(f"\\addlegendentry{{$t_v={tv}$}}")
    lines += [r"\end{semilogyaxis}", r"\end{tikzpicture}", r"\end{document}", ""]
    with open(os.path.join(OUTDIR, "ler_vs_p_tv1-5.tex"), "w") as fh:
        fh.write("\n".join(lines))

    # ---- console summary ----
    print("FINALIZED LER vs p (standard LER, tv=1..5):")
    print(f"{'p':>7} {'N':>10} | " + "  ".join(f"tv{tv}" for tv in range(1, 6)))
    for p in ps:
        row = "  ".join(f"{data[p][tv]['ler']:.2e}" for tv in range(1, 6))
        print(f"{p:>7} {data[p][5]['N']:>10} | {row}")
    print(f"\nWrote PNG, TikZ, CSV to {OUTDIR}")
    print(f"Wrote {result_txt}")


if __name__ == "__main__":
    main()
