#!/usr/bin/env python3
"""Generate LER / cost plots for the bb144 p-bit decoder campaign.

Four figures (each saved standalone, plus a 2x2 combined panel):
  1. p vs LER                 (end-to-end, per shot, 12 rounds, 12 observables)
  2. p vs LER/12              (per QEC cycle:        rounds=12)
  3. p vs LER/144             (per cycle per logical qubit: 12 rounds x 12 qubits)
  4. p vs cost                (single-GPU decode latency, ms/shot)

Data are the finalized 50-error campaign points (p=0.003,0.004,0.005); p=0.002
is shown as a provisional point (autonomous 10-error job still running).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "plots"
OUT.mkdir(exist_ok=True)

ROUNDS = 12          # QEC cycles in the memory experiment
LOGICAL_QUBITS = 12  # k for the [[144,12,12]] gross code

# --- finalized points (errors/shots -> LER, stderr) -------------------------
# p=0.002: 10 errors / 1,060,000 shots; others 50-error campaigns.
p_final = np.array([0.002, 0.003, 0.004, 0.005])
ler_final = np.array([9.434e-6, 7.85e-5, 1.08e-3, 1.05e-2])
err_final = np.array([2.98e-6, 1.1e-5, 1.3e-4, 1.3e-3])  # stderr on LER

# --- provisional point (none; p=0.002 is now finalized) ---------------------
p_prov = np.array([])
ler_prov = np.array([])
err_prov = np.array([])

# --- per-shot decode cost (single-GPU latency, ms/shot) ---------------------
p_cost = np.array([0.002, 0.003, 0.004, 0.005])
cost_ms = np.array([72.0, 100.0, 236.0, 489.0])

BLUE = "#1f77b4"
GREY = "#888888"


def _ler_axis(ax, divisor, ylabel, title):
    yf = ler_final / divisor
    ef = err_final / divisor
    yp = ler_prov / divisor
    ep = err_prov / divisor
    ax.errorbar(
        p_final, yf, yerr=ef, fmt="o-", color=BLUE, capsize=3,
        lw=1.8, ms=7, label="measured (≥10-error campaigns)",
    )
    if p_prov.size:
        ax.errorbar(
            p_prov, yp, yerr=ep, fmt="o", mfc="white", mec=GREY, ecolor=GREY,
            capsize=3, ms=7, label="provisional (job running)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate  p")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")


def _cost_axis(ax):
    ax.plot(p_cost, cost_ms, "s-", color="#d62728", lw=1.8, ms=7)
    for x, y in zip(p_cost, cost_ms):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("physical error rate  p")
    ax.set_ylabel("decode cost  (ms / shot, single A100)")
    ax.set_title("p vs decode cost  (chains=128)")
    ax.grid(True, which="both", ls=":", alpha=0.5)


SPECS = [
    ("ler", 1, "Logical error rate per shot (12 rounds)",
     "p vs LER  (end-to-end, gross code)", "01_p_vs_ler.png"),
    ("ler_per_cycle", ROUNDS, "LER per QEC cycle  (LER / 12)",
     "p vs LER per cycle  (gross code)", "02_p_vs_ler_per_cycle.png"),
    ("ler_per_cycle_qubit", ROUNDS * LOGICAL_QUBITS,
     "LER per cycle per logical qubit  (LER / 144)",
     "p vs LER per cycle per logical qubit  (gross code)",
     "03_p_vs_ler_per_cycle_per_qubit.png"),
]


def main() -> None:
    # standalone LER figures
    for _key, div, ylabel, title, fname in SPECS:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        _ler_axis(ax, div, ylabel, title)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=150)
        plt.close(fig)

    # standalone cost figure
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    _cost_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "04_p_vs_cost.png", dpi=150)
    plt.close(fig)

    # combined 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _ler_axis(axes[0, 0], SPECS[0][1], SPECS[0][2], SPECS[0][3])
    _ler_axis(axes[0, 1], SPECS[1][1], SPECS[1][2], SPECS[1][3])
    _ler_axis(axes[1, 0], SPECS[2][1], SPECS[2][2], SPECS[2][3])
    _cost_axis(axes[1, 1])
    fig.suptitle("bb144 [[144,12,12]] memory_X — p-bit ColoredBlockGibbsDecoder",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "00_combined.png", dpi=150)
    plt.close(fig)

    print(f"wrote plots to {OUT}")
    for f in sorted(OUT.glob("*.png")):
        print(" -", f.name)


if __name__ == "__main__":
    main()
