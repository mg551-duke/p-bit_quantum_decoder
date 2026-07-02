"""Publication plots for the annealing-schedule tuning sweeps.

Consumes a ``schedule_summary.csv`` produced by :mod:`pbit_qec.schedule_tuning`
and renders, as available from the swept grid:

- ``schedule_shapes.png`` -- the geometric/linear/sigmoid beta ramps (what the
  "temperature schedule" actually does over sweeps);
- ``ler_vs_convergence.png`` -- the logical-error-rate vs convergence-speed
  (first-valid-sweep p95) trade-off, the central tuning figure;
- ``rho_sensitivity.png`` -- logical error rate vs ``rho_scale`` (when swept);
- ``ler_vs_budget.png`` -- logical error rate vs sweep budget per ``beta_start``
  (when the sweep count is varied), visualizing the convergence/cost win.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pbit_qec.decoders import _annealing_schedule  # noqa: E402

_SHAPE_COLORS = {
    "geometric": "#2563eb",
    "linear": "#0f766e",
    "sigmoid": "#b45309",
}
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _beta_label(start: float, stop: float) -> str:
    return f"{start:g}-{stop:g}"


def make_schedule_plots(
    summary_path: Path,
    figures_dir: Path,
    demo_betas: Optional[Sequence[float]] = None,
) -> List[Path]:
    summary = pd.read_csv(summary_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    if demo_betas is None:
        demo_betas = (
            float(summary["beta_start"].median()),
            float(summary["beta_stop"].median()),
        )
    demo_steps = int(summary["sweeps"].median())
    paths.append(_shape_demo_plot(figures_dir / "schedule_shapes.png", demo_betas, demo_steps))
    paths.append(_ler_vs_convergence_plot(summary, figures_dir / "ler_vs_convergence.png"))

    if summary["rho_scale"].nunique() >= 2:
        paths.append(_rho_sensitivity_plot(summary, figures_dir / "rho_sensitivity.png"))
    if summary["sweeps"].nunique() >= 2:
        paths.append(_ler_vs_budget_plot(summary, figures_dir / "ler_vs_budget.png"))

    _write_manifest(paths, figures_dir)
    return paths


def _shape_demo_plot(path: Path, betas: Sequence[float], steps: int) -> Path:
    start, stop = float(betas[0]), float(betas[1])
    steps = max(int(steps), 8)
    sweeps = np.arange(steps)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for shape, color in _SHAPE_COLORS.items():
        ax.plot(
            sweeps,
            _annealing_schedule(start, stop, steps, shape),
            label=shape,
            color=color,
            linewidth=1.8,
        )
    ax.set_title(f"Annealing schedules (beta {start:g} -> {stop:g}, {steps} sweeps)")
    ax.set_xlabel("sweep")
    ax.set_ylabel("inverse temperature beta")
    ax.grid(True, alpha=0.3)
    ax.legend(title="shape")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _ler_vs_convergence_plot(summary: pd.DataFrame, path: Path) -> Path:
    p_values = sorted(summary["p"].unique())
    fig, axes = plt.subplots(
        1, len(p_values), figsize=(6.0 * len(p_values), 4.8), squeeze=False
    )
    beta_pairs = sorted(
        {_beta_label(r.beta_start, r.beta_stop) for r in summary.itertuples(index=False)}
    )
    marker_for = {pair: _MARKERS[i % len(_MARKERS)] for i, pair in enumerate(beta_pairs)}
    for col, p in enumerate(p_values):
        ax = axes[0][col]
        sub = summary[summary["p"] == p]
        for row in sub.itertuples(index=False):
            ax.scatter(
                row.fvs_p95,
                row.ler,
                color=_SHAPE_COLORS.get(row.shape, "#64748b"),
                marker=marker_for[_beta_label(row.beta_start, row.beta_stop)],
                s=46,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )
        ax.set_title(f"p = {p:g}")
        ax.set_xlabel("first-valid sweep (p95) — faster convergence ->")
        ax.set_ylabel("logical error rate")
        ax.grid(True, alpha=0.3)
    shape_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=shape)
        for shape, color in _SHAPE_COLORS.items()
    ]
    beta_handles = [
        plt.Line2D([], [], marker=marker_for[pair], linestyle="", color="#475569", label=pair)
        for pair in beta_pairs
    ]
    first = axes[0][0].legend(handles=shape_handles, title="shape", loc="upper right", fontsize=8)
    axes[0][0].add_artist(first)
    axes[0][-1].legend(handles=beta_handles, title="beta start-stop", loc="upper right", fontsize=8)
    fig.suptitle("Logical error rate vs convergence speed")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _rho_sensitivity_plot(summary: pd.DataFrame, path: Path) -> Path:
    p_values = sorted(summary["p"].unique())
    fig, axes = plt.subplots(
        1, len(p_values), figsize=(6.0 * len(p_values), 4.8), squeeze=False
    )
    shapes = sorted(summary["shape"].unique())
    preferred = "geometric" if "geometric" in shapes else shapes[0]
    view = summary[summary["shape"] == preferred]
    beta_pairs = sorted(
        {_beta_label(r.beta_start, r.beta_stop) for r in view.itertuples(index=False)}
    )
    for col, p in enumerate(p_values):
        ax = axes[0][col]
        sub = view[view["p"] == p]
        for pair in beta_pairs:
            start, stop = (float(value) for value in pair.split("-"))
            line = sub[(sub["beta_start"] == start) & (sub["beta_stop"] == stop)]
            line = line.sort_values("rho_scale")
            if line.empty:
                continue
            yerr = [
                (line["ler"] - line["ler_ci_low"]).to_numpy(),
                (line["ler_ci_high"] - line["ler"]).to_numpy(),
            ]
            ax.errorbar(
                line["rho_scale"].to_numpy(),
                line["ler"].to_numpy(),
                yerr=yerr,
                marker="o",
                capsize=2.5,
                linewidth=1.4,
                label=pair,
            )
        ax.set_title(f"p = {p:g} ({preferred})")
        ax.set_xlabel("rho_scale")
        ax.set_ylabel("logical error rate")
        ax.grid(True, alpha=0.3)
        ax.legend(title="beta start-stop", fontsize=8)
    fig.suptitle("rho sensitivity (Wilson 95% intervals)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _ler_vs_budget_plot(summary: pd.DataFrame, path: Path) -> Path:
    p_values = sorted(summary["p"].unique())
    fig, axes = plt.subplots(
        1, len(p_values), figsize=(6.0 * len(p_values), 4.8), squeeze=False
    )
    beta_starts = sorted(summary["beta_start"].unique())
    for col, p in enumerate(p_values):
        ax = axes[0][col]
        sub = summary[summary["p"] == p]
        for start in beta_starts:
            line = sub[sub["beta_start"] == start].sort_values("sweeps")
            if line.empty:
                continue
            yerr = [
                (line["ler"] - line["ler_ci_low"]).to_numpy(),
                (line["ler_ci_high"] - line["ler"]).to_numpy(),
            ]
            ax.errorbar(
                line["sweeps"].to_numpy(),
                line["ler"].to_numpy(),
                yerr=yerr,
                marker="o",
                capsize=2.5,
                linewidth=1.4,
                label=f"beta_start={start:g}",
            )
        ax.set_title(f"p = {p:g}")
        ax.set_xlabel("sweeps (annealing budget)")
        ax.set_ylabel("logical error rate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Logical error rate vs sweep budget")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_manifest(paths: Sequence[Path], figures_dir: Path) -> None:
    lines = ["# Schedule Figure Manifest", ""]
    lines.extend(f"- `{path.name}`" for path in paths)
    (figures_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    parser.add_argument(
        "--demo-beta",
        nargs=2,
        type=float,
        default=None,
        metavar=("START", "STOP"),
        help="beta endpoints for the schedule-shape illustration",
    )
    args = parser.parse_args(argv)
    for path in make_schedule_plots(args.summary, args.figures, args.demo_beta):
        print(path)


if __name__ == "__main__":
    main()
