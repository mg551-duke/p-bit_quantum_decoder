"""Generate publication-oriented plots from experiment CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_plots(results_dir: Path, figures_dir: Path) -> List[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        paths.extend(
            [
                _line_plot(
                    summary,
                    figures_dir / "logical_error_rate.png",
                    y="logical_error_rate",
                    ylabel="logical error rate",
                    title="Logical error rate",
                    y_low="logical_error_rate_ci_low",
                    y_high="logical_error_rate_ci_high",
                ),
                _line_plot(
                    summary,
                    figures_dir / "exact_coset_agreement.png",
                    y="exact_agreement",
                    ylabel="agreement with exact max coset",
                    title="Exact coset agreement",
                    y_low="exact_agreement_ci_low",
                    y_high="exact_agreement_ci_high",
                ),
                _line_plot(
                    summary,
                    figures_dir / "valid_sample_rate.png",
                    y="mean_valid_sample_rate",
                    ylabel="valid sample rate",
                    title="Syndrome-valid sample rate",
                ),
                _line_plot(
                    summary,
                    figures_dir / "distinct_logical_classes.png",
                    y="mean_distinct_logicals",
                    ylabel="distinct logical classes",
                    title="Logical classes visited",
                ),
                _line_plot(
                    summary,
                    figures_dir / "coset_entropy.png",
                    y="mean_coset_entropy",
                    ylabel="coset entropy (bits)",
                    title="Empirical coset entropy",
                ),
            ]
        )
        optional_plots = [
            (
                "first_valid_rate",
                "first_valid_rate.png",
                "first-valid rate",
                "First-valid convergence rate",
            ),
            (
                "mean_first_valid_sweep",
                "first_valid_sweep.png",
                "mean first-valid sweep",
                "First-valid convergence sweep",
            ),
            (
                "mean_best_residual_weight",
                "best_residual_weight.png",
                "mean best residual weight",
                "Best residual weight",
            ),
            (
                "mean_final_residual_weight",
                "final_residual_weight.png",
                "mean final residual weight",
                "Final residual weight",
            ),
            (
                "mean_distinct_valid_solutions",
                "distinct_valid_solutions.png",
                "mean distinct valid solutions",
                "Distinct valid solutions",
            ),
            (
                "mean_selected_weight",
                "selected_weight.png",
                "mean selected correction weight",
                "Selected correction weight",
            ),
        ]
        for column, filename, ylabel, title in optional_plots:
            if column in summary.columns and not summary[column].dropna().empty:
                paths.append(
                    _line_plot(
                        summary,
                        figures_dir / filename,
                        y=column,
                        ylabel=ylabel,
                        title=title,
                    )
                )
    hard_recovery = results_dir / "hard_case_recovery.csv"
    hard_summary = results_dir / "hard_case_summary.csv"
    if hard_recovery.exists() and hard_recovery.stat().st_size:
        recovery = pd.read_csv(hard_recovery)
        if not recovery.empty:
            summary = None
            if hard_summary.exists() and hard_summary.stat().st_size:
                summary = pd.read_csv(hard_summary)
            paths.append(_hard_case_plot(recovery, figures_dir / "hard_case_recovery.png", summary))
    if not paths:
        raise FileNotFoundError(f"No supported CSV inputs found in {results_dir}")
    _write_plot_manifest(paths, figures_dir)
    return paths


def _line_plot(
    summary: pd.DataFrame,
    path: Path,
    y: str,
    ylabel: str,
    title: str,
    y_low: Optional[str] = None,
    y_high: Optional[str] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    data = summary.dropna(subset=[y]).copy()
    for (family, size, decoder), group in data.groupby(["code_family", "code_size", "decoder"]):
        group = group.sort_values("p")
        ax.plot(
            group["p"],
            group[y],
            marker="o",
            linewidth=1.4,
            markersize=3.2,
            label=f"{family} {size} {decoder}",
        )
        if y_low and y_high and {y_low, y_high}.issubset(group.columns):
            band = group.dropna(subset=[y_low, y_high])
            if not band.empty:
                ax.fill_between(
                    band["p"].to_numpy(),
                    band[y_low].to_numpy(),
                    band[y_high].to_numpy(),
                    alpha=0.08,
                )
    ax.set_title(title)
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _hard_case_plot(
    recovery: pd.DataFrame,
    path: Path,
    summary: Optional[pd.DataFrame] = None,
) -> Path:
    if summary is not None and not summary.empty:
        grouped = summary.rename(columns={"recovered_exact": "recovery_rate"}).copy()
    else:
        grouped = (
            recovery.groupby(["code_family", "code_size", "p", "decoder"], dropna=False)
            .agg(recovery_rate=("recovered_exact", "mean"), cases=("syndrome_id", "nunique"))
            .reset_index()
        )
    group_columns = ["code_family", "code_size", "p"]
    grouped = grouped.sort_values(group_columns + ["decoder"])
    cases = grouped[group_columns].drop_duplicates().reset_index(drop=True)
    decoders = sorted(grouped["decoder"].unique())
    indexed = grouped.set_index(group_columns + ["decoder"], drop=False)

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    x = np.arange(len(cases))
    width = min(0.78 / max(len(decoders), 1), 0.25)
    colors = {
        "nullspace_coset_sampler": "#64748b",
        "pbit_coset_sampler": "#0f766e",
        "pbit_nullspace_sampler": "#2563eb",
        "pbit_annealed_optimizer": "#8a5a12",
    }
    ci_columns = {"recovered_exact_ci_low", "recovered_exact_ci_high"}

    for decoder_index, decoder in enumerate(decoders):
        values = []
        lower = []
        upper = []
        for case in cases.itertuples(index=False):
            key = (case.code_family, case.code_size, case.p, decoder)
            row = indexed.loc[key]
            values.append(row["recovery_rate"])
            if ci_columns.issubset(grouped.columns):
                lower.append(row["recovery_rate"] - row["recovered_exact_ci_low"])
                upper.append(row["recovered_exact_ci_high"] - row["recovery_rate"])
        offset = (decoder_index - (len(decoders) - 1) / 2) * width
        positions = x + offset
        ax.bar(
            positions,
            values,
            width=width,
            label=decoder,
            color=colors.get(decoder, "#3b82f6"),
        )
        if lower and upper:
            ax.errorbar(
                positions,
                values,
                yerr=[np.asarray(lower), np.asarray(upper)],
                fmt="none",
                ecolor="#172033",
                elinewidth=0.8,
                capsize=2.0,
            )

    labels = [
        f"{row.code_family} {row.code_size}\np={row.p:g}" for row in cases.itertuples(index=False)
    ]
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("hard-case exact-coset recovery")
    ax.set_title("Hard-case recovery behavior")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_plot_manifest(paths: Sequence[Path], figures_dir: Path) -> None:
    lines = ["# Figure Manifest", ""]
    lines.extend(f"- `{path.name}`" for path in paths)
    (figures_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    args = parser.parse_args(argv)
    for path in make_plots(args.results, args.figures):
        print(path)


if __name__ == "__main__":
    main()
