"""Mine and benchmark syndromes where min-energy and max-coset disagree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import pbit_qec.gf2 as gf2
from pbit_qec.analysis import analyze_cosets
from pbit_qec.codes import make_code
from pbit_qec.experiments import build_decoder, load_config
from pbit_qec.simulation import required_trials_for_wilson_half_width, wilson_interval

HARD_CASE_COLUMNS = [
    "code_family",
    "code_size",
    "p",
    "syndrome_id",
    "exact_best_coset",
    "min_energy_coset",
    "exact_coset_weights_json",
    "min_energy_by_coset_json",
    "posterior_gap",
    "energy_gap",
    "valid_count",
    "nullity",
]

RECOVERY_COLUMNS = [
    "code_family",
    "code_size",
    "p",
    "syndrome_id",
    "decoder",
    "repeat",
    "sampled_coset",
    "exact_best_coset",
    "min_energy_coset",
    "recovered_exact",
    "matched_min_energy",
    "valid",
    "valid_samples",
    "total_samples",
    "sampled_coset_counts_json",
]


def run_hard_cases(
    config: Mapping[str, object],
    config_path: Optional[Path] = None,
) -> Dict[str, Path]:
    output_dir = Path(str(config.get("output_dir", "results/run")))
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = config.get("decoder_settings", {})
    if not isinstance(settings, Mapping):
        raise ValueError("decoder_settings must be an object")
    p_values = [float(p) for p in config.get("p_values", [])]
    max_nullity = int(settings.get("exact_coset", {}).get("max_nullity", 24))  # type: ignore[union-attr]
    max_cases_per_sweep = int(config.get("max_cases_per_sweep", 12))
    recovery_repeats = int(config.get("recovery_repeats", 8))
    base_seed = int(config.get("seed", 4444))
    recovery_decoders = [str(name) for name in config.get("recovery_decoders", [])]

    hard_rows: List[Dict[str, object]] = []
    recovery_rows: List[Dict[str, object]] = []

    for sweep_idx, sweep in enumerate(config.get("sweeps", [])):
        if not isinstance(sweep, Mapping):
            raise ValueError("each sweep must be an object")
        code = make_code(str(sweep["family"]), int(sweep["size"]))
        if code.n - gf2.rank(code.H) > max_nullity:
            continue
        max_syndromes = min(int(sweep.get("max_syndromes", 1 << code.m)), 1 << code.m)
        for p_idx, p in enumerate(p_values):
            found: List[Dict[str, object]] = []
            analyses = {}
            for syndrome_id in range(max_syndromes):
                syndrome = gf2.unpack_bits(syndrome_id, code.m)
                try:
                    analysis = analyze_cosets(code, syndrome, p, max_nullity=max_nullity)
                except ValueError:
                    continue
                analyses[syndrome_id] = analysis
                if analysis.best_coset == analysis.min_energy_coset:
                    continue
                row = {
                    "code_family": code.family,
                    "code_size": code.size,
                    "p": p,
                    "syndrome_id": syndrome_id,
                    "exact_best_coset": analysis.best_coset,
                    "min_energy_coset": analysis.min_energy_coset,
                    "exact_coset_weights_json": json.dumps(
                        analysis.coset_log_weights, sort_keys=True
                    ),
                    "min_energy_by_coset_json": json.dumps(
                        analysis.min_energy_by_coset, sort_keys=True
                    ),
                    "posterior_gap": analysis.posterior_gap,
                    "energy_gap": analysis.energy_gap,
                    "valid_count": analysis.valid_count,
                    "nullity": analysis.nullity,
                }
                found.append(row)
            found.sort(key=lambda row: (float(row["posterior_gap"]), -float(row["energy_gap"])))
            selected = found[:max_cases_per_sweep]
            hard_rows.extend(selected)

            for case_idx, row in enumerate(selected):
                syndrome_id = int(row["syndrome_id"])
                syndrome = gf2.unpack_bits(syndrome_id, code.m)
                analysis = analyses[syndrome_id]
                for decoder_name in recovery_decoders:
                    decoder = build_decoder(decoder_name, settings)
                    for repeat in range(recovery_repeats):
                        seed = (
                            base_seed
                            + sweep_idx * 1_000_000
                            + p_idx * 10_000
                            + case_idx * 100
                            + repeat
                        )
                        result = decoder.decode(code, syndrome, p, np.random.default_rng(seed))
                        sampled_coset = gf2.bit_string(result.logical_class)
                        recovery_rows.append(
                            {
                                "code_family": code.family,
                                "code_size": code.size,
                                "p": p,
                                "syndrome_id": syndrome_id,
                                "decoder": decoder.name,
                                "repeat": repeat,
                                "sampled_coset": sampled_coset,
                                "exact_best_coset": analysis.best_coset,
                                "min_energy_coset": analysis.min_energy_coset,
                                "recovered_exact": sampled_coset == analysis.best_coset,
                                "matched_min_energy": sampled_coset == analysis.min_energy_coset,
                                "valid": result.valid,
                                "valid_samples": result.valid_samples,
                                "total_samples": result.total_samples,
                                "sampled_coset_counts_json": json.dumps(
                                    result.coset_counts, sort_keys=True
                                ),
                            }
                        )

    hard_path = output_dir / "hard_cases.csv"
    recovery_path = output_dir / "hard_case_recovery.csv"
    summary_path = output_dir / "hard_case_summary.csv"
    pd.DataFrame(hard_rows, columns=HARD_CASE_COLUMNS).to_csv(hard_path, index=False)
    recovery_df = pd.DataFrame(recovery_rows, columns=RECOVERY_COLUMNS)
    recovery_df.to_csv(recovery_path, index=False)
    summarize_hard_case_recovery(recovery_df).to_csv(summary_path, index=False)
    manifest_path = output_dir / "hard_cases_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config_path": str(config_path) if config_path else None,
                "hard_cases": len(hard_rows),
                "recovery_rows": len(recovery_rows),
                "summary_rows": int(len(summarize_hard_case_recovery(recovery_df))),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "hard_cases": hard_path,
        "hard_case_recovery": recovery_path,
        "hard_case_summary": summary_path,
    }


def summarize_hard_case_recovery(recovery: pd.DataFrame) -> pd.DataFrame:
    if recovery.empty:
        return pd.DataFrame(
            columns=[
                "code_family",
                "code_size",
                "p",
                "decoder",
                "attempts",
                "hard_cases",
                "recovered_exact",
                "recovered_exact_ci_low",
                "recovered_exact_ci_high",
                "recovered_exact_ci_half_width",
                "valid_rate",
                "matched_min_energy_rate",
                "mean_valid_samples",
                "target_attempts_5pct_ci",
                "additional_attempts_5pct_ci",
            ]
        )
    group_cols = ["code_family", "code_size", "p", "decoder"]
    rows = []
    for keys, group in recovery.groupby(group_cols, dropna=False):
        attempts = int(len(group))
        recovered = int(group["recovered_exact"].sum())
        low, high = wilson_interval(recovered, attempts)
        target_attempts = required_trials_for_wilson_half_width(
            recovered / attempts if attempts else 0.5
        )
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "attempts": attempts,
                "hard_cases": int(group["syndrome_id"].nunique()),
                "recovered_exact": recovered / attempts if attempts else float("nan"),
                "recovered_exact_ci_low": low,
                "recovered_exact_ci_high": high,
                "recovered_exact_ci_half_width": (high - low) / 2.0,
                "valid_rate": float(group["valid"].mean()),
                "matched_min_energy_rate": float(group["matched_min_energy"].mean()),
                "mean_valid_samples": float(group["valid_samples"].mean()),
                "target_attempts_5pct_ci": target_attempts,
                "additional_attempts_5pct_ci": max(0, target_attempts - attempts),
            }
        )
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = run_hard_cases(load_config(args.config), args.config)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
