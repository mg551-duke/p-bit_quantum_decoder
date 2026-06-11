import json
from pathlib import Path

import pandas as pd

from pbit_qec.experiments import load_config, run_config
from pbit_qec.hard_cases import run_hard_cases
from pbit_qec.plotting import make_plots


def test_smoke_experiment_csv_schema_and_plots(tmp_path: Path) -> None:
    config = load_config(Path("experiments/configs/v1_smoke.json"))
    config["output_dir"] = str(tmp_path / "results")

    paths = run_config(config)

    trials = pd.read_csv(paths["trials"])
    summary = pd.read_csv(paths["summary"])
    cosets = pd.read_csv(paths["coset_distributions"])
    assert {"decoder", "logical_success", "valid_sample_rate"}.issubset(trials.columns)
    syndrome_rows = trials[trials["decoder"] == "pbit_syndrome"]
    assert not syndrome_rows.empty
    syndrome_metadata = syndrome_rows["metadata_json"].map(json.loads)
    assert all(row["used_linear_solve"] is False for row in syndrome_metadata)
    assert all("final_residual_weight" in row for row in syndrome_metadata)
    assert {
        "logical_error_rate",
        "logical_error_rate_ci_low",
        "logical_error_rate_ci_high",
        "logical_error_rate_ci_half_width",
        "logical_error_target_trials_5pct_ci",
        "logical_error_additional_trials_5pct_ci",
        "mean_runtime_s",
        "se_runtime_s",
        "first_valid_rate",
        "mean_first_valid_sweep",
        "mean_best_residual_weight",
        "mean_final_residual_weight",
        "mean_distinct_valid_solutions",
        "mean_selected_weight",
    }.issubset(summary.columns)
    assert {"logical", "sample_count"}.issubset(cosets.columns)

    hard_config = load_config(Path("experiments/configs/v1_hard_cases.json"))
    hard_config["output_dir"] = str(tmp_path / "results")
    hard_config["p_values"] = [0.1]
    hard_config["max_cases_per_sweep"] = 2
    hard_config["recovery_repeats"] = 1
    hard_config["sweeps"] = [{"family": "toric", "size": 3, "max_syndromes": 32}]
    hard_paths = run_hard_cases(hard_config)
    hard_summary = pd.read_csv(hard_paths["hard_case_summary"])
    assert {
        "recovered_exact",
        "recovered_exact_ci_low",
        "recovered_exact_ci_high",
        "target_attempts_5pct_ci",
        "additional_attempts_5pct_ci",
    }.issubset(hard_summary.columns)

    figure_paths = make_plots(tmp_path / "results", tmp_path / "figures")
    assert figure_paths
    assert all(path.exists() and path.stat().st_size > 0 for path in figure_paths)


def test_sweep_specific_p_values(tmp_path: Path) -> None:
    config = load_config(Path("experiments/configs/v1_smoke.json"))
    config["output_dir"] = str(tmp_path / "focused")
    config["p_values"] = [0.06, 0.1]
    config["sweeps"] = [
        {"family": "toric", "size": 3, "trials": 2, "exact": True, "p_values": [0.06]},
        {"family": "surface", "size": 3, "trials": 2, "exact": True, "p_values": [0.1]},
    ]

    paths = run_config(config)
    trials = pd.read_csv(paths["trials"])

    assert set(trials.loc[trials["code_family"] == "toric", "p"]) == {0.06}
    assert set(trials.loc[trials["code_family"] == "surface", "p"]) == {0.1}


def test_sweep_specific_decoders(tmp_path: Path) -> None:
    config = load_config(Path("experiments/configs/v1_smoke.json"))
    config["output_dir"] = str(tmp_path / "decoder_override")
    config["p_values"] = [0.06]
    config["decoders"] = ["pbit_syndrome"]
    config["sweeps"] = [
        {
            "family": "toric",
            "size": 3,
            "trials": 1,
            "exact": True,
            "decoders": ["exact_coset", "pbit_syndrome"],
        },
        {
            "family": "surface",
            "size": 3,
            "trials": 1,
            "exact": False,
            "decoders": ["pbit_nullspace_sampler"],
        },
    ]

    paths = run_config(config)
    trials = pd.read_csv(paths["trials"])

    assert set(trials.loc[trials["code_family"] == "toric", "decoder"]) == {
        "exact_coset",
        "pbit_syndrome",
    }
    assert set(trials.loc[trials["code_family"] == "surface", "decoder"]) == {
        "pbit_nullspace_sampler"
    }
