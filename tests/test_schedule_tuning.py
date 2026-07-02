import json

import numpy as np
import pandas as pd

from pbit_qec.codes import make_toric_code
from pbit_qec.schedule_tuning import (
    ScheduleConfig,
    build_configs,
    evaluate_config,
    run_schedule_tuning,
    wilson_interval,
)


def test_wilson_interval_basic_properties() -> None:
    low, high = wilson_interval(0, 100)
    assert abs(low) < 1e-9
    assert 0.0 < high < 0.1
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high
    # Degenerate sample size yields NaNs rather than raising.
    assert all(np.isnan(value) for value in wilson_interval(0, 0))


def test_build_configs_is_full_cartesian_product() -> None:
    configs = build_configs(
        rho_scales=[0.5, 1.0],
        beta_pairs=[(0.3, 5.0), (0.8, 3.5)],
        shapes=["geometric", "sigmoid"],
        chains=[64],
        sweeps=[40, 80],
    )
    assert len(configs) == 2 * 2 * 2 * 1 * 2
    assert len({c.label for c in configs}) == len(configs)


def test_evaluate_config_returns_summary_and_paired_rows() -> None:
    code = make_toric_code(3)
    cfg = ScheduleConfig(
        rho_scale=1.0, beta_start=0.3, beta_stop=3.0, shape="geometric", chains=4, sweeps=20
    )
    summary, rows = evaluate_config(
        code, p=0.08, trials=5, base_seed=123, p_index=0, cfg=cfg, base_settings={
            "update": "gibbs",
            "selection": "min_weight",
            "weight_reduction": True,
            "collect_valid_samples": True,
            "stop_on_first_valid": False,
        }
    )
    assert summary["trials"] == 5
    assert 0.0 <= summary["ler"] <= 1.0
    assert summary["ler_ci_low"] <= summary["ler"] <= summary["ler_ci_high"]
    assert 0.0 <= summary["valid_rate"] <= 1.0
    assert len(rows) == 5
    assert summary["config"] == cfg.label


def test_run_schedule_tuning_writes_tables_with_paired_errors(tmp_path) -> None:
    configs = build_configs(
        rho_scales=[1.0],
        beta_pairs=[(0.3, 3.0)],
        shapes=["geometric", "sigmoid"],
        chains=[4],
        sweeps=[20],
    )
    paths = run_schedule_tuning(
        family="toric",
        size=3,
        p_values=[0.08],
        trials=6,
        configs=configs,
        base_seed=2024,
        output_dir=tmp_path,
        workers=1,
    )
    for key in ("summary", "trials", "best_by_p", "manifest"):
        assert paths[key].exists()

    summary = pd.read_csv(paths["summary"])
    assert len(summary) == len(configs)  # one row per config at the single p
    for column in ("ler", "ler_ci_low", "valid_rate", "fvs_p95", "secs_per_decode", "rho"):
        assert column in summary.columns

    best = pd.read_csv(paths["best_by_p"])
    assert len(best) == 1  # one recommended config per p

    # Errors must be identical (paired) across configs for each (p, trial).
    trials = pd.read_csv(paths["trials"])
    per_trial = trials.groupby(["p", "trial"])["error_seed"].nunique()
    assert (per_trial == 1).all()

    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["num_configs"] == len(configs)
    assert manifest["num_points"] == len(configs)
    assert "seed_scheme" in manifest
    assert len(manifest["configs"]) == len(configs)
