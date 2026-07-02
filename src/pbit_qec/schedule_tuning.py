"""Temperature-schedule and penalty/temperature hyperparameter tuning.

Sweeps the annealing controls that govern convergence speed and solution quality
for :class:`~pbit_qec.decoders.PBitSyndromeDecoder` -- the syndrome-penalty weight
``rho`` (through ``rho_scale``), the inverse-temperature endpoints
``beta_start``/``beta_stop``, the schedule shape (geometric/linear/sigmoid), and
the sampling budget (chains x sweeps) -- and reports, per configuration:

- logical error rate with a Wilson score interval,
- syndrome-valid rate,
- convergence speed via the first-valid-sweep distribution (mean/median/p95/max),
- mean selected correction weight,
- wall-clock seconds per decode.

Every configuration is evaluated on identical (paired) error realizations, so
config-to-config differences are variance-reduced. Full provenance -- the grid,
seed scheme, git commit, code and base-decoder parameters -- is written next to
the raw per-trial table and the per-config summary, and a per-p "best" table
distils the recommended operating point.

Run:
    .venv/bin/python -m pbit_qec.schedule_tuning --family surface --size 7 \\
        --p 0.05 0.10 --trials 300 --chains 256 --sweeps 150 \\
        --rho-scales 0.5 1.0 1.5 --beta-pairs 0.5:3.5 0.8:3.5 0.8:5.0 \\
        --shapes geometric linear sigmoid --workers 8 --name sched_surface_d7
"""

from __future__ import annotations

import os

# Pin BLAS to one thread per process *before* numpy initializes, so the
# config-level process pool gets parallelism without thread oversubscription.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pbit_qec.codes import make_code  # noqa: E402
from pbit_qec.decoders import PBitSyndromeDecoder  # noqa: E402
from pbit_qec.simulation import sample_error  # noqa: E402
from pbit_qec.types import CSSCode  # noqa: E402

# Fixed decoder settings that define the operating regime under study; only the
# annealing controls (rho/beta/shape/budget) vary across the swept grid.
BASE_SETTINGS: Dict[str, object] = {
    "update": "gibbs",
    "selection": "min_weight",
    "weight_reduction": True,
    "collect_valid_samples": True,
    "stop_on_first_valid": False,
}

_ERROR_SEED_OFFSET = 0
_DECODE_SEED_OFFSET = 500_000_000
_P_SEED_STRIDE = 1_000_000


@dataclass(frozen=True)
class ScheduleConfig:
    """One point in the annealing-control grid."""

    rho_scale: float
    beta_start: float
    beta_stop: float
    shape: str
    chains: int
    sweeps: int

    @property
    def label(self) -> str:
        return (
            f"rs{self.rho_scale:g}_b{self.beta_start:g}-{self.beta_stop:g}"
            f"_{self.shape}_{self.chains}x{self.sweeps}"
        )


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_config(
    code: CSSCode,
    p: float,
    trials: int,
    base_seed: int,
    p_index: int,
    cfg: ScheduleConfig,
    base_settings: Mapping[str, object],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Evaluate one schedule configuration over ``trials`` paired error draws.

    Returns a per-config summary row and the list of per-trial rows. The error
    and decoder seeds depend only on ``(p_index, trial)`` -- never on ``cfg`` --
    so every configuration sees the identical error ensemble.
    """
    decoder = PBitSyndromeDecoder(
        chains=cfg.chains,
        sweeps=cfg.sweeps,
        beta_start=cfg.beta_start,
        beta_stop=cfg.beta_stop,
        beta_schedule_shape=cfg.shape,
        rho_scale=cfg.rho_scale,
        **dict(base_settings),
    )
    fails = 0
    invalid = 0
    first_valid: List[int] = []
    weights: List[int] = []
    rho_value = float("nan")
    rows: List[Dict[str, object]] = []
    start = time.perf_counter()
    for trial in range(trials):
        error_seed = base_seed + _ERROR_SEED_OFFSET + p_index * _P_SEED_STRIDE + trial
        error = sample_error(code.n, p, np.random.default_rng(error_seed))
        syndrome = code.syndrome(error)
        decode_seed = base_seed + _DECODE_SEED_OFFSET + p_index * _P_SEED_STRIDE + trial
        result = decoder.decode(code, syndrome, p, np.random.default_rng(decode_seed))
        success = bool(result.valid and not code.logical(error ^ result.correction).any())
        fails += not success
        invalid += not result.valid
        rho_value = float(result.metadata["rho"])
        selected_weight = int(result.metadata["selected_weight"])
        weights.append(selected_weight)
        fvs = result.first_valid_sweep
        if fvs is not None:
            first_valid.append(int(fvs))
        rows.append(
            {
                "config": cfg.label,
                "rho_scale": cfg.rho_scale,
                "beta_start": cfg.beta_start,
                "beta_stop": cfg.beta_stop,
                "shape": cfg.shape,
                "chains": cfg.chains,
                "sweeps": cfg.sweeps,
                "p": p,
                "trial": trial,
                "error_seed": error_seed,
                "decode_seed": decode_seed,
                "valid": bool(result.valid),
                "logical_success": success,
                "first_valid_sweep": int(fvs) if fvs is not None else None,
                "selected_weight": selected_weight,
                "runtime_s": float(result.runtime_s),
            }
        )
    elapsed = time.perf_counter() - start
    low, high = wilson_interval(fails, trials)
    fvs_arr = np.asarray(first_valid, dtype=float)
    has_fvs = fvs_arr.size > 0
    summary = {
        "config": cfg.label,
        "rho_scale": cfg.rho_scale,
        "beta_start": cfg.beta_start,
        "beta_stop": cfg.beta_stop,
        "shape": cfg.shape,
        "chains": cfg.chains,
        "sweeps": cfg.sweeps,
        "p": p,
        "trials": trials,
        "logical_failures": fails,
        "ler": fails / trials,
        "ler_ci_low": low,
        "ler_ci_high": high,
        "ler_ci_halfwidth": (high - low) / 2.0,
        "valid_rate": 1.0 - invalid / trials,
        "never_valid_rate": invalid / trials,
        "rho": rho_value,
        "fvs_mean": float(fvs_arr.mean()) if has_fvs else float("nan"),
        "fvs_median": float(np.median(fvs_arr)) if has_fvs else float("nan"),
        "fvs_p95": float(np.percentile(fvs_arr, 95)) if has_fvs else float("nan"),
        "fvs_max": float(fvs_arr.max()) if has_fvs else float("nan"),
        "selected_weight_mean": float(np.mean(weights)) if weights else float("nan"),
        "secs_per_decode": elapsed / trials,
    }
    return summary, rows


def _run_job(job: Tuple) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    family, size, p, p_index, trials, base_seed, cfg, base_settings = job
    code = make_code(family, size)
    return evaluate_config(code, p, trials, base_seed, p_index, cfg, base_settings)


def build_configs(
    rho_scales: Sequence[float],
    beta_pairs: Sequence[Tuple[float, float]],
    shapes: Sequence[str],
    chains: Sequence[int],
    sweeps: Sequence[int],
) -> List[ScheduleConfig]:
    """Cartesian product of the swept annealing controls."""
    configs: List[ScheduleConfig] = []
    for rho_scale, (beta_start, beta_stop), shape, chain_count, sweep_count in itertools.product(
        rho_scales, beta_pairs, shapes, chains, sweeps
    ):
        configs.append(
            ScheduleConfig(
                rho_scale=rho_scale,
                beta_start=beta_start,
                beta_stop=beta_stop,
                shape=shape,
                chains=chain_count,
                sweeps=sweep_count,
            )
        )
    return configs


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def run_schedule_tuning(
    family: str,
    size: int,
    p_values: Sequence[float],
    trials: int,
    configs: Sequence[ScheduleConfig],
    base_seed: int,
    output_dir: Path,
    workers: int = 1,
    base_settings: Optional[Mapping[str, object]] = None,
) -> Dict[str, Path]:
    """Evaluate every (config, p) point and write the publishable result tables."""
    settings = dict(BASE_SETTINGS if base_settings is None else base_settings)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (family, size, float(p), p_index, trials, base_seed, cfg, settings)
        for p_index, p in enumerate(p_values)
        for cfg in configs
    ]

    summaries: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for summary, job_rows in executor.map(_run_job, jobs):
                summaries.append(summary)
                rows.extend(job_rows)
    else:
        code_cache: Dict[Tuple[str, int], CSSCode] = {}
        for job in jobs:
            key = (job[0], job[1])
            if key not in code_cache:
                code_cache[key] = make_code(*key)
            summary, job_rows = evaluate_config(
                code_cache[key], job[2], job[4], job[5], job[3], job[6], job[7]
            )
            summaries.append(summary)
            rows.extend(job_rows)

    summary_df = (
        pd.DataFrame(summaries)
        .sort_values(["p", "ler", "fvs_p95", "secs_per_decode"])
        .reset_index(drop=True)
    )
    trials_df = pd.DataFrame(rows)
    best_df = (
        summary_df.sort_values(["ler", "fvs_p95", "secs_per_decode"])
        .groupby("p", as_index=False)
        .first()
    )

    summary_path = output_dir / "schedule_summary.csv"
    trials_path = output_dir / "schedule_trials.csv"
    best_path = output_dir / "schedule_best_by_p.csv"
    summary_df.to_csv(summary_path, index=False)
    trials_df.to_csv(trials_path, index=False)
    best_df.to_csv(best_path, index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "family": family,
        "size": size,
        "p_values": [float(p) for p in p_values],
        "trials": trials,
        "base_seed": base_seed,
        "workers": workers,
        "base_settings": settings,
        "seed_scheme": {
            "error_seed": "base_seed + p_index*1_000_000 + trial",
            "decode_seed": "base_seed + 500_000_000 + p_index*1_000_000 + trial",
            "note": "depends only on (p_index, trial); paired across configs",
        },
        "num_configs": len(configs),
        "num_points": len(jobs),
        "configs": [
            {
                "label": cfg.label,
                "rho_scale": cfg.rho_scale,
                "beta_start": cfg.beta_start,
                "beta_stop": cfg.beta_stop,
                "shape": cfg.shape,
                "chains": cfg.chains,
                "sweeps": cfg.sweeps,
            }
            for cfg in configs
        ],
        "outputs": {
            "summary": summary_path.name,
            "trials": trials_path.name,
            "best_by_p": best_path.name,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "summary": summary_path,
        "trials": trials_path,
        "best_by_p": best_path,
        "manifest": manifest_path,
    }


def _parse_beta_pairs(specs: Sequence[str]) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"beta pair {spec!r} must be 'start:stop'")
        start_s, stop_s = spec.split(":", 1)
        pairs.append((float(start_s), float(stop_s)))
    return pairs


def _print_summary(summary: pd.DataFrame) -> None:
    cols = [
        "p",
        "config",
        "ler",
        "ler_ci_halfwidth",
        "valid_rate",
        "fvs_median",
        "fvs_p95",
        "secs_per_decode",
    ]
    view = summary[cols].copy()
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(view.to_string(index=False))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--family", default="surface")
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("--p", nargs="+", type=float, required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=20260620)
    parser.add_argument("--chains", nargs="+", type=int, default=[256])
    parser.add_argument("--sweeps", nargs="+", type=int, default=[150])
    parser.add_argument("--rho-scales", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--beta-pairs", nargs="+", default=["0.3:5.0"])
    parser.add_argument(
        "--shapes", nargs="+", default=["geometric", "linear", "sigmoid"]
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)

    beta_pairs = _parse_beta_pairs(args.beta_pairs)
    configs = build_configs(args.rho_scales, beta_pairs, args.shapes, args.chains, args.sweeps)
    name = args.name or f"sched_{args.family}_{args.size}"
    output_dir = args.output_dir or (Path("results") / name)

    print(
        f"{args.family} d={args.size} | {len(configs)} configs x {len(args.p)} p "
        f"x {args.trials} trials | workers={args.workers}"
    )
    paths = run_schedule_tuning(
        family=args.family,
        size=args.size,
        p_values=args.p,
        trials=args.trials,
        configs=configs,
        base_seed=args.base_seed,
        output_dir=output_dir,
        workers=args.workers,
    )
    _print_summary(pd.read_csv(paths["summary"]))
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
