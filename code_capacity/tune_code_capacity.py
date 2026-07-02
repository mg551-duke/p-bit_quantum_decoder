#!/usr/bin/env python3
"""Per-p decoder tuning for external CSS codes (code-capacity bit-flip model).

For each (code, p) this sweeps the annealing-control grid of
:class:`pbit_qec.decoders.PBitSyndromeDecoder` -- ``rho_scale``, the inverse-temperature
endpoints ``beta_start:beta_stop``, the schedule ``shape``, and the sampling budget
``chains x sweeps`` -- on *paired* error realizations (identical errors across configs
for variance reduction) and records, per config: legacy LER (``(logic+nonconv)/trials``,
matching the campaign's stop metric), syndrome-valid rate, first-valid-sweep stats,
mean selected correction weight, and seconds/decode.

The per-p "best" config is chosen by ranking on::

    (ler, selected_weight_mean, fvs_p95, secs_per_decode)

so that in the low-p regime -- where logical errors are too rare to separate configs --
the *mean selected weight* (a strong proxy for ML-coset quality) and convergence speed
break the tie. The winners are written to ``config_map.json`` keyed by ``code -> "p"``,
which :mod:`run_code_capacity` consumes via ``--config-map`` to launch the production
LER-vs-p campaign with each point's own tuned parameters.

This reuses the variance-reduced ``evaluate_config`` from
:mod:`pbit_qec.schedule_tuning`, but loads the code from an ``.npz`` (Hx,Hz,Lx,Lz)
instead of ``make_code``; the X-error sector is ``H=Hz, A=Lz`` (see ``load_css_code``).

Example:
    .venv/bin/python code_capacity/tune_code_capacity.py \\
        --codes code_capacity/codes/B1_HxHzLxLz.npz code_capacity/codes/B2_HxHzLxLz.npz \\
        --p 0.04 0.05 0.06 0.07 0.08 0.09 0.10 --trials 160 \\
        --rho-scales 0.5 1.0 2.0 --beta-pairs 0.5:5.0 0.5:8.0 --shapes linear \\
        --chains 256 --sweeps 100 --workers 80 --name tune_v1
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pbit_qec.decoders import PBitSyndromeDecoder  # noqa: E402
from pbit_qec.schedule_tuning import (  # noqa: E402
    _DECODE_SEED_OFFSET,
    _ERROR_SEED_OFFSET,
    _P_SEED_STRIDE,
    BASE_SETTINGS,
    ScheduleConfig,
    build_configs,
    wilson_interval,
)
from pbit_qec.simulation import sample_error  # noqa: E402
from run_code_capacity import load_css_code  # noqa: E402

# Keys that define a decoder operating point in the emitted config map.
_CONFIG_KEYS = (
    "chains",
    "sweeps",
    "beta_start",
    "beta_stop",
    "beta_schedule_shape",
    "rho_scale",
)
# Per-code seed offset so paired error draws are independent across codes.
_CODE_SEED_STRIDE = 7_919_000_000


def _chunk_job(args: Tuple) -> Dict[str, object]:
    """Decode trials ``[t0, t0+nt)`` of one (code, p, config); return aggregates.

    Uses the identical paired-seed scheme as :func:`pbit_qec.schedule_tuning.evaluate_config`
    (``error_seed`` / ``decode_seed`` depend only on ``(p_index, trial)``), so splitting a
    config's trials into chunks across workers yields exactly the same ensemble -- just
    finer-grained for responsiveness and load balancing.
    """
    (npz_path, family, tag, p, p_index, base_seed, cfg, settings, t0, nt) = args
    code = load_css_code(Path(npz_path), family)
    decoder = PBitSyndromeDecoder(
        chains=cfg.chains,
        sweeps=cfg.sweeps,
        beta_start=cfg.beta_start,
        beta_stop=cfg.beta_stop,
        beta_schedule_shape=cfg.shape,
        rho_scale=cfg.rho_scale,
        **dict(settings),
    )
    fails = 0
    invalid = 0
    weights: List[int] = []
    fvs: List[int] = []
    runtime = 0.0
    rho_value = float("nan")
    for trial in range(t0, t0 + nt):
        error_seed = base_seed + _ERROR_SEED_OFFSET + p_index * _P_SEED_STRIDE + trial
        error = sample_error(code.n, p, np.random.default_rng(error_seed))
        syndrome = code.syndrome(error)
        decode_seed = base_seed + _DECODE_SEED_OFFSET + p_index * _P_SEED_STRIDE + trial
        result = decoder.decode(code, syndrome, p, np.random.default_rng(decode_seed))
        success = bool(result.valid and not code.logical(error ^ result.correction).any())
        fails += not success
        invalid += not result.valid
        weights.append(int(result.metadata["selected_weight"]))
        if result.first_valid_sweep is not None:
            fvs.append(int(result.first_valid_sweep))
        runtime += float(result.runtime_s)
        rho_value = float(result.metadata["rho"])
    return {
        "key": (tag, f"{p:g}", cfg.label),
        "code": tag,
        "source": npz_path,
        "p": float(p),
        "config": cfg.label,
        "rho_scale": cfg.rho_scale,
        "beta_start": cfg.beta_start,
        "beta_stop": cfg.beta_stop,
        "shape": cfg.shape,
        "chains": cfg.chains,
        "sweeps": cfg.sweeps,
        "n": int(nt),
        "fails": int(fails),
        "invalid": int(invalid),
        "weights": weights,
        "fvs": fvs,
        "runtime_s": runtime,
        "rho": rho_value,
    }


def _finalize_summary(acc: Dict[str, object]) -> Dict[str, object]:
    """Build an evaluate_config-compatible summary row from accumulated chunks."""
    n = int(acc["n"])
    fails = int(acc["fails"])
    invalid = int(acc["invalid"])
    low, high = wilson_interval(fails, n)
    fvs_arr = np.asarray(acc["fvs"], dtype=float)
    has_fvs = fvs_arr.size > 0
    weights = acc["weights"]
    return {
        "code": acc["code"],
        "source": acc["source"],
        "config": acc["config"],
        "rho_scale": acc["rho_scale"],
        "beta_start": acc["beta_start"],
        "beta_stop": acc["beta_stop"],
        "shape": acc["shape"],
        "chains": acc["chains"],
        "sweeps": acc["sweeps"],
        "p": acc["p"],
        "trials": n,
        "logical_failures": fails,
        "ler": fails / n if n else float("nan"),
        "ler_ci_low": low,
        "ler_ci_high": high,
        "ler_ci_halfwidth": (high - low) / 2.0,
        "valid_rate": 1.0 - invalid / n if n else float("nan"),
        "never_valid_rate": invalid / n if n else float("nan"),
        "rho": acc["rho"],
        "fvs_mean": float(fvs_arr.mean()) if has_fvs else float("nan"),
        "fvs_median": float(np.median(fvs_arr)) if has_fvs else float("nan"),
        "fvs_p95": float(np.percentile(fvs_arr, 95)) if has_fvs else float("nan"),
        "fvs_max": float(fvs_arr.max()) if has_fvs else float("nan"),
        "selected_weight_mean": float(np.mean(weights)) if weights else float("nan"),
        "secs_per_decode": float(acc["runtime_s"]) / n if n else float("nan"),
    }


def _parse_beta_pairs(specs: Sequence[str]) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"beta pair {spec!r} must be 'start:stop'")
        start_s, stop_s = spec.split(":", 1)
        pairs.append((float(start_s), float(stop_s)))
    return pairs


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def _config_kwargs(row: pd.Series) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "chains": int(row["chains"]),
        "sweeps": int(row["sweeps"]),
        "beta_start": float(row["beta_start"]),
        "beta_stop": float(row["beta_stop"]),
        "beta_schedule_shape": str(row["shape"]),
        "rho_scale": float(row["rho_scale"]),
    }
    kwargs.update(dict(BASE_SETTINGS))
    return kwargs


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--codes", nargs="+", required=True)
    parser.add_argument(
        "--p", nargs="+", type=float, default=[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    )
    parser.add_argument("--trials", type=int, default=160)
    parser.add_argument("--rho-scales", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--beta-pairs", nargs="+", default=["0.5:5.0", "0.5:8.0"])
    parser.add_argument("--shapes", nargs="+", default=["linear"])
    parser.add_argument("--chains", nargs="+", type=int, default=[256])
    parser.add_argument("--sweeps", nargs="+", type=int, default=[100])
    parser.add_argument(
        "--chunk",
        type=int,
        default=8,
        help="trials per work unit; trials are split into chunks for finer parallelism",
    )
    parser.add_argument("--workers", type=int, default=min(80, os.cpu_count() or 1))
    parser.add_argument("--base-seed", type=int, default=20260618)
    parser.add_argument("--name", default="tune_v1")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    beta_pairs = _parse_beta_pairs(args.beta_pairs)
    configs = build_configs(args.rho_scales, beta_pairs, args.shapes, args.chains, args.sweeps)
    settings = dict(BASE_SETTINGS)

    code_specs = []
    for path in args.codes:
        code = load_css_code(Path(path))
        tag = Path(path).stem.split("_")[0]
        code_specs.append((str(Path(path)), code.family, tag))
    output_dir = args.output_dir or (_HERE / "results" / args.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for code_index, (npz_path, family, tag) in enumerate(code_specs):
        code_seed = args.base_seed + code_index * _CODE_SEED_STRIDE
        for p_index, p in enumerate(args.p):
            for cfg in configs:
                for t0 in range(0, args.trials, max(1, args.chunk)):
                    nt = min(args.chunk, args.trials - t0)
                    jobs.append(
                        (npz_path, family, tag, float(p), p_index, code_seed, cfg, settings, t0, nt)
                    )

    total = len(jobs)
    n_points = len(code_specs) * len(args.p) * len(configs)
    print(
        f"tuning: {len(code_specs)} codes x {len(args.p)} p x {len(configs)} configs "
        f"x {args.trials} trials = {n_points} points / {total} chunks "
        f"({n_points * args.trials} decodes), workers={args.workers}",
        flush=True,
    )

    # Accumulate chunk aggregates into per-(code,p,config) points.
    accum: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    decodes_done = 0
    decodes_total = n_points * args.trials

    def _merge(chunk: Dict[str, object]) -> None:
        key = chunk["key"]
        acc = accum.get(key)
        if acc is None:
            acc = {
                field: chunk[field]
                for field in (
                    "code", "source", "p", "config", "rho_scale", "beta_start",
                    "beta_stop", "shape", "chains", "sweeps", "rho",
                )
            }
            acc.update({"n": 0, "fails": 0, "invalid": 0, "runtime_s": 0.0,
                        "weights": [], "fvs": []})
            accum[key] = acc
        acc["n"] += chunk["n"]
        acc["fails"] += chunk["fails"]
        acc["invalid"] += chunk["invalid"]
        acc["runtime_s"] += chunk["runtime_s"]
        acc["weights"].extend(chunk["weights"])
        acc["fvs"].extend(chunk["fvs"])
        acc["rho"] = chunk["rho"]

    start = time.perf_counter()
    done = 0
    partial_path = output_dir / "tune_summary_partial.csv"
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for chunk in executor.map(_chunk_job, jobs):
            _merge(chunk)
            done += 1
            decodes_done += int(chunk["n"])
            elapsed = time.perf_counter() - start
            rate = decodes_done / elapsed if elapsed else 0.0
            eta = (decodes_total - decodes_done) / rate if rate else float("nan")
            print(
                f"[chunk {done}/{total} | {decodes_done}/{decodes_total} decodes] "
                f"{chunk['code']} p={chunk['p']:g} {chunk['config']} "
                f"| {rate * 60:.1f} dec/min elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
            if done % max(1, args.workers) == 0 or done == total:
                pd.DataFrame([_finalize_summary(a) for a in accum.values()]).to_csv(
                    partial_path, index=False
                )

    summaries = [_finalize_summary(acc) for acc in accum.values()]
    summary_df = pd.DataFrame(summaries)
    rank_cols = ["ler", "selected_weight_mean", "fvs_p95", "secs_per_decode"]
    summary_df = summary_df.sort_values(["code", "p", *rank_cols]).reset_index(drop=True)
    best_df = (
        summary_df.sort_values(rank_cols).groupby(["code", "p"], as_index=False).first()
    )

    summary_path = output_dir / "tune_summary.csv"
    best_path = output_dir / "tune_best_by_p.csv"
    summary_df.to_csv(summary_path, index=False)
    best_df.to_csv(best_path, index=False)

    config_map: Dict[str, Dict[str, Dict[str, object]]] = {}
    for _, row in best_df.iterrows():
        config_map.setdefault(str(row["code"]), {})[f"{float(row['p']):g}"] = _config_kwargs(row)
    map_path = output_dir / "config_map.json"
    map_path.write_text(json.dumps(config_map, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "name": args.name,
        "codes": [{"source": s, "family": f, "tag": t} for s, f, t in code_specs],
        "p_values": [float(p) for p in args.p],
        "trials": args.trials,
        "rho_scales": list(args.rho_scales),
        "beta_pairs": args.beta_pairs,
        "shapes": list(args.shapes),
        "chains": list(args.chains),
        "sweeps": list(args.sweeps),
        "chunk": args.chunk,
        "num_configs": len(configs),
        "workers": args.workers,
        "base_seed": args.base_seed,
        "ranking": rank_cols,
        "base_settings": settings,
        "outputs": {
            "summary": summary_path.name,
            "best_by_p": best_path.name,
            "config_map": map_path.name,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\n=== best config per (code, p) ===", flush=True)
    show = best_df[
        ["code", "p", "config", "ler", "valid_rate", "selected_weight_mean", "fvs_p95",
         "secs_per_decode"]
    ]
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(show.to_string(index=False))
    print(f"\nconfig_map: {map_path}\nsummary:    {summary_path}\nbest_by_p:  {best_path}")


if __name__ == "__main__":
    main()
