"""Efficiency sweep for PBitSyndromeDecoder min_weight.

The failure-mode diagnostics showed the production operating points are
fundamental-limited (min-weight vs maximum-likelihood), not coverage-limited, so the
chains x sweeps sampling budget is likely over-provisioned. This harness A/Bs budget
configs on identical error realizations and reports logical error rate, valid rate,
mean runtime, and speedup versus the first (baseline) config, so we can pick the
smallest budget that holds the frontier LER and valid_rate = 1.0.

Run:
    .venv/bin/python -u experiments/tune_efficiency.py --family surface --size 7 \
        --p 0.10 --trials 100 --configs 80x160 80x80 40x160 40x80 40x40 20x80
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from pbit_qec.codes import make_code
from pbit_qec.decoders import PBitSyndromeDecoder
from pbit_qec.simulation import sample_error

SURFACE_BASE = dict(
    beta_start=0.8, beta_stop=3.5, rho_scale=1.0, update="gibbs",
    stop_on_first_valid=False, collect_valid_samples=True, weight_reduction=True,
    selection="min_weight",
)
BB_BASE = dict(
    beta_start=0.3, beta_stop=5.0, rho_scale=1.0, update="gibbs",
    stop_on_first_valid=False, collect_valid_samples=True, weight_reduction=True,
    selection="min_weight",
)


def wilson_halfwidth(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    phat = k / n
    denom = 1 + z * z / n
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return margin


def evaluate(code, p, trials, base_seed, base_kw, chains, sweeps) -> dict:
    kw = {**base_kw, "chains": chains, "sweeps": sweeps}
    fails = invalid = 0
    t0 = time.perf_counter()
    for t in range(trials):
        seed = base_seed + t
        e_true = sample_error(code.n, p, np.random.default_rng(seed))
        syn = code.syndrome(e_true)
        drng = np.random.default_rng(seed + 3027)
        res = PBitSyndromeDecoder(**kw).decode(code, syn, p, drng)
        ok = bool(res.valid and not code.logical(e_true ^ res.correction).any())
        fails += not ok
        invalid += not res.valid
    dt = time.perf_counter() - t0
    return {
        "ler": fails / trials,
        "ler_hw": wilson_halfwidth(fails, trials),
        "valid_rate": 1.0 - invalid / trials,
        "secs_per_decode": dt / trials,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="surface")
    ap.add_argument("--size", type=int, default=7)
    ap.add_argument("--p", type=float, default=0.10)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=20260620)
    ap.add_argument("--configs", nargs="+", default=["80x160", "80x80", "40x160", "40x80"])
    args = ap.parse_args()

    code = make_code(args.family, args.size)
    base_kw = SURFACE_BASE if args.family == "surface" else BB_BASE

    print(f"\n{args.family} d={args.size} p={args.p} trials={args.trials}")
    header = f"{'config':<12} {'work':>8} {'LER':>14} {'valid':>7} {'s/dec':>7} {'speedup':>8}"
    print(header)
    base_secs = None
    for spec in args.configs:
        chains, sweeps = (int(x) for x in spec.lower().split("x"))
        m = evaluate(code, args.p, args.trials, args.base_seed, base_kw, chains, sweeps)
        if base_secs is None:
            base_secs = m["secs_per_decode"]
        speedup = base_secs / m["secs_per_decode"] if m["secs_per_decode"] else float("nan")
        ler_str = f"{m['ler']:.3f}+/-{m['ler_hw']:.3f}"
        print(
            f"{spec:<12} {chains * sweeps:>8} {ler_str:>14} {m['valid_rate']:>7.3f} "
            f"{m['secs_per_decode']:>7.2f} {speedup:>7.1f}x"
        )


if __name__ == "__main__":
    main()
