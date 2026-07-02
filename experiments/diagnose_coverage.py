"""A/B coverage levers for PBitSyndromeDecoder min_weight on a target code.

The BB144 diagnostic showed the dominant min_weight failure mode is coset coverage:
the sampler never reaches the true logical class, so no selection rule can recover it.
This script measures, on identical error realizations, the true-coset-reach rate and
logical error rate for the production config vs coverage-boosting overrides
(more sweeps, a moderate-beta measurement phase, or more chains).

Run:
    .venv/bin/python experiments/diagnose_coverage.py --family bb --size 144 \
        --p 0.08 --trials 40
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import pbit_qec.gf2 as gf2
from pbit_qec.codes import make_code
from pbit_qec.decoders import PBitSyndromeDecoder
from pbit_qec.simulation import sample_error

D7 = dict(
    chains=80, sweeps=160, beta_start=0.8, beta_stop=3.5, rho_scale=1.0, update="gibbs",
    stop_on_first_valid=False, collect_valid_samples=True, weight_reduction=True,
    selection="min_weight",
)
BB = dict(
    chains=256, sweeps=250, beta_start=0.3, beta_stop=5.0, rho_scale=1.0, update="gibbs",
    stop_on_first_valid=False, collect_valid_samples=True, weight_reduction=True,
    selection="min_weight",
)

VARIANTS = {
    "baseline": {},
    "sweeps2x": {"sweeps": 500},
    "measure": {"measure_sweeps": 200, "beta_measure": 1.2},
    "chains2x": {"chains": 512},
}


def evaluate(code, p, trials, base_seed, base_kw, overrides) -> dict:
    kw = {**base_kw, **overrides}
    fails = reached = 0
    reached_but_failed = 0
    t0 = time.perf_counter()
    for t in range(trials):
        seed = base_seed + t
        e_true = sample_error(code.n, p, np.random.default_rng(seed))
        syn = code.syndrome(e_true)
        true_key = gf2.bit_string(code.logical(e_true))
        res = PBitSyndromeDecoder(**kw).decode(code, syn, p, np.random.default_rng(seed + 3027))
        ok = bool(res.valid and not code.logical(e_true ^ res.correction).any())
        in_pool = true_key in res.metadata.get("ml_coset_logz", {})
        fails += not ok
        reached += in_pool
        reached_but_failed += in_pool and not ok
    dt = time.perf_counter() - t0
    return {
        "ler": fails / trials,
        "reach": reached / trials,
        "reached_but_failed": reached_but_failed,
        "secs_per_decode": dt / trials,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="bb")
    ap.add_argument("--size", type=int, default=144)
    ap.add_argument("--p", type=float, default=0.08)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=20260620)
    ap.add_argument("--variants", nargs="+", default=["baseline", "measure", "chains2x"])
    args = ap.parse_args()

    code = make_code(args.family, args.size)
    base_kw = D7 if args.family == "surface" else BB

    print(f"\n{args.family} d={args.size} p={args.p} trials={args.trials}")
    print(f"{'variant':<12} {'LER':>7} {'reach':>7} {'reach&fail':>11} {'s/dec':>7}  overrides")
    for name in args.variants:
        ov = VARIANTS[name]
        m = evaluate(code, args.p, args.trials, args.base_seed, base_kw, ov)
        print(
            f"{name:<12} {m['ler']:>7.3f} {m['reach']:>7.3f} {m['reached_but_failed']:>11d} "
            f"{m['secs_per_decode']:>7.2f}  {ov}"
        )


if __name__ == "__main__":
    main()
