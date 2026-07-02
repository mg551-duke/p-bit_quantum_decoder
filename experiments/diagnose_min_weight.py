"""Root-cause diagnostic for PBitSyndromeDecoder min_weight failures.

Selection is pure post-processing over an identical sampled ensemble, so decoding
the *same* syndrome with the same decoder RNG under selection="min_weight" vs
selection="ml_coset" isolates the effect of the selection rule alone. This script
quantifies whether degeneracy-aware (free-energy) selection would fix the residual
min_weight failures, and whether the true coset is even reachable.

Run:
    .venv/bin/python experiments/diagnose_min_weight.py --family surface --size 7 \
        --p 0.10 --trials 60
"""

from __future__ import annotations

import argparse

import numpy as np

import pbit_qec.gf2 as gf2
from pbit_qec.codes import make_code
from pbit_qec.decoders import PBitSyndromeDecoder
from pbit_qec.simulation import sample_error

D7 = dict(
    chains=80, sweeps=160, beta_start=0.8, beta_stop=3.5, rho_scale=1.0, update="gibbs",
    stop_on_first_valid=False, collect_valid_samples=True, weight_reduction=True,
)
BB = dict(
    chains=256, sweeps=250, beta_start=0.3, beta_stop=5.0, rho_scale=1.0, update="gibbs",
    stop_on_first_valid=False, collect_valid_samples=True, weight_reduction=True,
)


def success(code, e_true, result) -> bool:
    return bool(result.valid and not code.logical(e_true ^ result.correction).any())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="surface")
    ap.add_argument("--size", type=int, default=7)
    ap.add_argument("--p", type=float, default=0.10)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--base-seed", type=int, default=20260620)
    args = ap.parse_args()

    code = make_code(args.family, args.size)
    base_kw = D7 if args.family == "surface" else BB

    mw_fail = ml_fail = both_fail = 0
    mw_fail_ml_ok = 0  # min_weight failed but ml_coset succeeded (degeneracy-fixable)
    ml_fail_mw_ok = 0  # ml_coset failed but min_weight succeeded (weight helps)
    true_unreached = 0  # min_weight failure where true coset never sampled
    true_not_top_logz = 0  # true coset sampled but not the max-logz coset

    for t in range(args.trials):
        seed = args.base_seed + t
        e_true = sample_error(code.n, args.p, np.random.default_rng(seed))
        syn = code.syndrome(e_true)
        true_key = gf2.bit_string(code.logical(e_true))
        drng = seed + 3027  # identical sampling stream for both selection rules

        res_mw = PBitSyndromeDecoder(selection="min_weight", **base_kw).decode(
            code, syn, args.p, np.random.default_rng(drng)
        )
        res_ml = PBitSyndromeDecoder(selection="ml_coset", **base_kw).decode(
            code, syn, args.p, np.random.default_rng(drng)
        )
        ok_mw = success(code, e_true, res_mw)
        ok_ml = success(code, e_true, res_ml)
        mw_fail += not ok_mw
        ml_fail += not ok_ml
        both_fail += (not ok_mw) and (not ok_ml)
        mw_fail_ml_ok += (not ok_mw) and ok_ml
        ml_fail_mw_ok += (not ok_ml) and ok_mw

        if not ok_mw:
            logz = res_mw.metadata.get("ml_coset_logz", {})
            if true_key not in logz:
                true_unreached += 1
            elif logz and max(logz, key=logz.get) != true_key:
                true_not_top_logz += 1

    n = args.trials
    print(f"\n{args.family} d={args.size} p={args.p} trials={n}")
    print(f"  min_weight failures: {mw_fail}/{n} = {mw_fail / n:.3f}")
    print(f"  ml_coset   failures: {ml_fail}/{n} = {ml_fail / n:.3f}")
    print(f"  both fail          : {both_fail}/{n}")
    print(f"  min_weight FAIL but ml_coset OK (degeneracy-fixable): {mw_fail_ml_ok}")
    print(f"  ml_coset FAIL but min_weight OK (weight helps)      : {ml_fail_mw_ok}")
    print("  --- among min_weight failures ---")
    print(f"  true coset never sampled (coverage): {true_unreached}/{mw_fail}")
    print(f"  true coset sampled but not max-logz: {true_not_top_logz}/{mw_fail}")


if __name__ == "__main__":
    main()
