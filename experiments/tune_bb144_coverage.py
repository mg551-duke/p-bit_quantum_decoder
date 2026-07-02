#!/usr/bin/env python3
"""Tune p-bit ensemble size / annealing for BB144 coverage.

Runs the ``PBitSyndromeDecoder`` over a grid of (chains, sweeps, beta) configs on
the BB ``[[144,12,12]]`` code at a fixed physical error rate, and decomposes every
logical failure into a *coverage* failure (the true coset was never visited) or a
*selection* failure (the true coset was visited but a different one was returned).

A BP-OSD reference is reported for the same error set. Results print one line per
config (flushed) so a long background run can be polled incrementally.

Usage:
    python experiments/tune_bb144_coverage.py --p 0.05 --trials 8 --seed 0
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

from pbit_qec import gf2
from pbit_qec.codes import make_bb144_code
from pbit_qec.decoders import BPOSDDecoder, PBitSyndromeDecoder

# Each config: name -> kwargs passed to PBitSyndromeDecoder. ``selection`` defaults
# to ml_coset (so coverage via ml_coset_logz is always populated) but may be
# overridden per-config. coset_logz is populated for every selection mode, so the
# coverage decomposition is valid regardless.
#
# Current grid: A/B the over-penalized default rho against the p-adaptive
# rho_scale=1.0 (rho = max lambda) at the production min_weight + weight reduction
# ensemble, to quantify the coverage/success gain from the softer penalty.
_ENSEMBLE = {"sweeps": 250, "beta_start": 0.3, "beta_stop": 5.0,
             "selection": "min_weight", "weight_reduction": True}
CONFIGS: List[Dict] = [
    {"name": "default_128", **_ENSEMBLE, "chains": 128},
    {"name": "rho_scale1.0_128", **_ENSEMBLE, "chains": 128, "rho_scale": 1.0},
    {"name": "rho_scale1.0_256", **_ENSEMBLE, "chains": 256, "rho_scale": 1.0},
]


def build(cfg: Dict) -> PBitSyndromeDecoder:
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    kwargs.setdefault("selection", "ml_coset")
    return PBitSyndromeDecoder(
        update="gibbs",
        stop_on_first_valid=False,
        collect_valid_samples=True,
        vectorized=True,
        **kwargs,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    code = make_bb144_code()
    n = code.n
    rng = np.random.default_rng(args.seed)
    errors = [(rng.random(n) < args.p).astype(np.uint8) for _ in range(args.trials)]
    syndromes = [code.syndrome(e) for e in errors]
    true_keys = [gf2.bit_string(code.logical(e)) for e in errors]
    mean_true_w = float(np.mean([int(e.sum()) for e in errors]))

    print(f"# BB144  p={args.p}  trials={args.trials}  seed={args.seed}  "
          f"mean_true_weight={mean_true_w:.2f}", flush=True)

    # BP-OSD reference on the same errors.
    bposd = BPOSDDecoder(max_iter=100, osd_order=10)
    osd_succ = 0
    osd_w = 0.0
    t0 = time.perf_counter()
    for syn, key in zip(syndromes, true_keys):
        r = bposd.decode(code, syn, args.p, np.random.default_rng(7))
        if r.valid and gf2.bit_string(code.logical(r.correction)) == key:
            osd_succ += 1
        osd_w += int(r.correction.sum())
    print(f"{'bp_osd':<22} succ={osd_succ}/{args.trials}  "
          f"selW={osd_w/args.trials:5.2f}  t={(time.perf_counter()-t0)/args.trials:5.2f}s",
          flush=True)

    print(f"{'config':<22} {'succ':>9} {'cover':>9} {'covFail':>8} {'selFail':>8} "
          f"{'selW':>6} {'t/dec':>7}", flush=True)
    for cfg in CONFIGS:
        dec = build(cfg)
        succ = cover = cov_fail = sel_fail = 0
        wsum = 0.0
        t0 = time.perf_counter()
        for syn, key in zip(syndromes, true_keys):
            r = dec.decode(code, syn, args.p, np.random.default_rng(11))
            covered = key in r.metadata.get("ml_coset_logz", {})
            ok = r.valid and gf2.bit_string(code.logical(r.correction)) == key
            cover += covered
            wsum += int(r.metadata.get("selected_weight", r.correction.sum()))
            if ok:
                succ += 1
            elif covered:
                sel_fail += 1
            else:
                cov_fail += 1
        dt = (time.perf_counter() - t0) / args.trials
        print(f"{cfg['name']:<22} {succ:>6}/{args.trials} {cover:>6}/{args.trials} "
              f"{cov_fail:>8} {sel_fail:>8} {wsum/args.trials:>6.2f} {dt:>6.2f}s",
              flush=True)


if __name__ == "__main__":
    main()
