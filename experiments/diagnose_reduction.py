"""Test whether stronger within-coset weight reduction would fix min_weight failures.

The current `_reduce_weight` is greedy single-generator descent and can stop at a
local minimum that a two-generator move would escape. Because `e_true` is itself a
valid member of the true logical coset, reducing it gives a near-minimum true-coset
weight to compare against the decoder's selected (possibly wrong-coset) correction.

For each failure we classify:
  - reduction-fixable: true coset's strongly-reduced weight < selected coset's
    strongly-reduced weight (better reduction would flip min_weight to the true coset)
  - fundamental: true coset is genuinely no lighter even under strong reduction.

"strong" reduction augments the generator pool with pairwise stabilizer sums so greedy
can take two-generator moves, then re-runs greedy.

Run:
    .venv/bin/python experiments/diagnose_reduction.py --p 0.08 --trials 40
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from pbit_qec.codes import make_code
from pbit_qec.decoders import PBitSyndromeDecoder, _reduce_weight, _stabilizer_rows
from pbit_qec.simulation import sample_error

BB = dict(
    chains=256, sweeps=250, beta_start=0.3, beta_stop=5.0, rho_scale=1.0, update="gibbs",
    stop_on_first_valid=False, collect_valid_samples=True, weight_reduction=True,
    selection="min_weight",
)


def augment_pairwise(stabs: np.ndarray, max_pairs: int = 6000) -> np.ndarray:
    """Stabilizer pool plus pairwise XOR sums (enables two-generator greedy moves)."""
    rows = [stabs]
    pairs = list(itertools.combinations(range(stabs.shape[0]), 2))
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    if pairs:
        combo = np.array([stabs[i] ^ stabs[j] for i, j in pairs], dtype=np.uint8)
        rows.append(combo)
    return np.unique(np.vstack(rows), axis=0)


def reduce_strong(e: np.ndarray, stabs: np.ndarray, aug: np.ndarray) -> np.ndarray:
    cur = _reduce_weight(e, stabs)
    for _ in range(4):
        nxt = _reduce_weight(cur, aug)
        if int(nxt.sum()) >= int(cur.sum()):
            break
        cur = nxt
    return cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="bb")
    ap.add_argument("--size", type=int, default=144)
    ap.add_argument("--p", type=float, default=0.08)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=20260620)
    args = ap.parse_args()

    code = make_code(args.family, args.size)
    stabs = _stabilizer_rows(code)
    aug = augment_pairwise(stabs)
    print(f"stabilizers={stabs.shape[0]} augmented={aug.shape[0]}")

    fails = 0
    fixable = 0  # strong reduction of true coset beats strong reduction of selected coset
    fixable_vs_current = 0  # true-strong beats the decoder's current selected weight
    sampling_gap = 0  # true coset already lighter under plain greedy (sampling, not reduction)
    fundamental = 0

    for t in range(args.trials):
        seed = args.base_seed + t
        e_true = sample_error(code.n, args.p, np.random.default_rng(seed))
        syn = code.syndrome(e_true)
        drng = np.random.default_rng(seed + 3027)
        res = PBitSyndromeDecoder(**BB).decode(code, syn, args.p, drng)
        ok = bool(res.valid and not code.logical(e_true ^ res.correction).any())
        if ok:
            continue
        fails += 1
        sel = np.asarray(res.correction, dtype=np.uint8)
        w_sel = int(sel.sum())
        w_sel_strong = int(reduce_strong(sel, stabs, aug).sum())
        w_true_greedy = int(_reduce_weight(e_true, stabs).sum())
        w_true_strong = int(reduce_strong(e_true, stabs, aug).sum())
        if w_true_strong < w_sel_strong:
            fixable += 1
        else:
            fundamental += 1
        if w_true_strong < w_sel:
            fixable_vs_current += 1
        if w_true_greedy < w_sel:
            sampling_gap += 1

    print(f"\n{args.family} d={args.size} p={args.p} trials={args.trials}")
    print(f"  failures: {fails}")
    print(f"  reduction-fixable (true_strong < sel_strong): {fixable}/{fails}")
    print(f"  fundamental       (true_strong >= sel_strong): {fundamental}/{fails}")
    print(f"  true_strong < current selected weight        : {fixable_vs_current}/{fails}")
    print(f"  true already lighter under plain greedy      : {sampling_gap}/{fails}")


if __name__ == "__main__":
    main()
