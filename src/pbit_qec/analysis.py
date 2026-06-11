"""Exact coset diagnostics used by decoders and experiment scripts."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import numpy as np
from scipy.special import logsumexp

import pbit_qec.gf2 as gf2
from pbit_qec.decoders_common import prior_energy, prior_lambdas
from pbit_qec.types import BitArray, CSSCode


@dataclass(frozen=True)
class CosetAnalysis:
    syndrome_id: int
    nullity: int
    valid_count: int
    best_coset: str
    min_energy_coset: str
    posterior_gap: float
    energy_gap: float
    coset_counts: Dict[str, int]
    coset_log_weights: Dict[str, float]
    min_energy_by_coset: Dict[str, float]
    best_rep_by_coset: Dict[str, BitArray]


def logsumexp_pair(a: Optional[float], b: float) -> float:
    if a is None:
        return b
    hi = max(a, b)
    return hi + math.log(math.exp(a - hi) + math.exp(b - hi))


def analyze_cosets(
    code: CSSCode,
    syndrome: Sequence[int],
    p: Union[float, Sequence[float]],
    max_nullity: int = 24,
) -> CosetAnalysis:
    lambdas = prior_lambdas(p, code.n)
    target = gf2.as_binary_vector(syndrome)
    x0, basis = gf2.solve_affine(code.H, target)
    if x0 is None:
        raise ValueError("syndrome has no affine solution")
    if basis.shape[0] > max_nullity:
        raise ValueError(f"exact enumeration nullity {basis.shape[0]} exceeds {max_nullity}")
    if basis.shape[0] <= 20:
        return _analyze_cosets_vectorized(code, target, x0, basis, lambdas)

    log_weights: Dict[str, float] = {}
    counts: Dict[str, int] = defaultdict(int)
    best_rep_by_coset: Dict[str, BitArray] = {}
    min_energy_by_coset: Dict[str, float] = {}

    for e in gf2.enumerate_affine(x0, basis, max_nullity):
        key = gf2.bit_string(code.logical(e))
        e_prior = prior_energy(e, lambdas)
        log_weights[key] = logsumexp_pair(log_weights.get(key), -e_prior)
        counts[key] += 1
        if key not in min_energy_by_coset or e_prior < min_energy_by_coset[key]:
            min_energy_by_coset[key] = e_prior
            best_rep_by_coset[key] = e.copy()

    if not log_weights:
        raise ValueError("exact enumeration produced no states")

    sorted_log_weights = sorted(log_weights.values(), reverse=True)
    sorted_min_energies = sorted(min_energy_by_coset.values())
    best_coset = max(log_weights, key=lambda key: log_weights[key])
    min_energy_coset = min(min_energy_by_coset, key=lambda key: min_energy_by_coset[key])
    posterior_gap = (
        sorted_log_weights[0] - sorted_log_weights[1] if len(sorted_log_weights) > 1 else math.inf
    )
    energy_gap = (
        sorted_min_energies[1] - sorted_min_energies[0]
        if len(sorted_min_energies) > 1
        else math.inf
    )

    return CosetAnalysis(
        syndrome_id=gf2.pack_bits(target),
        nullity=int(basis.shape[0]),
        valid_count=sum(counts.values()),
        best_coset=best_coset,
        min_energy_coset=min_energy_coset,
        posterior_gap=float(posterior_gap),
        energy_gap=float(energy_gap),
        coset_counts=dict(counts),
        coset_log_weights=log_weights,
        min_energy_by_coset=min_energy_by_coset,
        best_rep_by_coset=best_rep_by_coset,
    )


def _analyze_cosets_vectorized(
    code: CSSCode,
    target: BitArray,
    x0: BitArray,
    basis: BitArray,
    lambdas: np.ndarray,
) -> CosetAnalysis:
    dim = int(basis.shape[0])
    count = 1 << dim
    masks = np.arange(count, dtype=np.uint64)
    if dim:
        shifts = np.arange(dim, dtype=np.uint64)
        coeffs = ((masks[:, None] >> shifts) & 1).astype(np.uint8)
        states = (coeffs @ basis % 2).astype(np.uint8)
        states ^= x0
    else:
        states = x0[None, :].copy()

    logicals = (states @ code.A.T % 2).astype(np.uint8)
    logical_ids = np.array([gf2.pack_bits(row) for row in logicals], dtype=np.int64)
    energies = states @ lambdas
    unique_ids = np.unique(logical_ids)

    counts: Dict[str, int] = {}
    log_weights: Dict[str, float] = {}
    min_energy_by_coset: Dict[str, float] = {}
    best_rep_by_coset: Dict[str, BitArray] = {}

    for logical_id in unique_ids:
        mask = logical_ids == logical_id
        key = gf2.bit_string(gf2.unpack_bits(int(logical_id), code.k))
        coset_energies = energies[mask]
        coset_indices = np.flatnonzero(mask)
        min_local = int(np.argmin(coset_energies))
        min_global = int(coset_indices[min_local])
        counts[key] = int(mask.sum())
        log_weights[key] = float(logsumexp(-coset_energies))
        min_energy_by_coset[key] = float(coset_energies[min_local])
        best_rep_by_coset[key] = states[min_global].copy()

    sorted_log_weights = sorted(log_weights.values(), reverse=True)
    sorted_min_energies = sorted(min_energy_by_coset.values())
    best_coset = max(log_weights, key=lambda key: log_weights[key])
    min_energy_coset = min(min_energy_by_coset, key=lambda key: min_energy_by_coset[key])
    posterior_gap = (
        sorted_log_weights[0] - sorted_log_weights[1]
        if len(sorted_log_weights) > 1
        else math.inf
    )
    energy_gap = (
        sorted_min_energies[1] - sorted_min_energies[0]
        if len(sorted_min_energies) > 1
        else math.inf
    )
    return CosetAnalysis(
        syndrome_id=gf2.pack_bits(target),
        nullity=dim,
        valid_count=count,
        best_coset=best_coset,
        min_energy_coset=min_energy_coset,
        posterior_gap=float(posterior_gap),
        energy_gap=float(energy_gap),
        coset_counts=counts,
        coset_log_weights=log_weights,
        min_energy_by_coset=min_energy_by_coset,
        best_rep_by_coset=best_rep_by_coset,
    )
