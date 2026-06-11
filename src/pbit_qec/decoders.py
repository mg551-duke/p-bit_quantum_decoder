"""Exact, stochastic, annealed, and hard-constrained coset decoders."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union

import numpy as np

import pbit_qec.gf2 as gf2
from pbit_qec.analysis import analyze_cosets
from pbit_qec.decoders_common import default_rho, prior_energy, prior_lambdas
from pbit_qec.types import BitArray, CSSCode, DecodeResult


def soft_energy(
    error: Sequence[int],
    syndrome: Sequence[int],
    code: CSSCode,
    lambdas: np.ndarray,
    rho: float,
) -> float:
    e = np.asarray(error, dtype=np.uint8)
    residual = code.syndrome(e) ^ np.asarray(syndrome, dtype=np.uint8)
    return prior_energy(e, lambdas) + rho * int(residual.sum())


def _coset_key(bits: Sequence[int]) -> str:
    return gf2.bit_string(bits)


def _empty_result(
    code: CSSCode,
    syndrome: Sequence[int],
    start: float,
    lambdas: np.ndarray,
) -> DecodeResult:
    correction = np.zeros(code.n, dtype=np.uint8)
    valid = code.is_valid_correction(correction, syndrome)
    logical = gf2.bit_tuple(code.logical(correction))
    return DecodeResult(
        correction=correction,
        logical_class=logical,
        valid=valid,
        energy=prior_energy(correction, lambdas),
        runtime_s=time.perf_counter() - start,
    )


@dataclass
class ExactCosetDecoder:
    max_nullity: int = 24
    name: str = "exact_coset"

    def decode(
        self,
        code: CSSCode,
        syndrome: Sequence[int],
        p: Union[float, Sequence[float]],
        rng: np.random.Generator,
    ) -> DecodeResult:
        del rng
        start = time.perf_counter()
        lambdas = prior_lambdas(p, code.n)
        try:
            analysis = analyze_cosets(code, syndrome, p, self.max_nullity)
        except ValueError as exc:
            if "exceeds" in str(exc):
                raise
            return _empty_result(code, syndrome, start, lambdas)

        best_key = analysis.best_coset
        correction = analysis.best_rep_by_coset[best_key]
        logical = gf2.bit_tuple(code.logical(correction))
        return DecodeResult(
            correction=correction,
            logical_class=logical,
            valid=True,
            energy=analysis.min_energy_by_coset[best_key],
            runtime_s=time.perf_counter() - start,
            valid_samples=analysis.valid_count,
            total_samples=analysis.valid_count,
            first_valid_sweep=0,
            distinct_logicals=len(analysis.coset_counts),
            coset_counts=analysis.coset_counts,
            coset_log_weights=analysis.coset_log_weights,
            metadata={
                "nullity": analysis.nullity,
                "min_energy_coset": analysis.min_energy_coset,
                "posterior_gap": analysis.posterior_gap,
                "energy_gap": analysis.energy_gap,
            },
        )


@dataclass
class PBitCosetSamplerDecoder:
    chains: int = 16
    burn_in_sweeps: int = 200
    sampling_sweeps: int = 1000
    thinning: int = 5
    beta: float = 1.0
    rho: Optional[float] = None
    update: str = "gibbs"
    initial_state: str = "random"
    name: str = "pbit_coset_sampler"

    def decode(
        self,
        code: CSSCode,
        syndrome: Sequence[int],
        p: Union[float, Sequence[float]],
        rng: np.random.Generator,
    ) -> DecodeResult:
        start = time.perf_counter()
        lambdas = prior_lambdas(p, code.n)
        rho = default_rho(lambdas) if self.rho is None else self.rho
        target = np.asarray(syndrome, dtype=np.uint8) % 2
        col_checks = [np.flatnonzero(code.H[:, j]).astype(np.int64) for j in range(code.n)]

        counts: Dict[str, int] = defaultdict(int)
        best_rep_by_coset: Dict[str, BitArray] = {}
        best_rep_energy: Dict[str, float] = {}
        total_samples = 0
        first_valid_sweep: Optional[int] = None
        best_valid: Optional[BitArray] = None
        best_valid_energy = float("inf")

        affine_seed = _affine_seed_state(code, target) if self.initial_state == "affine" else None
        if self.initial_state not in {"random", "affine"}:
            raise ValueError(f"unknown p-bit initial_state {self.initial_state!r}")

        for _ in range(self.chains):
            if affine_seed is None:
                e = rng.integers(0, 2, size=code.n, dtype=np.uint8)
            else:
                e = _sample_affine_state(affine_seed[0], affine_seed[1], rng)
            residual = (code.H @ e % 2).astype(np.uint8) ^ target
            invalid = int(residual.sum())
            for sweep in range(self.burn_in_sweeps + self.sampling_sweeps):
                for j in rng.permutation(code.n):
                    checks = col_checks[int(j)]
                    before = int(residual[checks].sum())
                    after = int(len(checks) - before)
                    delta = (1 - 2 * int(e[j])) * lambdas[j] + rho * (after - before)
                    if _should_flip(delta, int(e[j]), self.beta, rng, self.update):
                        e[j] ^= 1
                        residual[checks] ^= 1
                        invalid += after - before
                if sweep < self.burn_in_sweeps:
                    continue
                sample_sweep = sweep - self.burn_in_sweeps
                if sample_sweep % self.thinning != 0:
                    continue
                total_samples += 1
                if invalid != 0:
                    continue
                if first_valid_sweep is None:
                    first_valid_sweep = sweep
                e_prior = prior_energy(e, lambdas)
                key = _coset_key(code.logical(e))
                counts[key] += 1
                if key not in best_rep_energy or e_prior < best_rep_energy[key]:
                    best_rep_energy[key] = e_prior
                    best_rep_by_coset[key] = e.copy()
                if e_prior < best_valid_energy:
                    best_valid_energy = e_prior
                    best_valid = e.copy()

        if counts:
            best_key = max(counts, key=lambda key: (counts[key], -best_rep_energy[key]))
            correction = best_rep_by_coset[best_key]
        elif best_valid is not None:
            correction = best_valid
        else:
            correction = np.zeros(code.n, dtype=np.uint8)
        valid = code.is_valid_correction(correction, target)
        logical = gf2.bit_tuple(code.logical(correction))
        return DecodeResult(
            correction=correction,
            logical_class=logical,
            valid=valid,
            energy=prior_energy(correction, lambdas),
            runtime_s=time.perf_counter() - start,
            valid_samples=sum(counts.values()),
            total_samples=total_samples,
            first_valid_sweep=first_valid_sweep,
            distinct_logicals=len(counts),
            coset_counts=dict(counts),
            metadata={
                "rho": rho,
                "beta": self.beta,
                "update": self.update,
                "initial_state": self.initial_state,
            },
        )


@dataclass
class PBitAnnealedOptimizerDecoder:
    chains: int = 32
    sweeps: int = 1500
    beta_start: float = 0.2
    beta_stop: float = 6.0
    rho: Optional[float] = None
    update: str = "gibbs"
    name: str = "pbit_annealed_optimizer"

    def decode(
        self,
        code: CSSCode,
        syndrome: Sequence[int],
        p: Union[float, Sequence[float]],
        rng: np.random.Generator,
    ) -> DecodeResult:
        start = time.perf_counter()
        lambdas = prior_lambdas(p, code.n)
        rho = default_rho(lambdas) if self.rho is None else self.rho
        target = np.asarray(syndrome, dtype=np.uint8) % 2
        col_checks = [np.flatnonzero(code.H[:, j]).astype(np.int64) for j in range(code.n)]
        betas = np.geomspace(self.beta_start, self.beta_stop, self.sweeps)

        best_valid: Optional[BitArray] = None
        best_valid_energy = float("inf")
        best_any: Optional[BitArray] = None
        best_any_energy = float("inf")
        valid_hits = 0
        first_valid_sweep: Optional[int] = None

        for _ in range(self.chains):
            e = rng.integers(0, 2, size=code.n, dtype=np.uint8)
            residual = (code.H @ e % 2).astype(np.uint8) ^ target
            invalid = int(residual.sum())
            energy = prior_energy(e, lambdas) + rho * invalid
            for sweep, beta in enumerate(betas):
                for j in rng.permutation(code.n):
                    checks = col_checks[int(j)]
                    before = int(residual[checks].sum())
                    after = int(len(checks) - before)
                    delta = (1 - 2 * int(e[j])) * lambdas[j] + rho * (after - before)
                    if _should_flip(delta, int(e[j]), float(beta), rng, self.update):
                        e[j] ^= 1
                        residual[checks] ^= 1
                        invalid += after - before
                        energy += delta
                if energy < best_any_energy:
                    best_any_energy = energy
                    best_any = e.copy()
                if invalid == 0:
                    valid_hits += 1
                    e_prior = prior_energy(e, lambdas)
                    if first_valid_sweep is None:
                        first_valid_sweep = sweep
                    if e_prior < best_valid_energy:
                        best_valid_energy = e_prior
                        best_valid = e.copy()

        correction = best_valid if best_valid is not None else best_any
        if correction is None:
            correction = np.zeros(code.n, dtype=np.uint8)
        valid = code.is_valid_correction(correction, target)
        logical = gf2.bit_tuple(code.logical(correction))
        counts = {_coset_key(logical): valid_hits} if valid and valid_hits else {}
        return DecodeResult(
            correction=correction,
            logical_class=logical,
            valid=valid,
            energy=prior_energy(correction, lambdas) if valid else best_any_energy,
            runtime_s=time.perf_counter() - start,
            valid_samples=valid_hits,
            total_samples=self.chains * self.sweeps,
            first_valid_sweep=first_valid_sweep,
            distinct_logicals=len(counts),
            coset_counts=counts,
            metadata={
                "rho": rho,
                "beta_start": self.beta_start,
                "beta_stop": self.beta_stop,
                "update": self.update,
            },
        )


@dataclass
class MWPMDecoder:
    name: str = "mwpm"
    _matching_cache: Dict[bytes, object] = field(default_factory=dict, init=False, repr=False)

    def decode(
        self,
        code: CSSCode,
        syndrome: Sequence[int],
        p: Union[float, Sequence[float]],
        rng: np.random.Generator,
    ) -> DecodeResult:
        del rng
        start = time.perf_counter()
        lambdas = prior_lambdas(p, code.n)
        if int(code.H.sum(axis=0).max()) > 2:
            raise ValueError(
                "MWPMDecoder requires a graphlike check matrix with column weight <= 2"
            )
        cache_key = lambdas.tobytes()
        if cache_key not in self._matching_cache:
            import pymatching

            self._matching_cache[cache_key] = pymatching.Matching.from_check_matrix(
                code.H,
                weights=lambdas,
                use_virtual_boundary_node=True,
            )
        matching = self._matching_cache[cache_key]
        target = np.asarray(syndrome, dtype=np.uint8) % 2
        correction = np.asarray(matching.decode(target), dtype=np.uint8) % 2
        valid = code.is_valid_correction(correction, target)
        logical = gf2.bit_tuple(code.logical(correction))
        key = _coset_key(logical)
        return DecodeResult(
            correction=correction,
            logical_class=logical,
            valid=valid,
            energy=prior_energy(correction, lambdas),
            runtime_s=time.perf_counter() - start,
            valid_samples=int(valid),
            total_samples=1,
            first_valid_sweep=0 if valid else None,
            distinct_logicals=1 if valid else 0,
            coset_counts={key: 1} if valid else {},
            metadata={"weights": "log((1-p)/p)", "backend": "pymatching"},
        )


@dataclass
class PBitNullspaceCosetSamplerDecoder:
    samples: int = 2000
    burn_in: int = 100
    updates_per_sample: int = 1
    beta: float = 1.0
    max_nullity: int = 64
    initial_state: str = "random"
    name: str = "pbit_nullspace_sampler"

    def decode(
        self,
        code: CSSCode,
        syndrome: Sequence[int],
        p: Union[float, Sequence[float]],
        rng: np.random.Generator,
    ) -> DecodeResult:
        start = time.perf_counter()
        lambdas = prior_lambdas(p, code.n)
        x0, basis = gf2.solve_affine(code.H, syndrome)
        if x0 is None:
            return _empty_result(code, syndrome, start, lambdas)
        if basis.shape[0] > self.max_nullity:
            raise ValueError(f"nullity {basis.shape[0]} exceeds {self.max_nullity}")
        if self.initial_state not in {"particular", "random"}:
            raise ValueError(f"unknown nullspace p-bit initial_state {self.initial_state!r}")
        if self.updates_per_sample <= 0:
            raise ValueError("updates_per_sample must be positive")
        if basis.shape[0] == 0:
            logical = gf2.bit_tuple(code.logical(x0))
            key = _coset_key(logical)
            return DecodeResult(
                correction=x0,
                logical_class=logical,
                valid=True,
                energy=prior_energy(x0, lambdas),
                runtime_s=time.perf_counter() - start,
                valid_samples=self.samples,
                total_samples=self.samples,
                first_valid_sweep=0,
                distinct_logicals=1,
                coset_counts={key: self.samples},
                metadata={
                    "nullity": 0,
                    "beta": self.beta,
                    "updates_per_sample": self.updates_per_sample,
                    "update": "gibbs",
                    "state_space": "affine_nullspace",
                    "initial_state": self.initial_state,
                },
            )

        z = np.zeros(basis.shape[0], dtype=np.uint8)
        e = x0.copy()
        if self.initial_state == "random":
            z = rng.integers(0, 2, size=basis.shape[0], dtype=np.uint8)
            selected = z.astype(bool)
            if np.any(selected):
                e ^= np.bitwise_xor.reduce(basis[selected], axis=0)
        energy = prior_energy(e, lambdas)
        basis_indices = [np.flatnonzero(row).astype(np.int64) for row in basis]
        signed_lambdas = lambdas * (1 - 2 * e.astype(np.int8))

        counts: Dict[str, int] = defaultdict(int)
        best_rep_by_coset: Dict[str, BitArray] = {}
        best_rep_energy: Dict[str, float] = {}

        for step in range(self.burn_in + self.samples):
            for _ in range(self.updates_per_sample):
                i = int(rng.integers(0, basis.shape[0]))
                indices = basis_indices[i]
                delta = _basis_flip_delta(signed_lambdas, indices)
                delta_one_minus_zero = delta if z[i] == 0 else -delta
                probability_one = _pbit_probability_one(delta_one_minus_zero, self.beta)
                next_bit = int(rng.random() < probability_one)
                if next_bit != int(z[i]):
                    e[indices] ^= 1
                    signed_lambdas[indices] *= -1
                    energy += delta
                    z[i] = next_bit
            if step < self.burn_in:
                continue
            key = _coset_key(code.logical(e))
            counts[key] += 1
            if key not in best_rep_energy or energy < best_rep_energy[key]:
                best_rep_energy[key] = energy
                best_rep_by_coset[key] = e.copy()

        best_key = max(counts, key=lambda key: (counts[key], -best_rep_energy[key]))
        correction = best_rep_by_coset[best_key]
        logical = gf2.bit_tuple(code.logical(correction))
        return DecodeResult(
            correction=correction,
            logical_class=logical,
            valid=True,
            energy=prior_energy(correction, lambdas),
            runtime_s=time.perf_counter() - start,
            valid_samples=sum(counts.values()),
            total_samples=self.samples,
            first_valid_sweep=0,
            distinct_logicals=len(counts),
            coset_counts=dict(counts),
            metadata={
                "nullity": int(basis.shape[0]),
                "beta": self.beta,
                "updates_per_sample": self.updates_per_sample,
                "update": "gibbs",
                "state_space": "affine_nullspace",
                "initial_state": self.initial_state,
            },
        )


@dataclass
class NullspaceCosetSamplerDecoder:
    samples: int = 2000
    burn_in: int = 100
    beta: float = 1.0
    max_nullity: int = 64
    name: str = "nullspace_coset_sampler"

    def decode(
        self,
        code: CSSCode,
        syndrome: Sequence[int],
        p: Union[float, Sequence[float]],
        rng: np.random.Generator,
    ) -> DecodeResult:
        start = time.perf_counter()
        lambdas = prior_lambdas(p, code.n)
        x0, basis = gf2.solve_affine(code.H, syndrome)
        if x0 is None:
            return _empty_result(code, syndrome, start, lambdas)
        if basis.shape[0] > self.max_nullity:
            raise ValueError(f"nullity {basis.shape[0]} exceeds {self.max_nullity}")
        if basis.shape[0] == 0:
            logical = gf2.bit_tuple(code.logical(x0))
            key = _coset_key(logical)
            return DecodeResult(
                correction=x0,
                logical_class=logical,
                valid=True,
                energy=prior_energy(x0, lambdas),
                runtime_s=time.perf_counter() - start,
                valid_samples=self.samples,
                total_samples=self.samples,
                first_valid_sweep=0,
                distinct_logicals=1,
                coset_counts={key: self.samples},
                metadata={"nullity": 0},
            )

        e = x0.copy()
        counts: Dict[str, int] = defaultdict(int)
        best_rep_by_coset: Dict[str, BitArray] = {}
        best_rep_energy: Dict[str, float] = {}
        for step in range(self.burn_in + self.samples):
            b = basis[int(rng.integers(0, basis.shape[0]))]
            proposal = e ^ b
            delta = prior_energy(proposal, lambdas) - prior_energy(e, lambdas)
            if _accept(delta, self.beta, rng):
                e = proposal
            if step < self.burn_in:
                continue
            key = _coset_key(code.logical(e))
            e_prior = prior_energy(e, lambdas)
            counts[key] += 1
            if key not in best_rep_energy or e_prior < best_rep_energy[key]:
                best_rep_energy[key] = e_prior
                best_rep_by_coset[key] = e.copy()

        best_key = max(counts, key=lambda key: (counts[key], -best_rep_energy[key]))
        correction = best_rep_by_coset[best_key]
        logical = gf2.bit_tuple(code.logical(correction))
        return DecodeResult(
            correction=correction,
            logical_class=logical,
            valid=True,
            energy=prior_energy(correction, lambdas),
            runtime_s=time.perf_counter() - start,
            valid_samples=sum(counts.values()),
            total_samples=self.samples,
            first_valid_sweep=0,
            distinct_logicals=len(counts),
            coset_counts=dict(counts),
            metadata={"nullity": int(basis.shape[0]), "beta": self.beta},
        )


def _accept(delta: float, beta: float, rng: np.random.Generator) -> bool:
    if delta <= 0:
        return True
    return bool(rng.random() < np.exp(-beta * delta))


def _should_flip(
    flip_delta: float,
    current_bit: int,
    beta: float,
    rng: np.random.Generator,
    update: str,
) -> bool:
    if update == "metropolis":
        return _accept(flip_delta, beta, rng)
    if update != "gibbs":
        raise ValueError(f"unknown p-bit update rule {update!r}")

    delta_one_minus_zero = flip_delta if current_bit == 0 else -flip_delta
    probability_one = _pbit_probability_one(delta_one_minus_zero, beta)
    next_bit = int(rng.random() < probability_one)
    return next_bit != current_bit


def _pbit_probability_one(delta_one_minus_zero: float, beta: float) -> float:
    """Return P(e_j = 1 | rest) for Delta = E(e_j=1)-E(e_j=0)."""

    x = beta * delta_one_minus_zero
    if x >= 0:
        z = np.exp(-x)
        return float(z / (1.0 + z))
    return float(1.0 / (1.0 + np.exp(x)))


def _basis_flip_delta(signed_lambdas: np.ndarray, indices: np.ndarray) -> float:
    return float(signed_lambdas[indices].sum())


def _affine_seed_state(
    code: CSSCode,
    syndrome: Sequence[int],
) -> Optional[tuple[BitArray, BitArray]]:
    x0, basis = gf2.solve_affine(code.H, syndrome)
    if x0 is None:
        return None
    return x0, basis


def _sample_affine_state(x0: BitArray, basis: BitArray, rng: np.random.Generator) -> BitArray:
    state = x0.copy()
    if basis.shape[0] == 0:
        return state
    coeffs = rng.integers(0, 2, size=basis.shape[0], dtype=np.uint8)
    selected = coeffs.astype(bool)
    if np.any(selected):
        state ^= np.bitwise_xor.reduce(basis[selected], axis=0)
    return state
