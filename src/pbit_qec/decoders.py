"""Exact, stochastic, annealed, and hard-constrained coset decoders."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

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


def _prior_probabilities(p: Union[float, Sequence[float]], n: int) -> np.ndarray:
    probs = np.asarray(p, dtype=float)
    if probs.ndim == 0:
        probs = np.full(n, float(probs))
    if probs.shape != (n,):
        raise ValueError(f"expected scalar p or vector shape {(n,)}, got {probs.shape}")
    if np.any(probs <= 0.0) or np.any(probs >= 0.5):
        raise ValueError("p must be in the open interval (0, 0.5)")
    return probs


def _stabilizer_rows(code: CSSCode) -> np.ndarray:
    """Return stabilizer generators: rows v with H v = 0 and A v = 0 over GF(2).

    Such vectors preserve both the syndrome and the logical coset, so XORing any
    combination into a valid correction keeps it valid and in the same coset.
    Prefers the low-weight complementary CSS checks (``metadata['h_x']``) when
    available; otherwise falls back to a computed nullspace basis of ``[H; A]``.
    """
    hx = code.metadata.get("h_x")
    if hx is not None:
        rows = np.asarray(hx, dtype=np.uint8) % 2
        if rows.ndim == 2 and rows.shape[1] == code.n:
            rows = rows[rows.any(axis=1)]
            if rows.size:
                return np.unique(rows, axis=0)
    basis = gf2.nullspace(np.vstack([code.H, code.A]).astype(np.uint8))
    return basis.astype(np.uint8)


def _reduce_weight(error: BitArray, stabs: np.ndarray, max_passes: int = 1000) -> BitArray:
    """Greedily XOR stabilizers into ``error`` to lower its Hamming weight.

    Each pass applies the single generator that decreases the weight most and
    stops at a local minimum where no generator helps. XORing stabilizer ``s``
    changes the weight by ``|s| - 2 * <s, e>``.
    """
    e = np.asarray(error, dtype=np.uint8).copy()
    if stabs.size == 0:
        return e
    stab_int = stabs.astype(np.int64)
    stab_weight = stabs.sum(axis=1).astype(np.int64)
    for _ in range(max_passes):
        delta = stab_weight - 2 * (stab_int @ e.astype(np.int64))
        j = int(np.argmin(delta))
        if delta[j] >= 0:
            break
        e = (e ^ stabs[j]).astype(np.uint8)
    return e


def _apply_weight_reduction(
    selection: str,
    best_rep_by_coset: Dict[str, BitArray],
    best_rep_energy: Dict[str, float],
    coset_logz: Dict[str, float],
    stabs: np.ndarray,
    lambdas: np.ndarray,
    best_valid: Optional[BitArray] = None,
) -> Optional[BitArray]:
    """Reduce coset representatives to low weight and refresh derived stats.

    Updates ``best_rep_by_coset``/``best_rep_energy`` in place and folds each newly
    discovered lower-energy representative into the ``coset_logz`` free-energy
    estimate (benefitting ``ml_coset`` selection).

    Returns the best global correction for the active ``selection`` rule:
    lowest Hamming weight (``min_weight``, energy tie-break) or lowest prior energy
    (otherwise). The incoming ``best_valid`` and its own reduction are always
    included among the candidates, so the result can never regress below the
    correction that plain selection would have returned.
    """
    for ckey in list(best_rep_by_coset):
        reduced = _reduce_weight(best_rep_by_coset[ckey], stabs)
        reduced_energy = prior_energy(reduced, lambdas)
        if reduced_energy < best_rep_energy[ckey]:
            best_rep_by_coset[ckey] = reduced
            best_rep_energy[ckey] = reduced_energy
            previous = coset_logz.get(ckey)
            coset_logz[ckey] = (
                -reduced_energy
                if previous is None
                else float(np.logaddexp(previous, -reduced_energy))
            )
    candidates: List[BitArray] = list(best_rep_by_coset.values())
    if best_valid is not None:
        candidates.append(np.asarray(best_valid, dtype=np.uint8))
        candidates.append(_reduce_weight(best_valid, stabs))
    if not candidates:
        return None
    if selection == "min_weight":
        return min(candidates, key=lambda v: (int(v.sum()), prior_energy(v, lambdas)))
    return min(candidates, key=lambda v: prior_energy(v, lambdas))


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


def _annealing_schedule(start: float, stop: float, steps: int, shape: str) -> np.ndarray:
    """Inverse-temperature (beta) annealing schedule of length ``steps``.

    Every shape ramps beta from ``start`` to ``stop``; only the spacing differs:

    - ``geometric``: constant multiplicative steps (default; equal spacing in
      log-beta). Spends most sweeps cold, then climbs quickly.
    - ``linear``: constant additive steps in beta.
    - ``sigmoid``: logistic ramp in log-beta space that dwells longer at the cold
      (exploration) and hot (commitment) ends and crosses the middle quickly.

    ``geometric`` and ``linear`` reproduce ``np.geomspace``/``np.linspace`` exactly,
    so the default keeps prior behavior unchanged.
    """
    steps = max(int(steps), 1)
    if shape == "geometric":
        return np.geomspace(start, stop, steps)
    if shape == "linear":
        return np.linspace(start, stop, steps)
    if shape == "sigmoid":
        if steps == 1:
            return np.array([start], dtype=float)
        x = np.linspace(-6.0, 6.0, steps)
        s = 1.0 / (1.0 + np.exp(-x))
        s = (s - s[0]) / (s[-1] - s[0])
        return np.exp(np.log(start) + s * (np.log(stop) - np.log(start)))
    raise ValueError(f"unknown beta_schedule_shape {shape!r}")


@dataclass
class PBitSyndromeDecoder:
    chains: int = 8
    sweeps: int = 500
    beta: float = 0.6
    beta_start: Optional[float] = None
    beta_stop: Optional[float] = None
    beta_schedule_shape: str = "geometric"
    anneal_prior: bool = False
    prior_scale: float = 1.0
    gamma_active: Optional[float] = None
    gamma_far: Optional[float] = None
    idle_update_rate: float = 1.0
    rho: Optional[float] = None
    rho_scale: Optional[float] = None
    update: str = "gibbs"
    initial_probability: Optional[float] = 0.0
    stop_on_first_valid: bool = False
    collect_valid_samples: bool = True
    selection: str = "min_weight"
    target_valid_solutions: int = 10
    vectorized: bool = True
    measure_sweeps: int = 0
    beta_measure: float = 1.0
    weight_reduction: bool = True
    coloring: bool = False
    simultaneous: bool = False
    sync_damping: float = 1.0
    name: str = "pbit_syndrome"

    def _resolve_gammas(self) -> tuple[float, float]:
        """Resolve the active/far prior-sharpening factors for the active-region
        search. Each defaults to ``prior_scale`` when unset, so the knobs are a
        no-op (identical flip field) unless explicitly tuned."""
        base = float(self.prior_scale)
        g_active = base if self.gamma_active is None else float(self.gamma_active)
        g_far = base if self.gamma_far is None else float(self.gamma_far)
        return g_active, g_far

    def _color_layout(
        self,
        code: CSSCode,
        col_pad: np.ndarray,
        deg_mask: np.ndarray,
        col_deg: np.ndarray,
    ) -> list[tuple]:
        """Greedy-color ``code.H`` columns into check-disjoint blocks for block Gibbs.

        Reuses :func:`pbit_qec.circuit_decoder.greedy_coloring` (lazily imported so
        the core decoder import stays free of the circuit/DEM/scipy stack). Columns
        in a color share no check, hence are conditionally independent and can be
        resampled together. Only the ``p``-independent structure is returned (and
        cached per ``H`` identity); the per-block ``lambda`` slice is rebuilt by the
        caller each decode so a reused instance stays correct across error rates.

        Returns a list of ``(cols, deg_c, cp_c, vm_c, fchk_c, flcol_c)`` per color:
            cols    -- global column indices in the color, ``(k,)``
            deg_c   -- per-column check degree, ``(k,)``
            cp_c    -- padded incident-check indices ``col_pad[cols]``, ``(k, max_deg)``
            vm_c    -- validity mask ``deg_mask[cols]``, ``(k, max_deg)``
            fchk_c  -- flat incident-check indices (unique within color), ``(L,)``
            flcol_c -- local column (0..k-1) owning each ``fchk_c`` entry, ``(L,)``
        """
        cache = getattr(self, "_color_cache", None)
        key = id(code.H)
        if cache is not None and cache[0] == key:
            return cache[1]
        from scipy import sparse  # lazy: keep scipy/circuit deps off the import path

        from pbit_qec.circuit_decoder import greedy_coloring

        colors = greedy_coloring(sparse.csc_matrix(np.asarray(code.H, dtype=np.uint8)))
        struct: list[tuple] = []
        for cols in colors:
            cols = np.asarray(cols, dtype=np.int64)
            k = int(cols.size)
            deg_c = col_deg[cols]
            cp_c = col_pad[cols]
            vm_c = deg_mask[cols]
            # cp_c[vm_c] flattens row-major -> incident checks grouped by column,
            # matching np.repeat(arange(k), deg_c). Checks are unique within a color
            # (greedy_coloring guarantees check-disjointness) so the residual scatter
            # XOR cannot alias.
            fchk_c = cp_c[vm_c].astype(np.int64)
            flcol_c = np.repeat(np.arange(k), deg_c).astype(np.int64)
            struct.append((cols, deg_c, cp_c, vm_c, fchk_c, flcol_c))
        self._color_cache = (key, struct)
        return struct

    def _row_layout(self, code: CSSCode) -> tuple[np.ndarray, np.ndarray]:
        """Per-check incident-bit index pad for the synchronous (Jacobi) update.

        Returns ``(row_pad, row_mask)`` where ``row_pad[i, :]`` lists the error bits
        incident to check ``i`` (zero-padded, ``row_mask`` marks the valid entries). A
        simultaneous sweep flips every chain from one state snapshot, so each check
        toggles by the *parity* of how many of its incident bits flipped -- one padded
        gather over checks, no scatter. Cached on ``id(code.H)`` (``p``-independent).
        """
        cache = getattr(self, "_row_cache", None)
        key = id(code.H)
        if cache is not None and cache[0] == key:
            return cache[1]
        hz = np.asarray(code.H, dtype=np.uint8)
        row_deg = hz.sum(axis=1).astype(np.int64)
        max_row = int(row_deg.max()) if hz.size else 0
        row_pad = np.zeros((code.m, max_row), dtype=np.int64)
        row_mask = np.zeros((code.m, max_row), dtype=bool)
        for i in range(code.m):
            bits = np.flatnonzero(hz[i])
            row_pad[i, : bits.size] = bits
            row_mask[i, : bits.size] = True
        layout = (row_pad, row_mask)
        self._row_cache = (key, layout)
        return layout

    def decode(
        self,
        code: CSSCode,
        syndrome: Sequence[int],
        p: Union[float, Sequence[float]],
        rng: np.random.Generator,
    ) -> DecodeResult:
        start = time.perf_counter()
        lambdas = prior_lambdas(p, code.n)
        probabilities = _prior_probabilities(p, code.n)
        rho = self._resolve_rho(lambdas)
        target = np.asarray(syndrome, dtype=np.uint8) % 2
        col_checks = [np.flatnonzero(code.H[:, j]).astype(np.int64) for j in range(code.n)]
        beta_schedule = self._beta_schedule()
        initial_probability = self._initial_probability(probabilities)
        if self.selection not in {"best_energy", "counts", "min_weight", "ml_coset"}:
            raise ValueError(f"unknown p-bit syndrome selection {self.selection!r}")
        if self.stop_on_first_valid and self.selection != "best_energy":
            raise ValueError(
                "stop_on_first_valid=True returns the first valid sample in chain order, "
                f"which defeats selection={self.selection!r} (it aggregates over the full "
                "ensemble). Set stop_on_first_valid=False, or use target_valid_solutions to "
                "cap exploration while still selecting across the collected samples."
            )
        if self.target_valid_solutions < 0:
            raise ValueError("target_valid_solutions must be nonnegative")
        if self.measure_sweeps < 0:
            raise ValueError("measure_sweeps must be nonnegative")
        if self.beta_measure <= 0.0:
            raise ValueError("beta_measure must be positive")
        if self.prior_scale <= 0.0:
            raise ValueError("prior_scale must be positive")
        if self.gamma_active is not None and self.gamma_active <= 0.0:
            raise ValueError("gamma_active must be positive")
        if self.gamma_far is not None and self.gamma_far <= 0.0:
            raise ValueError("gamma_far must be positive")
        if not 0.0 <= self.idle_update_rate <= 1.0:
            raise ValueError("idle_update_rate must be in [0, 1]")
        if self.coloring and not self.vectorized:
            raise ValueError("coloring requires vectorized=True (chromatic block Gibbs)")
        if self.simultaneous and not self.vectorized:
            raise ValueError("simultaneous requires vectorized=True (synchronous Jacobi sweep)")
        if self.simultaneous and self.coloring:
            raise ValueError("coloring and simultaneous are mutually exclusive update schedules")
        if not 0.0 < self.sync_damping <= 1.0:
            raise ValueError("sync_damping must be in (0, 1]")
        measure_start: Optional[int] = None
        if self.measure_sweeps > 0:
            measure_start = int(beta_schedule.shape[0])
            beta_schedule = np.concatenate(
                [beta_schedule, np.full(self.measure_sweeps, float(self.beta_measure))]
            )
        stab_rows = _stabilizer_rows(code) if self.weight_reduction else None

        if self.vectorized:
            return self._decode_batched(
                code=code,
                target=target,
                lambdas=lambdas,
                rho=rho,
                col_checks=col_checks,
                beta_schedule=beta_schedule,
                initial_probability=initial_probability,
                start=start,
                rng=rng,
                measure_start=measure_start,
                stab_rows=stab_rows,
            )

        counts: Dict[str, int] = defaultdict(int)
        coset_logz: Dict[str, float] = {}
        best_rep_by_coset: Dict[str, BitArray] = {}
        best_rep_energy: Dict[str, float] = {}
        best_valid: Optional[BitArray] = None
        best_valid_energy = float("inf")
        best_valid_weight: Optional[int] = None
        best_any: Optional[BitArray] = None
        best_any_energy = float("inf")
        valid_solution_keys: set[str] = set()
        best_residual_weight = code.m + 1
        final_residual_weight = code.m
        first_valid_sweep: Optional[int] = None
        first_valid_chain: Optional[int] = None
        valid_hits = 0
        total_samples = 0
        measure_valid = 0
        initial_residual_weight: Optional[int] = None
        total_sweeps = self.sweeps if measure_start is None else int(beta_schedule.shape[0])
        # Active-region knobs (no-op when unset): bits adjacent to a firing check use
        # gamma_active, far/idle bits (no firing incident check) use gamma_far, and
        # idle bits are attempted only with probability idle_rate (idle_gate draws
        # are taken only when idle_rate < 1 so the default RNG stream is unchanged).
        g_active, g_far = self._resolve_gammas()
        idle_rate = float(self.idle_update_rate)
        idle_gate = idle_rate < 1.0

        def record_valid(e_vec, logical_vec, e_prior, e_weight, sweep, chain):
            """Fold a syndrome-satisfying state into the candidate pool.

            Updates the best correction (per ``selection``), the distinct-solution
            set, the per-coset representative, and its free-energy estimate. Called
            both at sweep boundaries and immediately after any accepted flip that
            lands the chain on the target syndrome, so a valid correction reached
            inside a sweep is never lost. Consumes no randomness, so the trajectory
            is identical to before. Returns the coset key for sampling counts.
            """
            nonlocal best_valid, best_valid_energy, best_valid_weight
            nonlocal first_valid_sweep, first_valid_chain
            if first_valid_sweep is None:
                first_valid_sweep = sweep
                first_valid_chain = chain
            logical_key = _coset_key(logical_vec)
            if logical_key not in best_rep_energy or e_prior < best_rep_energy[logical_key]:
                best_rep_energy[logical_key] = e_prior
                best_rep_by_coset[logical_key] = e_vec.copy()
            config_key = e_vec.tobytes()
            if config_key not in valid_solution_keys:
                valid_solution_keys.add(config_key)
                prev_logz = coset_logz.get(logical_key)
                coset_logz[logical_key] = (
                    -e_prior
                    if prev_logz is None
                    else float(np.logaddexp(prev_logz, -e_prior))
                )
            is_better_valid = (
                self.selection == "min_weight"
                and (
                    best_valid_weight is None
                    or e_weight < best_valid_weight
                    or (e_weight == best_valid_weight and e_prior < best_valid_energy)
                )
            ) or (self.selection != "min_weight" and e_prior < best_valid_energy)
            if is_better_valid:
                best_valid_weight = e_weight
                best_valid_energy = e_prior
                best_valid = e_vec.copy()
            return logical_key

        for chain in range(self.chains):
            e = (rng.random(code.n) < initial_probability).astype(np.uint8)
            residual = (code.H @ e % 2).astype(np.uint8) ^ target
            invalid = int(residual.sum())
            if initial_residual_weight is None:
                initial_residual_weight = invalid
            energy = prior_energy(e, lambdas) + rho * invalid

            for sweep in range(total_sweeps + 1):
                total_samples += 1
                final_residual_weight = invalid
                if invalid < best_residual_weight or (
                    invalid == best_residual_weight and energy < best_any_energy
                ):
                    best_residual_weight = invalid
                    best_any_energy = energy
                    best_any = e.copy()
                if invalid == 0:
                    valid_hits += 1
                    logical_key = record_valid(
                        e, code.logical(e), prior_energy(e, lambdas), int(e.sum()), sweep, chain
                    )
                    if self.collect_valid_samples or self.stop_on_first_valid:
                        counts[logical_key] += 1
                    if measure_start is not None and sweep > measure_start:
                        measure_valid += 1
                    if self.stop_on_first_valid or (
                        self.target_valid_solutions
                        and len(valid_solution_keys) >= self.target_valid_solutions
                    ):
                        if self.weight_reduction and stab_rows is not None and stab_rows.size:
                            reduced_best = _apply_weight_reduction(
                                self.selection,
                                best_rep_by_coset,
                                best_rep_energy,
                                coset_logz,
                                stab_rows,
                                lambdas,
                                best_valid=best_valid,
                            )
                            if reduced_best is not None:
                                best_valid = reduced_best
                        correction = best_valid if best_valid is not None else e.copy()
                        return self._result(
                            code=code,
                            correction=correction,
                            syndrome=target,
                            lambdas=lambdas,
                            rho=rho,
                            start=start,
                            valid_hits=valid_hits,
                            total_samples=total_samples,
                            first_valid_sweep=first_valid_sweep,
                            first_valid_chain=first_valid_chain,
                            counts=dict(counts),
                            initial_residual_weight=initial_residual_weight,
                            final_residual_weight=0,
                            best_residual_weight=0,
                            initial_probability=initial_probability,
                            distinct_valid_solutions=len(valid_solution_keys),
                            coset_logz=dict(coset_logz),
                            measure_valid=measure_valid,
                        )
                if sweep == total_sweeps:
                    break
                beta = float(beta_schedule[sweep])
                # Calibrated mode (anneal_prior=False): keep the channel prior at
                # full scale and anneal only the syndrome penalty via kappa = beta*rho,
                # so the per-sweep marginal stays the physical posterior (a bit far
                # from the syndrome sets with probability ~p, not ~0.5, when hot).
                # prior_scale (gamma) > 1 additionally sharpens the prior so a far bit
                # sets with probability ~sigmoid(-gamma*lambda) << p, biasing the
                # search toward lighter (MAP-leaning) corrections. It scales only the
                # flip field; ``energy``/weight bookkeeping stays the physical value.
                flip_beta = beta if self.anneal_prior else 1.0
                syn_coeff = rho if self.anneal_prior else beta * rho
                for j in rng.permutation(code.n):
                    checks = col_checks[int(j)]
                    before = int(residual[checks].sum())
                    after = int(len(checks) - before)
                    bit = int(e[j])
                    # Active iff set (bit==1) or touching a violated check (before>0);
                    # only idle bits (bit==0 and before==0) are soft-gated.
                    if idle_gate and bit == 0 and before == 0 and rng.random() >= idle_rate:
                        continue
                    delta = (1 - 2 * bit) * lambdas[j] + rho * (after - before)
                    gamma = self.prior_scale if bit == 1 else (g_active if before > 0 else g_far)
                    field = (
                        delta
                        if self.anneal_prior
                        else gamma * (1 - 2 * bit) * lambdas[j]
                        + syn_coeff * (after - before)
                    )
                    if _should_flip(field, bit, flip_beta, rng, self.update):
                        e[j] ^= 1
                        residual[checks] ^= 1
                        invalid += after - before
                        energy += delta
                        if invalid == 0:
                            record_valid(
                                e,
                                code.logical(e),
                                prior_energy(e, lambdas),
                                int(e.sum()),
                                sweep,
                                chain,
                            )

        if self.weight_reduction and stab_rows is not None and stab_rows.size:
            reduced_best = _apply_weight_reduction(
                self.selection,
                best_rep_by_coset,
                best_rep_energy,
                coset_logz,
                stab_rows,
                lambdas,
                best_valid=best_valid,
            )
            if reduced_best is not None:
                best_valid = reduced_best
        if self.selection == "ml_coset" and coset_logz:
            best_key = max(coset_logz, key=lambda key: (coset_logz[key], -best_rep_energy[key]))
            correction = best_rep_by_coset[best_key]
        elif self.collect_valid_samples and counts and self.selection == "counts":
            best_key = max(counts, key=lambda key: (counts[key], -best_rep_energy[key]))
            correction = best_rep_by_coset[best_key]
        elif best_valid is not None:
            correction = best_valid
        elif best_any is not None:
            correction = best_any
        else:
            correction = np.zeros(code.n, dtype=np.uint8)
        final_residual_weight = int((code.syndrome(correction) ^ target).sum())
        return self._result(
            code=code,
            correction=correction,
            syndrome=target,
            lambdas=lambdas,
            rho=rho,
            start=start,
            valid_hits=valid_hits,
            total_samples=total_samples,
            first_valid_sweep=first_valid_sweep,
            first_valid_chain=first_valid_chain,
            counts=dict(counts),
            initial_residual_weight=initial_residual_weight or 0,
            final_residual_weight=final_residual_weight,
            best_residual_weight=best_residual_weight,
            initial_probability=initial_probability,
            distinct_valid_solutions=len(valid_solution_keys),
            coset_logz=dict(coset_logz),
            measure_valid=measure_valid,
        )

    def _decode_batched(
        self,
        code: CSSCode,
        target: BitArray,
        lambdas: np.ndarray,
        rho: float,
        col_checks: list[np.ndarray],
        beta_schedule: np.ndarray,
        initial_probability: Union[float, np.ndarray],
        start: float,
        rng: np.random.Generator,
        measure_start: Optional[int] = None,
        stab_rows: Optional[np.ndarray] = None,
    ) -> DecodeResult:
        """Chain-parallel reimplementation of the scalar p-bit sweep.

        The independent chains are advanced together as rows of an (chains, n) state
        matrix. Within a sweep, bits are still updated sequentially (asynchronous
        Gibbs/Metropolis) so the per-chain dynamics match the scalar path; only the
        per-bit work is vectorized across chains. Results are deterministic given the
        seed but use a different random stream than the scalar implementation.
        """
        n = code.n
        chains = self.chains
        g_active, g_far = self._resolve_gammas()
        active_on = (g_active != self.prior_scale) or (g_far != self.prior_scale)
        idle_rate = float(self.idle_update_rate)
        idle_gate = idle_rate < 1.0
        lam = np.asarray(lambdas, dtype=float)
        h_transpose = code.H.T.astype(np.uint8)
        a_transpose = code.A.T.astype(np.int64)
        max_deg = max((checks.size for checks in col_checks), default=0)
        col_pad = np.zeros((n, max_deg), dtype=np.int64)
        deg_mask = np.zeros((n, max_deg), dtype=bool)
        col_deg = np.empty(n, dtype=np.int64)
        for j, checks in enumerate(col_checks):
            col_deg[j] = checks.size
            if checks.size:
                col_pad[j, : checks.size] = checks
                deg_mask[j, : checks.size] = True
        chain_rows = np.arange(chains)
        if self.coloring and max_deg:
            color_blocks = [
                (cols, lam[cols], deg_c, cp_c, vm_c, fchk_c, flcol_c)
                for (cols, deg_c, cp_c, vm_c, fchk_c, flcol_c) in self._color_layout(
                    code, col_pad, deg_mask, col_deg
                )
            ]
        else:
            color_blocks = None
        if self.simultaneous and max_deg:
            sync_row_pad, sync_row_mask = self._row_layout(code)
            sync_damping = float(self.sync_damping)
        else:
            sync_row_pad = sync_row_mask = None

        counts: Dict[str, int] = defaultdict(int)
        coset_logz: Dict[str, float] = {}
        best_rep_by_coset: Dict[str, BitArray] = {}
        best_rep_energy: Dict[str, float] = {}
        best_valid: Optional[BitArray] = None
        best_valid_energy = float("inf")
        best_valid_weight: Optional[int] = None
        best_any: Optional[BitArray] = None
        best_any_energy = float("inf")
        valid_solution_keys: set[str] = set()
        best_residual_weight = code.m + 1
        first_valid_sweep: Optional[int] = None
        first_valid_chain: Optional[int] = None
        valid_hits = 0
        total_samples = 0
        measure_valid = 0

        if isinstance(initial_probability, np.ndarray):
            states = (rng.random((chains, n)) < initial_probability[None, :]).astype(np.uint8)
        else:
            states = (rng.random((chains, n)) < float(initial_probability)).astype(np.uint8)
        residual = ((states @ h_transpose) & 1).astype(np.uint8) ^ target[None, :]
        invalid = residual.sum(axis=1).astype(np.int64)
        energy = states @ lam + rho * invalid
        initial_residual_weight = int(invalid[0]) if chains else 0

        def record_valid(e_vec, logical_vec, e_prior, e_weight, sweep, chain):
            """Fold a syndrome-satisfying state into the candidate pool.

            Mirror of the scalar helper: called at sweep boundaries and immediately
            after any accepted flip that lands a chain on the target syndrome, so a
            transient valid correction is never lost. Consumes no randomness.
            Returns the coset key for sampling counts.
            """
            nonlocal best_valid, best_valid_energy, best_valid_weight
            nonlocal first_valid_sweep, first_valid_chain
            if first_valid_sweep is None:
                first_valid_sweep = sweep
                first_valid_chain = chain
            logical_key = _coset_key(logical_vec)
            if logical_key not in best_rep_energy or e_prior < best_rep_energy[logical_key]:
                best_rep_energy[logical_key] = e_prior
                best_rep_by_coset[logical_key] = e_vec.copy()
            config_key = e_vec.tobytes()
            if config_key not in valid_solution_keys:
                valid_solution_keys.add(config_key)
                prev_logz = coset_logz.get(logical_key)
                coset_logz[logical_key] = (
                    -e_prior
                    if prev_logz is None
                    else float(np.logaddexp(prev_logz, -e_prior))
                )
            is_better_valid = (
                self.selection == "min_weight"
                and (
                    best_valid_weight is None
                    or e_weight < best_valid_weight
                    or (e_weight == best_valid_weight and e_prior < best_valid_energy)
                )
            ) or (self.selection != "min_weight" and e_prior < best_valid_energy)
            if is_better_valid:
                best_valid_weight = e_weight
                best_valid_energy = e_prior
                best_valid = e_vec.copy()
            return logical_key

        sweeps = self.sweeps if measure_start is None else int(beta_schedule.shape[0])
        for sweep in range(sweeps + 1):
            # Boundary scan: fold every chain's current state into the candidate pool.
            # best_any (lowest residual, then energy) is tracked with a vectorized argmin
            # and the expensive per-chain bookkeeping (coset matmul + record_valid) is
            # restricted to syndrome-satisfying chains -- so a sweep with no valid chain
            # costs an argmin instead of a (chains x A^T) int matmul plus a Python loop
            # over every chain. Consumes no randomness, so results are unchanged.
            total_samples += chains
            best_i = int(invalid.min())
            if best_i <= best_residual_weight:
                cand = np.flatnonzero(invalid == best_i)
                k = int(cand[np.argmin(energy[cand])])
                ek = float(energy[k])
                if best_i < best_residual_weight or ek < best_any_energy:
                    best_residual_weight = best_i
                    best_any_energy = ek
                    best_any = states[k].copy()
            valid_idx = np.flatnonzero(invalid == 0)
            if valid_idx.size:
                logical_valid = (states[valid_idx].astype(np.int64) @ a_transpose) & 1
                priors_valid = states[valid_idx] @ lam
                weights_valid = states[valid_idx].sum(axis=1)
                for vi in range(valid_idx.size):
                    chain = int(valid_idx[vi])
                    valid_hits += 1
                    e_vec = states[chain]
                    logical_key = record_valid(
                        e_vec,
                        logical_valid[vi],
                        float(priors_valid[vi]),
                        int(weights_valid[vi]),
                        sweep,
                        chain,
                    )
                    if self.collect_valid_samples or self.stop_on_first_valid:
                        counts[logical_key] += 1
                    if measure_start is not None and sweep > measure_start:
                        measure_valid += 1
                    if self.stop_on_first_valid or (
                        self.target_valid_solutions
                        and len(valid_solution_keys) >= self.target_valid_solutions
                    ):
                        # Match the scalar loop's sample count: it would have visited only
                        # chains 0..chain before returning, not the whole ensemble.
                        total_samples -= chains - chain - 1
                        if self.weight_reduction and stab_rows is not None and stab_rows.size:
                            reduced_best = _apply_weight_reduction(
                                self.selection,
                                best_rep_by_coset,
                                best_rep_energy,
                                coset_logz,
                                stab_rows,
                                lambdas,
                                best_valid=best_valid,
                            )
                            if reduced_best is not None:
                                best_valid = reduced_best
                        correction = best_valid if best_valid is not None else e_vec.copy()
                        return self._result(
                            code=code,
                            correction=correction,
                            syndrome=target,
                            lambdas=lambdas,
                            rho=rho,
                            start=start,
                            valid_hits=valid_hits,
                            total_samples=total_samples,
                            first_valid_sweep=first_valid_sweep,
                            first_valid_chain=first_valid_chain,
                            counts=dict(counts),
                            initial_residual_weight=initial_residual_weight,
                            final_residual_weight=0,
                            best_residual_weight=0,
                            initial_probability=initial_probability,
                            distinct_valid_solutions=len(valid_solution_keys),
                            coset_logz=dict(coset_logz),
                            measure_valid=measure_valid,
                        )
            if sweep == sweeps:
                break
            beta = float(beta_schedule[sweep])
            flip_beta = beta if self.anneal_prior else 1.0
            syn_coeff = rho if self.anneal_prior else beta * rho
            if sync_row_pad is not None:
                # Synchronous (Jacobi) sweep: every bit computes its field from the
                # SAME state snapshot, then all chains flip at once. This is not
                # detailed-balance Gibbs -- it is a BP-like parallel relaxation -- but as
                # an annealed optimizer the freezing schedule keeps check-sharing bits
                # from flipping together once flip probabilities drop. One vectorized
                # pass over all n bits + one parity gather over all m checks per sweep.
                # Valid states are captured by the next iteration's boundary scan (a
                # single update per sweep leaves no transient state to lose).
                before = (residual[:, col_pad] * deg_mask[None]).sum(axis=2).astype(np.int64)
                after = col_deg[None, :] - before
                bits = states.astype(np.int64)
                sign = 1 - 2 * bits
                syn_term = after - before
                delta = sign * lam[None, :] + rho * syn_term
                if self.anneal_prior:
                    field = delta
                else:
                    prior_coeff = (
                        np.where(bits == 1, self.prior_scale, np.where(before > 0, g_active, g_far))
                        if active_on
                        else self.prior_scale
                    )
                    field = prior_coeff * sign * lam[None, :] + syn_coeff * syn_term
                flip = _should_flip_batch(
                    field, bits, flip_beta, rng.random((chains, n)), self.update
                )
                if sync_damping < 1.0:
                    # Stochastic damping: each bit is eligible to flip with prob q per
                    # sweep, so the expected number of incident bits that flip on an
                    # unsatisfied check is ~deg*q*p_flip. Tuned to ~1 it makes parallel
                    # flips resolve checks one-at-a-time (like async) while staying fully
                    # vectorized -- the knob that turns the Jacobi sweep into a stable,
                    # BP-like relaxation.
                    flip = flip & (rng.random((chains, n)) < sync_damping)
                if idle_gate:
                    flip = flip & ((bits == 1) | (before > 0) | (rng.random((chains, n)) < idle_rate))
                if not flip.any():
                    continue
                states ^= flip.astype(np.uint8)
                toggled = (flip[:, sync_row_pad] & sync_row_mask[None]).sum(axis=2) & 1
                residual ^= toggled.astype(np.uint8)
                invalid = residual.sum(axis=1).astype(np.int64)
                energy = states @ lam + rho * invalid
                continue
            if color_blocks is not None:
                # Chromatic block Gibbs: resample whole check-disjoint color classes
                # at once. Identical per-bit field/energy math as the sequential scan,
                # but ~n/#colors fewer Python steps per sweep (~80x for B1/B2). Columns
                # in a color share no check, so their simultaneous flips are independent
                # and the residual toggle is collision-free.
                for cols, lam_c, deg_c, cp_c, vm_c, fchk_c, flcol_c in color_blocks:
                    k = cols.size
                    before = (residual[:, cp_c] * vm_c[None]).sum(axis=2).astype(np.int64)
                    after = deg_c[None, :] - before
                    bits = states[:, cols].astype(np.int64)
                    sign = 1 - 2 * bits
                    syn_term = after - before
                    delta = sign * lam_c[None, :] + rho * syn_term
                    if self.anneal_prior:
                        field = delta
                    else:
                        prior_coeff = (
                            np.where(bits == 1, self.prior_scale, np.where(before > 0, g_active, g_far))
                            if active_on
                            else self.prior_scale
                        )
                        field = prior_coeff * sign * lam_c[None, :] + syn_coeff * syn_term
                    flip = _should_flip_batch(
                        field, bits, flip_beta, rng.random((chains, k)), self.update
                    )
                    if idle_gate:
                        flip = flip & ((bits == 1) | (before > 0) | (rng.random((chains, k)) < idle_rate))
                    if not flip.any():
                        continue
                    flip_u8 = flip.astype(np.uint8)
                    states[:, cols] ^= flip_u8
                    residual[:, fchk_c] ^= flip_u8[:, flcol_c]
                    invalid += (syn_term * flip).sum(axis=1)
                    energy += (delta * flip).sum(axis=1)
                    chains_flipped = np.flatnonzero(flip.any(axis=1))
                    newly_valid = chains_flipped[invalid[chains_flipped] == 0]
                    for c in newly_valid:
                        c = int(c)
                        e_vec = states[c]
                        record_valid(
                            e_vec,
                            (e_vec.astype(np.int64) @ a_transpose) & 1,
                            float(e_vec @ lam),
                            int(e_vec.sum()),
                            sweep,
                            c,
                        )
                continue
            order = np.argsort(rng.random((chains, n)), axis=1)
            uniforms = rng.random((chains, n))
            idle_uniforms = rng.random((chains, n)) if idle_gate else None
            lam_order = lam[order]
            deg_order = col_deg[order]
            if max_deg:
                cp_order = col_pad[order]
                vm_order = deg_mask[order]
            for step in range(n):
                j = order[:, step]
                if max_deg:
                    cp = cp_order[:, step]
                    vm = vm_order[:, step]
                    res_vals = residual[chain_rows[:, None], cp]
                    before = (res_vals * vm).sum(axis=1).astype(np.int64)
                else:
                    before = np.zeros(chains, dtype=np.int64)
                after = deg_order[:, step] - before
                bits = states[chain_rows, j].astype(np.int64)
                delta = (1 - 2 * bits) * lam_order[:, step] + rho * (after - before)
                if self.anneal_prior:
                    field = delta
                else:
                    prior_coeff = (
                        np.where(bits == 1, self.prior_scale, np.where(before > 0, g_active, g_far))
                        if active_on
                        else self.prior_scale
                    )
                    field = (
                        prior_coeff * (1 - 2 * bits) * lam_order[:, step]
                        + syn_coeff * (after - before)
                    )
                flip = _should_flip_batch(field, bits, flip_beta, uniforms[:, step], self.update)
                if idle_gate:
                    flip = flip & ((bits == 1) | (before > 0) | (idle_uniforms[:, step] < idle_rate))
                flipped = np.flatnonzero(flip)
                if flipped.size:
                    states[flipped, j[flipped]] ^= 1
                    if max_deg:
                        sel = vm[flipped].reshape(-1)
                        rows_idx = np.repeat(flipped, max_deg)[sel]
                        cols_idx = cp[flipped].reshape(-1)[sel]
                        residual[rows_idx, cols_idx] ^= 1
                    invalid[flipped] += (after - before)[flipped]
                    energy[flipped] += delta[flipped]
                    newly_valid = flipped[invalid[flipped] == 0]
                    for c in newly_valid:
                        c = int(c)
                        e_vec = states[c]
                        record_valid(
                            e_vec,
                            (e_vec.astype(np.int64) @ a_transpose) & 1,
                            float(e_vec @ lam),
                            int(e_vec.sum()),
                            sweep,
                            c,
                        )

        if self.weight_reduction and stab_rows is not None and stab_rows.size:
            reduced_best = _apply_weight_reduction(
                self.selection,
                best_rep_by_coset,
                best_rep_energy,
                coset_logz,
                stab_rows,
                lambdas,
                best_valid=best_valid,
            )
            if reduced_best is not None:
                best_valid = reduced_best
        if self.selection == "ml_coset" and coset_logz:
            best_key = max(coset_logz, key=lambda key: (coset_logz[key], -best_rep_energy[key]))
            correction = best_rep_by_coset[best_key]
        elif self.collect_valid_samples and counts and self.selection == "counts":
            best_key = max(counts, key=lambda key: (counts[key], -best_rep_energy[key]))
            correction = best_rep_by_coset[best_key]
        elif best_valid is not None:
            correction = best_valid
        elif best_any is not None:
            correction = best_any
        else:
            correction = np.zeros(code.n, dtype=np.uint8)
        final_residual_weight = int((code.syndrome(correction) ^ target).sum())
        return self._result(
            code=code,
            correction=correction,
            syndrome=target,
            lambdas=lambdas,
            rho=rho,
            start=start,
            valid_hits=valid_hits,
            total_samples=total_samples,
            first_valid_sweep=first_valid_sweep,
            first_valid_chain=first_valid_chain,
            counts=dict(counts),
            initial_residual_weight=initial_residual_weight,
            final_residual_weight=final_residual_weight,
            best_residual_weight=best_residual_weight,
            initial_probability=initial_probability,
            distinct_valid_solutions=len(valid_solution_keys),
            coset_logz=dict(coset_logz),
            measure_valid=measure_valid,
        )

    def _resolve_rho(self, lambdas: np.ndarray) -> float:
        """Resolve the syndrome-penalty weight ``rho``.

        Precedence: an explicit ``rho`` wins; otherwise a p-adaptive
        ``rho_scale * max(lambda)`` is used when ``rho_scale`` is set; otherwise the
        shared ``default_rho`` heuristic. Empirically ``rho_scale`` near 1.0 (i.e.
        rho ~ the per-qubit log-likelihood) greatly improves coverage of the true
        low-weight coset versus the much larger ``default_rho``.
        """
        if self.rho is not None:
            return float(self.rho)
        if self.rho_scale is not None:
            if self.rho_scale <= 0.0:
                raise ValueError("rho_scale must be positive")
            return float(self.rho_scale) * float(np.max(lambdas))
        return default_rho(lambdas)

    def _beta_schedule(self) -> np.ndarray:
        if self.chains <= 0:
            raise ValueError("chains must be positive")
        if self.sweeps < 0:
            raise ValueError("sweeps must be nonnegative")
        if self.beta_start is None and self.beta_stop is None:
            if self.beta <= 0.0:
                raise ValueError("beta must be positive")
            return np.full(max(self.sweeps, 1), float(self.beta))
        if self.beta_start is None or self.beta_stop is None:
            raise ValueError("beta_start and beta_stop must be provided together")
        if self.beta_start <= 0.0 or self.beta_stop <= 0.0:
            raise ValueError("beta_start and beta_stop must be positive")
        return _annealing_schedule(
            float(self.beta_start),
            float(self.beta_stop),
            max(self.sweeps, 1),
            self.beta_schedule_shape,
        )

    def _initial_probability(self, probabilities: np.ndarray) -> Union[float, np.ndarray]:
        if self.initial_probability is None:
            return probabilities
        if self.initial_probability < 0.0 or self.initial_probability > 1.0:
            raise ValueError("initial_probability must be in [0, 1]")
        return float(self.initial_probability)

    def _result(
        self,
        code: CSSCode,
        correction: BitArray,
        syndrome: BitArray,
        lambdas: np.ndarray,
        rho: float,
        start: float,
        valid_hits: int,
        total_samples: int,
        first_valid_sweep: Optional[int],
        first_valid_chain: Optional[int],
        counts: Dict[str, int],
        initial_residual_weight: int,
        final_residual_weight: int,
        best_residual_weight: int,
        initial_probability: Union[float, np.ndarray],
        distinct_valid_solutions: int,
        coset_logz: Optional[Dict[str, float]] = None,
        measure_valid: int = 0,
    ) -> DecodeResult:
        valid = code.is_valid_correction(correction, syndrome)
        logical = gf2.bit_tuple(code.logical(correction))
        key = _coset_key(logical)
        if valid and not counts:
            counts = {key: max(valid_hits, 1)}
        beta_metadata: Dict[str, float]
        if self.beta_start is None and self.beta_stop is None:
            beta_metadata = {"beta": self.beta}
        else:
            beta_metadata = {
                "beta_start": float(self.beta_start),
                "beta_stop": float(self.beta_stop),
            }
        if isinstance(initial_probability, np.ndarray):
            initial_probability_metadata: Union[float, str] = "channel"
        else:
            initial_probability_metadata = float(initial_probability)
        return DecodeResult(
            correction=correction,
            logical_class=logical,
            valid=valid,
            energy=(
                prior_energy(correction, lambdas)
                if valid
                else prior_energy(correction, lambdas) + rho * final_residual_weight
            ),
            runtime_s=time.perf_counter() - start,
            valid_samples=valid_hits,
            total_samples=total_samples,
            first_valid_sweep=first_valid_sweep,
            distinct_logicals=len(counts),
            coset_counts=counts,
            metadata={
                **beta_metadata,
                "beta_schedule_shape": self.beta_schedule_shape,
                "anneal_prior": self.anneal_prior,
                "prior_scale": float(self.prior_scale),
                "gamma_active": (None if self.gamma_active is None else float(self.gamma_active)),
                "gamma_far": (None if self.gamma_far is None else float(self.gamma_far)),
                "idle_update_rate": float(self.idle_update_rate),
                "rho": rho,
                "update": self.update,
                "chains": self.chains,
                "sweeps": self.sweeps,
                "initial_probability": initial_probability_metadata,
                "initial_residual_weight": initial_residual_weight,
                "final_residual_weight": final_residual_weight,
                "first_valid_sweep": first_valid_sweep,
                "first_valid_chain": first_valid_chain,
                "valid_hits": valid_hits,
                "distinct_valid_solutions": distinct_valid_solutions,
                "measure_sweeps": self.measure_sweeps,
                "measure_samples": int(measure_valid),
                "ml_coset_logz": (
                    {key: round(value, 4) for key, value in coset_logz.items()}
                    if coset_logz
                    else {}
                ),
                "selected_weight": int(correction.sum()),
                "best_residual_weight": best_residual_weight,
                "rho_scale": self.rho_scale,
                "stop_on_first_valid": self.stop_on_first_valid,
                "collect_valid_samples": self.collect_valid_samples,
                "selection": self.selection,
                "target_valid_solutions": self.target_valid_solutions,
                "used_linear_solve": False,
                "persistent_disorder": False,
                "state_space": "full_error_space",
                "symmetry_breaking": "bernoulli_randomness",
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
class BPOSDDecoder:
    max_iter: int = 100
    bp_method: str = "minimum_sum"
    ms_scaling_factor: float = 0.625
    schedule: str = "parallel"
    osd_method: str = "OSD_CS"
    osd_order: int = 10
    omp_thread_count: int = 1
    name: str = "bp_osd"
    _decoder_cache: Dict[bytes, object] = field(default_factory=dict, init=False, repr=False)

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
        target = np.asarray(syndrome, dtype=np.uint8) % 2
        decoder = self._cached_decoder(code, p)
        correction = np.asarray(decoder.decode(target), dtype=np.uint8) % 2
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
            metadata={
                "backend": "ldpc.BpOsdDecoder",
                "max_iter": self.max_iter,
                "bp_method": self.bp_method,
                "ms_scaling_factor": self.ms_scaling_factor,
                "schedule": self.schedule,
                "osd_method": self.osd_method,
                "osd_order": self.osd_order,
                "converge": bool(getattr(decoder, "converge", False)),
                "iterations": int(getattr(decoder, "iter", 0)),
            },
        )

    def _cached_decoder(self, code: CSSCode, p: Union[float, Sequence[float]]) -> object:
        probabilities = _prior_probabilities(p, code.n)
        cache_key = b"|".join(
            [
                code.H.shape[0].to_bytes(4, "little"),
                code.H.shape[1].to_bytes(4, "little"),
                code.H.tobytes(),
                probabilities.tobytes(),
                repr(
                    (
                        self.max_iter,
                        self.bp_method,
                        self.ms_scaling_factor,
                        self.schedule,
                        self.osd_method,
                        self.osd_order,
                        self.omp_thread_count,
                    )
                ).encode("utf-8"),
            ]
        )
        if cache_key not in self._decoder_cache:
            from ldpc import BpOsdDecoder as LdpcBpOsdDecoder

            if np.all(probabilities == probabilities[0]):
                channel_kwargs = {"error_rate": float(probabilities[0])}
            else:
                channel_kwargs = {"error_channel": probabilities.astype(float).tolist()}
            self._decoder_cache[cache_key] = LdpcBpOsdDecoder(
                code.H,
                max_iter=self.max_iter,
                bp_method=self.bp_method,
                ms_scaling_factor=self.ms_scaling_factor,
                schedule=self.schedule,
                osd_method=self.osd_method,
                osd_order=self.osd_order,
                omp_thread_count=self.omp_thread_count,
                **channel_kwargs,
            )
        return self._decoder_cache[cache_key]


@dataclass
class BPDecoder:
    max_iter: int = 100
    bp_method: str = "minimum_sum"
    ms_scaling_factor: float = 0.625
    schedule: str = "parallel"
    omp_thread_count: int = 1
    name: str = "bp"
    _decoder_cache: Dict[bytes, object] = field(default_factory=dict, init=False, repr=False)

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
        target = np.asarray(syndrome, dtype=np.uint8) % 2
        decoder = self._cached_decoder(code, p)
        correction = np.asarray(decoder.decode(target), dtype=np.uint8) % 2
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
            metadata={
                "backend": "ldpc.BpDecoder",
                "uses_osd": False,
                "max_iter": self.max_iter,
                "bp_method": self.bp_method,
                "ms_scaling_factor": self.ms_scaling_factor,
                "schedule": self.schedule,
                "converge": bool(getattr(decoder, "converge", False)),
                "iterations": int(getattr(decoder, "iter", 0)),
            },
        )

    def _cached_decoder(self, code: CSSCode, p: Union[float, Sequence[float]]) -> object:
        probabilities = _prior_probabilities(p, code.n)
        cache_key = b"|".join(
            [
                code.H.shape[0].to_bytes(4, "little"),
                code.H.shape[1].to_bytes(4, "little"),
                code.H.tobytes(),
                probabilities.tobytes(),
                repr(
                    (
                        self.max_iter,
                        self.bp_method,
                        self.ms_scaling_factor,
                        self.schedule,
                        self.omp_thread_count,
                    )
                ).encode("utf-8"),
            ]
        )
        if cache_key not in self._decoder_cache:
            from ldpc import BpDecoder as LdpcBpDecoder

            if np.all(probabilities == probabilities[0]):
                channel_kwargs = {"error_rate": float(probabilities[0])}
            else:
                channel_kwargs = {"error_channel": probabilities.astype(float).tolist()}
            self._decoder_cache[cache_key] = LdpcBpDecoder(
                code.H,
                max_iter=self.max_iter,
                bp_method=self.bp_method,
                ms_scaling_factor=self.ms_scaling_factor,
                schedule=self.schedule,
                omp_thread_count=self.omp_thread_count,
                **channel_kwargs,
            )
        return self._decoder_cache[cache_key]


@dataclass
class PBitNullspaceCosetSamplerDecoder:
    samples: int = 2000
    burn_in: int = 100
    updates_per_sample: int = 1
    beta: float = 1.0
    target_beta: float = 1.0
    chains: int = 1
    max_nullity: int = 64
    initial_state: str = "random"
    update_schedule: str = "random"
    selection: str = "counts"
    guide: str = "none"
    guide_max_iter: int = 100
    guide_bp_method: str = "minimum_sum"
    guide_ms_scaling_factor: float = 0.625
    guide_schedule: str = "parallel"
    guide_osd_method: str = "OSD_CS"
    guide_osd_order: int = 10
    guide_omp_thread_count: int = 1
    name: str = "pbit_nullspace_sampler"
    _guide_cache: Dict[bytes, object] = field(default_factory=dict, init=False, repr=False)

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
        if self.guide not in {"none", "bp", "bp_osd"}:
            raise ValueError(f"unknown nullspace p-bit guide {self.guide!r}")
        if self.updates_per_sample <= 0:
            raise ValueError("updates_per_sample must be positive")
        if self.chains <= 0:
            raise ValueError("chains must be positive")
        if self.beta <= 0.0 or self.target_beta <= 0.0:
            raise ValueError("beta and target_beta must be positive")
        if self.update_schedule not in {"random", "sweep"}:
            raise ValueError(f"unknown nullspace p-bit update_schedule {self.update_schedule!r}")
        if self.selection not in {"counts", "importance"}:
            raise ValueError(f"unknown nullspace p-bit selection {self.selection!r}")

        guide_valid = False
        if self.guide != "none":
            guided = self._guided_particular_solution(code, syndrome, p)
            if guided is not None:
                x0 = guided
                guide_valid = True

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
                    "target_beta": self.target_beta,
                    "chains": self.chains,
                    "updates_per_sample": self.updates_per_sample,
                    "selection": self.selection,
                    "guide": self.guide,
                    "guide_valid": guide_valid,
                    "update": "gibbs",
                    "update_schedule": self.update_schedule,
                    "state_space": "affine_nullspace",
                    "initial_state": self.initial_state,
                },
            )

        basis_indices = [np.flatnonzero(row).astype(np.int64) for row in basis]

        counts: Dict[str, int] = defaultdict(int)
        log_scores: Dict[str, float] = {}
        best_rep_by_coset: Dict[str, BitArray] = {}
        best_rep_energy: Dict[str, float] = {}

        sample_counts = _split_samples(self.samples, self.chains)
        for chain_samples in sample_counts:
            if chain_samples == 0:
                continue
            z = np.zeros(basis.shape[0], dtype=np.uint8)
            e = x0.copy()
            if self.initial_state == "random":
                z = rng.integers(0, 2, size=basis.shape[0], dtype=np.uint8)
                selected = z.astype(bool)
                if np.any(selected):
                    e ^= np.bitwise_xor.reduce(basis[selected], axis=0)
            energy = prior_energy(e, lambdas)
            signed_lambdas = lambdas * (1 - 2 * e.astype(np.int8))

            for step in range(self.burn_in + chain_samples):
                for i in _pbit_update_indices(
                    basis.shape[0],
                    self.updates_per_sample,
                    self.update_schedule,
                    rng,
                ):
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
                if self.selection == "importance":
                    log_weight = -(self.target_beta - self.beta) * energy
                    log_scores[key] = (
                        log_weight
                        if key not in log_scores
                        else float(np.logaddexp(log_scores[key], log_weight))
                    )
                if key not in best_rep_energy or energy < best_rep_energy[key]:
                    best_rep_energy[key] = energy
                    best_rep_by_coset[key] = e.copy()

        if self.selection == "counts":
            best_key = max(counts, key=lambda key: (counts[key], -best_rep_energy[key]))
        else:
            best_key = max(log_scores, key=lambda key: (log_scores[key], -best_rep_energy[key]))
        correction = best_rep_by_coset[best_key]
        logical = gf2.bit_tuple(code.logical(correction))
        score_log_weight = log_scores.get(best_key)
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
                "target_beta": self.target_beta,
                "chains": self.chains,
                "updates_per_sample": self.updates_per_sample,
                "selection": self.selection,
                "guide": self.guide,
                "guide_valid": guide_valid,
                "update": "gibbs",
                "update_schedule": self.update_schedule,
                "state_space": "affine_nullspace",
                "initial_state": self.initial_state,
                "selected_log_importance_score": score_log_weight,
            },
        )

    def _guided_particular_solution(
        self,
        code: CSSCode,
        syndrome: Sequence[int],
        p: Union[float, Sequence[float]],
    ) -> Optional[BitArray]:
        decoder = self._cached_guide_decoder(code, p)
        target = np.asarray(syndrome, dtype=np.uint8) % 2
        correction = np.asarray(decoder.decode(target), dtype=np.uint8) % 2
        if not code.is_valid_correction(correction, target):
            return None
        return correction

    def _cached_guide_decoder(self, code: CSSCode, p: Union[float, Sequence[float]]) -> object:
        probabilities = _prior_probabilities(p, code.n)
        cache_key = b"|".join(
            [
                self.guide.encode("utf-8"),
                code.H.shape[0].to_bytes(4, "little"),
                code.H.shape[1].to_bytes(4, "little"),
                code.H.tobytes(),
                probabilities.tobytes(),
                repr(
                    (
                        self.guide_max_iter,
                        self.guide_bp_method,
                        self.guide_ms_scaling_factor,
                        self.guide_schedule,
                        self.guide_osd_method,
                        self.guide_osd_order,
                        self.guide_omp_thread_count,
                    )
                ).encode("utf-8"),
            ]
        )
        if cache_key not in self._guide_cache:
            if np.all(probabilities == probabilities[0]):
                channel_kwargs = {"error_rate": float(probabilities[0])}
            else:
                channel_kwargs = {"error_channel": probabilities.astype(float).tolist()}
            if self.guide == "bp":
                from ldpc import BpDecoder as LdpcBpDecoder

                self._guide_cache[cache_key] = LdpcBpDecoder(
                    code.H,
                    max_iter=self.guide_max_iter,
                    bp_method=self.guide_bp_method,
                    ms_scaling_factor=self.guide_ms_scaling_factor,
                    schedule=self.guide_schedule,
                    omp_thread_count=self.guide_omp_thread_count,
                    **channel_kwargs,
                )
            elif self.guide == "bp_osd":
                from ldpc import BpOsdDecoder as LdpcBpOsdDecoder

                self._guide_cache[cache_key] = LdpcBpOsdDecoder(
                    code.H,
                    max_iter=self.guide_max_iter,
                    bp_method=self.guide_bp_method,
                    ms_scaling_factor=self.guide_ms_scaling_factor,
                    schedule=self.guide_schedule,
                    osd_method=self.guide_osd_method,
                    osd_order=self.guide_osd_order,
                    omp_thread_count=self.guide_omp_thread_count,
                    **channel_kwargs,
                )
            else:
                raise ValueError(f"unknown nullspace p-bit guide {self.guide!r}")
        return self._guide_cache[cache_key]


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


def _pbit_probability_one_batch(delta_one_minus_zero: np.ndarray, beta: float) -> np.ndarray:
    """Vectorized, numerically stable counterpart of ``_pbit_probability_one``."""

    x = beta * np.asarray(delta_one_minus_zero, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    z_pos = np.exp(-x[pos])
    out[pos] = z_pos / (1.0 + z_pos)
    out[~pos] = 1.0 / (1.0 + np.exp(x[~pos]))
    return out


def _should_flip_batch(
    flip_delta: np.ndarray,
    bits: np.ndarray,
    beta: float,
    uniforms: np.ndarray,
    update: str,
) -> np.ndarray:
    """Vectorized flip decision matching ``_should_flip`` for a batch of chains."""

    delta = np.asarray(flip_delta, dtype=float)
    bits_int = np.asarray(bits, dtype=np.int64)
    if update == "metropolis":
        accept = delta <= 0.0
        with np.errstate(over="ignore"):
            threshold = np.exp(-beta * delta)
        return accept | (uniforms < threshold)
    if update != "gibbs":
        raise ValueError(f"unknown p-bit update rule {update!r}")
    delta_one_minus_zero = delta * (1 - 2 * bits_int)
    probability_one = _pbit_probability_one_batch(delta_one_minus_zero, beta)
    next_bit = (uniforms < probability_one).astype(np.int64)
    return next_bit != bits_int


def _basis_flip_delta(signed_lambdas: np.ndarray, indices: np.ndarray) -> float:
    return float(signed_lambdas[indices].sum())


def _split_samples(samples: int, chains: int) -> list[int]:
    base = samples // chains
    remainder = samples % chains
    return [base + int(chain < remainder) for chain in range(chains)]


def _pbit_update_indices(
    width: int,
    updates: int,
    schedule: str,
    rng: np.random.Generator,
) -> list[int]:
    if schedule == "random":
        return [int(rng.integers(0, width)) for _ in range(updates)]
    if schedule != "sweep":
        raise ValueError(f"unknown p-bit update schedule {schedule!r}")
    order = []
    while len(order) < updates:
        remaining = updates - len(order)
        permutation = rng.permutation(width)
        order.extend(int(i) for i in permutation[:remaining])
    return order


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
