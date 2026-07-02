"""Scalable sparse p-bit decoder for circuit-level (DEM) decoding.

This adapts the code-capacity :class:`~pbit_qec.decoders.PBitSyndromeDecoder`
algorithm -- annealed multi-chain Gibbs sampling of the energy

    E(e) = sum_j lambda_j e_j + rho * | H e XOR s |_1 ,

with ``lambda_j = log((1 - p_j) / p_j)`` -- to the size of circuit-level detector
error models (tens of thousands of mechanisms, thousands of detectors). Three
changes make that feasible:

1. **Chromatic block Gibbs.** Mechanism columns are greedily colored so two
   columns sharing a detector never share a color. Same-color columns are then
   conditionally independent given the rest, so an entire color class is
   resampled in one vectorized step. This replaces the ``O(n)`` per-bit Python
   sweep with an ``O(#colors)`` loop (~300 vs ~68000 for the gross code), while
   keeping the sampler exact (standard graph-colored block Gibbs).

2. **Sparse residual algebra.** The check matrix is kept sparse; the per-color
   "violated-check count" gather and residual toggle are ``scipy`` sparse
   matrix-vector products, costing ``O(nnz)`` rather than the ``O(m n)`` dense
   matmuls that a dense submatrix formulation would need.

3. **Shot x chain batching with an ``e = 0`` start.** Every chain starts from the
   all-zero mechanism activation, so the initial residual is exactly the observed
   syndrome (no startup matmul) and the prior energy starts at zero and is tracked
   incrementally. Many shots are decoded at once by stacking their chains as extra
   rows, amortizing Python/scipy overhead across a whole batch.

The most-likely correction is selected by lowest prior energy among
syndrome-consistent samples (equivalently weighted-minimum-weight), with a
lowest-residual fallback when no chain reaches a valid configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from pbit_qec.dem import DEMMatrices


@dataclass(frozen=True)
class _ColorBlock:
    """Precomputed layout for one check-disjoint color class.

    Within a color no two columns share a detector, so every touched detector is
    flipped by exactly one column. The per-column violated-check count is gathered
    with a sparse ``uint8`` mat-vec (``gather @ residual``, the fast scipy CSR
    kernel), and the simultaneous toggles are scattered back with a single
    fancy-indexed XOR over only the detectors this color touches (no modulo): each
    touched detector appears exactly once in ``det_idx`` so the XOR cannot collide.

    Attributes:
        cols: global mechanism-column indices in this color, shape ``(k,)``.
        lam: per-column log-likelihood ratios, shape ``(k,)``.
        degree: per-column detector degree (as float for the energy term), ``(k,)``.
        gather: ``H[:, cols].T`` as a ``uint8`` CSR matrix, shape ``(k, m)``; the
            row sums of ``gather @ residual`` are the per-column violated checks.
        det_idx: global detector indices this color touches, grouped by column,
            shape ``(L,)`` with ``L = sum(degree)``.
        col_of_det: local column index (0..k-1) owning each ``det_idx`` entry, ``(L,)``.
    """

    cols: np.ndarray
    lam: np.ndarray
    degree: np.ndarray
    gather: sparse.csr_matrix
    det_idx: np.ndarray
    col_of_det: np.ndarray
    col_det_indptr: np.ndarray


def greedy_coloring(check_matrix: sparse.spmatrix) -> List[np.ndarray]:
    """Greedily color mechanism columns so same-color columns share no detector.

    Two columns conflict iff they have a common nonzero row (detector). The
    returned color classes are therefore *independent sets* of the conflict
    graph, which is exactly the condition that makes their p-bits conditionally
    independent given the other columns (and hence block-Gibbs updatable).

    The implementation never materializes the full conflict graph (which has up
    to ``sum_d deg(d)^2`` edges). Instead it keeps, per detector, the set of
    colors already used by colored columns touching that detector; a column's
    forbidden colors are the union over its detectors. Columns are colored
    largest-degree-first (Welsh-Powell-style) to keep the color count low.

    Args:
        check_matrix: GF(2) ``(num_detectors, num_mechanisms)`` sparse matrix.

    Returns:
        A list of integer arrays, one per color, partitioning ``range(num_mechanisms)``.
    """
    h_csc = check_matrix.tocsc()
    h_csr = h_csc.tocsr()
    num_detectors, num_mechanisms = h_csc.shape
    if num_mechanisms == 0:
        return []

    col_indptr, col_rows = h_csc.indptr, h_csc.indices
    row_degree = np.diff(h_csr.indptr).astype(np.int64)
    # Conflict degree proxy: sum of detector degrees over a column's detectors.
    col_priority = np.asarray(h_csc.T @ row_degree).ravel()
    order = np.argsort(-col_priority, kind="stable")

    color_of = np.full(num_mechanisms, -1, dtype=np.int64)
    detector_colors: List[set] = [set() for _ in range(num_detectors)]

    for col in order:
        detectors = col_rows[col_indptr[col] : col_indptr[col + 1]]
        forbidden: set = set()
        for d in detectors:
            forbidden |= detector_colors[d]
        color = 0
        while color in forbidden:
            color += 1
        color_of[col] = color
        for d in detectors:
            detector_colors[d].add(color)

    num_colors = int(color_of.max()) + 1
    return [np.flatnonzero(color_of == c) for c in range(num_colors)]


def _sigmoid_neg(x: np.ndarray) -> np.ndarray:
    """Branch-free, overflow-safe ``sigmoid(-x) = 0.5 (1 - tanh(x/2))``."""
    return 0.5 * (1.0 - np.tanh(0.5 * x))


# Fused per-color GPU sweep: one launch replaces the spmm gather + ~12 cupy
# elementwise/reduction/fancy-index kernels of the reference path. Each thread
# owns one (local column i, chain/row j) pair, recomputes the violated-check
# count by summing the column's detectors in the residual (folding the gather
# in), draws the Gibbs/Metropolis flip from the supplied uniform, and on a flip
# toggles its state bit, the touched residual entries (check-disjoint within a
# color, so collision-free) and accumulates prior energy / invalid via double
# atomics. Math is identical to ``_sweep``; residual stays exactly 0/1 so the
# in-kernel gather is exact.
_FUSED_SWEEP_SRC = r"""
extern "C" __global__
void fused_color_sweep(
    unsigned char* __restrict__ states,
    float* __restrict__ residual,
    double* __restrict__ prior_energy,
    double* __restrict__ invalid,
    const float* __restrict__ uniforms,
    const long long* __restrict__ cols,
    const float* __restrict__ lam,
    const float* __restrict__ degree,
    const long long* __restrict__ col_det_indptr,
    const long long* __restrict__ col_det_indices,
    const float* __restrict__ uniforms_idle,
    const int k, const int rows,
    const float gamma, const float gamma_active, const float gamma_far,
    const float idle_rate, const float kappa, const int metropolis)
{
    const long long total = (long long)k * (long long)rows;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += stride) {
        const int i = (int)(t / rows);
        const int j = (int)(t - (long long)i * rows);
        const long long s = col_det_indptr[i];
        const long long e = col_det_indptr[i + 1];
        float bef = 0.0f;
        for (long long p = s; p < e; ++p)
            bef += residual[col_det_indices[p] * (long long)rows + j];
        const long long sidx = cols[i] * (long long)rows + j;
        const unsigned char bit = states[sidx];
        const float sign = 1.0f - 2.0f * (float)bit;
        const float syn_delta = degree[i] - 2.0f * bef;
        // Active-region rule: a mechanism is active iff it is set (bit==1) or
        // touches a violated detector (bef>0). Idle bits use gamma_far and are
        // gated below; active bits use gamma (set) or gamma_active (near syndrome).
        const bool active = (bit != 0) || (bef > 0.0f);
        const float g = (bit != 0) ? gamma : (bef > 0.0f ? gamma_active : gamma_far);
        const float flip_delta = g * sign * lam[i] + kappa * syn_delta;
        const float u = uniforms[t];
        bool flip;
        if (metropolis) {
            flip = (flip_delta <= 0.0f) || (u < expf(-flip_delta));
        } else {
            const float p1 = 0.5f * (1.0f - tanhf(0.5f * flip_delta * sign));
            flip = (u < p1) ^ (bit != 0);
        }
        // Soft idle gating: idle bits (bit==0 && bef==0) flip only with prob
        // idle_rate. Prior/energy bookkeeping below stays physical (uses lam[i],
        // syn_delta), so only the flip decision is biased.
        if (flip && !active && idle_rate < 1.0f && !(uniforms_idle[t] < idle_rate)) {
            flip = false;
        }
        if (flip) {
            states[sidx] = bit ^ (unsigned char)1;
            atomicAdd(&prior_energy[j], (double)(sign * lam[i]));
            atomicAdd(&invalid[j], (double)syn_delta);
            for (long long p = s; p < e; ++p) {
                float* rp = &residual[col_det_indices[p] * (long long)rows + j];
                *rp = 1.0f - *rp;
            }
        }
    }
}
"""


@dataclass(frozen=True)
class CircuitDecodeStats:
    """Aggregate diagnostics from a batched decode call."""

    num_shots: int
    valid_fraction: float
    mean_selected_weight: float
    num_colors: int
    mean_legs: float = 1.0
    mean_sweeps: float = 0.0
    # Worst-case / tail cost across shots (the chunk wall time is driven by the
    # hardest shot because legs are synchronized over the active batch).
    max_legs_used: int = 1
    p99_legs_used: float = 1.0
    max_sweeps_used: int = 0
    p99_sweeps_used: float = 0.0
    # Per-shot boolean: True iff the shot reached at least one syndrome-valid
    # correction. Shots with valid_mask=False fell back to a lowest-residual
    # (syndrome-violating) guess and MUST be scored as logical failures.
    valid_mask: Optional[np.ndarray] = None


class ColoredBlockGibbsDecoder:
    """Sparse, chain-parallel, annealed chromatic-block-Gibbs DEM decoder."""

    def __init__(
        self,
        matrices: DEMMatrices,
        *,
        chains: int = 64,
        sweeps: int = 100,
        beta: float = 1.0,
        beta_start: Optional[float] = None,
        beta_stop: Optional[float] = None,
        beta_schedule_shape: str = "geometric",
        rho: Optional[float] = None,
        rho_scale: Optional[float] = 1.0,
        gamma: float = 1.0,
        idle_update_rate: float = 1.0,
        gamma_active: Optional[float] = None,
        gamma_far: Optional[float] = None,
        update: str = "gibbs",
        selection: str = "best_energy",
        capture: str = "sweep",
        max_legs: int = 1,
        target_valid: Optional[int] = None,
        compact: bool = True,
        compact_interval: int = 32,
        initial_probability: Optional[float] = None,
        seed: int = 0,
        state_budget_bytes: int = 2 * (1 << 30),
        dtype: np.dtype = np.float32,
        device: str = "cpu",
    ) -> None:
        if chains <= 0:
            raise ValueError("chains must be positive")
        if sweeps < 0:
            raise ValueError("sweeps must be nonnegative")
        if update not in ("gibbs", "metropolis"):
            raise ValueError(f"unknown update rule {update!r}")
        if selection not in ("best_energy", "min_weight"):
            raise ValueError(f"unknown selection {selection!r}")
        if capture not in ("sweep", "block"):
            raise ValueError(f"unknown capture mode {capture!r}")
        if beta_schedule_shape not in ("geometric", "linear"):
            raise ValueError(f"unknown beta_schedule_shape {beta_schedule_shape!r}")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if not (0.0 <= idle_update_rate <= 1.0):
            raise ValueError("idle_update_rate must be in [0, 1] (0 = hard gating: freeze idle bits)")
        if gamma_active is not None and gamma_active <= 0.0:
            raise ValueError("gamma_active must be positive")
        if gamma_far is not None and gamma_far <= 0.0:
            raise ValueError("gamma_far must be positive")
        if max_legs < 1:
            raise ValueError("max_legs must be >= 1")
        if target_valid is not None and target_valid < 1:
            raise ValueError("target_valid must be None or >= 1")
        if compact_interval < 1:
            raise ValueError("compact_interval must be >= 1")

        # Backend selection. The CPU path uses numpy + scipy.sparse exactly as
        # before; the GPU path swaps in cupy + cupyx.scipy.sparse with identical
        # math (only the residual dtype / scatter differ, see _sweep).
        self.device = str(device)
        self._on_gpu = self.device in ("cuda", "gpu")
        if self._on_gpu:
            try:
                import cupy as _cp
                import cupyx.scipy.sparse as _csp
            except ImportError as exc:  # pragma: no cover - optional GPU dep
                raise ImportError(
                    "device='cuda' requires cupy (e.g. pip install cupy-cuda12x)"
                ) from exc
            self.xp = _cp
            self._xsp = _csp
            self._fused_kernel = _cp.RawKernel(_FUSED_SWEEP_SRC, "fused_color_sweep")
        else:
            self.xp = np
            self._xsp = sparse
            self._fused_kernel = None
        # GPU sweeps use the fused kernel by default; flip to False to fall back
        # to the reference cupy elementwise path (same math, used for validation).
        self._fused = self._on_gpu

        self.matrices = matrices
        self.chains = int(chains)
        self.sweeps = int(sweeps)
        self.beta = float(beta)
        self.beta_start = beta_start
        self.beta_stop = beta_stop
        self.beta_schedule_shape = str(beta_schedule_shape)
        self.gamma = float(gamma)
        # Active-region knobs (no-op at defaults): a mechanism is "active" iff it
        # is currently set (e_j=1) or touches a violated detector (v_j>0). Idle
        # bits (e_j=0, v_j=0) are resampled only with probability idle_update_rate:
        # rate=1 no gating (default), rate in (0,1) SOFT gating, rate=0 HARD gating
        # (idle bits are frozen until a neighbour violates one of their detectors).
        # The prior sharpness can also differ by region -- gamma_active near the
        # syndrome vs gamma_far in quiet regions. Defaults (rate=1, gammas=gamma)
        # reproduce the uniform sweep exactly.
        self.idle_update_rate = float(idle_update_rate)
        self.gamma_active = self.gamma if gamma_active is None else float(gamma_active)
        self.gamma_far = self.gamma if gamma_far is None else float(gamma_far)
        self._idle_gate = self.idle_update_rate < 1.0
        self._active_on = (self.gamma_active != self.gamma) or (self.gamma_far != self.gamma)
        self.update = update
        self.selection = selection
        self.capture = capture
        # Adaptive-legs (RelayBP-style ensemble): a "leg" is a wave of `chains`
        # fresh chains annealed for up to `sweeps` sweeps. A chain that reaches a
        # valid (syndrome-consistent) state contributes one ensemble member;
        # `target_valid` members stop a shot early (and stop the wasted sweeps),
        # and shots short of `target_valid` get another leg, up to `max_legs`.
        # Defaults (max_legs=1, target_valid=None) reproduce the single fixed-
        # budget pass. `target_valid=None` runs full sweeps but still retries a
        # shot with a fresh leg until it has any valid sample.
        self.max_legs = int(max_legs)
        self.target_valid = None if target_valid is None else int(target_valid)
        # Mid-leg compaction: drop shots that have reached the stop threshold from
        # the annealed working set so the remaining (hard) shots run at reduced
        # width. Only active when a capture/target stop is in play; behaviour is
        # statistically identical (snapshots freeze at crossing time) but the RNG
        # stream shifts, so it is not bit-for-bit vs compact=False.
        self._compact = bool(compact)
        self._compact_interval = max(1, int(compact_interval))
        self.initial_probability = initial_probability
        self.dtype = dtype
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._device_rng = self.xp.random.default_rng(seed) if self._on_gpu else None

        self.m = matrices.num_detectors
        self.num_observables = matrices.num_observables

        # Detector-less mechanisms are invisible to the syndrome; the maximum
        # likelihood choice (prior < 1/2) always leaves them off, so we drop them
        # from the sampled state. ``kept_columns`` maps surviving columns back to
        # the original DEM ordering for callers that need it.
        check_matrix = matrices.check_matrix.tocsc()
        column_degree = np.asarray(check_matrix.sum(axis=0)).ravel()
        keep = column_degree > 0
        self.kept_columns = np.flatnonzero(keep)
        self._check_matrix = check_matrix[:, self.kept_columns].tocsc()
        self.n = int(self.kept_columns.size)

        priors = np.clip(matrices.priors[self.kept_columns].astype(np.float64), 1e-15, 0.5 - 1e-9)
        self.lambdas = np.log((1.0 - priors) / priors).astype(dtype)
        self.rho = self._resolve_rho(rho, rho_scale)

        observables = matrices.observables_matrix.tocsc()[:, self.kept_columns]
        self._observables_csr = self._to_device_sparse(observables.tocsr())
        self._colors = self._build_colors()

        # Rows = shots * chains processed per chunk, bounded by a memory budget on
        # the (rows x n) state matrix (the largest array we allocate).
        per_shot_bytes = max(self.n * self.chains, 1)
        self.max_shots_per_chunk = max(1, int(state_budget_bytes) // per_shot_bytes)

    @property
    def num_colors(self) -> int:
        return len(self._colors)

    def _resolve_rho(self, rho: Optional[float], rho_scale: Optional[float]) -> float:
        if rho is not None:
            if rho <= 0.0:
                raise ValueError("rho must be positive")
            return float(rho)
        max_lambda = float(np.max(self.lambdas)) if self.n else 1.0
        if rho_scale is not None:
            if rho_scale <= 0.0:
                raise ValueError("rho_scale must be positive")
            return float(rho_scale) * max_lambda
        return float(max(8.0, 4.0 * max_lambda))

    def _to_device(self, array: np.ndarray):
        """Move a host array to the active backend (no-op on the CPU backend)."""
        return self.xp.asarray(array) if self._on_gpu else array

    def _to_device_sparse(self, csr: sparse.csr_matrix):
        """Backend sparse matrix: float32 cuSPARSE CSR on GPU, uint8 CSR on CPU.

        cuSPARSE only accepts floating dtypes for sparse @ dense, so GPU gathers
        and the observables map are stored as ``float32`` (entries are 0/1, so the
        products are exact integers).
        """
        if self._on_gpu:
            return self._xsp.csr_matrix(csr.astype(np.float32))
        return csr.astype(np.uint8)

    def _build_colors(self) -> List[_ColorBlock]:
        colors: List[_ColorBlock] = []
        h_csc = self._check_matrix
        for cols in greedy_coloring(h_csc):
            sub = h_csc[:, cols].tocsc()  # (m x k), one detector touched by one column
            det_idx = sub.indices.astype(np.intp)
            degree = np.diff(sub.indptr).astype(self.dtype)
            col_of_det = np.repeat(np.arange(cols.size, dtype=np.intp), np.diff(sub.indptr))
            colors.append(
                _ColorBlock(
                    cols=self._to_device(cols.astype(np.int64)),
                    lam=self._to_device(self.lambdas[cols].astype(self.dtype)),
                    degree=self._to_device(degree),
                    gather=self._to_device_sparse(sub.T.tocsr()),
                    det_idx=self._to_device(det_idx.astype(np.int64)),
                    col_of_det=self._to_device(col_of_det),
                    col_det_indptr=self._to_device(sub.indptr.astype(np.int64)),
                )
            )
        return colors

    def _beta_schedule(self) -> np.ndarray:
        steps = max(self.sweeps, 1)
        if self.beta_start is None and self.beta_stop is None:
            if self.beta <= 0.0:
                raise ValueError("beta must be positive")
            return np.full(steps, self.beta, dtype=self.dtype)
        if self.beta_start is None or self.beta_stop is None:
            raise ValueError("beta_start and beta_stop must be provided together")
        if self.beta_start <= 0.0 or self.beta_stop <= 0.0:
            raise ValueError("beta_start and beta_stop must be positive")
        if self.beta_schedule_shape == "linear":
            sched = np.linspace(self.beta_start, self.beta_stop, steps)
        else:
            sched = np.geomspace(self.beta_start, self.beta_stop, steps)
        return sched.astype(self.dtype)

    def decode(
        self,
        syndromes: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Decode a batch of detector syndromes into observable predictions.

        Args:
            syndromes: ``(num_shots, num_detectors)`` array of 0/1 detection bits.
            rng: Optional generator; defaults to the engine's seeded generator.

        Returns:
            ``(num_shots, num_observables)`` uint8 array of predicted observable flips.
        """
        predictions, _ = self.decode_with_stats(syndromes, rng=rng)
        return predictions

    def decode_with_stats(
        self,
        syndromes: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, CircuitDecodeStats]:
        """Like :meth:`decode` but also return :class:`CircuitDecodeStats`."""
        rng = self._rng if rng is None else rng
        # On GPU we draw from a persistent cupy generator; the CPU path uses the
        # provided (or engine) numpy generator unchanged.
        sample_rng = self._device_rng if self._on_gpu else rng
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes[None, :]
        if syndromes.shape[1] != self.m:
            raise ValueError(
                f"expected syndromes with {self.m} detectors, got {syndromes.shape[1]}"
            )
        num_shots = syndromes.shape[0]
        predictions = np.zeros((num_shots, self.num_observables), dtype=np.uint8)
        weight_total = 0
        legs_parts: List[np.ndarray] = []
        sweeps_parts: List[np.ndarray] = []
        valid_parts: List[np.ndarray] = []

        for begin in range(0, num_shots, self.max_shots_per_chunk):
            end = min(begin + self.max_shots_per_chunk, num_shots)
            chunk_pred, chunk_valid, chunk_weight, chunk_legs, chunk_sweeps, _snaps, _corr = (
                self._decode_chunk(syndromes[begin:end], sample_rng)
            )
            predictions[begin:end] = chunk_pred
            valid_parts.append(chunk_valid)
            weight_total += chunk_weight
            legs_parts.append(chunk_legs)
            sweeps_parts.append(chunk_sweeps)

        legs_all = (
            np.concatenate(legs_parts) if legs_parts else np.zeros(0, dtype=np.int64)
        )
        sweeps_all = (
            np.concatenate(sweeps_parts) if sweeps_parts else np.zeros(0, dtype=np.int64)
        )
        valid_all = (
            np.concatenate(valid_parts) if valid_parts else np.zeros(0, dtype=bool)
        )
        valid_total = int(valid_all.sum())
        stats = CircuitDecodeStats(
            num_shots=num_shots,
            valid_fraction=valid_total / num_shots if num_shots else 0.0,
            mean_selected_weight=weight_total / num_shots if num_shots else 0.0,
            num_colors=self.num_colors,
            mean_legs=float(legs_all.mean()) if legs_all.size else 0.0,
            mean_sweeps=float(sweeps_all.mean()) if sweeps_all.size else 0.0,
            max_legs_used=int(legs_all.max()) if legs_all.size else 0,
            p99_legs_used=float(np.percentile(legs_all, 99)) if legs_all.size else 0.0,
            max_sweeps_used=int(sweeps_all.max()) if sweeps_all.size else 0,
            p99_sweeps_used=(
                float(np.percentile(sweeps_all, 99)) if sweeps_all.size else 0.0
            ),
            valid_mask=valid_all,
        )
        return predictions, stats

    def decode_corrections(
        self,
        syndromes: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, CircuitDecodeStats]:
        """Decode a batch and return the SELECTED correction bit-strings.

        Returns ``(corrections, predictions, stats)`` where ``corrections`` is a
        ``(num_shots, num_mechanisms)`` uint8 array of the chosen error pattern per
        shot -- the lowest-weight syndrome-valid sample, or the lowest-residual
        fallback when no valid sample was found (``stats.valid_mask`` flags which).
        ``predictions`` is the usual ``(num_shots, num_observables)`` observable
        projection. This is the entry point for code-capacity scoring, which needs
        the full residual ``error ^ correction`` to split same/degen/logic.
        """
        rng = self._rng if rng is None else rng
        sample_rng = self._device_rng if self._on_gpu else rng
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes[None, :]
        if syndromes.shape[1] != self.m:
            raise ValueError(
                f"expected syndromes with {self.m} detectors, got {syndromes.shape[1]}"
            )
        num_shots = syndromes.shape[0]
        n_mech = self.matrices.num_mechanisms
        corrections = np.zeros((num_shots, n_mech), dtype=np.uint8)
        predictions = np.zeros((num_shots, self.num_observables), dtype=np.uint8)
        weight_total = 0
        legs_parts: List[np.ndarray] = []
        sweeps_parts: List[np.ndarray] = []
        valid_parts: List[np.ndarray] = []

        for begin in range(0, num_shots, self.max_shots_per_chunk):
            end = min(begin + self.max_shots_per_chunk, num_shots)
            chunk_pred, chunk_valid, chunk_weight, chunk_legs, chunk_sweeps, _snaps, chunk_corr = (
                self._decode_chunk(
                    syndromes[begin:end], sample_rng, want_corrections=True
                )
            )
            predictions[begin:end] = chunk_pred
            corrections[begin:end] = chunk_corr
            valid_parts.append(chunk_valid)
            weight_total += chunk_weight
            legs_parts.append(chunk_legs)
            sweeps_parts.append(chunk_sweeps)

        legs_all = (
            np.concatenate(legs_parts) if legs_parts else np.zeros(0, dtype=np.int64)
        )
        sweeps_all = (
            np.concatenate(sweeps_parts) if sweeps_parts else np.zeros(0, dtype=np.int64)
        )
        valid_all = (
            np.concatenate(valid_parts) if valid_parts else np.zeros(0, dtype=bool)
        )
        valid_total = int(valid_all.sum())
        stats = CircuitDecodeStats(
            num_shots=num_shots,
            valid_fraction=valid_total / num_shots if num_shots else 0.0,
            mean_selected_weight=weight_total / num_shots if num_shots else 0.0,
            num_colors=self.num_colors,
            mean_legs=float(legs_all.mean()) if legs_all.size else 0.0,
            mean_sweeps=float(sweeps_all.mean()) if sweeps_all.size else 0.0,
            max_legs_used=int(legs_all.max()) if legs_all.size else 0,
            p99_legs_used=float(np.percentile(legs_all, 99)) if legs_all.size else 0.0,
            max_sweeps_used=int(sweeps_all.max()) if sweeps_all.size else 0,
            p99_sweeps_used=(
                float(np.percentile(sweeps_all, 99)) if sweeps_all.size else 0.0
            ),
            valid_mask=valid_all,
        )
        return corrections, predictions, stats

    def decode_multi(
        self,
        syndromes: np.ndarray,
        targets: Sequence[int],
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], CircuitDecodeStats]:
        """Decode once at ``target_valid = max(targets)`` and return, for every
        ``k`` in ``targets``, the prediction/valid-mask that a ``target_valid=k``
        run would have produced (recovered by snapshotting the monotonically
        improving best-valid correction when each shot first reaches ``k`` valid
        members). Returns ``(predictions_by_k, valid_mask_by_k, stats)`` where the
        stats describe the single underlying max-target run. ``k=max(targets)`` is
        exact; smaller ``k`` are a slight upper bound on a dedicated batch run
        (see ``_decode_chunk``)."""
        rng = self._rng if rng is None else rng
        sample_rng = self._device_rng if self._on_gpu else rng
        ks = sorted({int(k) for k in targets if int(k) >= 1})
        if not ks:
            raise ValueError("targets must contain integers >= 1")
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes[None, :]
        if syndromes.shape[1] != self.m:
            raise ValueError(
                f"expected syndromes with {self.m} detectors, got {syndromes.shape[1]}"
            )
        num_shots = syndromes.shape[0]
        preds = {k: np.zeros((num_shots, self.num_observables), dtype=np.uint8) for k in ks}
        valids = {k: np.zeros(num_shots, dtype=bool) for k in ks}
        weight_total = 0
        legs_parts: List[np.ndarray] = []
        sweeps_parts: List[np.ndarray] = []
        valid_parts: List[np.ndarray] = []

        kmax = ks[-1]
        for begin in range(0, num_shots, self.max_shots_per_chunk):
            end = min(begin + self.max_shots_per_chunk, num_shots)
            chosen_pred, chunk_valid, chunk_weight, chunk_legs, chunk_sweeps, snaps, _corr = (
                self._decode_chunk(syndromes[begin:end], sample_rng, capture_targets=ks)
            )
            assert snaps is not None
            for k in ks:
                if k == kmax:
                    # The run's stop threshold IS k=max, so the final selected
                    # correction is the exact target_valid=kmax result (no snapshot
                    # approximation). Smaller k use the monotone snapshot.
                    pred_k, valid_k = chosen_pred, chunk_valid
                else:
                    pred_k, valid_k = snaps[k]
                preds[k][begin:end] = pred_k
                valids[k][begin:end] = valid_k
            valid_parts.append(chunk_valid)
            weight_total += chunk_weight
            legs_parts.append(chunk_legs)
            sweeps_parts.append(chunk_sweeps)

        legs_all = np.concatenate(legs_parts) if legs_parts else np.zeros(0, dtype=np.int64)
        sweeps_all = np.concatenate(sweeps_parts) if sweeps_parts else np.zeros(0, dtype=np.int64)
        valid_all = np.concatenate(valid_parts) if valid_parts else np.zeros(0, dtype=bool)
        stats = CircuitDecodeStats(
            num_shots=num_shots,
            valid_fraction=int(valid_all.sum()) / num_shots if num_shots else 0.0,
            mean_selected_weight=weight_total / num_shots if num_shots else 0.0,
            num_colors=self.num_colors,
            mean_legs=float(legs_all.mean()) if legs_all.size else 0.0,
            mean_sweeps=float(sweeps_all.mean()) if sweeps_all.size else 0.0,
            max_legs_used=int(legs_all.max()) if legs_all.size else 0,
            p99_legs_used=float(np.percentile(legs_all, 99)) if legs_all.size else 0.0,
            max_sweeps_used=int(sweeps_all.max()) if sweeps_all.size else 0,
            p99_sweeps_used=float(np.percentile(sweeps_all, 99)) if sweeps_all.size else 0.0,
            valid_mask=valid_all,
        )
        return preds, valids, stats

    def _decode_chunk(
        self,
        syndromes: np.ndarray,
        rng: np.random.Generator,
        capture_targets: Optional[Sequence[int]] = None,
        want_corrections: bool = False,
    ) -> Tuple[
        np.ndarray, np.ndarray, int, np.ndarray, np.ndarray,
        Optional[dict], Optional[np.ndarray],
    ]:
        """Adaptive-legs decode of one shot chunk (RelayBP-style ensemble).

        Runs up to ``max_legs`` waves ("legs") of ``chains`` fresh chains. Within a
        leg, chains anneal for up to ``sweeps`` sweeps and each chain that reaches a
        valid state contributes one ensemble member; the leg stops as soon as every
        still-active shot has collected ``target_valid`` members, so easy shots burn
        only a handful of sweeps instead of the whole budget. After a leg, shots
        that reached the target drop out and the rest get another leg. The
        lowest-weight valid sample across all legs is selected per shot (lowest-
        residual fallback if a shot never goes valid).

        ``max_legs=1, target_valid=None`` reproduces the original single fixed-
        budget pass exactly. Returns ``(predictions, valid_mask, weight_total,
        legs_used, sweeps_used)`` where ``valid_mask`` is a per-shot bool (True iff a
        syndrome-valid correction was found) and the last two are per-shot arrays of
        legs and *executed* sweeps (so the caller can report mean and worst-case).
        """
        xp = self.xp
        shots = syndromes.shape[0]
        chains = self.chains
        n = self.n
        invalid_dtype = xp.float64 if self._on_gpu else xp.int32

        # Global, per-shot best trackers (variable x shot columns), shared across
        # legs so the ensemble minimum survives between legs.
        best_valid_score = xp.full(shots, float("inf"))
        best_valid_states = xp.zeros((n, shots), dtype=xp.uint8)
        have_valid = xp.zeros(shots, dtype=bool)
        best_any_score = xp.full(shots, float("inf"))
        best_any_states = xp.zeros((n, shots), dtype=xp.uint8)

        # Multi-target snapshot capture. When ``capture_targets`` is given, the run
        # uses the largest target as its stop threshold and additionally freezes,
        # per shot, the current best-valid correction at the moment that shot first
        # accumulates k valid members (k in capture_targets). This recovers every
        # smaller-tv answer from a single (largest-tv) trajectory, since best_valid
        # only improves monotonically. (Per-shot stop semantics: a slight upper
        # bound on the batch-global tv result, which keeps annealing easy shots
        # while waiting on hard ones in the same leg.)
        if capture_targets is not None:
            cap_targets = sorted({int(k) for k in capture_targets if int(k) >= 1})
            if not cap_targets:
                raise ValueError("capture_targets must contain integers >= 1")
            snap_states = {k: xp.zeros((n, shots), dtype=xp.uint8) for k in cap_targets}
            snap_have = {k: xp.zeros(shots, dtype=bool) for k in cap_targets}
            snap_done = {k: np.zeros(shots, dtype=bool) for k in cap_targets}
        else:
            cap_targets = None

        beta_schedule = self._beta_schedule()
        # The largest capture target overrides target_valid as the stop threshold.
        target_valid = cap_targets[-1] if cap_targets is not None else self.target_valid
        # A shot is done once it has this many valid members (>=1 when target_valid
        # is None, i.e. just keep retrying with fresh legs until it is solved).
        need = 1 if target_valid is None else target_valid

        # Host-side ensemble bookkeeping that drives the Python leg control flow.
        valid_collected = np.zeros(shots, dtype=np.int64)
        legs_used = np.zeros(shots, dtype=np.int64)
        sweeps_used = np.zeros(shots, dtype=np.int64)
        active = np.arange(shots)  # global shot indices still seeking members

        for _leg in range(self.max_legs):
            if active.size == 0:
                break
            leg_active = active  # shots entering this leg (fixed, for leg/sweep accounting)
            cur_global = np.asarray(active).copy()  # survivors still annealing this leg
            n_active = int(cur_global.size)
            shot_index = xp.arange(n_active)
            gindex = xp.asarray(cur_global)  # local active shot -> global shot column

            states, residual, prior_energy, invalid = self._init_state(
                syndromes[cur_global], chains, rng, invalid_dtype
            )
            ever_valid = xp.zeros(n_active * chains, dtype=bool)
            prior = xp.asarray(valid_collected[cur_global])  # members from earlier legs

            def capture_valid() -> None:
                if self.selection == "min_weight":
                    sc = states.sum(axis=0, dtype=xp.int64).astype(xp.float64)
                else:
                    sc = prior_energy
                self._update_best(
                    xp.where(invalid == 0, sc, float("inf")), states, chains,
                    n_active, shot_index, best_valid_score, best_valid_states,
                    have_valid, gindex,
                )

            # capture='block' also captures after every color commit inside _sweep;
            # the default only captures at sweep boundaries (no per-color overhead).
            on_block = capture_valid if self.capture == "block" else None

            leg_sweeps = 0
            for sweep in range(self.sweeps + 1):
                capture_valid()
                self._update_best(
                    invalid.astype(xp.float64) * 1e7 + prior_energy, states, chains,
                    n_active, shot_index, best_any_score, best_any_states, None,
                    gindex,
                )
                ever_valid |= invalid == 0
                leg_sweeps = sweep
                # Only touch the host (snapshot bookkeeping, all-done test, and
                # compaction) on compaction-interval boundaries (and the final
                # sweep). Between boundaries every sweep stays fully on-device, so
                # the GPU pipelines them instead of stalling on a per-sweep
                # device->host sync -- the dominant cost once compaction has shrunk
                # the working set to a few straggler columns.
                if target_valid is not None and (
                    sweep % self._compact_interval == 0 or sweep == self.sweeps
                ):
                    collected = prior + ever_valid.reshape(n_active, chains).sum(axis=1)
                    if cap_targets is not None:
                        coll_host = (
                            xp.asnumpy(collected) if self._on_gpu
                            else np.asarray(collected)
                        )
                        for k in cap_targets:
                            newly = np.flatnonzero(
                                (coll_host >= k) & (~snap_done[k][cur_global])
                            )
                            if newly.size:
                                gsel = cur_global[newly]
                                gdev = xp.asarray(gsel)
                                snap_states[k][:, gdev] = best_valid_states[:, gdev]
                                snap_have[k][gdev] = have_valid[gdev]
                                snap_done[k][gsel] = True
                    if bool((collected >= target_valid).all()):
                        break
                    # Mid-leg compaction: shots that have collected target_valid
                    # members are done (their tv snapshots are already frozen above),
                    # so drop their columns and keep annealing only the stragglers.
                    # The fused kernel reads its row count from the array shape, so a
                    # narrower working set makes every later sweep proportionally
                    # cheaper -- the big win deep below threshold, where most shots
                    # solve in a handful of sweeps but a rare hard shot would
                    # otherwise hold the whole chunk at full width for the budget.
                    if (
                        self._compact and cap_targets is not None
                        and sweep > 0 and sweep % self._compact_interval == 0
                    ):
                        keep_local = coll_host < target_valid
                        if not keep_local.all():
                            members_leg = ever_valid.reshape(n_active, chains).sum(axis=1)
                            members_leg = (
                                xp.asnumpy(members_leg) if self._on_gpu else members_leg
                            ).astype(np.int64)
                            drop = np.flatnonzero(~keep_local)
                            valid_collected[cur_global[drop]] += members_leg[drop]
                            keep_idx = np.flatnonzero(keep_local)
                            col_keep = xp.asarray(
                                (keep_idx[:, None] * chains
                                 + np.arange(chains)[None, :]).ravel()
                            )
                            states = states[:, col_keep]
                            residual = residual[:, col_keep]
                            prior_energy = prior_energy[col_keep]
                            invalid = invalid[col_keep]
                            ever_valid = ever_valid[col_keep]
                            prior = prior[xp.asarray(keep_idx)]
                            cur_global = cur_global[keep_idx]
                            n_active = int(cur_global.size)
                            shot_index = xp.arange(n_active)
                            gindex = xp.asarray(cur_global)
                if sweep == self.sweeps:
                    break
                self._sweep(
                    states, residual, prior_energy, invalid,
                    float(beta_schedule[sweep]), rng, on_block,
                )

            members = ever_valid.reshape(n_active, chains).sum(axis=1)
            members = (xp.asnumpy(members) if self._on_gpu else members).astype(np.int64)
            valid_collected[cur_global] += members
            legs_used[leg_active] += 1
            sweeps_used[leg_active] += leg_sweeps
            active = leg_active[valid_collected[leg_active] < need]

        chosen = xp.where(have_valid[None, :], best_valid_states, best_any_states)

        def _obs_predict(state_mat):
            if self._on_gpu:
                obs = self._observables_csr @ state_mat.astype(xp.float32)
                return (xp.asnumpy(obs) % 2).astype(np.uint8).T
            return (np.asarray(self._observables_csr @ state_mat) % 2).astype(np.uint8).T

        predictions = _obs_predict(chosen)
        weight_total = int(chosen.sum(dtype=xp.int64))
        valid_mask = (
            xp.asnumpy(have_valid) if self._on_gpu else np.asarray(have_valid)
        ).astype(bool)

        snaps: Optional[dict] = None
        if cap_targets is not None:
            snaps = {}
            for k in cap_targets:
                # Shots that never reached k members fall back to the run's final
                # choice (best valid, or lowest-residual guess) -- exactly what a
                # dedicated tv=k run would report after exhausting max_legs.
                pending = ~snap_done[k]
                if pending.any():
                    pdev = xp.asarray(np.flatnonzero(pending))
                    snap_states[k][:, pdev] = chosen[:, pdev]
                    snap_have[k][pdev] = have_valid[pdev]
                snaps[k] = (
                    _obs_predict(snap_states[k]),
                    (
                        xp.asnumpy(snap_have[k]) if self._on_gpu
                        else np.asarray(snap_have[k])
                    ).astype(bool),
                )
        corrections_full: Optional[np.ndarray] = None
        if want_corrections:
            chosen_host = xp.asnumpy(chosen) if self._on_gpu else np.asarray(chosen)
            corrections_full = np.zeros(
                (chosen_host.shape[1], self.matrices.num_mechanisms), dtype=np.uint8
            )
            corrections_full[:, self.kept_columns] = chosen_host.T.astype(np.uint8)
        return (
            predictions, valid_mask, weight_total,
            legs_used, sweeps_used, snaps, corrections_full,
        )

    def _init_state(self, syndromes, chains, rng, invalid_dtype):
        """Build ``(states, residual, prior_energy, invalid)`` for one leg.

        Column ``c = local_shot * chains + chain``; each shot's target is tiled
        across its chains and arrays are stored (variable x column) so the per-color
        gathers and per-chain selections stay contiguous. cuSPARSE needs a floating
        residual, so the GPU backend stores it float32 (entries stay 0/1); the CPU
        backend keeps the uint8 XOR formulation.
        """
        xp = self.xp
        n = self.n
        res_dtype = xp.float32 if self._on_gpu else xp.uint8
        target_mr = xp.asarray(
            np.repeat(syndromes.T.astype(np.uint8), chains, axis=1)
        ).astype(res_dtype)  # (m, rows)
        rows = target_mr.shape[1]
        if self.initial_probability is None:
            states = xp.zeros((n, rows), dtype=xp.uint8)
            residual = target_mr.copy()  # H @ 0 == 0, so residual == syndrome
            prior_energy = xp.zeros(rows, dtype=xp.float64)
        elif self._on_gpu:
            raise NotImplementedError(
                "initial_probability warm start is only implemented on the CPU backend"
            )
        else:
            states = (rng.random((n, rows)) < float(self.initial_probability)).astype(np.uint8)
            syndrome_of_state = (self._check_matrix.astype(np.int32) @ states.astype(np.int32)) % 2
            residual = (syndrome_of_state.astype(np.uint8) ^ target_mr)
            prior_energy = self.lambdas.astype(np.float64) @ states
        # Maintained incrementally by _sweep (exact: same-color columns are
        # check-disjoint), so the residual is never re-summed after this point.
        invalid = residual.sum(axis=0, dtype=invalid_dtype)  # (rows,)
        return states, residual, prior_energy, invalid

    def _update_best(
        self,
        per_row_score, states, chains, shots, shot_index,
        best_score, best_states, have_valid, global_index,
    ) -> None:
        """Keep, per shot, the chain column minimizing ``per_row_score``.

        ``shots`` / ``shot_index`` index the current leg's (active) shots while the
        best trackers are global, so ``global_index`` maps each active shot to its
        global column (``global_index = arange(shots)`` is the identity used when a
        single leg covers every shot).
        """
        candidate = per_row_score.reshape(shots, chains)
        best_chain = candidate.argmin(axis=1)
        chain_score = candidate[shot_index, best_chain]
        improved = chain_score < best_score[global_index]
        if not self._on_gpu and not improved.any():
            # CPU fast path; on GPU this guard syncs, and the masked writes below
            # are no-ops when nothing improved.
            return
        cols_sel = (shot_index * chains + best_chain)[improved]
        gsel = global_index[improved]
        best_states[:, gsel] = states[:, cols_sel]
        best_score[gsel] = chain_score[improved]
        if have_valid is not None:
            have_valid[gsel] = True

    def _sweep(
        self,
        states: np.ndarray,
        residual: np.ndarray,
        prior_energy: np.ndarray,
        invalid: np.ndarray,
        beta: float,
        rng: np.random.Generator,
        on_block=None,
    ) -> None:
        """Advance every chain by one chromatic block Gibbs/Metropolis sweep.

        ``states`` (n x rows uint8), ``residual`` (m x rows), ``prior_energy``
        (rows,) and ``invalid`` (rows,) are all mutated in place. For each
        check-disjoint color the per-column violated-check count is gathered with a
        sparse mat-vec (``gather @ residual``) and the simultaneous toggles are
        scattered back with a single fancy-indexed XOR over only the detectors this
        color touches (``det_idx`` is unique within a color, so the XOR cannot
        collide). Because the columns are check-disjoint the per-color violation
        change ``sum(flip * syn_delta)`` is exact, so ``invalid`` is tracked
        incrementally instead of re-summing the residual every sweep.

        The flip field decouples the physical prior from the syndrome pressure::

            syn_delta  = degree - 2 * before
            flip_delta = gamma * sign * lam + kappa_sweep * syn_delta

        A fixed ``gamma`` keeps quiet mechanisms near their prior rate while
        ``kappa_sweep = rho * beta`` carries the (optionally annealed) syndrome
        strength; ``beta`` no longer heats the prior term. When ``on_block`` is
        given it runs after each color commit so the caller can capture transient
        valid states (``capture='block'``).
        """
        if self._on_gpu and self._fused:
            self._sweep_gpu_fused(
                states, residual, prior_energy, invalid, beta, rng, on_block
            )
            return
        xp = self.xp
        metropolis = self.update == "metropolis"
        rows = states.shape[1]
        kappa_sweep = self.rho * beta
        active_on = self._active_on
        idle_gate = self._idle_gate
        idle_rate_f = self.dtype(self.idle_update_rate)
        if active_on:
            g_e1 = self.dtype(self.gamma)
            g_active = self.dtype(self.gamma_active)
            g_far = self.dtype(self.gamma_far)
        for block in self._colors:
            cols = block.cols
            lam = block.lam
            before = (block.gather @ residual).astype(self.dtype)  # (k, rows)
            bits = states[cols]  # (k, rows) uint8
            sign = 1.0 - 2.0 * bits.astype(self.dtype)
            syn_delta = block.degree[:, None] - 2.0 * before
            if active_on:
                prior_coeff = xp.where(
                    bits != 0,
                    g_e1,
                    xp.where(before > 0, g_active, g_far),
                )
                flip_delta = prior_coeff * sign * lam[:, None] + kappa_sweep * syn_delta
            else:
                flip_delta = self.gamma * sign * lam[:, None] + kappa_sweep * syn_delta
            uniforms = rng.random((cols.size, rows), dtype=self.dtype)
            if metropolis:
                with np.errstate(over="ignore"):
                    threshold = xp.exp(-flip_delta)
                flip = (flip_delta <= 0.0) | (uniforms < threshold)
            else:
                delta_one_minus_zero = flip_delta * sign
                probability_one = 0.5 * (1.0 - xp.tanh(0.5 * delta_one_minus_zero))
                flip = (uniforms < probability_one) ^ bits.astype(bool)
            if idle_gate:
                # Match the fused kernel: draw the idle-gate uniforms right after
                # the flip uniforms so both paths share one RNG stream.
                active = (bits != 0) | (before > 0)
                idle_uniforms = rng.random((cols.size, rows), dtype=self.dtype)
                flip = flip & (active | (idle_uniforms < idle_rate_f))
            if not self._on_gpu and not flip.any():
                # CPU-only fast path: skip the scatter when nothing toggled (invalid
                # is unchanged). On GPU this guard would force a device->host sync
                # every color (~90k per decode) and dominate runtime, so we always
                # apply (XOR-ing an all-false mask is a no-op, and the incremental
                # invalid update adds zero).
                continue
            flip_u8 = flip.astype(xp.uint8)
            states[cols] ^= flip_u8
            prior_energy += (flip_u8 * (sign * lam[:, None])).sum(axis=0)
            invalid += (flip_u8 * syn_delta).sum(axis=0).astype(invalid.dtype)
            if self._on_gpu:
                # float32 residual: XOR via parity-preserving (a + b) % 2 over the
                # detectors this color touches (det_idx is unique within a color).
                det = block.det_idx
                residual[det] = (residual[det] + flip.astype(self.dtype)[block.col_of_det]) % 2.0
            else:
                residual[block.det_idx] ^= flip_u8[block.col_of_det]
            if on_block is not None:
                on_block()

    def _sweep_gpu_fused(
        self,
        states,
        residual,
        prior_energy,
        invalid,
        beta: float,
        rng,
        on_block=None,
    ) -> None:
        """GPU sweep using the fused per-color kernel (one launch per color).

        Mathematically identical to the reference loop in :meth:`_sweep`: per
        color it draws the same ``(k, rows)`` uniform block from ``rng`` (so the
        random stream matches the reference path) and the kernel recomputes the
        violated-check count, flip decision, state/residual toggle and the
        ``prior_energy`` / ``invalid`` accumulation. ``residual`` stays exactly
        0/1, so the in-kernel detector sum reproduces ``gather @ residual``.
        """
        xp = self.xp
        rows = int(states.shape[1])
        gamma = np.float32(self.gamma)
        gamma_active = np.float32(self.gamma_active)
        gamma_far = np.float32(self.gamma_far)
        idle_rate = np.float32(self.idle_update_rate)
        idle_gate = self._idle_gate
        kappa = np.float32(self.rho * beta)
        metro = np.int32(1 if self.update == "metropolis" else 0)
        rows32 = np.int32(rows)
        threads = 256
        for block in self._colors:
            k = int(block.cols.size)
            if k == 0:
                continue
            uniforms = rng.random((k, rows), dtype=xp.float32)
            # Draw the idle-gate uniforms in the same per-color order as the
            # reference _sweep so the streams match. Only when gating is active,
            # so defaults stay bit-for-bit identical to the pre-gate kernel; when
            # off, the kernel never dereferences the placeholder (idle_rate>=1).
            idle_uniforms = rng.random((k, rows), dtype=xp.float32) if idle_gate else uniforms
            total = k * rows
            blocks = min((total + threads - 1) // threads, 65535)
            self._fused_kernel(
                (blocks,),
                (threads,),
                (
                    states,
                    residual,
                    prior_energy,
                    invalid,
                    uniforms,
                    block.cols,
                    block.lam,
                    block.degree,
                    block.col_det_indptr,
                    block.det_idx,
                    idle_uniforms,
                    np.int32(k),
                    rows32,
                    gamma,
                    gamma_active,
                    gamma_far,
                    idle_rate,
                    kappa,
                    metro,
                ),
            )
            if on_block is not None:
                on_block()
