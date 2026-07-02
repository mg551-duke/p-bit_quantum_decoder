#!/usr/bin/env python3
"""Parallel circuit-level benchmark for the p-bit DEM decoder.

Samples detector/observable shots from a Stim circuit, decodes them with the
sparse :class:`~pbit_qec.circuit_decoder.ColoredBlockGibbsDecoder` across a
process pool, and reports the logical error rate (LER), the *valid fraction*
(shots for which at least one chain reached a syndrome-consistent correction),
the mean selected correction weight, and decode throughput.

This is the honest pure-p-bit baseline harness: no OSD / BP post-processing. An
optional ``--bposd`` reference uses the SAME ``H``/``A``/priors so the comparison
isolates the solver rather than the DEM conversion.

Examples:
    # Tune the penalty strength on the gross code memory_X circuit.
    python experiments/bench_circuit_pbit.py --shots 2000 --chains 128 \
        --sweeps 300 --beta-start 0.5 --beta-stop 8.0 --rho-scales 0.4,0.5,0.6,0.7

    # Headline benchmark vs BP-OSD.
    python experiments/bench_circuit_pbit.py --shots 20000 --chains 128 \
        --sweeps 300 --beta-start 0.5 --beta-stop 8.0 --rho-scales 0.5 --bposd
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP to a single thread per process *before* importing numpy: this
# harness gets its parallelism from the process pool, so library-level threading
# would only oversubscribe the cores and slow every worker down.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import stim

from pbit_qec.circuit_decoder import ColoredBlockGibbsDecoder
from pbit_qec.dem import DEMMatrices, dem_to_matrices
from pbit_qec.detector_filter import filter_detectors_by_basis

DEFAULT_CIRCUIT = (
    "/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate/"
    "circuit=bicycle_bivariate_144_12_12_memory_X,distance=12,rounds=12,"
    "error_rate=0.003,noise_model=uniform_circuit,basis=CX,"
    "A=x^3+y+y^2,B=y^3+x+x^2.stim"
)

# Per-process global engine, built once by the pool initializer.
_ENGINE: Optional[ColoredBlockGibbsDecoder] = None


def _init_worker(matrices: DEMMatrices, cfg: dict) -> None:
    global _ENGINE
    _ENGINE = ColoredBlockGibbsDecoder(matrices, **cfg)


def _decode_batch(
    args: Tuple[np.ndarray, np.ndarray, int],
) -> Tuple[int, int, float, float, float, int, int, float, int, float, int, int]:
    """Decode one batch; return (errors, valid, weightW, legsW, sweepsW, shots,
    max_legs, p99_legs, max_sweeps, p99_sweeps, invalid, errors_strict).

    ``errors`` is the standard sinter/RelayBP logical-error count: a shot fails
    iff its predicted observable flips differ from the actual ones
    (``A.e_hat != A.e``), scored for every shot including the ones that fell back
    to a lowest-residual (syndrome-violating) guess. This is the apples-to-apples
    metric for comparing against RelayBP, which scores its own non-converged hard
    decisions the same way and reports convergence separately.

    ``errors_strict`` is a conservative variant that ALSO counts every invalid
    (no syndrome-valid correction) shot as a failure -- an upper bound, reported
    as a diagnostic alongside the validity rate.

    The ``*W`` entries are shot-weighted sums so the caller can average exactly;
    the worst-case entries are per-batch tails (caller takes the max across
    batches, so the pooled p99 is a conservative max-of-batch approximation).
    """
    dets, obs, seed = args
    assert _ENGINE is not None
    predictions, stats = _ENGINE.decode_with_stats(dets, rng=np.random.default_rng(seed))
    shots = dets.shape[0]
    valid_mask = (
        stats.valid_mask if stats.valid_mask is not None
        else np.ones(shots, dtype=bool)
    )
    mismatched = np.any(predictions != obs, axis=1)
    errors = int(np.sum(mismatched))
    errors_strict = int(np.sum(mismatched | ~valid_mask))
    invalid = int(np.sum(~valid_mask))
    valid = int(np.sum(valid_mask))
    return (
        errors, valid, stats.mean_selected_weight * shots,
        stats.mean_legs * shots, stats.mean_sweeps * shots, shots,
        stats.max_legs_used, stats.p99_legs_used,
        stats.max_sweeps_used, stats.p99_sweeps_used, invalid, errors_strict,
    )


@dataclass
class RunResult:
    label: str
    shots: int
    errors: int
    valid: int
    mean_weight: float
    seconds: float
    mean_legs: float = 1.0
    mean_sweeps: float = 0.0
    max_legs: int = 0
    p99_legs: float = 0.0
    max_sweeps: int = 0
    p99_sweeps: float = 0.0
    invalid: int = 0
    errors_strict: int = 0

    @property
    def ler(self) -> float:
        return self.errors / self.shots if self.shots else float("nan")

    @property
    def ler_stderr(self) -> float:
        p = self.ler
        return math.sqrt(p * (1.0 - p) / self.shots) if self.shots else float("nan")

    @property
    def ler_strict(self) -> float:
        return self.errors_strict / self.shots if self.shots else float("nan")

    def line(self) -> str:
        # sw_exec = mean *executed* sweeps/shot (adaptive legs stop early), so it is
        # directly comparable to a BP decoder's iteration count.
        extra = (
            f"  legs={self.mean_legs:.2f} sw_exec={self.mean_sweeps:.1f}"
            f"  legs_max={self.max_legs} legs_p99={self.p99_legs:.0f}"
            f"  sw_max={self.max_sweeps} sw_p99={self.p99_sweeps:.0f}"
            if self.mean_sweeps > 0 else ""
        )
        # LER is the sinter/RelayBP convention (predicted != actual observables).
        # LER_strict additionally fails every invalid shot (upper bound).
        strict = (
            f"  LER_strict={self.ler_strict:.4f} ({self.errors_strict}/{self.shots})"
            if self.invalid else ""
        )
        return (
            f"{self.label:<26} LER={self.ler:.4f}+-{self.ler_stderr:.4f} "
            f"({self.errors}/{self.shots})  valid={self.valid/self.shots:.3f} "
            f"invalid={self.invalid}{strict}  "
            f"meanW={self.mean_weight:.1f}  {1e3 * self.seconds / self.shots:.2f} ms/shot "
            f"(wall {self.seconds:.0f}s){extra}"
        )


def _batched(dets: np.ndarray, obs: np.ndarray, batch: int, seed0: int):
    for i, begin in enumerate(range(0, dets.shape[0], batch)):
        end = min(begin + batch, dets.shape[0])
        yield dets[begin:end], obs[begin:end], seed0 + i


def run_pbit(
    matrices: DEMMatrices,
    dets: np.ndarray,
    obs: np.ndarray,
    cfg: dict,
    procs: int,
    batch: int,
    seed0: int,
    label: str,
) -> RunResult:
    tasks = list(_batched(dets, obs, batch, seed0))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=procs, initializer=_init_worker, initargs=(matrices, cfg)
    ) as pool:
        results = list(pool.map(_decode_batch, tasks))
    seconds = time.perf_counter() - t0
    errors = sum(r[0] for r in results)
    valid = sum(r[1] for r in results)
    weight_sum = sum(r[2] for r in results)
    legs_sum = sum(r[3] for r in results)
    sweeps_sum = sum(r[4] for r in results)
    shots = sum(r[5] for r in results)
    max_legs = max((r[6] for r in results), default=0)
    p99_legs = max((r[7] for r in results), default=0.0)
    max_sweeps = max((r[8] for r in results), default=0)
    p99_sweeps = max((r[9] for r in results), default=0.0)
    invalid = sum(r[10] for r in results)
    errors_strict = sum(r[11] for r in results)
    return RunResult(
        label, shots, errors, valid, weight_sum / shots, seconds,
        mean_legs=legs_sum / shots, mean_sweeps=sweeps_sum / shots,
        max_legs=max_legs, p99_legs=p99_legs,
        max_sweeps=max_sweeps, p99_sweeps=p99_sweeps, invalid=invalid,
        errors_strict=errors_strict,
    )


def run_pbit_inproc(
    matrices: DEMMatrices,
    dets: np.ndarray,
    obs: np.ndarray,
    cfg: dict,
    label: str,
) -> RunResult:
    """Single-process decode (used for device='cuda'): one engine, all shots.

    cupy + process-pool fork do not mix, so the GPU path builds a single engine
    in this process and lets the engine chunk shots internally by its memory
    budget. Throughput is therefore the whole-batch wall time on one device.
    """
    engine = ColoredBlockGibbsDecoder(matrices, **cfg)
    t0 = time.perf_counter()
    predictions, stats = engine.decode_with_stats(
        dets, rng=np.random.default_rng(cfg.get("seed", 0))
    )
    seconds = time.perf_counter() - t0
    shots = dets.shape[0]
    valid_mask = (
        stats.valid_mask if stats.valid_mask is not None
        else np.ones(shots, dtype=bool)
    )
    mismatched = np.any(predictions != obs, axis=1)
    errors = int(np.sum(mismatched))
    errors_strict = int(np.sum(mismatched | ~valid_mask))
    invalid = int(np.sum(~valid_mask))
    valid = int(np.sum(valid_mask))
    return RunResult(
        label, shots, errors, valid, stats.mean_selected_weight, seconds,
        mean_legs=stats.mean_legs, mean_sweeps=stats.mean_sweeps,
        max_legs=stats.max_legs_used, p99_legs=stats.p99_legs_used,
        max_sweeps=stats.max_sweeps_used, p99_sweeps=stats.p99_sweeps_used,
        invalid=invalid, errors_strict=errors_strict,
    )


def run_pbit_multi_inproc(
    matrices: DEMMatrices,
    dets: np.ndarray,
    obs: np.ndarray,
    cfg: dict,
    targets: List[int],
    label_prefix: str,
) -> List[RunResult]:
    """Single GPU run at target_valid=max(targets) that scores EVERY tv in
    ``targets`` from one trajectory (see ``ColoredBlockGibbsDecoder.decode_multi``).
    Returns one :class:`RunResult` per tv. The decode wall time is attributed to
    every tv (they share the single run); tv=max is exact, smaller tv are a slight
    upper bound on a dedicated run."""
    engine = ColoredBlockGibbsDecoder(matrices, **cfg)
    t0 = time.perf_counter()
    preds_by_k, valid_by_k, stats = engine.decode_multi(
        dets, targets, rng=np.random.default_rng(cfg.get("seed", 0))
    )
    seconds = time.perf_counter() - t0
    shots = dets.shape[0]
    out: List[RunResult] = []
    for k in sorted(preds_by_k):
        predictions = preds_by_k[k]
        valid_mask = valid_by_k[k]
        mismatched = np.any(predictions != obs, axis=1)
        errors = int(np.sum(mismatched))
        errors_strict = int(np.sum(mismatched | ~valid_mask))
        invalid = int(np.sum(~valid_mask))
        valid = int(np.sum(valid_mask))
        out.append(RunResult(
            f"{label_prefix} tv={k}", shots, errors, valid,
            stats.mean_selected_weight, seconds,
            mean_legs=stats.mean_legs, mean_sweeps=stats.mean_sweeps,
            max_legs=stats.max_legs_used, p99_legs=stats.p99_legs_used,
            max_sweeps=stats.max_sweeps_used, p99_sweeps=stats.p99_sweeps_used,
            invalid=invalid, errors_strict=errors_strict,
        ))
    return out


def run_bposd(
    matrices: DEMMatrices,
    dets: np.ndarray,
    obs: np.ndarray,
    osd_order: int,
    max_iter: int,
) -> RunResult:
    from ldpc import bposd_decoder

    H = matrices.check_matrix.tocsr().astype(np.uint8)
    A = matrices.observables_matrix.tocsr().astype(np.uint8)
    dec = bposd_decoder(
        H,
        channel_probs=list(matrices.priors),
        max_iter=max_iter,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_cs",
        osd_order=osd_order,
    )
    t0 = time.perf_counter()
    errors = 0
    weight_sum = 0
    for i in range(dets.shape[0]):
        e = dec.decode(dets[i])
        weight_sum += int(e.sum())
        if np.any((A @ e) % 2 != obs[i]):
            errors += 1
    seconds = time.perf_counter() - t0
    n = dets.shape[0]
    return RunResult(
        f"bposd(osd_cs{osd_order})", n, errors, n, weight_sum / n, seconds
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--circuit", default=DEFAULT_CIRCUIT)
    ap.add_argument("--basis-filter", default=None, choices=["X", "Z"],
                    help="keep only detectors sensitive to the opposite-basis errors "
                         "(Z -> X-sensitive detectors, pair with memory_Z; X -> "
                         "Z-sensitive detectors, pair with memory_X). None = full DEM.")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=0, help="0 = os.cpu_count()")
    ap.add_argument("--batch", type=int, default=0, help="shots per task; 0 = auto")
    ap.add_argument("--chains", type=int, default=128)
    ap.add_argument("--sweeps", type=int, default=300)
    ap.add_argument("--beta-start", type=float, default=0.5)
    ap.add_argument("--beta-stop", type=float, default=8.0)
    ap.add_argument("--rho-scales", default="0.6", help="comma-separated list")
    ap.add_argument("--update", default="gibbs", choices=["gibbs", "metropolis"])
    ap.add_argument("--selection", default="min_weight", choices=["best_energy", "min_weight"])
    ap.add_argument("--device", default="cpu", help="cpu (process pool) or cuda (single in-process engine)")
    ap.add_argument("--gamma", type=float, default=1.0, help="prior scale, decoupled from kappa")
    ap.add_argument("--idle-update-rate", type=float, default=1.0,
                    help="active-region gating: idle bits (e_j=0, v_j=0) are resampled "
                         "only with this probability. 1.0 = no gating (default); "
                         "0<r<1 = SOFT gating; 0.0 = HARD gating (freeze idle bits).")
    ap.add_argument("--gamma-active", type=float, default=None,
                    help="active-region prior scale for near-syndrome bits (e_j=0, v_j>0); "
                         "default = gamma.")
    ap.add_argument("--gamma-far", type=float, default=None,
                    help="active-region prior scale for far/idle bits (e_j=0, v_j=0); "
                         "default = gamma.")
    ap.add_argument("--capture", default="sweep", choices=["sweep", "block"])
    ap.add_argument("--max-legs", type=int, default=1,
                    help="adaptive-legs: max waves of fresh chains per shot")
    ap.add_argument("--target-valid", type=int, default=None,
                    help="adaptive-legs: stop a shot once it collects this many valid "
                         "samples (early-stop); None = run full sweeps, retry until >=1")
    ap.add_argument("--capture-targets", default=None,
                    help="comma-separated tv values (e.g. '1,2,3,4,5'): one GPU run at "
                         "max(tv) scores every tv via snapshot. device=cuda only.")
    ap.add_argument("--no-compact", action="store_true",
                    help="disable mid-leg compaction (drop finished shots from the "
                         "annealed working set); default on for target/capture stops")
    ap.add_argument("--compact-interval", type=int, default=32,
                    help="sweeps between mid-leg compaction passes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bposd", action="store_true")
    ap.add_argument("--bposd-shots", type=int, default=0, help="0 = same as --shots")
    ap.add_argument("--bposd-order", type=int, default=10)
    args = ap.parse_args()

    import os

    procs = args.procs or os.cpu_count() or 1

    circuit = stim.Circuit.from_file(args.circuit)
    if args.basis_filter:
        circuit = filter_detectors_by_basis(circuit, args.basis_filter)
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    matrices = dem_to_matrices(dem)
    sampler = circuit.compile_detector_sampler(seed=args.seed)
    dets, obs = sampler.sample(args.shots, separate_observables=True)
    dets = dets.astype(np.uint8)
    obs = obs.astype(np.uint8)

    batch = args.batch or max(1, args.shots // (procs * 4))

    print(
        f"# circuit={os.path.basename(args.circuit)}\n"
        f"# detectors={matrices.num_detectors} mechanisms={matrices.num_mechanisms} "
        f"observables={matrices.num_observables} basis_filter={args.basis_filter}\n"
        f"# shots={args.shots} procs={procs} batch={batch} chains={args.chains} "
        f"sweeps={args.sweeps} beta={args.beta_start}->{args.beta_stop} "
        f"update={args.update} selection={args.selection} device={args.device} "
        f"gamma={args.gamma} capture={args.capture} "
        f"max_legs={args.max_legs} target_valid={args.target_valid}\n"
        f"# mean detector weight/shot={dets.sum(axis=1).mean():.1f}",
        flush=True,
    )

    for rs in [float(x) for x in args.rho_scales.split(",")]:
        cfg = dict(
            chains=args.chains,
            sweeps=args.sweeps,
            beta_start=args.beta_start,
            beta_stop=args.beta_stop,
            rho_scale=rs,
            gamma=args.gamma,
            idle_update_rate=args.idle_update_rate,
            gamma_active=args.gamma_active,
            gamma_far=args.gamma_far,
            update=args.update,
            selection=args.selection,
            capture=args.capture,
            max_legs=args.max_legs,
            target_valid=args.target_valid,
            compact=not args.no_compact,
            compact_interval=args.compact_interval,
            seed=args.seed,
        )
        if args.capture_targets is not None:
            if args.device not in ("cuda", "gpu"):
                raise SystemExit("--capture-targets requires --device cuda")
            cfg["device"] = args.device
            targets = sorted({int(x) for x in args.capture_targets.split(",")})
            for res in run_pbit_multi_inproc(
                matrices, dets, obs, cfg, targets,
                label_prefix=f"pbit-{args.device} rho_scale={rs}",
            ):
                print(res.line(), flush=True)
            continue
        if args.device in ("cuda", "gpu"):
            cfg["device"] = args.device
            res = run_pbit_inproc(
                matrices, dets, obs, cfg, label=f"pbit-{args.device} rho_scale={rs}",
            )
        else:
            res = run_pbit(
                matrices, dets, obs, cfg, procs, batch, args.seed * 1000 + 1,
                label=f"pbit rho_scale={rs}",
            )
        print(res.line(), flush=True)

    if args.bposd:
        nb = args.bposd_shots or args.shots
        res = run_bposd(matrices, dets[:nb], obs[:nb], args.bposd_order, max_iter=30)
        print(res.line(), flush=True)


if __name__ == "__main__":
    main()
