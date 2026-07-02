#!/usr/bin/env python3
"""Run ``sinter.collect`` with the p-bit custom decoder and report LER/throughput.

This is the parallel-CPU counterpart to ``bench_circuit_pbit.py --device cuda``:
both decode the SAME circuit/DEM (``decompose_errors=False``) with the same
hyperparameters, so the Sinter (multi-process CPU) path can be compared
head-to-head against the in-process single-GPU run.

Example:
    python experiments/run_sinter_pbit.py --shots 1000 --chains 128 \
        --sweeps 300 --beta-start 0.5 --beta-stop 8.0 --rho-scale 0.6
"""
from __future__ import annotations

import os

# Single-thread the math libs: parallelism comes from Sinter's worker pool, so
# library threading would only oversubscribe the cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import time

import sinter
import stim

from pbit_qec.detector_filter import filter_detectors_by_basis
from pbit_qec.sinter_decoder import PBitSinterDecoder

DEFAULT_CIRCUIT = (
    "/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate/"
    "circuit=bicycle_bivariate_144_12_12_memory_X,distance=12,rounds=12,"
    "error_rate=0.003,noise_model=uniform_circuit,basis=CX,"
    "A=x^3+y+y^2,B=y^3+x+x^2.stim"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--circuit", default=DEFAULT_CIRCUIT)
    ap.add_argument("--basis-filter", default=None, choices=["X", "Z"],
                    help="Z pairs with memory_Z, X with memory_X; None = full DEM")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--max-errors", type=int, default=0,
                    help="stop after this many logical errors; 0 = only --shots")
    ap.add_argument("--procs", type=int, default=0, help="0 = os.cpu_count()")
    ap.add_argument("--chains", type=int, default=128)
    ap.add_argument("--sweeps", type=int, default=300)
    ap.add_argument("--beta-start", type=float, default=0.5)
    ap.add_argument("--beta-stop", type=float, default=8.0)
    ap.add_argument("--rho-scale", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--capture", default="sweep", choices=["sweep", "block"])
    ap.add_argument("--update", default="gibbs", choices=["gibbs", "metropolis"])
    ap.add_argument("--selection", default="min_weight", choices=["best_energy", "min_weight"])
    ap.add_argument("--max-legs", type=int, default=1,
                    help="adaptive-legs: max waves of fresh chains per shot")
    ap.add_argument("--target-valid", type=int, default=None,
                    help="adaptive-legs: stop a shot once it collects this many valid "
                         "samples (early-stop); None = run full sweeps, retry until >=1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    procs = args.procs or os.cpu_count() or 1
    circuit = stim.Circuit.from_file(args.circuit)
    if args.basis_filter:
        circuit = filter_detectors_by_basis(circuit, args.basis_filter)
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    task = sinter.Task(
        circuit=circuit,
        detector_error_model=dem,
        json_metadata={"circuit": os.path.basename(args.circuit)},
    )
    decoder = PBitSinterDecoder(
        chains=args.chains,
        sweeps=args.sweeps,
        beta_start=args.beta_start,
        beta_stop=args.beta_stop,
        rho_scale=args.rho_scale,
        gamma=args.gamma,
        update=args.update,
        selection=args.selection,
        capture=args.capture,
        max_legs=args.max_legs,
        target_valid=args.target_valid,
        seed=args.seed,
    )

    print(
        f"# sinter.collect circuit={os.path.basename(args.circuit)}\n"
        f"# detectors={dem.num_detectors} observables={dem.num_observables}\n"
        f"# shots={args.shots} procs={procs} chains={args.chains} sweeps={args.sweeps} "
        f"beta={args.beta_start}->{args.beta_stop} rho_scale={args.rho_scale} "
        f"gamma={args.gamma} capture={args.capture} update={args.update} "
        f"selection={args.selection} max_legs={args.max_legs} "
        f"target_valid={args.target_valid}",
        flush=True,
    )

    t0 = time.perf_counter()
    results = sinter.collect(
        num_workers=procs,
        tasks=[task],
        decoders=["pbit"],
        custom_decoders={"pbit": decoder},
        max_shots=args.shots,
        max_errors=args.max_errors or args.shots,
        print_progress=False,
    )
    seconds = time.perf_counter() - t0

    for r in results:
        shots = r.shots
        errors = r.errors
        ler = errors / shots if shots else float("nan")
        stderr = math.sqrt(ler * (1.0 - ler) / shots) if shots else float("nan")
        ms_per_shot = 1e3 * seconds / shots if shots else float("nan")
        print(
            f"sinter pbit  LER={ler:.4f}+-{stderr:.4f} ({errors}/{shots})  "
            f"{ms_per_shot:.2f} ms/shot (wall {seconds:.0f}s, {procs} procs)",
            flush=True,
        )


if __name__ == "__main__":
    main()
