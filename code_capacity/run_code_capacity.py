#!/usr/bin/env python3
"""Code-capacity bit-flip (X-error) decoder-performance campaign for external CSS codes.

Loads a CSS code from an ``.npz`` that stores ``Hx, Hz, Lx, Lz`` and builds the
**bit-flip (X-error) sector**:

    H (syndrome matrix) = Hz        # Z-checks detect X errors
    A (logical action)  = Lz        # Lz @ e detects logical X content
    stabilizer rows     = Hx        # X-stabilizers (coset-preserving weight reduction)

Each shot draws i.i.d. bit-flips ``error ~ Bernoulli(p)`` on the n data qubits
(perfect syndrome extraction -- pure code capacity), decodes with the tuned
:class:`pbit_qec.decoders.PBitSyndromeDecoder`, and classifies the residual
``residual = (correction + error) % 2`` exactly as in the reference scoring:

    if (Hz @ residual) % 2 == 0:               # back in the code space
        if residual == 0:           -> same    # recovered the exact error
        elif (Lz @ residual) % 2==0:-> degen   # differ by a stabilizer (harmless)
        else:                       -> logic   # logical failure
    else:                           -> nonconv # correction does not match syndrome

Additionally (to match the spirit of circuit-level scoring) even nonconvergent
residuals are tested for Lz commutation: a nonconvergent residual that anticommutes
with Lz is ``nonconv_logic`` (it carries logical content despite leaving the code
space); the rest are ``nonconv_plain``. ``nonconv = nonconv_logic + nonconv_plain``.

Reported per (code, p):
    runs, same, degen, logic, nonconv, nonconv_logic
    same/run, degen/run, logic/run, nonconv/run
    (logic + nonconv) / run               # legacy LER (logic + nonconv)
    nonconv_logic / run
    (logic + nonconv_logic) / run         # logical-content LER

Errors are accumulated per (code, p) until ``--target-errors`` events of the chosen
``--target-metric`` are observed (default: the legacy ``logic + nonconv``) or until
``--max-shots`` is reached, whichever comes first.

Example:
    .venv/bin/python code_capacity/run_code_capacity.py \\
        --codes code_capacity/codes/B1_HxHzLxLz.npz code_capacity/codes/B2_HxHzLxLz.npz \\
        --p 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \\
        --target-errors 50 --workers 64 --name v1
"""

from __future__ import annotations

import os

# Pin BLAS to one thread per process *before* numpy initializes, so the worker
# process pool gets real parallelism without thread oversubscription.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

# Allow running straight from a checkout without an editable install.
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from pbit_qec.decoders import PBitSyndromeDecoder  # noqa: E402
from pbit_qec.types import CSSCode  # noqa: E402

# Disjoint seed sub-spaces so the error stream and the decoder stream never collide
# and (code, p, shot) maps to a unique, reproducible pair of seeds.
_CODE_STRIDE = 1_000_000_000
_P_STRIDE = 1_000_000
_DECODE_OFFSET = 500_000_000_000

CATEGORIES = ("same", "degen", "logic", "nonconv_logic", "nonconv_plain")

# Tuned code-capacity operating point (bivariate-bicycle / qLDPC regime), matching
# experiments/configs/v1_efficient.json + the schedule-tuning recommendations.
DEFAULT_DECODER: Dict[str, object] = {
    "chains": 256,
    "sweeps": 150,
    "beta_start": 0.5,
    "beta_stop": 5.0,
    "beta_schedule_shape": "linear",
    "rho_scale": 1.0,
    "update": "gibbs",
    "selection": "min_weight",
    "weight_reduction": True,
    "collect_valid_samples": True,
    "stop_on_first_valid": False,
    "target_valid_solutions": 0,
    # Chromatic block-Gibbs spin updates (check-disjoint color classes resampled
    # together): same energy/field math as the sequential scan, ~5x faster per
    # decode on B1/B2. The tuned rho/beta/chains/sweeps carry over unchanged.
    "coloring": True,
}


# --------------------------------------------------------------------------- code IO


def _as_dense_gf2(arr) -> np.ndarray:
    """Coerce a stored matrix (dense, scipy-sparse, or 0-d object) to dense GF(2)."""
    a = np.asarray(arr)
    if a.dtype == object:
        obj = a.item() if a.shape == () else a
        if hasattr(obj, "toarray"):
            a = obj.toarray()
        else:
            a = np.asarray(obj)
    if hasattr(a, "toarray"):
        a = a.toarray()
    return (np.asarray(a) % 2).astype(np.uint8)


def _pick(npz: "np.lib.npyio.NpzFile", *names: str) -> np.ndarray:
    keys = {k.lower(): k for k in npz.files}
    for name in names:
        if name.lower() in keys:
            return npz[keys[name.lower()]]
    raise KeyError(f"none of {names} found in npz (keys present: {list(npz.files)})")


def load_css_code(npz_path: Path, family: Optional[str] = None) -> CSSCode:
    """Build the X-error sector CSSCode (H=Hz, A=Lz, h_x=Hx) from an npz."""
    npz_path = Path(npz_path)
    family = family or npz_path.stem.split("_")[0]
    with np.load(npz_path, allow_pickle=True) as data:
        hx = _as_dense_gf2(_pick(data, "Hx", "H_x", "hx"))
        hz = _as_dense_gf2(_pick(data, "Hz", "H_z", "hz"))
        lx = _as_dense_gf2(_pick(data, "Lx", "L_x", "lx"))
        lz = _as_dense_gf2(_pick(data, "Lz", "L_z", "lz"))
    n = int(hz.shape[1])
    for label, mat in (("Hx", hx), ("Hz", hz), ("Lx", lx), ("Lz", lz)):
        if mat.ndim != 2 or mat.shape[1] != n:
            raise ValueError(f"{label} has shape {mat.shape}; expected (*, {n})")
    if bool((hx @ hz.T % 2).any()):
        raise ValueError(f"{npz_path.name}: Hx and Hz do not commute (Hx @ Hz^T != 0)")
    k = int(lz.shape[0])
    return CSSCode(
        H=hz,
        A=lz,
        qubit_labels=tuple(f"q{i}" for i in range(n)),
        check_labels=tuple(f"z{i}" for i in range(int(hz.shape[0]))),
        logical_labels=tuple(f"L{i}" for i in range(k)),
        family=family,
        size=n,
        metadata={
            "h_x": hx,
            "h_z": hz,
            "l_x": lx,
            "l_z": lz,
            "sector": "x_errors",
            "source": str(npz_path),
        },
    )


def describe_code(code: CSSCode) -> Dict[str, object]:
    hx = np.asarray(code.metadata["h_x"], dtype=np.uint8)
    hz = np.asarray(code.H, dtype=np.uint8)
    lz = np.asarray(code.A, dtype=np.uint8)
    return {
        "family": code.family,
        "n": code.n,
        "k": code.k,
        "m_z_checks": int(hz.shape[0]),
        "m_x_checks": int(hx.shape[0]),
        "hz_max_col_weight": int(hz.sum(axis=0).max()) if hz.size else 0,
        "hz_max_row_weight": int(hz.sum(axis=1).max()) if hz.size else 0,
        "lz_rows": int(lz.shape[0]),
    }


# ------------------------------------------------------------------------- scoring


def classify(error: np.ndarray, correction: np.ndarray, hz: np.ndarray, lz: np.ndarray) -> str:
    """Classify one shot's residual into the reference categories."""
    residual = (error ^ correction) & 1
    in_codespace = not bool((hz @ residual % 2).any())
    has_logical = bool((lz @ residual % 2).any())
    if in_codespace:
        if not residual.any():
            return "same"
        return "logic" if has_logical else "degen"
    return "nonconv_logic" if has_logical else "nonconv_plain"


def _target_count(counts: Dict[str, int], metric: str) -> int:
    logic = counts["logic"]
    nonconv = counts["nonconv_logic"] + counts["nonconv_plain"]
    if metric == "legacy":  # logic + nonconv  (historical LER numerator)
        return logic + nonconv
    if metric == "logic":
        return logic
    if metric == "logical_content":  # logic + nonconv_logic
        return logic + counts["nonconv_logic"]
    raise ValueError(f"unknown target metric {metric!r}")


# --------------------------------------------------------------------------- worker


def _decode_block(args: Tuple) -> Dict[str, object]:
    """Decode a contiguous block of shots; return category counts + diagnostics."""
    (npz_path, family, decoder_kwargs, p, p_index, base_seed, code_index, shot_start, n_shots) = args
    code = load_css_code(Path(npz_path), family)
    hz = np.asarray(code.H, dtype=np.uint8)
    lz = np.asarray(code.A, dtype=np.uint8)
    decoder = PBitSyndromeDecoder(**decoder_kwargs)
    counts = {category: 0 for category in CATEGORIES}
    runtime = 0.0
    sel_weight_sum = 0
    base_p = base_seed + code_index * _CODE_STRIDE + p_index * _P_STRIDE
    for shot in range(shot_start, shot_start + n_shots):
        error = (np.random.default_rng(base_p + shot).random(code.n) < p).astype(np.uint8)
        syndrome = (hz @ error % 2).astype(np.uint8)
        result = decoder.decode(
            code, syndrome, p, np.random.default_rng(_DECODE_OFFSET + base_p + shot)
        )
        correction = np.asarray(result.correction, dtype=np.uint8)
        counts[classify(error, correction, hz, lz)] += 1
        runtime += float(result.runtime_s)
        sel_weight_sum += int(correction.sum())
    return {
        "counts": counts,
        "runtime_s": runtime,
        "shots": int(n_shots),
        "sel_weight_sum": int(sel_weight_sum),
    }


# --------------------------------------------------------------------------- driver


def _rates(counts: Dict[str, int]) -> Dict[str, float]:
    run = sum(counts.values())
    same = counts["same"]
    degen = counts["degen"]
    logic = counts["logic"]
    nonconv_logic = counts["nonconv_logic"]
    nonconv = nonconv_logic + counts["nonconv_plain"]
    legacy = logic + nonconv
    content = logic + nonconv_logic

    def rate(k: int) -> float:
        return k / run if run else float("nan")

    def se(k: int) -> float:
        return (math.sqrt(k) / run) if (run and k) else 0.0

    return {
        "runs": run,
        "same": same,
        "degen": degen,
        "logic": logic,
        "nonconv": nonconv,
        "nonconv_logic": nonconv_logic,
        "nonconv_plain": counts["nonconv_plain"],
        "same_rate": rate(same),
        "degen_rate": rate(degen),
        "logic_rate": rate(logic),
        "nonconv_rate": rate(nonconv),
        "nonconv_logic_rate": rate(nonconv_logic),
        # 2x2 outcome grid: {valid=convergent / invalid=nonconvergent} x
        # {correct = no logical content in residual / incorrect = logical failure}.
        # valid_correct=same+degen, valid_incorrect=logic,
        # invalid_correct=nonconv_plain, invalid_incorrect=nonconv_logic.
        "valid_correct": same + degen,
        "valid_incorrect": logic,
        "invalid_correct": counts["nonconv_plain"],
        "invalid_incorrect": nonconv_logic,
        "valid_correct_rate": rate(same + degen),
        "valid_incorrect_rate": rate(logic),
        "invalid_correct_rate": rate(counts["nonconv_plain"]),
        "invalid_incorrect_rate": rate(nonconv_logic),
        "ler_legacy": rate(legacy),  # (logic + nonconv) / run
        "ler_legacy_se": se(legacy),
        "ler_content": rate(content),  # (logic + nonconv_logic) / run
        "ler_content_se": se(content),
    }


def accumulate_for_p(
    executor: ProcessPoolExecutor,
    npz_path: str,
    family: str,
    decoder_kwargs: Dict[str, object],
    p: float,
    p_index: int,
    base_seed: int,
    code_index: int,
    target_errors: int,
    target_metric: str,
    max_shots: int,
    workers: int,
    shots_per_task: int,
    progress: "ProgressLog",
    tag: str,
) -> Dict[str, object]:
    counts = {category: 0 for category in CATEGORIES}
    runtime = 0.0
    sel_weight_sum = 0
    shots_done = 0
    round_index = 0
    round_shots = max(1, workers) * max(1, shots_per_task)
    start = time.perf_counter()
    while _target_count(counts, target_metric) < target_errors and shots_done < max_shots:
        round_index += 1
        remaining = max_shots - shots_done
        this_round = min(round_shots, remaining)
        tasks = []
        offset = shots_done
        per = max(1, math.ceil(this_round / max(1, workers)))
        placed = 0
        while placed < this_round:
            block = min(per, this_round - placed)
            tasks.append(
                (
                    npz_path,
                    family,
                    decoder_kwargs,
                    p,
                    p_index,
                    base_seed,
                    code_index,
                    offset + placed,
                    block,
                )
            )
            placed += block
        for res in executor.map(_decode_block, tasks):
            for category in CATEGORIES:
                counts[category] += res["counts"][category]
            runtime += res["runtime_s"]
            sel_weight_sum += res["sel_weight_sum"]
        shots_done += this_round
        hit = _target_count(counts, target_metric)
        vc = counts["same"] + counts["degen"]
        vi = counts["logic"]
        ic = counts["nonconv_plain"]
        ii = counts["nonconv_logic"]
        progress.log(
            f"{tag} p={p:g} round={round_index} runs={shots_done} "
            f"valid_correct={vc} valid_incorrect={vi} "
            f"invalid_correct={ic} invalid_incorrect={ii} "
            f"LERlegacy={(vi + ic + ii) / shots_done:.3e} "
            f"LERcontent={(vi + ii) / shots_done:.3e} {target_metric}={hit}/{target_errors}"
        )
    wall = time.perf_counter() - start
    row = {
        "code": tag,
        "source": npz_path,
        "p": p,
        **_rates(counts),
        "target_metric": target_metric,
        "target_errors": target_errors,
        "reached_target": _target_count(counts, target_metric) >= target_errors,
        "mean_selected_weight": (sel_weight_sum / shots_done) if shots_done else float("nan"),
        "ms_per_shot_1core": (1000.0 * runtime / shots_done) if shots_done else float("nan"),
        "wall_s": wall,
    }
    return row


# ----------------------------------------------------------------------- GPU driver


def _build_gpu_decoder(
    code: CSSCode,
    p: float,
    kwargs: Dict[str, object],
    *,
    max_legs: int,
    target_valid: Optional[int],
    state_budget_bytes: int,
    seed: int,
):
    """Map the code-capacity decoder kwargs onto the GPU ColoredBlockGibbs engine.

    The bit-flip sector ``(H=Hz, A=Lz, prior=p)`` is wrapped as a ``DEMMatrices`` so
    the sparse, fused-kernel chromatic block-Gibbs decoder (with RelayBP-style
    adaptive restart legs) can decode whole shot batches on the GPU. ``rho_scale``
    has identical ``rho_scale * max(lambda)`` semantics in both engines, so the
    tuned operating point carries over; selection is forced to a circuit-engine
    mode (``min_weight``/``best_energy``). ``weight_reduction``/``ml_coset`` are not
    ported -- they do not change the LER (only the same/degen split), and for the
    convergence-limited regime min_weight over the valid pool suffices.
    """
    from scipy import sparse

    from pbit_qec.circuit_decoder import ColoredBlockGibbsDecoder
    from pbit_qec.dem import DEMMatrices

    hz = np.asarray(code.H, dtype=np.uint8)
    lz = np.asarray(code.A, dtype=np.uint8)
    selection = str(kwargs.get("selection", "min_weight"))
    if selection not in ("min_weight", "best_energy"):
        selection = "min_weight"
    matrices = DEMMatrices(
        check_matrix=sparse.csc_matrix(hz),
        observables_matrix=sparse.csc_matrix(lz),
        priors=np.full(code.n, float(p), dtype=np.float64),
    )
    return ColoredBlockGibbsDecoder(
        matrices,
        chains=int(kwargs.get("chains", 256)),
        sweeps=int(kwargs.get("sweeps", 1500)),
        beta_start=float(kwargs.get("beta_start", 0.5)),
        beta_stop=float(kwargs.get("beta_stop", 5.0)),
        beta_schedule_shape=str(kwargs.get("beta_schedule_shape", "geometric")),
        rho_scale=float(kwargs.get("rho_scale", 1.0)),
        gamma=float(kwargs.get("prior_scale", 1.0)),
        update=str(kwargs.get("update", "gibbs")),
        selection=selection,
        max_legs=int(max_legs),
        target_valid=(None if target_valid is None else int(target_valid)),
        seed=int(seed) % (2 ** 63),
        state_budget_bytes=int(state_budget_bytes),
        device="cuda",
    )


def accumulate_for_p_gpu(
    *,
    decoder,
    code: CSSCode,
    source: str,
    p: float,
    base_p: int,
    target_errors: int,
    target_metric: str,
    max_shots: int,
    round_shots: int,
    progress: "ProgressLog",
    tag: str,
) -> Dict[str, object]:
    """GPU analogue of :func:`accumulate_for_p`: batch shots through one engine.

    Errors use the SAME per-shot seeding as the CPU path (``base_p + shot``), so a
    GPU run sees the identical error realizations a CPU run would. The selected
    correction is returned per shot and classified with the exact same residual
    rules as :func:`classify`, fully vectorized over the batch.
    """
    hz_t = np.asarray(code.H, dtype=np.int32).T
    lz_t = np.asarray(code.A, dtype=np.int32).T
    n = code.n
    counts = {category: 0 for category in CATEGORIES}
    shots_done = 0
    round_index = 0
    weight_sum = 0
    valid_total = 0
    legs_max = 0
    sweeps_max = 0
    decode_s = 0.0
    start = time.perf_counter()
    while _target_count(counts, target_metric) < target_errors and shots_done < max_shots:
        round_index += 1
        this_round = int(min(round_shots, max_shots - shots_done))
        errors = np.empty((this_round, n), dtype=np.uint8)
        for i in range(this_round):
            errors[i] = (
                np.random.default_rng(base_p + shots_done + i).random(n) < p
            ).astype(np.uint8)
        errors_i = errors.astype(np.int32)
        syndromes = ((errors_i @ hz_t) % 2).astype(np.uint8)
        t0 = time.perf_counter()
        corrections, _predictions, stats = decoder.decode_corrections(syndromes)
        decode_s += time.perf_counter() - t0
        residual = ((errors ^ corrections) & 1).astype(np.int32)
        valid = stats.valid_mask
        if valid is None:
            valid = ~((residual @ hz_t) % 2).any(axis=1)
        has_log = ((residual @ lz_t) % 2).any(axis=1)
        nz = residual.any(axis=1)
        same = valid & ~nz
        degen = valid & nz & ~has_log
        logic = valid & has_log
        nclogic = (~valid) & has_log
        ncplain = (~valid) & ~has_log
        counts["same"] += int(same.sum())
        counts["degen"] += int(degen.sum())
        counts["logic"] += int(logic.sum())
        counts["nonconv_logic"] += int(nclogic.sum())
        counts["nonconv_plain"] += int(ncplain.sum())
        weight_sum += int(corrections.sum())
        valid_total += int(valid.sum())
        legs_max = max(legs_max, int(stats.max_legs_used))
        sweeps_max = max(sweeps_max, int(stats.max_sweeps_used))
        shots_done += this_round
        hit = _target_count(counts, target_metric)
        vc = counts["same"] + counts["degen"]
        vi = counts["logic"]
        ic = counts["nonconv_plain"]
        ii = counts["nonconv_logic"]
        progress.log(
            f"{tag} p={p:g} [gpu] runs={shots_done} "
            f"valid_correct={vc} valid_incorrect={vi} "
            f"invalid_correct={ic} invalid_incorrect={ii} "
            f"valid={valid_total / shots_done:.5f} legs<={legs_max} "
            f"LERlegacy={(vi + ic + ii) / shots_done:.3e} "
            f"LERcontent={(vi + ii) / shots_done:.3e} {target_metric}={hit}/{target_errors}"
        )
    wall = time.perf_counter() - start
    row = {
        "code": tag,
        "source": source,
        "p": p,
        **_rates(counts),
        "target_metric": target_metric,
        "target_errors": target_errors,
        "reached_target": _target_count(counts, target_metric) >= target_errors,
        "mean_selected_weight": (weight_sum / shots_done) if shots_done else float("nan"),
        "ms_per_shot_1core": (1000.0 * decode_s / shots_done) if shots_done else float("nan"),
        "wall_s": wall,
        "device": "cuda",
        "valid_fraction": (valid_total / shots_done) if shots_done else float("nan"),
        "max_legs_used": legs_max,
        "max_sweeps_used": sweeps_max,
    }
    return row


class ProgressLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def _write_result_txt(path: Path, rows: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    for row in rows:
        lines.append(
            f"=== {row['code']}  p={row['p']:g}  runs={row['runs']}  "
            f"target={row['target_metric']} {'REACHED' if row['reached_target'] else 'CAP'} ==="
        )
        lines.append(
            f"  same={row['same']} ({row['same_rate']:.3e})  "
            f"degen={row['degen']} ({row['degen_rate']:.3e})  "
            f"logic={row['logic']} ({row['logic_rate']:.3e})  "
            f"nonconv={row['nonconv']} ({row['nonconv_rate']:.3e})  "
            f"nonconv_logic={row['nonconv_logic']} ({row['nonconv_logic_rate']:.3e})"
        )
        lines.append(
            f"  valid  (convergent)  correct={row['valid_correct']} ({row['valid_correct_rate']:.3e})   "
            f"incorrect={row['valid_incorrect']} ({row['valid_incorrect_rate']:.3e})"
        )
        lines.append(
            f"  invalid(nonconverg.) correct={row['invalid_correct']} ({row['invalid_correct_rate']:.3e})   "
            f"incorrect={row['invalid_incorrect']} ({row['invalid_incorrect_rate']:.3e})"
        )
        lines.append(
            f"  LER(logic+nonconv)      = {row['ler_legacy']:.3e} +- {row['ler_legacy_se']:.1e}"
        )
        lines.append(
            f"  LER(logic+nonconv_logic)= {row['ler_content']:.3e} +- {row['ler_content_se']:.1e}"
        )
        lines.append(
            f"  meanW={row['mean_selected_weight']:.2f}  "
            f"{row['ms_per_shot_1core']:.1f} ms/shot/core  wall={row['wall_s']:.0f}s"
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--codes", nargs="+", required=True, help="paths to *_HxHzLxLz.npz")
    parser.add_argument(
        "--p", nargs="+", type=float, default=[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    )
    parser.add_argument("--target-errors", type=int, default=50)
    parser.add_argument(
        "--target-metric",
        choices=["legacy", "logic", "logical_content"],
        default="legacy",
        help="which event count drives the stop: legacy=logic+nonconv, "
        "logic=logic only, logical_content=logic+nonconv_logic",
    )
    parser.add_argument("--max-shots", type=int, default=20_000_000)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--shots-per-task", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260618)
    parser.add_argument("--name", default="v1")
    parser.add_argument("--output-dir", type=Path, default=None)
    # Decoder knobs (default to the placeholder code-capacity operating point).
    parser.add_argument(
        "--config-map",
        type=Path,
        default=None,
        help="JSON {code_tag: {\"p\": decoder_kwargs}} from tune_code_capacity.py; "
        "per-(code,p) winners override the defaults/CLI for that point.",
    )
    parser.add_argument("--decoder-json", type=str, default=None, help="JSON dict overrides")
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--sweeps", type=int, default=None)
    parser.add_argument("--beta-start", type=float, default=None)
    parser.add_argument("--beta-stop", type=float, default=None)
    parser.add_argument("--shape", type=str, default=None)
    parser.add_argument("--rho-scale", type=float, default=None)
    parser.add_argument(
        "--no-coloring",
        action="store_true",
        help="disable chromatic block-Gibbs updates (fall back to the sequential "
        "per-spin scan; same results, ~5x slower).",
    )
    parser.add_argument("--describe", action="store_true", help="print code dims and exit")
    # GPU backend (batched ColoredBlockGibbs engine via DEMMatrices(H=Hz, A=Lz, p)).
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="cpu = numpy process pool (default); cuda = batched GPU engine.",
    )
    parser.add_argument(
        "--max-legs",
        type=int,
        default=1,
        help="GPU only: adaptive restart legs per shot (RelayBP-style); >1 retries "
        "non-converging shots with fresh chains to clear the nonconv floor.",
    )
    parser.add_argument(
        "--target-valid",
        type=int,
        default=None,
        help="GPU only: stop a shot once it collects this many valid samples; "
        "None = run the full sweep budget per leg.",
    )
    parser.add_argument(
        "--state-budget-gb",
        type=float,
        default=2.0,
        help="GPU only: device-memory budget for the (n x shots*chains) state matrix.",
    )
    parser.add_argument(
        "--gpu-round",
        type=int,
        default=5000,
        help="GPU only: shots per accumulation round (progress/stop cadence).",
    )
    args = parser.parse_args(argv)

    decoder_kwargs = dict(DEFAULT_DECODER)
    if args.no_coloring:
        decoder_kwargs["coloring"] = False
    if args.decoder_json:
        decoder_kwargs.update(json.loads(args.decoder_json))
    for cli_name, key in (
        ("chains", "chains"),
        ("sweeps", "sweeps"),
        ("beta_start", "beta_start"),
        ("beta_stop", "beta_stop"),
        ("shape", "beta_schedule_shape"),
        ("rho_scale", "rho_scale"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            decoder_kwargs[key] = value

    config_map: Dict[str, Dict[str, Dict[str, object]]] = {}
    if args.config_map:
        config_map = json.loads(Path(args.config_map).read_text(encoding="utf-8"))

    def resolve_kwargs(tag: str, p: float) -> Dict[str, object]:
        """Defaults/CLI overlaid with the tuned per-(code,p) winner, if present."""
        merged = dict(decoder_kwargs)
        merged.update(config_map.get(tag, {}).get(f"{p:g}", {}))
        return merged

    code_specs = []
    for path in args.codes:
        code = load_css_code(Path(path))
        code_specs.append((str(Path(path)), code.family, describe_code(code)))
        print(f"loaded {Path(path).name}: {describe_code(code)}")
    if args.describe:
        return

    output_dir = args.output_dir or (Path(__file__).resolve().parent / "results" / args.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLog(output_dir / "progress.txt")
    progress.log(
        f"START name={args.name} codes={[Path(p).name for p in args.codes]} p={args.p} "
        f"target={args.target_errors} metric={args.target_metric} workers={args.workers} "
        f"decoder={decoder_kwargs}"
    )

    rows: List[Dict[str, object]] = []
    summary_path = output_dir / "summary.csv"
    result_path = output_dir / "RESULT.txt"

    def _persist() -> None:
        # Persist after every (code, p) so partial progress is never lost.
        import pandas as pd

        pd.DataFrame(rows).to_csv(summary_path, index=False)
        _write_result_txt(result_path, rows)

    if args.device == "cuda":
        gpu_round = max(1, int(args.gpu_round))
        state_budget_bytes = int(args.state_budget_gb * (1 << 30))
        for code_index, (npz_path, family, _info) in enumerate(code_specs):
            code = load_css_code(Path(npz_path), family)
            tag = Path(npz_path).stem.split("_")[0]
            for p_index, p in enumerate(args.p):
                pp_kwargs = resolve_kwargs(tag, float(p))
                base_p = args.base_seed + code_index * _CODE_STRIDE + p_index * _P_STRIDE
                decoder = _build_gpu_decoder(
                    code,
                    float(p),
                    pp_kwargs,
                    max_legs=args.max_legs,
                    target_valid=args.target_valid,
                    state_budget_bytes=state_budget_bytes,
                    seed=_DECODE_OFFSET + base_p,
                )
                progress.log(
                    f"CONFIG[gpu] {tag} p={p:g}: chains={pp_kwargs.get('chains')} "
                    f"sweeps={pp_kwargs.get('sweeps')} "
                    f"beta={pp_kwargs.get('beta_start')}->{pp_kwargs.get('beta_stop')} "
                    f"rho_scale={pp_kwargs.get('rho_scale')} sel={pp_kwargs.get('selection')} "
                    f"max_legs={args.max_legs} target_valid={args.target_valid} "
                    f"colors={decoder.num_colors}"
                )
                row = accumulate_for_p_gpu(
                    decoder=decoder,
                    code=code,
                    source=npz_path,
                    p=float(p),
                    base_p=base_p,
                    target_errors=args.target_errors,
                    target_metric=args.target_metric,
                    max_shots=args.max_shots,
                    round_shots=gpu_round,
                    progress=progress,
                    tag=tag,
                )
                row["decoder"] = json.dumps(
                    {
                        **pp_kwargs,
                        "device": "cuda",
                        "max_legs": args.max_legs,
                        "target_valid": args.target_valid,
                    },
                    sort_keys=True,
                )
                rows.append(row)
                _persist()
                progress.log(
                    f"DONE {tag} p={p:g}: "
                    f"LER(legacy)={row['ler_legacy']:.3e} "
                    f"LER(content)={row['ler_content']:.3e} runs={row['runs']}"
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for code_index, (npz_path, family, _info) in enumerate(code_specs):
                tag = Path(npz_path).stem.split("_")[0]
                for p_index, p in enumerate(args.p):
                    pp_kwargs = resolve_kwargs(tag, float(p))
                    progress.log(f"CONFIG {tag} p={p:g}: {pp_kwargs}")
                    row = accumulate_for_p(
                        executor=executor,
                        npz_path=npz_path,
                        family=family,
                        decoder_kwargs=pp_kwargs,
                        p=float(p),
                        p_index=p_index,
                        base_seed=args.base_seed,
                        code_index=code_index,
                        target_errors=args.target_errors,
                        target_metric=args.target_metric,
                        max_shots=args.max_shots,
                        workers=args.workers,
                        shots_per_task=args.shots_per_task,
                        progress=progress,
                        tag=tag,
                    )
                    row["decoder"] = json.dumps(pp_kwargs, sort_keys=True)
                    rows.append(row)
                    _persist()
                    progress.log(
                        f"DONE {tag} p={p:g}: "
                        f"LER(legacy)={row['ler_legacy']:.3e} "
                        f"LER(content)={row['ler_content']:.3e} runs={row['runs']}"
                    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "name": args.name,
        "codes": [{"source": s, "family": f, **i} for s, f, i in code_specs],
        "p_values": [float(p) for p in args.p],
        "target_errors": args.target_errors,
        "target_metric": args.target_metric,
        "max_shots": args.max_shots,
        "base_seed": args.base_seed,
        "workers": args.workers,
        "shots_per_task": args.shots_per_task,
        "decoder_defaults": decoder_kwargs,
        "config_map": str(args.config_map) if args.config_map else None,
        "sector": "x_errors (bit-flip): H=Hz, A=Lz, stabilizers=Hx",
        "seed_scheme": {
            "error": "base_seed + code_index*1e9 + p_index*1e6 + shot",
            "decode": "5e11 + base_seed + code_index*1e9 + p_index*1e6 + shot",
        },
        "outputs": {"summary": summary_path.name, "result": result_path.name},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    progress.log(f"ALLDONE -> {summary_path}")
    print(f"\nsummary: {summary_path}\nresult:  {result_path}")


if __name__ == "__main__":
    main()
