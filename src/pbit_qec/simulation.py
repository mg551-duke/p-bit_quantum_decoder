"""Trial simulation and result aggregation."""

from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import pbit_qec.gf2 as gf2
from pbit_qec.decoders import ExactCosetDecoder
from pbit_qec.types import CSSCode, Decoder, DecodeResult, TrialResult


def sample_error(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) < p).astype(np.uint8)


def coset_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        entropy -= prob * math.log(prob, 2)
    return entropy


def run_trial(
    code: CSSCode,
    p: float,
    trial: int,
    seed: int,
    decoders: Sequence[Decoder],
    exact_decoder: Optional[ExactCosetDecoder] = None,
    exact_cache: Optional[Dict[str, DecodeResult]] = None,
) -> tuple[List[TrialResult], List[Dict[str, object]]]:
    rng_error = np.random.default_rng(seed)
    e_true = sample_error(code.n, p, rng_error)
    syndrome = code.syndrome(e_true)
    syndrome_id = gf2.pack_bits(syndrome)
    true_logical_bits = code.logical(e_true)
    true_logical = gf2.bit_string(true_logical_bits)

    exact_key: Optional[str] = None
    exact_weights: Dict[str, float] = {}
    exact_result: Optional[DecodeResult] = None
    if exact_decoder is not None:
        cache_key = _exact_cache_key(code, p, syndrome_id)
        if exact_cache is not None and cache_key in exact_cache:
            exact_result = exact_cache[cache_key]
        else:
            exact_result = exact_decoder.decode(code, syndrome, p, np.random.default_rng(seed + 17))
            if exact_cache is not None:
                exact_cache[cache_key] = exact_result
        exact_key = gf2.bit_string(exact_result.logical_class)
        exact_weights = exact_result.coset_log_weights

    rows: List[TrialResult] = []
    coset_rows: List[Dict[str, object]] = []
    for idx, decoder in enumerate(decoders):
        if decoder.name == "exact_coset" and exact_result is not None:
            result = exact_result
        else:
            decoder_rng = np.random.default_rng(seed + 1009 * (idx + 1))
            result = decoder.decode(code, syndrome, p, decoder_rng)
        decoded_logical = gf2.bit_string(result.logical_class)
        residual_logical = code.logical(e_true ^ result.correction)
        logical_success = bool(result.valid and not residual_logical.any())
        total = result.total_samples or 0
        valid_rate = result.valid_samples / total if total else float(result.valid)
        row = TrialResult(
            code_family=code.family,
            code_size=code.size,
            p=p,
            trial=trial,
            seed=seed,
            decoder=decoder.name,
            syndrome_id=syndrome_id,
            true_logical=true_logical,
            decoded_logical=decoded_logical,
            valid=result.valid,
            logical_success=logical_success,
            runtime_s=result.runtime_s,
            valid_sample_rate=valid_rate,
            valid_samples=result.valid_samples,
            total_samples=result.total_samples,
            first_valid_sweep=result.first_valid_sweep,
            distinct_logicals=result.distinct_logicals,
            coset_entropy=coset_entropy(result.coset_counts),
            exact_best_logical=exact_key,
            exact_agreement=(decoded_logical == exact_key) if exact_key is not None else None,
            metadata=result.metadata,
        )
        rows.append(row)
        for logical, count in result.coset_counts.items():
            coset_rows.append(
                {
                    "code_family": code.family,
                    "code_size": code.size,
                    "p": p,
                    "trial": trial,
                    "decoder": decoder.name,
                    "syndrome_id": syndrome_id,
                    "logical": logical,
                    "sample_count": count,
                    "exact_log_weight": exact_weights.get(logical),
                }
            )
    return rows, coset_rows


def _exact_cache_key(code: CSSCode, p: float, syndrome_id: int) -> str:
    return f"{code.family}:{code.size}:{p:.12g}:{syndrome_id}"


def trial_results_to_frame(results: Iterable[TrialResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "code_family": result.code_family,
            "code_size": result.code_size,
            "p": result.p,
            "trial": result.trial,
            "seed": result.seed,
            "decoder": result.decoder,
            "syndrome_id": result.syndrome_id,
            "true_logical": result.true_logical,
            "decoded_logical": result.decoded_logical,
            "valid": result.valid,
            "logical_success": result.logical_success,
            "runtime_s": result.runtime_s,
            "valid_sample_rate": result.valid_sample_rate,
            "valid_samples": result.valid_samples,
            "total_samples": result.total_samples,
            "first_valid_sweep": result.first_valid_sweep,
            "distinct_logicals": result.distinct_logicals,
            "coset_entropy": result.coset_entropy,
            "exact_best_logical": result.exact_best_logical,
            "exact_agreement": result.exact_agreement,
            "metadata_json": json.dumps(result.metadata, sort_keys=True),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    group_cols = ["code_family", "code_size", "p", "decoder"]
    rows = []
    for keys, group in trials.groupby(group_cols, dropna=False):
        key_data = dict(zip(group_cols, keys))
        n = int(len(group))
        logical_successes = int(group["logical_success"].sum())
        logical_failures = n - logical_successes
        logical_error_low, logical_error_high = wilson_interval(logical_failures, n)
        logical_target_trials = required_trials_for_wilson_half_width(
            logical_failures / n if n else 0.5
        )
        valid_successes = int(group["valid"].sum())
        valid_low, valid_high = wilson_interval(valid_successes, n)
        exact_present = group["exact_agreement"].dropna()
        exact_n = int(len(exact_present))
        exact_successes = int(exact_present.sum()) if exact_n else 0
        exact_low, exact_high = wilson_interval(exact_successes, exact_n)
        exact_target_trials = required_trials_for_wilson_half_width(
            exact_successes / exact_n if exact_n else 0.5
        )

        row = {
            **key_data,
            "trials": n,
            "logical_successes": logical_successes,
            "logical_failures": logical_failures,
            "logical_error_rate": logical_failures / n if n else float("nan"),
            "logical_error_rate_ci_low": logical_error_low,
            "logical_error_rate_ci_high": logical_error_high,
            "logical_error_rate_ci_half_width": _half_width(logical_error_low, logical_error_high),
            "logical_error_target_trials_5pct_ci": logical_target_trials,
            "logical_error_additional_trials_5pct_ci": max(0, logical_target_trials - n),
            "valid_successes": valid_successes,
            "valid_rate": valid_successes / n if n else float("nan"),
            "valid_rate_ci_low": valid_low,
            "valid_rate_ci_high": valid_high,
            "valid_rate_ci_half_width": _half_width(valid_low, valid_high),
            "mean_valid_sample_rate": float(group["valid_sample_rate"].mean()),
            "se_valid_sample_rate": _standard_error(group["valid_sample_rate"]),
            "mean_valid_samples": float(group["valid_samples"].mean()),
            "se_valid_samples": _standard_error(group["valid_samples"]),
            "mean_distinct_logicals": float(group["distinct_logicals"].mean()),
            "se_distinct_logicals": _standard_error(group["distinct_logicals"]),
            "mean_coset_entropy": float(group["coset_entropy"].mean()),
            "se_coset_entropy": _standard_error(group["coset_entropy"]),
            "mean_runtime_s": float(group["runtime_s"].mean()),
            "se_runtime_s": _standard_error(group["runtime_s"]),
            "exact_observations": exact_n,
            "exact_agreement_successes": exact_successes,
            "exact_agreement": exact_successes / exact_n if exact_n else float("nan"),
            "exact_agreement_ci_low": exact_low,
            "exact_agreement_ci_high": exact_high,
            "exact_agreement_ci_half_width": _half_width(exact_low, exact_high),
            "exact_agreement_target_trials_5pct_ci": exact_target_trials if exact_n else np.nan,
            "exact_agreement_additional_trials_5pct_ci": (
                max(0, exact_target_trials - exact_n) if exact_n else np.nan
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_optional_bool(values: pd.Series) -> float:
    present = [value for value in values if pd.notna(value)]
    if not present:
        return float("nan")
    return float(np.mean(present))


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def required_trials_for_wilson_half_width(
    proportion: float,
    target_half_width: float = 0.05,
    z: float = 1.959963984540054,
    max_trials: int = 100_000,
) -> int:
    """Smallest n whose Wilson interval half-width is at most target_half_width."""

    p = min(1.0, max(0.0, float(proportion)))

    def successes_for(n: int) -> int:
        return int(round(p * n))

    lo, hi = 1, 1
    while hi < max_trials:
        low, high = wilson_interval(successes_for(hi), hi, z)
        if _half_width(low, high) <= target_half_width:
            break
        hi *= 2
    hi = min(hi, max_trials)
    while lo < hi:
        mid = (lo + hi) // 2
        low, high = wilson_interval(successes_for(mid), mid, z)
        if _half_width(low, high) <= target_half_width:
            hi = mid
        else:
            lo = mid + 1
    return hi


def _half_width(low: float, high: float) -> float:
    if math.isnan(low) or math.isnan(high):
        return float("nan")
    return (high - low) / 2.0


def _standard_error(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))
