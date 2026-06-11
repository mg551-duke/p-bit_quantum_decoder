"""Calibration helpers for p-bit decoder settings."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import pbit_qec.gf2 as gf2
from pbit_qec.codes import make_code
from pbit_qec.decoders import PBitNullspaceCosetSamplerDecoder
from pbit_qec.experiments import load_config
from pbit_qec.types import CSSCode


def run_pbit_nullspace_calibration(
    config: Mapping[str, object],
    config_path: Optional[Path] = None,
) -> Path:
    del config_path
    hard_cases_path = Path(str(config["hard_cases_path"]))
    output_csv = Path(str(config["output_csv"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    hard_cases = pd.read_csv(hard_cases_path)
    cases = _prepare_cases(hard_cases)
    repeats = int(config.get("repeats", 12))
    base_seed = int(config.get("seed", 404040))
    base_settings = config.get("decoder_settings", {})
    if not isinstance(base_settings, Mapping):
        raise ValueError("decoder_settings must be an object")

    rows: List[Dict[str, object]] = []
    grid = config.get("grid", [])
    for setting_index, setting in enumerate(grid):
        if not isinstance(setting, Mapping):
            raise ValueError("each grid entry must be an object")
        beta = float(setting["beta"])
        updates_per_sample = int(setting["updates_per_sample"])
        recovered = 0
        attempts = 0
        for case_index, case in enumerate(cases):
            decoder = PBitNullspaceCosetSamplerDecoder(
                samples=int(base_settings.get("samples", 800)),
                burn_in=int(base_settings.get("burn_in", 120)),
                updates_per_sample=updates_per_sample,
                beta=beta,
                max_nullity=int(base_settings.get("max_nullity", 64)),
                initial_state=str(base_settings.get("initial_state", "random")),
            )
            for repeat in range(repeats):
                seed = base_seed + setting_index * 1_000_000 + case_index * 100 + repeat
                result = decoder.decode(
                    case["code"],
                    case["syndrome"],
                    case["p"],
                    np.random.default_rng(seed),
                )
                sampled = gf2.bit_string(result.logical_class).zfill(case["code"].k)
                recovered += int(sampled == case["exact_best"])
                attempts += 1
        low, high = _wilson_interval(recovered, attempts)
        rows.append(
            {
                "beta": beta,
                "updates_per_sample": updates_per_sample,
                "recovered": recovered,
                "attempts": attempts,
                "rate": recovered / attempts if attempts else float("nan"),
                "ci_low": low,
                "ci_high": high,
                "ci_half_width": (high - low) / 2.0,
            }
        )

    pd.DataFrame(rows).sort_values(["rate", "ci_low"], ascending=False).to_csv(
        output_csv,
        index=False,
    )
    return output_csv


def _prepare_cases(hard_cases: pd.DataFrame) -> List[Dict[str, object]]:
    code_cache: Dict[tuple[str, int], CSSCode] = {}
    cases = []
    for row in hard_cases.itertuples(index=False):
        key = (str(row.code_family), int(row.code_size))
        if key not in code_cache:
            code_cache[key] = make_code(*key)
        code = code_cache[key]
        cases.append(
            {
                "code": code,
                "p": float(row.p),
                "syndrome": gf2.unpack_bits(int(row.syndrome_id), code.m),
                "exact_best": str(row.exact_best_coset).zfill(code.k),
            }
        )
    return cases


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    path = run_pbit_nullspace_calibration(load_config(args.config), args.config)
    print(f"calibration: {path}")
