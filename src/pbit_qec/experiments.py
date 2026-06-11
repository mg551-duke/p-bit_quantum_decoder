"""Config-driven experiment runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

import pbit_qec.gf2 as gf2
from pbit_qec.codes import make_code
from pbit_qec.decoders import (
    ExactCosetDecoder,
    MWPMDecoder,
    NullspaceCosetSamplerDecoder,
    PBitAnnealedOptimizerDecoder,
    PBitCosetSamplerDecoder,
    PBitNullspaceCosetSamplerDecoder,
)
from pbit_qec.simulation import run_trial, summarize_trials, trial_results_to_frame
from pbit_qec.types import Decoder


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_decoder(name: str, settings: Mapping[str, object]) -> Decoder:
    kwargs = dict(settings.get(name, {})) if isinstance(settings.get(name, {}), dict) else {}
    if name == "exact_coset":
        return ExactCosetDecoder(**kwargs)
    if name == "mwpm":
        return MWPMDecoder(**kwargs)
    if name == "pbit_coset_sampler":
        return PBitCosetSamplerDecoder(**kwargs)
    if name == "pbit_annealed_optimizer":
        return PBitAnnealedOptimizerDecoder(**kwargs)
    if name == "pbit_nullspace_sampler":
        return PBitNullspaceCosetSamplerDecoder(**kwargs)
    if name == "nullspace_coset_sampler":
        return NullspaceCosetSamplerDecoder(**kwargs)
    raise ValueError(f"unknown decoder {name!r}")


def run_config(config: Mapping[str, object], config_path: Optional[Path] = None) -> Dict[str, Path]:
    output_dir = Path(str(config.get("output_dir", "results/run")))
    output_dir.mkdir(parents=True, exist_ok=True)

    global_p_values = [float(p) for p in config.get("p_values", [])]
    if not global_p_values:
        raise ValueError("config must provide p_values")
    decoder_names = [str(name) for name in config.get("decoders", [])]
    if not decoder_names:
        raise ValueError("config must provide decoders")
    settings = config.get("decoder_settings", {})
    if not isinstance(settings, Mapping):
        raise ValueError("decoder_settings must be an object")
    base_seed = int(config.get("seed", 12345))
    exact_settings = settings.get("exact_coset", {})
    exact_max_nullity = int(exact_settings.get("max_nullity", 24))  # type: ignore[union-attr]

    all_trials = []
    all_cosets: List[Dict[str, object]] = []
    exact_cache = {}

    for sweep_index, sweep in enumerate(config.get("sweeps", [])):
        if not isinstance(sweep, Mapping):
            raise ValueError("each sweep must be an object")
        family = str(sweep["family"])
        size = int(sweep["size"])
        trials = int(sweep["trials"])
        use_exact = bool(sweep.get("exact", False))
        p_values = [float(p) for p in sweep.get("p_values", global_p_values)]
        code = make_code(family, size)
        nullity = code.n - gf2.rank(code.H)
        exact_decoder = ExactCosetDecoder(max_nullity=exact_max_nullity) if use_exact else None
        if exact_decoder is not None and nullity > exact_max_nullity:
            exact_decoder = None
        decoders = [
            build_decoder(name, settings)
            for name in decoder_names
            if name != "exact_coset" or exact_decoder is not None
        ]
        for p_idx, p in enumerate(p_values):
            for trial in range(trials):
                seed = base_seed + sweep_index * 1_000_000 + p_idx * 10_000 + trial
                rows, cosets = run_trial(
                    code=code,
                    p=p,
                    trial=trial,
                    seed=seed,
                    decoders=decoders,
                    exact_decoder=exact_decoder,
                    exact_cache=exact_cache,
                )
                all_trials.extend(rows)
                all_cosets.extend(cosets)
    trials_df = trial_results_to_frame(all_trials)
    summary_df = summarize_trials(trials_df)
    cosets_df = pd.DataFrame(all_cosets)

    trials_path = output_dir / "trials.csv"
    summary_path = output_dir / "summary.csv"
    cosets_path = output_dir / "coset_distributions.csv"
    trials_df.to_csv(trials_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    cosets_df.to_csv(cosets_path, index=False)

    manifest = {
        "config_path": str(config_path) if config_path else None,
        "output_dir": str(output_dir),
        "rows": {
            "trials": int(len(trials_df)),
            "summary": int(len(summary_df)),
            "coset_distributions": int(len(cosets_df)),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "trials": trials_path,
        "summary": summary_path,
        "coset_distributions": cosets_path,
        "manifest": manifest_path,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = run_config(load_config(args.config), args.config)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
