"""Sinter integration for the sparse p-bit DEM decoder.

Exposes a :class:`sinter.Decoder` (:class:`PBitSinterDecoder`) that compiles a
Stim detector error model into the sparse
:class:`~pbit_qec.circuit_decoder.ColoredBlockGibbsDecoder` and answers Sinter's
bit-packed shot batches. Use it with
``sinter.collect(..., custom_decoders={"pbit": PBitSinterDecoder(...)})``.

Importing this module requires the optional ``sinter``/``stim`` dependencies; the
rest of ``pbit_qec`` does not import it, so those stay optional.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import sinter
import stim

from pbit_qec.circuit_decoder import ColoredBlockGibbsDecoder
from pbit_qec.dem import dem_to_matrices

# Per-worker state-matrix budget. Sinter typically runs many worker processes, so
# the default is modest; raise it for single-process throughput.
_DEFAULT_STATE_BUDGET_BYTES = 256 * (1 << 20)


class CompiledPBitDecoder(sinter.CompiledDecoder):
    """Sinter ``CompiledDecoder`` wrapping a built :class:`ColoredBlockGibbsDecoder`.

    Implements :meth:`decode_shots_bit_packed`: unpack the detector bits, run the
    sparse chromatic-block-Gibbs decoder, and bit-pack the predicted observables.
    """

    def __init__(
        self,
        engine: ColoredBlockGibbsDecoder,
        num_detectors: int,
        num_observables: int,
        rng: np.random.Generator,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.num_detectors = int(num_detectors)
        self.num_observables = int(num_observables)
        self.rng = rng

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        syndromes = np.unpackbits(
            np.asarray(bit_packed_detection_event_data, dtype=np.uint8),
            axis=1,
            bitorder="little",
        )
        syndromes = syndromes[:, : self.num_detectors]
        predictions = self.engine.decode(syndromes, rng=self.rng)
        return np.packbits(predictions, axis=1, bitorder="little")


class PBitSinterDecoder(sinter.Decoder):
    """Sinter ``Decoder`` for the sparse p-bit chromatic-block-Gibbs decoder.

    Hyperparameters mirror :class:`~pbit_qec.circuit_decoder.ColoredBlockGibbsDecoder`.
    A fresh, deterministically-seeded generator is created per compiled DEM so
    results are reproducible across Sinter worker processes.
    """

    def __init__(
        self,
        *,
        chains: int = 64,
        sweeps: int = 100,
        beta: float = 1.0,
        beta_start: Optional[float] = None,
        beta_stop: Optional[float] = None,
        rho: Optional[float] = None,
        rho_scale: Optional[float] = 1.0,
        gamma: float = 1.0,
        update: str = "gibbs",
        selection: str = "best_energy",
        capture: str = "sweep",
        max_legs: int = 1,
        target_valid: Optional[int] = None,
        initial_probability: Optional[float] = None,
        seed: int = 0,
        merge_duplicates: bool = True,
        state_budget_bytes: int = _DEFAULT_STATE_BUDGET_BYTES,
        device: str = "cpu",
    ) -> None:
        self.chains = chains
        self.sweeps = sweeps
        self.beta = beta
        self.beta_start = beta_start
        self.beta_stop = beta_stop
        self.rho = rho
        self.rho_scale = rho_scale
        self.gamma = gamma
        self.update = update
        self.selection = selection
        self.capture = capture
        self.max_legs = max_legs
        self.target_valid = target_valid
        self.initial_probability = initial_probability
        self.seed = seed
        self.merge_duplicates = merge_duplicates
        self.state_budget_bytes = state_budget_bytes
        self.device = device

    def build_engine(self, dem: stim.DetectorErrorModel) -> ColoredBlockGibbsDecoder:
        """Convert ``dem`` to sparse matrices and construct the decode engine."""
        matrices = dem_to_matrices(dem, merge_duplicates=self.merge_duplicates)
        return ColoredBlockGibbsDecoder(
            matrices,
            chains=self.chains,
            sweeps=self.sweeps,
            beta=self.beta,
            beta_start=self.beta_start,
            beta_stop=self.beta_stop,
            rho=self.rho,
            rho_scale=self.rho_scale,
            gamma=self.gamma,
            update=self.update,
            selection=self.selection,
            capture=self.capture,
            max_legs=self.max_legs,
            target_valid=self.target_valid,
            initial_probability=self.initial_probability,
            seed=self.seed,
            state_budget_bytes=self.state_budget_bytes,
            device=self.device,
        )

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> CompiledPBitDecoder:
        engine = self.build_engine(dem)
        rng = np.random.default_rng(self.seed)
        return CompiledPBitDecoder(
            engine, dem.num_detectors, dem.num_observables, rng
        )
