"""p-bit/QEC coset decoding research prototype."""

from pbit_qec.circuit_decoder import (
    CircuitDecodeStats,
    ColoredBlockGibbsDecoder,
    greedy_coloring,
)
from pbit_qec.codes import make_bb144_code, make_surface_code, make_toric_code
from pbit_qec.decoders import (
    BPDecoder,
    BPOSDDecoder,
    ExactCosetDecoder,
    MWPMDecoder,
    NullspaceCosetSamplerDecoder,
    PBitAnnealedOptimizerDecoder,
    PBitCosetSamplerDecoder,
    PBitNullspaceCosetSamplerDecoder,
    PBitSyndromeDecoder,
)
from pbit_qec.dem import (
    DEMMatrices,
    dem_to_csscode,
    dem_to_matrices,
    load_dem_from_circuit_file,
)
from pbit_qec.types import CSSCode, Decoder, DecodeResult, TrialResult

# NOTE: ``pbit_qec.sinter_decoder`` (the ``sinter.Decoder`` integration) is *not*
# imported here because it requires the optional ``sinter``/``stim`` packages.
# Import it explicitly with ``from pbit_qec.sinter_decoder import PBitSinterDecoder``.

__all__ = [
    "CSSCode",
    "BPDecoder",
    "BPOSDDecoder",
    "CircuitDecodeStats",
    "ColoredBlockGibbsDecoder",
    "DEMMatrices",
    "DecodeResult",
    "Decoder",
    "ExactCosetDecoder",
    "MWPMDecoder",
    "NullspaceCosetSamplerDecoder",
    "PBitAnnealedOptimizerDecoder",
    "PBitCosetSamplerDecoder",
    "PBitNullspaceCosetSamplerDecoder",
    "PBitSyndromeDecoder",
    "TrialResult",
    "dem_to_csscode",
    "dem_to_matrices",
    "greedy_coloring",
    "load_dem_from_circuit_file",
    "make_surface_code",
    "make_bb144_code",
    "make_toric_code",
]
