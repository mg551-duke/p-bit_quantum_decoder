"""p-bit/QEC coset decoding research prototype."""

from pbit_qec.codes import make_surface_code, make_toric_code
from pbit_qec.decoders import (
    ExactCosetDecoder,
    NullspaceCosetSamplerDecoder,
    PBitAnnealedOptimizerDecoder,
    PBitCosetSamplerDecoder,
)
from pbit_qec.types import CSSCode, Decoder, DecodeResult, TrialResult

__all__ = [
    "CSSCode",
    "DecodeResult",
    "Decoder",
    "ExactCosetDecoder",
    "NullspaceCosetSamplerDecoder",
    "PBitAnnealedOptimizerDecoder",
    "PBitCosetSamplerDecoder",
    "TrialResult",
    "make_surface_code",
    "make_toric_code",
]
