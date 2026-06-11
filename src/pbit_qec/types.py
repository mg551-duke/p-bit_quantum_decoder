"""Public data structures for the research prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

BitArray = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class CSSCode:
    """One Pauli-sector binary CSS code model.

    H maps an error vector to a syndrome, and A maps an error vector to a logical/coset
    label. This package uses one sector at a time, so errors are binary vectors.
    """

    H: BitArray
    A: BitArray
    qubit_labels: Tuple[str, ...]
    check_labels: Tuple[str, ...]
    logical_labels: Tuple[str, ...]
    family: str
    size: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "H", np.asarray(self.H, dtype=np.uint8) % 2)
        object.__setattr__(self, "A", np.asarray(self.A, dtype=np.uint8) % 2)
        if self.H.ndim != 2:
            raise ValueError("H must be a matrix")
        if self.A.ndim != 2:
            raise ValueError("A must be a matrix")
        if self.H.shape[1] != self.A.shape[1]:
            raise ValueError("H and A must have the same number of columns")
        if len(self.qubit_labels) != self.n:
            raise ValueError("qubit label count does not match H columns")
        if len(self.check_labels) != self.m:
            raise ValueError("check label count does not match H rows")
        if len(self.logical_labels) != self.k:
            raise ValueError("logical label count does not match A rows")

    @property
    def n(self) -> int:
        return int(self.H.shape[1])

    @property
    def m(self) -> int:
        return int(self.H.shape[0])

    @property
    def k(self) -> int:
        return int(self.A.shape[0])

    def syndrome(self, error: Sequence[int]) -> BitArray:
        e = np.asarray(error, dtype=np.uint8) % 2
        if e.shape != (self.n,):
            raise ValueError(f"expected error shape {(self.n,)}, got {e.shape}")
        return (self.H @ e % 2).astype(np.uint8)

    def logical(self, error: Sequence[int]) -> BitArray:
        e = np.asarray(error, dtype=np.uint8) % 2
        if e.shape != (self.n,):
            raise ValueError(f"expected error shape {(self.n,)}, got {e.shape}")
        return (self.A @ e % 2).astype(np.uint8)

    def is_valid_correction(self, correction: Sequence[int], syndrome: Sequence[int]) -> bool:
        target = np.asarray(syndrome, dtype=np.uint8) % 2
        return bool(np.array_equal(self.syndrome(correction), target))


@dataclass(frozen=True)
class DecodeResult:
    correction: BitArray
    logical_class: Tuple[int, ...]
    valid: bool
    energy: float
    runtime_s: float
    valid_samples: int = 0
    total_samples: int = 0
    first_valid_sweep: Optional[int] = None
    distinct_logicals: int = 0
    coset_counts: Dict[str, int] = field(default_factory=dict)
    coset_log_weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Decoder(Protocol):
    name: str

    def decode(
        self,
        code: CSSCode,
        syndrome: Sequence[int],
        p: Union[float, Sequence[float]],
        rng: np.random.Generator,
    ) -> DecodeResult:
        """Decode one syndrome with the given channel prior."""


@dataclass(frozen=True)
class TrialResult:
    code_family: str
    code_size: int
    p: float
    trial: int
    seed: int
    decoder: str
    syndrome_id: int
    true_logical: str
    decoded_logical: str
    valid: bool
    logical_success: bool
    runtime_s: float
    valid_sample_rate: float
    valid_samples: int
    total_samples: int
    first_valid_sweep: Optional[int]
    distinct_logicals: int
    coset_entropy: float
    exact_best_logical: Optional[str] = None
    exact_agreement: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
