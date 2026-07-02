"""Map a Stim ``DetectorErrorModel`` (DEM) to p-bit decoding matrices.

A circuit-level Stim DEM is a list of independent error *mechanisms*. Each
mechanism fires with some probability and, when it fires, flips a set of
detectors and a set of logical observables. Collecting one binary column per
mechanism gives two sparse GF(2) matrices

* ``check_matrix`` ``H`` of shape ``(num_detectors, num_mechanisms)`` -- the
  detector each mechanism flips, and
* ``observables_matrix`` ``A`` of shape ``(num_observables, num_mechanisms)`` --
  the logical observable each mechanism flips,

plus a per-mechanism prior probability vector. This is exactly the
``(H, A, p)`` triple the :class:`~pbit_qec.types.CSSCode` abstraction expects:
``H`` maps an error (here a *mechanism activation* vector) to a syndrome and
``A`` maps it to the logical/observable class. Decoding a circuit-level shot is
then "find a low-(weighted-)weight mechanism activation that reproduces the
observed detector syndrome", identical to the code-capacity p-bit problem but
with per-column priors.

This module is intentionally self-contained: it parses the DEM with ``stim``
alone (no ``beliefmatching``/``pymatching`` dependency) and builds ``scipy``
sparse matrices so it scales to the ~10^5-column DEMs produced by distance-12
circuits. Duplicate mechanisms (identical detector+observable support) are
merged and their probabilities XOR-combined, matching the standard
decomposition-free check-matrix construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from pbit_qec.types import CSSCode

if TYPE_CHECKING:  # pragma: no cover - typing only
    import stim

# Clamp priors away from {0, 1/2} so the log-likelihood ratio log((1-p)/p) that
# the p-bit decoder uses stays finite and positive (a prior >= 1/2 would make a
# mechanism "want" to fire, which never happens for real DEM noise channels).
_MIN_PRIOR = 1e-15
_MAX_PRIOR = 0.5 - 1e-9


@dataclass(frozen=True)
class DEMMatrices:
    """Sparse decoding matrices extracted from a Stim DEM.

    Attributes:
        check_matrix: GF(2) ``(num_detectors, num_mechanisms)`` CSC matrix ``H``.
        observables_matrix: GF(2) ``(num_observables, num_mechanisms)`` CSC matrix ``A``.
        priors: ``(num_mechanisms,)`` per-mechanism activation probabilities in (0, 1/2).
    """

    check_matrix: sparse.csc_matrix
    observables_matrix: sparse.csc_matrix
    priors: np.ndarray

    @property
    def num_detectors(self) -> int:
        return int(self.check_matrix.shape[0])

    @property
    def num_observables(self) -> int:
        return int(self.observables_matrix.shape[0])

    @property
    def num_mechanisms(self) -> int:
        return int(self.check_matrix.shape[1])


def _combine_independent(p_existing: float, p_new: float) -> float:
    """Probability that an odd number of two independent error events fires."""
    return p_existing * (1.0 - p_new) + p_new * (1.0 - p_existing)


def _iter_error_mechanisms(dem: "stim.DetectorErrorModel"):
    """Yield ``(detectors, observables, probability)`` for each error instruction.

    Loops/repeat blocks are expanded via :meth:`flattened`. Decomposition
    separators (``^``) are ignored: the whole instruction is treated as one
    undecomposed mechanism (the union of its detector/observable targets), which
    is the correct hyperedge column for a decomposition-free decoder.
    """
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        detectors: List[int] = []
        observables: List[int] = []
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.append(int(target.val))
            elif target.is_logical_observable_id():
                observables.append(int(target.val))
            # separator targets carry no id and are skipped
        yield detectors, observables, probability


def dem_to_matrices(
    dem: "stim.DetectorErrorModel",
    *,
    merge_duplicates: bool = True,
) -> DEMMatrices:
    """Build sparse ``(H, A, priors)`` decoding matrices from a Stim DEM.

    Args:
        dem: The detector error model. Hyperedge (undecomposed) DEMs are fully
            supported; build them with ``circuit.detector_error_model(
            decompose_errors=False)``.
        merge_duplicates: When True (default), mechanisms with identical detector
            and observable support are merged into one column whose probability is
            the XOR-combination of the merged probabilities. This matches the
            canonical check-matrix construction and shrinks the column count.

    Returns:
        A :class:`DEMMatrices` with CSC ``check_matrix``/``observables_matrix`` and
        a clamped per-mechanism ``priors`` vector.
    """
    num_detectors = int(dem.num_detectors)
    num_observables = int(dem.num_observables)

    # Column-major COO builders for H and A.
    h_rows: List[int] = []
    h_cols: List[int] = []
    a_rows: List[int] = []
    a_cols: List[int] = []
    priors: List[float] = []

    # Maps a (sorted detectors, sorted observables) support to its column index.
    support_to_col: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}

    for detectors, observables, probability in _iter_error_mechanisms(dem):
        det_key = tuple(sorted(set(detectors)))
        obs_key = tuple(sorted(set(observables)))

        if merge_duplicates:
            existing = support_to_col.get((det_key, obs_key))
            if existing is not None:
                priors[existing] = _combine_independent(priors[existing], probability)
                continue

        col = len(priors)
        if merge_duplicates:
            support_to_col[(det_key, obs_key)] = col
        priors.append(probability)
        for d in det_key:
            h_rows.append(d)
            h_cols.append(col)
        for o in obs_key:
            a_rows.append(o)
            a_cols.append(col)

    num_mechanisms = len(priors)
    prior_arr = np.clip(np.asarray(priors, dtype=np.float64), _MIN_PRIOR, _MAX_PRIOR)

    check_matrix = sparse.csc_matrix(
        (np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)),
        shape=(num_detectors, num_mechanisms),
    )
    observables_matrix = sparse.csc_matrix(
        (np.ones(len(a_rows), dtype=np.uint8), (a_rows, a_cols)),
        shape=(num_observables, num_mechanisms),
    )
    # COO->CSC may have summed coincident entries; reduce back to GF(2).
    check_matrix.data %= 2
    observables_matrix.data %= 2
    check_matrix.eliminate_zeros()
    observables_matrix.eliminate_zeros()

    return DEMMatrices(
        check_matrix=check_matrix,
        observables_matrix=observables_matrix,
        priors=prior_arr,
    )


def dem_to_csscode(
    dem: "stim.DetectorErrorModel",
    *,
    family: str = "dem",
    merge_duplicates: bool = True,
    max_dense_entries: int = 64_000_000,
) -> Tuple[CSSCode, np.ndarray]:
    """Build a dense :class:`CSSCode` plus a per-mechanism prior vector from a DEM.

    This is a convenience bridge for small DEMs (tests, tiny circuits) and for
    code that wants the established ``CSSCode`` interface. It densifies ``H`` and
    ``A``, so it guards against accidentally materializing a multi-gigabyte matrix
    for a large circuit-level DEM; use :func:`dem_to_matrices` plus the sparse
    circuit decoder for those.

    Args:
        dem: The detector error model.
        family: Value stored in :attr:`CSSCode.family`.
        merge_duplicates: Forwarded to :func:`dem_to_matrices`.
        max_dense_entries: Refuse to densify ``H`` if it would exceed this many
            entries (``num_detectors * num_mechanisms``).

    Returns:
        ``(code, priors)`` where ``code`` carries dense ``H``/``A`` and ``priors``
        is the ``(num_mechanisms,)`` activation-probability vector.
    """
    mats = dem_to_matrices(dem, merge_duplicates=merge_duplicates)
    dense_entries = mats.num_detectors * mats.num_mechanisms
    if dense_entries > max_dense_entries:
        raise ValueError(
            f"dense H would have {dense_entries} entries (> {max_dense_entries}); "
            "use dem_to_matrices() with the sparse circuit decoder for large DEMs"
        )

    h_dense = np.asarray(mats.check_matrix.todense(), dtype=np.uint8) % 2
    a_dense = np.asarray(mats.observables_matrix.todense(), dtype=np.uint8) % 2
    code = CSSCode(
        H=h_dense,
        A=a_dense,
        qubit_labels=tuple(f"m{j}" for j in range(mats.num_mechanisms)),
        check_labels=tuple(f"D{i}" for i in range(mats.num_detectors)),
        logical_labels=tuple(f"L{i}" for i in range(mats.num_observables)),
        family=family,
        size=mats.num_mechanisms,
        metadata={"source": "stim_dem", "merged_duplicates": merge_duplicates},
    )
    return code, mats.priors


def load_dem_from_circuit_file(
    path: str,
    *,
    decompose_errors: bool = False,
) -> "stim.DetectorErrorModel":
    """Load a ``.stim`` circuit file and return its (flattened) detector error model."""
    import stim

    circuit = stim.Circuit.from_file(path)
    return circuit.detector_error_model(
        decompose_errors=decompose_errors,
        flatten_loops=True,
    )
