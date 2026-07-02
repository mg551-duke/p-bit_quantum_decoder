"""Tests for the Stim DEM -> sparse decoding-matrix conversion."""

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from pbit_qec.dem import dem_to_matrices


def _dem(text: str) -> "stim.DetectorErrorModel":
    return stim.DetectorErrorModel(text)


def test_handcrafted_dem_builds_exact_matrices() -> None:
    mats = dem_to_matrices(
        _dem(
            """
            error(0.1) D0 D1 L0
            error(0.2) D1 D2
            """
        ),
        merge_duplicates=True,
    )
    assert mats.num_detectors == 3
    assert mats.num_observables == 1
    assert mats.num_mechanisms == 2

    h = np.asarray(mats.check_matrix.todense(), dtype=np.uint8)
    a = np.asarray(mats.observables_matrix.todense(), dtype=np.uint8)
    np.testing.assert_array_equal(h, [[1, 0], [1, 1], [0, 1]])
    np.testing.assert_array_equal(a, [[1, 0]])
    np.testing.assert_allclose(mats.priors, [0.1, 0.2])


def test_duplicate_mechanisms_are_xor_combined() -> None:
    mats = dem_to_matrices(
        _dem(
            """
            error(0.1) D0
            error(0.1) D0
            """
        ),
        merge_duplicates=True,
    )
    assert mats.num_mechanisms == 1
    # P(odd of two independent 0.1 events) = 0.1*0.9 + 0.1*0.9 = 0.18.
    np.testing.assert_allclose(mats.priors, [0.18])


def test_merge_duplicates_false_keeps_separate_columns() -> None:
    text = """
    error(0.1) D0
    error(0.1) D0
    """
    merged = dem_to_matrices(_dem(text), merge_duplicates=True)
    unmerged = dem_to_matrices(_dem(text), merge_duplicates=False)
    assert merged.num_mechanisms == 1
    assert unmerged.num_mechanisms == 2


def test_distinct_observable_support_not_merged() -> None:
    # Same detectors, different observable -> distinct columns.
    mats = dem_to_matrices(
        _dem(
            """
            error(0.1) D0 L0
            error(0.1) D0 L1
            """
        )
    )
    assert mats.num_mechanisms == 2


def test_matrices_are_binary_over_gf2() -> None:
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        before_round_data_depolarization=0.02,
        before_measure_flip_probability=0.02,
    )
    mats = dem_to_matrices(circuit.detector_error_model(decompose_errors=False))
    assert set(np.unique(mats.check_matrix.data)) <= {1}
    assert set(np.unique(mats.observables_matrix.data)) <= {1}
    assert mats.num_detectors == circuit.num_detectors
    assert mats.num_observables == circuit.num_observables
    assert np.all(mats.priors > 0.0) and np.all(mats.priors < 0.5)
