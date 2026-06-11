import numpy as np

from pbit_qec.codes import make_surface_code, make_toric_code


def test_toric_code_shape_and_labels() -> None:
    code = make_toric_code(3)
    assert code.H.shape == (9, 18)
    assert code.A.shape == (2, 18)
    assert len(code.qubit_labels) == 18
    assert all(weight == 4 for weight in code.H.sum(axis=1))


def test_surface_code_shape_and_valid_syndrome() -> None:
    code = make_surface_code(3)
    assert code.H.shape == (8, 24)
    assert code.A.shape == (1, 24)
    error = np.zeros(code.n, dtype=np.uint8)
    error[0] = 1
    syndrome = code.syndrome(error)
    assert syndrome.sum() >= 1
    assert code.is_valid_correction(error, syndrome)


def test_logical_success_uses_coset_not_physical_equality() -> None:
    code = make_surface_code(3)
    error = np.zeros(code.n, dtype=np.uint8)
    correction = np.zeros(code.n, dtype=np.uint8)
    assert not np.array_equal(error, correction ^ np.ones(code.n, dtype=np.uint8))
    assert not code.logical(error ^ correction).any()
