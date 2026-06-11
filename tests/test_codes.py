import numpy as np

import pbit_qec.gf2 as gf2
from pbit_qec.codes import make_bb144_code, make_surface_code, make_toric_code


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


def test_bb144_code_shape_commutation_and_logicals() -> None:
    code = make_bb144_code()
    h_x = code.metadata["h_x"]
    h_z = code.metadata["h_z"]

    assert code.H.shape == (72, 144)
    assert code.A.shape == (12, 144)
    assert code.n == 144
    assert code.k == 12
    assert gf2.rank(h_x) == 66
    assert gf2.rank(h_z) == 66
    assert not ((h_x @ h_z.T) % 2).any()
    assert not ((code.A @ h_x.T) % 2).any()
    assert gf2.rank(np.vstack([h_z, code.A])) == gf2.rank(h_z) + code.k
