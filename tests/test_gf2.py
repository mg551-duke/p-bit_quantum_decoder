import numpy as np

from pbit_qec import gf2


def test_rank_nullspace_and_affine_solution() -> None:
    h = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    b = np.array([1, 0], dtype=np.uint8)

    assert gf2.rank(h) == 2
    x0, basis = gf2.solve_affine(h, b)

    assert x0 is not None
    assert np.array_equal(h @ x0 % 2, b)
    assert basis.shape == (1, 3)
    assert np.array_equal(h @ basis[0] % 2, np.zeros(2, dtype=np.uint8))


def test_pack_unpack_round_trip() -> None:
    bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    value = gf2.pack_bits(bits)
    assert value == 13
    assert np.array_equal(gf2.unpack_bits(value, len(bits)), bits)


def test_bounded_affine_enumeration() -> None:
    x0 = np.array([1, 0, 0], dtype=np.uint8)
    basis = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    states = list(gf2.enumerate_affine(x0, basis))
    assert len(states) == 4
    assert len({tuple(int(x) for x in state) for state in states}) == 4
