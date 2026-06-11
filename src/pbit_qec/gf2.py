"""Small GF(2) linear algebra helpers."""

from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

BitArray = npt.NDArray[np.uint8]


def as_binary_matrix(matrix: Union[Sequence[Sequence[int]], npt.NDArray[np.integer]]) -> BitArray:
    return np.asarray(matrix, dtype=np.uint8) % 2


def as_binary_vector(vector: Union[Sequence[int], npt.NDArray[np.integer]]) -> BitArray:
    return np.asarray(vector, dtype=np.uint8).reshape(-1) % 2


def rref(
    matrix: Union[Sequence[Sequence[int]], npt.NDArray[np.integer]],
) -> Tuple[BitArray, Tuple[int, ...]]:
    """Return row-reduced echelon form over GF(2) and pivot columns."""

    a = as_binary_matrix(matrix).copy()
    rows, cols = a.shape
    pivot_cols = []
    r = 0
    for c in range(cols):
        pivot_rel = np.flatnonzero(a[r:, c])
        if pivot_rel.size == 0:
            continue
        pivot = int(pivot_rel[0] + r)
        if pivot != r:
            a[[r, pivot]] = a[[pivot, r]]
        for rr in range(rows):
            if rr != r and a[rr, c]:
                a[rr] ^= a[r]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    return a, tuple(pivot_cols)


def rank(matrix: Union[Sequence[Sequence[int]], npt.NDArray[np.integer]]) -> int:
    return len(rref(matrix)[1])


def solve_affine(
    matrix: Union[Sequence[Sequence[int]], npt.NDArray[np.integer]],
    rhs: Union[Sequence[int], npt.NDArray[np.integer]],
) -> Tuple[Optional[BitArray], BitArray]:
    """Solve H x = rhs over GF(2), returning one solution and a nullspace basis.

    The nullspace basis has shape (dimension, n). If no solution exists, the solution is None
    and the nullspace is still returned for H.
    """

    h = as_binary_matrix(matrix)
    b = as_binary_vector(rhs)
    if h.shape[0] != b.shape[0]:
        raise ValueError("rhs length must match matrix row count")
    augmented = np.concatenate([h.copy(), b[:, None]], axis=1)
    reduced, pivots = rref(augmented)
    n = h.shape[1]
    for row in range(reduced.shape[0]):
        if not reduced[row, :n].any() and reduced[row, n]:
            return None, nullspace(h)
    pivots_h = tuple(c for c in pivots if c < n)
    free_cols = [c for c in range(n) if c not in pivots_h]
    solution = np.zeros(n, dtype=np.uint8)
    for row, col in enumerate(pivots_h):
        solution[col] = reduced[row, n]
    basis = []
    for free in free_cols:
        vec = np.zeros(n, dtype=np.uint8)
        vec[free] = 1
        for row, col in enumerate(pivots_h):
            vec[col] = reduced[row, free]
        basis.append(vec)
    null = np.vstack(basis).astype(np.uint8) if basis else np.zeros((0, n), dtype=np.uint8)
    return solution, null


def nullspace(matrix: Union[Sequence[Sequence[int]], npt.NDArray[np.integer]]) -> BitArray:
    h = as_binary_matrix(matrix)
    zero = np.zeros(h.shape[0], dtype=np.uint8)
    _, basis = solve_affine(h, zero)
    return basis


def enumerate_affine(
    solution: Union[Sequence[int], npt.NDArray[np.integer]],
    basis: Union[Sequence[Sequence[int]], npt.NDArray[np.integer]],
    max_dim: int = 24,
) -> Iterator[BitArray]:
    """Yield all vectors in solution + span(basis) if basis dimension is bounded."""

    x0 = as_binary_vector(solution)
    b = as_binary_matrix(basis)
    if b.ndim != 2:
        raise ValueError("basis must be a matrix")
    dim = b.shape[0]
    if dim > max_dim:
        raise ValueError(f"nullity {dim} exceeds max_dim={max_dim}")
    for mask in range(1 << dim):
        x = x0.copy()
        bit = 0
        m = mask
        while m:
            if m & 1:
                x ^= b[bit]
            bit += 1
            m >>= 1
        yield x


def pack_bits(bits: Union[Sequence[int], npt.NDArray[np.integer]]) -> int:
    value = 0
    for i, bit in enumerate(as_binary_vector(bits)):
        value |= int(bit) << i
    return value


def unpack_bits(value: int, width: int) -> BitArray:
    if width < 0:
        raise ValueError("width must be non-negative")
    return np.array([(value >> i) & 1 for i in range(width)], dtype=np.uint8)


def bit_tuple(bits: Union[Sequence[int], npt.NDArray[np.integer]]) -> Tuple[int, ...]:
    return tuple(int(x) for x in as_binary_vector(bits))


def bit_string(bits: Union[Sequence[int], npt.NDArray[np.integer]]) -> str:
    arr = as_binary_vector(bits)
    if arr.size == 0:
        return "0"
    return "".join(str(int(x)) for x in arr)
