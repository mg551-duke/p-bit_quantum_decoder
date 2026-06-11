"""One-sector code constructors for topological and bivariate-bicycle codes."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

import pbit_qec.gf2 as gf2
from pbit_qec.types import CSSCode


def make_toric_code(L: int) -> CSSCode:
    """Build one sector of an L x L toric code on square-lattice edges.

    Checks are vertices, qubits are horizontal and vertical edges, and the two logical rows
    record representative winding parities.
    """

    if L < 2:
        raise ValueError("toric code requires L >= 2")
    qubits: List[Tuple[str, int, int]] = []
    index: Dict[Tuple[str, int, int], int] = {}
    for y in range(L):
        for x in range(L):
            index[("h", x, y)] = len(qubits)
            qubits.append(("h", x, y))
    for y in range(L):
        for x in range(L):
            index[("v", x, y)] = len(qubits)
            qubits.append(("v", x, y))

    H = np.zeros((L * L, 2 * L * L), dtype=np.uint8)
    check_labels: List[str] = []
    row = 0
    for y in range(L):
        for x in range(L):
            check_labels.append(f"vertex({x},{y})")
            incident = [
                ("h", x, y),
                ("h", (x - 1) % L, y),
                ("v", x, y),
                ("v", x, (y - 1) % L),
            ]
            for edge in incident:
                H[row, index[edge]] ^= 1
            row += 1

    A = np.zeros((2, 2 * L * L), dtype=np.uint8)
    for y in range(L):
        A[0, index[("h", 0, y)]] = 1
    for x in range(L):
        A[1, index[("v", x, 0)]] = 1

    return CSSCode(
        H=H,
        A=A,
        qubit_labels=tuple(f"{kind}({x},{y})" for kind, x, y in qubits),
        check_labels=tuple(check_labels),
        logical_labels=("wind_x", "wind_y"),
        family="toric",
        size=L,
        metadata={"periodic": True},
    )


def make_surface_code(d: int) -> CSSCode:
    """Build one sector of a planar surface-like square-lattice edge code.

    Vertices at x=0 and x=d are rough boundaries for this sector and are omitted from H.
    The logical row is a vertical dual cut using all horizontal edges crossing x=0.
    """

    if d < 3 or d % 2 == 0:
        raise ValueError("surface code requires odd d >= 3")
    qubits: List[Tuple[str, int, int]] = []
    index: Dict[Tuple[str, int, int], int] = {}
    for y in range(d + 1):
        for x in range(d):
            index[("h", x, y)] = len(qubits)
            qubits.append(("h", x, y))
    for y in range(d):
        for x in range(d + 1):
            index[("v", x, y)] = len(qubits)
            qubits.append(("v", x, y))

    check_positions = [(x, y) for y in range(d + 1) for x in range(1, d)]
    H = np.zeros((len(check_positions), len(qubits)), dtype=np.uint8)
    check_labels: List[str] = []
    for row, (x, y) in enumerate(check_positions):
        check_labels.append(f"vertex({x},{y})")
        if x > 0:
            H[row, index[("h", x - 1, y)]] ^= 1
        if x < d:
            H[row, index[("h", x, y)]] ^= 1
        if y > 0:
            H[row, index[("v", x, y - 1)]] ^= 1
        if y < d:
            H[row, index[("v", x, y)]] ^= 1

    A = np.zeros((1, len(qubits)), dtype=np.uint8)
    for y in range(d + 1):
        A[0, index[("h", 0, y)]] = 1

    return CSSCode(
        H=H,
        A=A,
        qubit_labels=tuple(f"{kind}({x},{y})" for kind, x, y in qubits),
        check_labels=tuple(check_labels),
        logical_labels=("rough_to_rough",),
        family="surface",
        size=d,
        metadata={"periodic": False, "rough_boundaries": "left-right"},
    )


def make_bb144_code() -> CSSCode:
    """Build the X-error sector of the BB ``[[144, 12, 12]]`` gross code.

    The bivariate-bicycle parameters are ``l = 12``, ``m = 6``,
    ``A = x^3 + y + y^2``, and ``B = y^3 + x + x^2``. For X errors, the
    syndrome matrix is ``H_Z = [B.T | A.T]`` and logical action rows are
    representatives of ``null(H_X) / row(H_Z)``.
    """

    ell, m = 12, 6
    a_poly = ((3, 0), (0, 1), (0, 2))
    b_poly = ((0, 3), (1, 0), (2, 0))
    a_matrix = _bb_polynomial_matrix(ell, m, a_poly)
    b_matrix = _bb_polynomial_matrix(ell, m, b_poly)
    h_x = np.concatenate([a_matrix, b_matrix], axis=1).astype(np.uint8)
    h_z = np.concatenate([b_matrix.T, a_matrix.T], axis=1).astype(np.uint8)
    logical_action = _css_logical_action_rows(syndrome_matrix=h_z, stabilizer_matrix=h_x)

    block = [(side, x, y) for side in ("L", "R") for y in range(m) for x in range(ell)]
    return CSSCode(
        H=h_z,
        A=logical_action,
        qubit_labels=tuple(f"{side}({x},{y})" for side, x, y in block),
        check_labels=tuple(f"z_check({x},{y})" for y in range(m) for x in range(ell)),
        logical_labels=tuple(f"logical_{i}" for i in range(logical_action.shape[0])),
        family="bb",
        size=144,
        metadata={
            "name": "gross",
            "parameters": "[[144,12,12]]",
            "l": ell,
            "m": m,
            "a_terms": a_poly,
            "b_terms": b_poly,
            "sector": "x_errors",
            "h_x": h_x,
            "h_z": h_z,
        },
    )


def _bb_polynomial_matrix(ell: int, m: int, terms: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    matrix = np.zeros((ell * m, ell * m), dtype=np.uint8)
    for x_shift, y_shift in terms:
        matrix ^= _bb_shift_matrix(ell, m, x_shift, y_shift)
    return matrix


def _bb_shift_matrix(ell: int, m: int, x_shift: int, y_shift: int) -> np.ndarray:
    matrix = np.zeros((ell * m, ell * m), dtype=np.uint8)
    for y in range(m):
        for x in range(ell):
            source = y * ell + x
            target = ((y + y_shift) % m) * ell + ((x + x_shift) % ell)
            matrix[target, source] = 1
    return matrix


def _css_logical_action_rows(
    syndrome_matrix: np.ndarray,
    stabilizer_matrix: np.ndarray,
) -> np.ndarray:
    if np.any((syndrome_matrix @ stabilizer_matrix.T) % 2):
        raise ValueError("CSS check matrices do not commute")
    target_rank = (
        syndrome_matrix.shape[1] - gf2.rank(syndrome_matrix) - gf2.rank(stabilizer_matrix)
    )
    current = gf2.rref(syndrome_matrix)[0]
    current = current[current.any(axis=1)]
    current_rank = gf2.rank(current) if current.size else 0
    logicals = []
    for candidate in gf2.nullspace(stabilizer_matrix):
        stacked = (
            np.vstack([current, candidate])
            if current.size
            else candidate.reshape(1, candidate.size)
        )
        new_rank = gf2.rank(stacked)
        if new_rank > current_rank:
            logicals.append(candidate.copy())
            current = stacked
            current_rank = new_rank
        if len(logicals) == target_rank:
            break
    if len(logicals) != target_rank:
        raise ValueError(f"expected {target_rank} logical rows, found {len(logicals)}")
    return np.vstack(logicals).astype(np.uint8)


def make_code(family: str, size: int) -> CSSCode:
    if family == "toric":
        return make_toric_code(size)
    if family == "surface":
        return make_surface_code(size)
    if family in {"bb", "bb144", "gross"}:
        if size != 144:
            raise ValueError("only the BB [[144,12,12]] code is implemented")
        return make_bb144_code()
    raise ValueError(f"unknown code family {family!r}")
