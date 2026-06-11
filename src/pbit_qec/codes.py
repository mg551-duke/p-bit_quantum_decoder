"""Toric and surface one-sector code constructors."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

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


def make_code(family: str, size: int) -> CSSCode:
    if family == "toric":
        return make_toric_code(size)
    if family == "surface":
        return make_surface_code(size)
    raise ValueError(f"unknown code family {family!r}")
