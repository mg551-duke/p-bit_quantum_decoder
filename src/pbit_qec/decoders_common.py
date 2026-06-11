"""Shared energy helpers for decoders and exact analysis."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np


def prior_lambdas(p: Union[float, Sequence[float]], n: int) -> np.ndarray:
    probs = np.asarray(p, dtype=float)
    if probs.ndim == 0:
        probs = np.full(n, float(probs))
    if probs.shape != (n,):
        raise ValueError(f"expected scalar p or vector shape {(n,)}, got {probs.shape}")
    if np.any(probs <= 0.0) or np.any(probs >= 0.5):
        raise ValueError("p must be in the open interval (0, 0.5)")
    return np.log((1.0 - probs) / probs)


def default_rho(lambdas: np.ndarray) -> float:
    return float(max(8.0, 4.0 * np.max(lambdas)))


def prior_energy(error: Sequence[int], lambdas: np.ndarray) -> float:
    e = np.asarray(error, dtype=np.uint8)
    return float(np.dot(lambdas, e))
