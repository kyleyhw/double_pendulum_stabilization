r"""
Hyperparameter schedules driven by training progress :math:`p \in [0, 1]`.

A schedule is a callable :math:`f: [0, 1] \to \mathbb R` returning the
parameter value for a given fraction of training completed.
"""
from __future__ import annotations

from typing import Callable


def constant(value: float) -> Callable[[float], float]:
    """Schedule that always returns ``value``."""
    def _f(_p: float) -> float:
        return float(value)
    return _f


def linear(start: float, end: float) -> Callable[[float], float]:
    r"""Linear interpolation :math:`f(p) = (1-p)\,\text{start} + p\,\text{end}`."""
    def _f(p: float) -> float:
        p = max(0.0, min(1.0, float(p)))
        return float((1.0 - p) * start + p * end)
    return _f


def cosine(start: float, end: float) -> Callable[[float], float]:
    r"""
    Cosine annealing :math:`f(p) = \mathrm{end} + 0.5(\mathrm{start} - \mathrm{end})(1 + \cos(\pi p))`.
    Decays smoothly with zero derivative at :math:`p = 0, 1`.
    """
    import math
    def _f(p: float) -> float:
        p = max(0.0, min(1.0, float(p)))
        return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * p)))
    return _f
