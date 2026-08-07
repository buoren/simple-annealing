"""Cooling schedules.

A schedule is just ``Callable[[float], float]`` — it takes progress through the
run in ``[0, 1]`` and returns a temperature. Nothing here is special-cased by the
annealer, so any function of that shape works, including a lambda or a closure
over your own state.

Progress is deliberately the input rather than the step number: a run bounded by
a wall-clock deadline doesn't know its step count in advance, and a schedule
written against progress behaves identically under either kind of budget.
"""
from __future__ import annotations

import math
from collections.abc import Callable

Schedule = Callable[[float], float]

__all__ = ["Schedule", "constant", "geometric", "linear"]


def geometric(t_max: float, t_min: float) -> Schedule:
    """Exponential decay from ``t_max`` to ``t_min`` — the usual choice.

    Temperature falls by a constant *factor* per unit of progress, so the run
    spends proportionally similar effort at each order of magnitude. That suits
    energy landscapes whose interesting structure spans several scales, which is
    most of them.
    """
    if t_max <= 0 or t_min <= 0:
        raise ValueError("geometric() needs positive temperatures; use linear() to reach 0")
    if t_min > t_max:
        raise ValueError(f"t_min ({t_min}) must not exceed t_max ({t_max})")
    ratio = t_min / t_max

    def schedule(progress: float) -> float:
        return float(t_max * (ratio ** _clamp(progress)))

    return schedule


def linear(t_max: float, t_min: float = 0.0) -> Schedule:
    """Straight-line decay. Reaches ``t_min`` exactly, including zero — at which
    point the search accepts only improvements and becomes plain hill climbing."""
    if t_max < 0 or t_min < 0:
        raise ValueError("temperatures must not be negative")
    if t_min > t_max:
        raise ValueError(f"t_min ({t_min}) must not exceed t_max ({t_max})")

    def schedule(progress: float) -> float:
        return t_max + (t_min - t_max) * _clamp(progress)

    return schedule


def constant(temperature: float) -> Schedule:
    """A fixed temperature — the Metropolis algorithm without annealing. Useful
    for sampling, and for isolating whether a disappointing result is the
    schedule's fault or the move function's."""
    if temperature < 0:
        raise ValueError("temperature must not be negative")

    def schedule(_progress: float) -> float:
        return temperature

    return schedule


def acceptance_probability(delta: float, temperature: float) -> float:
    """The Metropolis criterion: certainty for a downhill move, ``exp(-ΔE/T)``
    for an uphill one, and nothing at all once the temperature reaches zero.

    Exposed because it's the thing people most often want to reason about when a
    run behaves unexpectedly — and because calibration is defined in terms of it.
    """
    if delta <= 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta / temperature)


def _clamp(progress: float) -> float:
    return 0.0 if progress < 0 else 1.0 if progress > 1 else progress
