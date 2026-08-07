"""Sizing the temperature range from the landscape itself.

Good `t_max`/`t_min` values are in the units of *your* energy function, so a
library default is meaningless: 25,000 is scorching for a problem scored in
fractions and freezing for one scored in thousands. Rather than make you guess,
this takes a short random walk, measures how big the uphill steps actually are,
and picks temperatures that accept a stated fraction of them.

The sample is small (tens of moves) and counted against your budget, so it costs
a rounding error on any real run.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import TypeVar

S = TypeVar("S")

__all__ = ["calibrate"]


def calibrate(
    initial: S,
    *,
    energy: Callable[[S], float],
    move: Callable[[S, random.Random], object],
    rng: random.Random,
    samples: int = 64,
    accept_high: float = 0.8,
    accept_low: float = 0.01,
    copy: Callable[[S], S] | None = None,
) -> tuple[float, float]:
    """Return ``(t_max, t_min)`` for this landscape.

    ``t_max`` accepts a typical uphill move with probability ``accept_high``, so
    the run starts free to roam. ``t_min`` accepts the *smallest* observed uphill
    move with probability ``accept_low``, so it finishes almost greedy but not
    quite frozen.

    The walk accepts every move, which is the point: it's measuring the shape of
    the neighbourhood, not searching it.
    """
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if not 0 < accept_low < accept_high < 1:
        raise ValueError("need 0 < accept_low < accept_high < 1")

    state = initial if copy is None else copy(initial)
    current = float(energy(state))
    uphill: list[float] = []

    for _ in range(samples):
        result = move(state, rng)
        if isinstance(result, tuple) and len(result) == 2:
            candidate, delta = result[0], result[1]
        else:
            candidate, delta = result, None
        if delta is None:
            candidate_energy = float(energy(candidate))
            delta = candidate_energy - current
        else:
            delta = float(delta)
            candidate_energy = current + delta
        if delta > 0:
            uphill.append(delta)
        state, current = candidate, candidate_energy

    if not uphill:
        # Every sampled move was flat or downhill. Either the landscape really is
        # that smooth or the sample was unlucky; either way there's nothing to
        # size against, so use a low temperature and let the search behave
        # greedily rather than inventing a scale.
        return 1.0, 0.01

    typical = sum(uphill) / len(uphill)
    smallest = min(uphill)
    t_max = -typical / math.log(accept_high)
    t_min = -smallest / math.log(accept_low)
    # Degenerate landscapes (one distinct uphill size) can put these level or
    # inverted; keep a real range so the schedule still cools.
    t_min = min(t_min, t_max / 2)
    return t_max, max(t_min, t_max * 1e-6)
