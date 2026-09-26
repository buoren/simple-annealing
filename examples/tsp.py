"""A travelling-salesman run — the canonical annealing example.

Shows the three things that distinguish this library in one place: a private
seeded generator, a wall-clock budget, and progress through a callback instead of
a print. Run it with `python examples/tsp.py`.
"""
from __future__ import annotations

import math
import random

from simple_annealing import anneal

CITIES = {
    "Amsterdam": (52.37, 4.90), "Berlin": (52.52, 13.40), "Copenhagen": (55.68, 12.57),
    "Dublin": (53.35, -6.26), "Edinburgh": (55.95, -3.19), "Frankfurt": (50.11, 8.68),
    "Geneva": (46.20, 6.14), "Hamburg": (53.55, 9.99), "Innsbruck": (47.27, 11.39),
    "Jyvaskyla": (62.24, 25.75), "Krakow": (50.06, 19.94), "Lisbon": (38.72, -9.14),
    "Madrid": (40.42, -3.70), "Naples": (40.85, 14.27), "Oslo": (59.91, 10.75),
    "Prague": (50.08, 14.44), "Rome": (41.90, 12.50), "Stockholm": (59.33, 18.07),
    "Tallinn": (59.44, 24.75), "Vienna": (48.21, 16.37),
}


def tour_length(route: tuple[str, ...]) -> float:
    """Closed loop, so the last city returns to the first."""
    total = 0.0
    for a, b in zip(route, route[1:] + route[:1], strict=True):
        (ax, ay), (bx, by) = CITIES[a], CITIES[b]
        total += math.hypot(ax - bx, ay - by)
    return total


def reverse_segment(route: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    """The 2-opt move: reverse a slice. Far better than swapping two cities —
    it untangles crossings, which is what makes a tour long."""
    i = rng.randrange(len(route))
    j = rng.randrange(len(route))
    if i > j:
        i, j = j, i
    return route[:i] + route[i:j + 1][::-1] + route[j + 1:]


def main() -> None:
    start = tuple(CITIES)
    milestones: list[str] = []

    result = anneal(
        start,
        energy=tour_length,
        move=reverse_segment,
        seed=20260807,        # reproducible: run it twice, get the same tour
        time_budget=2.0,      # and it will not take longer than this
        on_progress=lambda p: milestones.append(
            f"  {p.step:>7,} steps  T={p.temperature:8.3f}  best={p.best_energy:7.2f}"
        ),
        progress_every=20_000,
    )

    print(f"start:  {tour_length(start):.2f}")
    print("\n".join(milestones))
    print(f"best:   {result.energy:.2f}  ({result.improvement:.2f} shorter)")
    print(f"        {result.steps:,} steps in {result.elapsed:.2f}s, stopped by {result.stopped_by}")
    print("route:  " + " → ".join(result.state))


if __name__ == "__main__":
    main()
