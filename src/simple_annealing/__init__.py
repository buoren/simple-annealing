"""simple_annealing — simulated annealing that is safe to run inside a server.

    from simple_annealing import anneal

    result = anneal(
        route,
        energy=tour_length,
        move=swap_two_cities,
        seed=1234,
        max_steps=50_000,
    )
    result.state, result.energy, result.stopped_by

Deterministic when you ask for it, bounded by a deadline when you need one,
silent by default, and free of global state — so several runs can share a
process without interfering with each other or with you.
"""
from .calibrate import calibrate
from .core import Progress, Result, anneal
from .schedules import Schedule, acceptance_probability, constant, geometric, linear

__version__ = "0.1.0"

__all__ = [
    "Progress",
    "Result",
    "Schedule",
    "acceptance_probability",
    "anneal",
    "calibrate",
    "constant",
    "geometric",
    "linear",
]
