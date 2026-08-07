"""Running in a worker thread — the case that motivated the library.

A web framework serves sync handlers from a threadpool, so an optimiser that
installs signal handlers, or that seeds the process-global RNG for determinism,
either crashes or produces results that depend on what else happened to be
running. Both properties are asserted here rather than assumed.
"""
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import energy, swap

from simannealing import anneal


def solve(seed, numbers):
    return anneal(numbers, energy=energy, move=swap, seed=seed, max_steps=2000)


def test_it_runs_off_the_main_thread(numbers):
    """`signal.signal` raises ValueError anywhere but the main thread, which is
    enough to make an optimiser unusable in a request handler."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(solve, 1, numbers).result()
    assert result.energy == pytest.approx(9.0)


def test_concurrent_runs_do_not_interfere(numbers):
    """Eight runs at once, each with its own seed. With a shared global generator
    they would interleave draws and none would be reproducible; with private ones
    each matches what it produces alone."""
    seeds = list(range(8))
    alone = {seed: solve(seed, numbers).state for seed in seeds}
    with ThreadPoolExecutor(max_workers=8) as pool:
        together = dict(zip(seeds, pool.map(lambda s: solve(s, numbers).state, seeds), strict=True))
    assert together == alone


def test_the_same_seed_is_reproducible_across_threads(numbers):
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = [f.result().state for f in [pool.submit(solve, 42, numbers) for _ in range(4)]]
    assert len(set(states)) == 1
