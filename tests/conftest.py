"""A tiny shared problem: order N numbers, energy = total adjacent difference.

Small, has a known optimum (the sorted order), and its moves are trivial — so a
failing test points at the annealer rather than at the fixture.
"""
import pytest


@pytest.fixture
def numbers():
    return (8, 3, 5, 1, 9, 2, 7, 4, 6, 0)


def energy(state):
    """Total absolute gap between neighbours. Minimised by sorting."""
    return float(sum(abs(b - a) for a, b in zip(state, state[1:], strict=False)))


def swap(state, rng):
    """Swap two positions, returning a NEW tuple — no copy hook needed."""
    i, j = rng.randrange(len(state)), rng.randrange(len(state))
    items = list(state)
    items[i], items[j] = items[j], items[i]
    return tuple(items)
