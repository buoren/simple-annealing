"""Cooling schedules, and sizing them from the landscape."""
import random

import pytest
from conftest import energy, swap

from simple_annealing import acceptance_probability, anneal, calibrate, constant, geometric, linear


# --- schedules ----------------------------------------------------------------
def test_geometric_spans_the_range_and_decays_by_a_constant_factor():
    schedule = geometric(100.0, 1.0)
    assert schedule(0.0) == pytest.approx(100.0)
    assert schedule(1.0) == pytest.approx(1.0)
    # Equal progress steps multiply the temperature by the same factor.
    assert schedule(0.5) / schedule(0.0) == pytest.approx(schedule(1.0) / schedule(0.5))


def test_linear_reaches_zero():
    schedule = linear(10.0, 0.0)
    assert schedule(0.5) == pytest.approx(5.0)
    assert schedule(1.0) == 0.0


def test_progress_outside_the_unit_interval_is_clamped():
    schedule = geometric(100.0, 1.0)
    assert schedule(-3.0) == pytest.approx(100.0)
    assert schedule(7.0) == pytest.approx(1.0)


def test_constant_never_cools():
    schedule = constant(4.0)
    assert {schedule(p) for p in (0.0, 0.5, 1.0)} == {4.0}


@pytest.mark.parametrize("factory", [geometric, linear])
def test_an_inverted_range_is_refused(factory):
    with pytest.raises(ValueError, match="t_min"):
        factory(1.0, 10.0)


def test_geometric_refuses_a_zero_floor():
    """It can only approach zero asymptotically; asking for it is a mistake worth
    naming rather than a silent division."""
    with pytest.raises(ValueError, match="positive"):
        geometric(10.0, 0.0)


# --- the acceptance rule ------------------------------------------------------
def test_downhill_moves_are_always_accepted():
    assert acceptance_probability(-5.0, 0.001) == 1.0
    assert acceptance_probability(0.0, 0.001) == 1.0


def test_uphill_acceptance_falls_with_temperature():
    warm = acceptance_probability(1.0, 10.0)
    cool = acceptance_probability(1.0, 0.1)
    assert 0 < cool < warm < 1


def test_at_zero_temperature_nothing_uphill_is_accepted():
    assert acceptance_probability(0.001, 0.0) == 0.0


# --- calibration --------------------------------------------------------------
def test_calibration_scales_with_the_energy_units(numbers):
    """The reason calibration exists: a fixed default temperature is meaningless
    when one problem is scored in units and another in thousands."""
    small = calibrate(numbers, energy=energy, move=swap, rng=random.Random(1), samples=200)
    big = calibrate(
        numbers,
        energy=lambda s: energy(s) * 1000,
        move=swap,
        rng=random.Random(1),
        samples=200,
    )
    assert big[0] == pytest.approx(small[0] * 1000, rel=1e-6)


def test_calibration_returns_a_usable_descending_range(numbers):
    t_max, t_min = calibrate(numbers, energy=energy, move=swap, rng=random.Random(2))
    assert t_max > t_min > 0


def test_a_flat_landscape_does_not_invent_a_scale(numbers):
    t_max, t_min = calibrate(
        numbers, energy=lambda _s: 1.0, move=swap, rng=random.Random(3), samples=20
    )
    assert (t_max, t_min) == (1.0, 0.01)


def test_calibration_leaves_an_in_place_state_alone(numbers):
    """It walks the neighbourhood to measure it; with mutating moves that walk
    must happen on a copy, or the caller's starting state is quietly destroyed."""
    state = list(numbers)

    def in_place(s, rng):
        i, j = rng.randrange(len(s)), rng.randrange(len(s))
        s[i], s[j] = s[j], s[i]
        return s

    calibrate(state, energy=energy, move=in_place, rng=random.Random(4), samples=30, copy=list)
    assert state == list(numbers)


def test_calibration_rejects_impossible_acceptance_bounds(numbers):
    with pytest.raises(ValueError, match="accept_low"):
        calibrate(numbers, energy=energy, move=swap, rng=random.Random(5), accept_high=0.1, accept_low=0.5)


def test_the_default_schedule_is_calibrated_and_works(numbers):
    """No schedule given: the annealer sizes one itself and still finds the
    optimum, which is the whole promise of not making people guess."""
    result = anneal(numbers, energy=energy, move=swap, seed=11, max_steps=4000)
    assert result.energy == pytest.approx(9.0)
