"""The core loop: does it search, does it stop when told, does it stay honest."""
import random
from dataclasses import FrozenInstanceError

import pytest
from conftest import energy, swap

from simannealing import Result, anneal, constant, linear


def run(numbers, **kwargs):
    kwargs.setdefault("max_steps", 4000)
    kwargs.setdefault("seed", 7)
    return anneal(numbers, energy=energy, move=swap, **kwargs)


def test_it_actually_optimises(numbers):
    result = run(numbers)
    assert result.energy < energy(numbers)
    assert result.improvement > 0
    # Ten values 0..9: sorted order costs exactly 9.
    assert result.energy == pytest.approx(9.0)


def test_the_result_is_the_best_seen_not_the_last(numbers):
    """Annealing accepts uphill moves, so the final state is routinely worse than
    the best one. Returning the last would silently throw away the answer."""
    result = run(numbers, schedule=constant(50.0))   # hot throughout: it wanders
    assert energy(result.state) == pytest.approx(result.energy)
    assert result.energy <= result.initial_energy


def test_energy_is_recomputed_from_the_returned_state(numbers):
    result = run(numbers)
    assert result.energy == pytest.approx(energy(result.state))


# --- determinism --------------------------------------------------------------
def test_same_seed_same_answer(numbers):
    first, second = run(numbers), run(numbers)
    assert first.state == second.state
    assert (first.energy, first.steps, first.accepted) == (second.energy, second.steps, second.accepted)


def test_different_seeds_explore_differently(numbers):
    a = run(numbers, seed=1, max_steps=60, schedule=constant(50.0))
    b = run(numbers, seed=2, max_steps=60, schedule=constant(50.0))
    assert a.state != b.state


def test_it_never_touches_the_global_random(numbers):
    """The whole reason for a private generator: a run must not perturb — or
    depend on — anyone else's random stream."""
    random.seed(99)
    before = [random.random() for _ in range(3)]
    random.seed(99)
    run(numbers)
    after = [random.random() for _ in range(3)]
    assert before == after


def test_an_injected_generator_is_used(numbers):
    result = anneal(numbers, energy=energy, move=swap, rng=random.Random(3), max_steps=500)
    again = anneal(numbers, energy=energy, move=swap, rng=random.Random(3), max_steps=500)
    assert result.state == again.state


def test_seed_and_rng_together_are_refused(numbers):
    with pytest.raises(ValueError, match="not both"):
        anneal(numbers, energy=energy, move=swap, rng=random.Random(1), seed=1, max_steps=10)


# --- stopping conditions ------------------------------------------------------
def test_a_run_with_no_budget_is_refused(numbers):
    with pytest.raises(ValueError, match="max_steps or time_budget"):
        anneal(numbers, energy=energy, move=swap)


def test_max_steps_is_exact(numbers):
    assert run(numbers, max_steps=250).steps == 250


def test_zero_steps_returns_the_initial_state(numbers):
    result = run(numbers, max_steps=0)
    assert result.state == numbers and result.steps == 0
    assert result.energy == pytest.approx(energy(numbers))


def test_a_deadline_stops_the_run(numbers):
    """A fake clock jumps past the budget, so this asserts the deadline logic
    rather than the speed of the machine running the tests."""
    ticks = iter([0.0, 0.0, 0.1, 0.2, 5.1] + [9.0] * 50)
    result = anneal(
        numbers, energy=energy, move=swap, seed=1,
        time_budget=5.0, schedule=linear(10, 0), clock=lambda: next(ticks),
    )
    assert result.stopped_by == "time_budget"
    assert result.steps < 5


def test_target_energy_ends_the_run_early(numbers):
    result = run(numbers, max_steps=100_000, target_energy=9.0)
    assert result.stopped_by == "target_energy"
    assert result.energy <= 9.0
    assert result.steps < 100_000


def test_stall_gives_up_when_nothing_improves(numbers):
    result = run(numbers, max_steps=100_000, stall_steps=200)
    assert result.stopped_by == "stall"


def test_steps_win_when_both_budgets_are_generous(numbers):
    assert run(numbers, max_steps=100, time_budget=60).stopped_by == "max_steps"


# --- deltas -------------------------------------------------------------------
def test_a_move_may_report_its_own_delta(numbers):
    """The optimisation that makes annealing viable on expensive objectives:
    skip the full energy call when the move already knows what it cost."""
    calls = []

    def counting_energy(state):
        calls.append(state)
        return energy(state)

    def move_with_delta(state, rng):
        nxt = swap(state, rng)
        return nxt, energy(nxt) - energy(state)

    result = anneal(
        numbers, energy=counting_energy, move=move_with_delta, seed=5,
        max_steps=300, schedule=linear(10, 0),
    )
    # One call to seed the run, one to verify the winner — not one per step.
    assert len(calls) == 2
    assert result.energy == pytest.approx(energy(result.state))


def test_a_none_delta_falls_back_to_the_energy_function(numbers):
    def sometimes(state, rng):
        nxt = swap(state, rng)
        return (nxt, None) if rng.random() < 0.5 else nxt

    result = anneal(numbers, energy=energy, move=sometimes, seed=2, max_steps=800)
    assert result.energy == pytest.approx(energy(result.state))


def test_a_lying_delta_is_reported_rather_than_believed(numbers):
    """Wrong deltas are the classic silent failure of delta-based annealing: the
    run optimises a number that has drifted away from the real objective. The
    tracked value gets replaced by the truth, and the gap is surfaced."""
    def lying(state, rng):
        return swap(state, rng), -1.0      # claims every move is an improvement

    result = anneal(
        numbers, energy=energy, move=lying, seed=3, max_steps=200, schedule=linear(1, 0),
    )
    assert result.delta_drift > 0
    assert result.energy == pytest.approx(energy(result.state))


def test_honest_deltas_show_no_drift(numbers):
    def honest(state, rng):
        nxt = swap(state, rng)
        return nxt, energy(nxt) - energy(state)

    assert anneal(
        numbers, energy=energy, move=honest, seed=4, max_steps=400,
    ).delta_drift == pytest.approx(0.0, abs=1e-9)


# --- in-place moves -----------------------------------------------------------
def test_copy_snapshots_the_best_and_only_on_improvement(numbers):
    """Mutating states are supported, and the copy happens O(improvements) rather
    than O(steps) — the difference between a cheap run and a copy-bound one."""
    copies = []

    def in_place(state, rng):
        i, j = rng.randrange(len(state)), rng.randrange(len(state))
        state[i], state[j] = state[j], state[i]
        return state

    def snapshot(state):
        copies.append(1)
        return list(state)

    result = anneal(
        list(numbers), energy=energy, move=in_place, copy=snapshot,
        seed=6, max_steps=1500, schedule=linear(20, 0),   # explicit: no calibration copy
    )
    assert result.energy == pytest.approx(energy(result.state))
    assert len(copies) == result.improved + 1          # the initial snapshot, then each best
    assert len(copies) < result.steps // 10            # nowhere near one per step


def test_calibration_copies_the_state_it_walks(numbers):
    """The default (calibrated) schedule takes one protective copy before its
    sampling walk, so an in-place move can't destroy the caller's start state."""
    copies = []

    def in_place(state, rng):
        i, j = rng.randrange(len(state)), rng.randrange(len(state))
        state[i], state[j] = state[j], state[i]
        return state

    start = list(numbers)
    anneal(
        start, energy=energy, move=in_place,
        copy=lambda s: (copies.append(1), list(s))[1],
        seed=6, max_steps=200,
    )
    assert len(copies) >= 2      # one for calibration's walk, one for the first best


# --- shapes the caller may bring ----------------------------------------------
def test_a_problem_class_works_through_bound_methods(numbers):
    """The class-based style needs no support from the library: bound methods ARE
    functions, so you keep the shared data and the "implement two methods" shape
    without inheriting a state-ownership contract. Taking functions is therefore
    the more general choice, not the more austere one."""
    class OrderingProblem:
        def __init__(self, weight):
            self.weight = weight          # shared by both methods, as on a base class
            self.evaluations = 0

        def energy(self, state):
            self.evaluations += 1
            return self.weight * energy(state)

        def move(self, state, rng):
            return swap(state, rng)

    problem = OrderingProblem(weight=2.0)
    result = anneal(numbers, energy=problem.energy, move=problem.move, seed=8, max_steps=3000)
    assert result.energy == pytest.approx(2.0 * 9.0)
    assert problem.evaluations > 0


def test_closures_carry_context_without_a_class(numbers):
    """The other half of the same point: a closure holds shared data too."""
    def weighted(weight):
        return lambda state: weight * energy(state)

    result = anneal(numbers, energy=weighted(3.0), move=swap, seed=9, max_steps=3000)
    assert result.energy == pytest.approx(3.0 * 9.0)


# --- reporting ----------------------------------------------------------------
def test_progress_is_reported_on_the_requested_cadence(numbers):
    seen = []
    run(numbers, max_steps=1000, progress_every=100, on_progress=seen.append)
    assert len(seen) == 10
    assert [p.step for p in seen] == [100 * (i + 1) for i in range(10)]
    assert all(p.best_energy <= p.energy or p.best_energy <= seen[0].energy for p in seen)
    assert seen[-1].fraction == pytest.approx(1.0, abs=0.01)


def test_nothing_is_written_to_stdout_or_stderr(numbers, capsys):
    """A library that prints is a library you can't put in a request handler."""
    run(numbers)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_no_signal_handler_is_installed(numbers):
    import signal

    before = signal.getsignal(signal.SIGINT)
    run(numbers)
    assert signal.getsignal(signal.SIGINT) is before


def test_the_result_is_immutable(numbers):
    result = run(numbers)
    assert isinstance(result, Result)
    with pytest.raises(FrozenInstanceError):
        result.energy = 0.0
