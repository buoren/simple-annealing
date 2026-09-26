"""The annealer.

One function, ``anneal()``. No base class to inherit, no global state, no signal
handlers, no output. See the README for why each of those is a deliberate choice
rather than an omission.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .calibrate import calibrate
from .schedules import Schedule, acceptance_probability, geometric

S = TypeVar("S")

# What a move function may return: the next state, or the next state paired with
# the energy change it caused (which lets the annealer skip a full energy call).
# A `None` delta means "I don't know" and is equivalent to returning the state
# alone — handy when only *some* moves can compute their delta cheaply.
Move = Callable[[S, random.Random], "S | tuple[S, float | None]"]

__all__ = ["Move", "Progress", "Result", "anneal"]


@dataclass(frozen=True)
class Progress:
    """Handed to ``on_progress``. Read-only: reporting must not steer the run."""

    step: int
    energy: float           # the current state's energy
    best_energy: float
    temperature: float
    accepted: int           # moves accepted so far, uphill ones included
    elapsed: float          # seconds since the run began
    fraction: float         # progress through the budget, 0..1


@dataclass(frozen=True)
class Result(Generic[S]):
    """What a finished run found, and how it got there."""

    state: S
    energy: float
    initial_energy: float
    steps: int
    accepted: int
    improved: int           # times a new best was found
    elapsed: float
    stopped_by: str         # max_steps | time_budget | stall | target_energy
    # |tracked energy - recomputed energy| for the returned state. Non-zero means
    # a move function's deltas disagree with its energy function — a bug that is
    # otherwise silent, and which quietly ruins a long run. See the README.
    delta_drift: float = 0.0

    @property
    def improvement(self) -> float:
        """How much energy the run removed. Negative would mean it got worse,
        which can't happen: the best state is tracked, never merely the last."""
        return self.initial_energy - self.energy


def anneal(  # noqa: C901 — one loop, kept in one place on purpose
    initial: S,
    *,
    energy: Callable[[S], float],
    move: Move[S],
    max_steps: int | None = None,
    time_budget: float | None = None,
    seed: int | None = None,
    rng: random.Random | None = None,
    schedule: Schedule | None = None,
    stall_steps: int | None = None,
    target_energy: float | None = None,
    copy: Callable[[S], S] | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    progress_every: int = 1000,
    calibration_samples: int = 64,
    clock: Callable[[], float] = time.monotonic,
) -> Result[S]:
    """Minimise ``energy`` over states reachable by ``move``.

    Args:
        initial: The starting state. Any object at all — the annealer never
            inspects it.
        energy: The objective, lower being better. Called once per step unless
            ``move`` returns deltas.
        move: ``move(state, rng) -> next_state`` or ``-> (next_state, delta)``.
            **Draw all randomness from the ``rng`` passed in**, never from the
            global ``random`` module, or the run stops being reproducible.
        max_steps: Stop after this many steps. Reproducible.
        time_budget: Stop after this many seconds. Bounds latency; costs exact
            reproducibility, since machines differ in how many steps they fit.
            At least one of ``max_steps`` / ``time_budget`` is required.
        seed: Seeds a private generator. Same seed + same ``max_steps`` + same
            inputs → identical result, with no effect on anyone else's RNG.
        rng: Supply your own generator instead of ``seed``.
        schedule: Temperature over progress. Defaults to a geometric schedule
            calibrated from the landscape itself (see ``calibrate``).
        stall_steps: Give up after this many steps with no new best.
        target_energy: Stop as soon as energy reaches this. For problems with a
            known floor — zero conflicts, say — this can end a run in
            milliseconds that would otherwise use its whole budget.
        copy: Only needed if ``move`` mutates and returns the *same* object.
            Called when a new best is found, so copying costs O(improvements),
            not O(steps). Leave it None if moves return fresh states.
        on_progress: Called every ``progress_every`` steps with a ``Progress``.
            The library itself never writes to a stream.
        progress_every: Steps between ``on_progress`` calls.
        calibration_samples: Moves sampled to size the default schedule. Ignored
            when ``schedule`` is given.
        clock: Monotonic time source; injectable so deadlines are testable.

    Returns:
        A ``Result`` holding the best state seen — never merely the last one.
    """
    if max_steps is None and time_budget is None:
        raise ValueError(
            "anneal() needs max_steps or time_budget: a search with no stopping "
            "condition is a hang, and a silent default would be a worse one"
        )
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must not be negative")
    if time_budget is not None and time_budget <= 0:
        raise ValueError("time_budget must be positive")
    if rng is not None and seed is not None:
        raise ValueError("pass seed or rng, not both")

    generator = rng if rng is not None else random.Random(seed)
    started = clock()
    deadline = started + time_budget if time_budget is not None else None

    current = initial
    current_energy = float(energy(current))
    best, best_energy = _snapshot(current, copy), current_energy
    initial_energy = current_energy

    if schedule is None:
        t_max, t_min = calibrate(
            initial,
            energy=energy,
            move=move,
            rng=generator,
            samples=calibration_samples,
            copy=copy,
        )
        schedule = geometric(t_max, t_min)

    steps = accepted = improved = 0
    since_improvement = 0
    stopped_by = "max_steps" if max_steps is not None else "time_budget"

    while True:
        if max_steps is not None and steps >= max_steps:
            stopped_by = "max_steps"
            break
        now = clock()
        if deadline is not None and now >= deadline:
            stopped_by = "time_budget"
            break
        if target_energy is not None and best_energy <= target_energy:
            stopped_by = "target_energy"
            break
        if stall_steps is not None and since_improvement >= stall_steps:
            stopped_by = "stall"
            break

        fraction = _fraction(steps, max_steps, now - started, time_budget)
        temperature = schedule(fraction)

        candidate, delta = _unpack(move(current, generator))
        if delta is None:
            candidate_energy = float(energy(candidate))
            delta = candidate_energy - current_energy
        else:
            candidate_energy = current_energy + delta

        if delta <= 0 or acceptance_probability(delta, temperature) > generator.random():
            current, current_energy = candidate, candidate_energy
            accepted += 1
            if current_energy < best_energy:
                best, best_energy = _snapshot(current, copy), current_energy
                improved += 1
                since_improvement = 0
            else:
                since_improvement += 1
        else:
            since_improvement += 1
            # Rejected. With functional moves the discarded candidate is simply
            # garbage; with in-place moves the caller is responsible for leaving
            # `current` usable, which is what `copy` exists to make safe.

        steps += 1
        if on_progress is not None and steps % progress_every == 0:
            on_progress(
                Progress(
                    step=steps,
                    energy=current_energy,
                    best_energy=best_energy,
                    temperature=temperature,
                    accepted=accepted,
                    elapsed=clock() - started,
                    fraction=fraction,
                )
            )

    # Recompute the winner's energy from scratch. With delta-returning moves the
    # tracked figure is a running sum, and a subtly wrong delta drifts away from
    # the truth without ever announcing itself. Reporting the recomputed value —
    # and how far it had wandered — turns a silent corruption into a number.
    true_energy = float(energy(best))
    return Result(
        state=best,
        energy=true_energy,
        initial_energy=initial_energy,
        steps=steps,
        accepted=accepted,
        improved=improved,
        elapsed=clock() - started,
        stopped_by=stopped_by,
        delta_drift=abs(true_energy - best_energy),
    )


def _unpack(result: S | tuple[S, float | None]) -> tuple[S, float | None]:
    """A move returned either a state or a (state, delta) pair."""
    if isinstance(result, tuple) and len(result) == 2:
        state, delta = result
        return state, (None if delta is None else float(delta))
    return result, None    # not a pair, so it IS the state


def _snapshot(state: S, copy: Callable[[S], S] | None) -> S:
    return state if copy is None else copy(state)


def _fraction(steps: int, max_steps: int | None, elapsed: float, time_budget: float | None) -> float:
    """How far through the budget we are. With both budgets set, whichever is
    running out faster drives the cooling — otherwise a run cut short by the
    deadline would end while still hot, throwing away the fine-grained search
    that the end of a schedule is for."""
    by_steps = steps / max_steps if max_steps else 0.0
    by_time = elapsed / time_budget if time_budget else 0.0
    return min(1.0, max(by_steps, by_time))
