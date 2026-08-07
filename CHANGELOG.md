# Changelog

## 0.1.0 — unreleased

First release.

- `anneal()` — simulated annealing over any state type, with `energy` and `move`
  as plain functions rather than an inherited base class.
- Private seeded RNG. The process-global `random` module is never read or
  written, so concurrent runs stay independent and reproducible.
- No signal handlers and no stream output, so a run is safe in a worker thread
  and silent unless you pass `on_progress`.
- Stopping conditions: `max_steps`, `time_budget`, `target_energy`,
  `stall_steps`. At least one budget is required — there is no default.
- `move()` may return `(state, delta)` to skip the energy call. The winner's
  energy is recomputed at the end and any disagreement surfaces as
  `result.delta_drift`.
- `copy=` for in-place moves, invoked once per improvement rather than per step.
- Schedules: `geometric`, `linear`, `constant`, or any
  `Callable[[float], float]`.
- `calibrate()` sizes `t_max`/`t_min` from a sample of the actual landscape;
  used automatically when no `schedule` is given.
