# simple-annealing

Simulated annealing that is safe to run **inside a server**.

Deterministic when you ask for it, bounded by a deadline when you need one, silent
by default, and free of global state — so several optimisations can share a
process without interfering with each other or with you.

Pure Python, no dependencies, one function.

```sh
pip install simple-annealing
```

---

## What it's for

Simulated annealing finds a good arrangement of things when there are too many
arrangements to try: routes, rosters, seating plans, timetables, packings. You
give it a way to score an arrangement and a way to nudge one into a neighbouring
arrangement, and it wanders the space — accepting the occasional worse
arrangement early on so it doesn't get stuck in the first decent one it finds, and
growing fussier as it goes.

**Reach for it when** the state is discrete and combinatorial, "good enough,
quickly" beats "provably optimal, eventually", and you can score an arrangement
cheaply.

**Don't** when you're optimising a smooth function of real numbers (use SciPy),
when your problem is small enough to solve exactly (use a real solver), or when
you're tuning hyperparameters (use Optuna). More on all three [below](#what-to-use-instead).

The algorithm itself is short — nobody needs a library for the loop. What's worth
packaging is the handful of decisions that go wrong when you run one as part of a
**live system** rather than in a script. That's what this is.

---

## How to use it

Two functions and a budget:

```python
from simple_annealing import anneal

def energy(route):                      # lower is better
    return sum(distance(a, b) for a, b in zip(route, route[1:]))

def move(route, rng):                   # one small random change
    i, j = rng.randrange(len(route)), rng.randrange(len(route))
    nudged = list(route)
    nudged[i], nudged[j] = nudged[j], nudged[i]
    return tuple(nudged)

result = anneal(
    initial_route,
    energy=energy,
    move=move,
    seed=1234,
    max_steps=50_000,
)

result.state        # the best arrangement found — never merely the last one tried
result.energy       # its score, recomputed from scratch rather than accumulated
result.improvement  # how much better than where you started
result.stopped_by   # "max_steps" | "time_budget" | "stall" | "target_energy"
```

That's the whole surface. The rest of this section is the detail behind those five
arguments.

### `energy(state) -> float`

Your objective. Lower is better; the units are yours. It's called once per step
unless your `move` returns deltas (see below), so if it's expensive, that's the
first thing to look at.

### `move(state, rng) -> state`

Returns a *neighbouring* arrangement. Two rules:

1. **Draw all randomness from the `rng` you're handed**, never from the global
   `random` module — otherwise the run stops being reproducible.
2. **Return a new state.** Nothing is copied if you do, which is what keeps the
   loop cheap. If you'd rather mutate in place, return the same object and pass
   `copy=` — that gets called once per improvement, not once per step.

### Budgets

At least one of `max_steps` or `time_budget` is required. There's no default,
because a search with no stopping condition is a hang, and a silent default step
count is a hang waiting for a slow objective. See
[choosing a budget](#choosing-a-budget) for which to pick.

You can also pass `target_energy` (stop once the answer is good enough — for
problems with a known floor, like "zero conflicts", this can end a run in
milliseconds) and `stall_steps` (give up after N steps with no improvement).

### Reporting

The library never writes to a stream. Pass `on_progress` to receive an immutable
`Progress` record every `progress_every` steps and route it wherever you like:

```python
anneal(..., on_progress=lambda p: log.info("step %d best %.2f", p.step, p.best_energy),
       progress_every=10_000)
```

### Bringing a class

There's no base class to inherit, but bound methods are functions, so modelling
the problem as a class works with no support from the library:

```python
class TourProblem:
    def __init__(self, cities):
        self.cities = cities          # shared by both methods

    def energy(self, route): ...
    def move(self, route, rng): ...

problem = TourProblem(cities)
result = anneal(start, energy=problem.energy, move=problem.move, seed=1, max_steps=50_000)
```

You keep the class and the shared data; you just don't inherit a constructor and a
state contract. See [why it's this way round](#8-functions-not-inheritance--but-classes-still-work).

---

## Writing a good `move()`

This matters more than any parameter here. A tuned schedule with a poor move
function loses to a default schedule with a good one, every time.

**Small.** A move should reach a *neighbour*, not a stranger. If a move changes
half the state, its energy is uncorrelated with the current one and you're doing
random sampling with extra steps.

**Reachable.** Every arrangement you'd accept as an answer must be reachable from
the start by some chain of moves. If your moves only ever swap adjacent items,
whole regions of the space are unreachable and the search can't find what's in
them.

**Roughly reversible.** If a move can be undone by another equally likely move, the
maths behind the acceptance rule holds. Strongly one-way moves bias the walk.

**Structure-aware beats generic.** The classic illustration is the travelling
salesman: swapping two cities is a legal move, but reversing a *segment* (2-opt)
is dramatically better, because a long tour's real problem is crossings and
reversing a segment is exactly what uncrosses them. Both examples in `examples/`
use 2-opt for that reason.

**Cheap.** It runs hundreds of thousands of times.

If a run disappoints, suspect the move function before the temperature. A quick
diagnostic: run with `schedule=constant(0)`, which accepts only improvements. If
that hill-climb gets nearly as far as annealing did, your moves aren't opening up
the space and no schedule will fix it.

---

## Choosing a budget

The one real trade-off in the library, stated plainly.

| | `max_steps` | `time_budget` |
|---|---|---|
| Bounds | work | latency |
| Same seed, same answer? | **Yes**, on any machine, under any load | No |
| Use when | you need reproducibility, or you're testing | you have a deadline to hit |

Both can be set: the run stops at whichever comes first, cooling is driven by
whichever is running out faster — so a run cut short by the clock still finishes
cold rather than mid-wander — and `result.stopped_by` tells you which one fired.

That last field is worth logging. A deployment that has quietly got slower shows
up as a shift from `max_steps` to `time_budget` in your metrics, rather than as
silence.

The difference is real and easy to see for yourself. From `examples/server.py`, 20
concurrent requests with the same seed:

```
step-bounded      distinct routes: 1     stopped_by: max_steps
deadline-bounded  distinct routes: 19    stopped_by: time_budget
```

Nothing is wrong in the second row: twenty runs contending for the same cores each
fit a different number of steps into the deadline, so each explores a different
amount. A wall-clock bound buys promptness with exactly that reproducibility.

---

## Why this one?

### 1. It can run in a worker thread

Web frameworks serve synchronous handlers from a threadpool. An optimiser that
registers a `SIGINT` handler in its constructor cannot be constructed there at all
— `signal.signal` raises `ValueError: signal only works in main thread of the main
interpreter` — leaving you to run a CPU-bound loop on the event loop and block
every other request.

This library installs no signal handlers and touches no interpreter-global state.
A test asserts `SIGINT`'s handler is untouched; another runs eight optimisations
concurrently across a threadpool.

### 2. Determinism that survives concurrency

The usual way to get a reproducible run is `random.seed(n)` — process-global. Two
requests annealing at once then interleave draws from the same generator, so
neither is reproducible, and seeding stamps on every other consumer of `random` in
the process.

Here, `seed=` creates a private `random.Random`. Same seed and same step budget
give an identical answer whether the run is alone or one of twenty; the global
generator is provably untouched.

This matters beyond testing. People stop trusting a button that answers
differently each time they press it, even when every answer is good.

### 3. Deadlines, not just step counts

See [choosing a budget](#choosing-a-budget). A run with no stopping condition at
all raises rather than defaulting.

### 4. It never prints

Nothing goes to stdout or stderr. Progress goes to a callback, or nowhere.
Asserted by a test.

### 5. Copies scale with improvements, not with steps

The conventional design mutates state in place and copies it constantly to track
current, previous and best — hundreds of thousands of copies in a 50,000-step run.

Here `move()` returns the next state, so nothing is copied at all. Prefer to
mutate? Pass `copy=`; it runs only when a new best is found, so copying costs
O(improvements) — typically a few dozen — instead of O(steps).

### 6. Deltas, with the usual footgun defused

An expensive objective makes full recomputation the bottleneck, so `move()` may
return `(next_state, delta)` and skip the `energy()` call entirely.

The trap is that a *wrong* delta is silent: the run diligently optimises a running
total that has drifted from the real objective, and reports a confident,
meaningless number. This library recomputes the winner's energy once at the end,
reports that as the truth, and exposes the discrepancy as `result.delta_drift`.
Non-zero means your delta and your energy disagree.

### 7. Temperatures sized from your problem

`t_max` and `t_min` are in the units of *your* energy function, so a library
default is meaningless — 25,000 is scorching for a problem scored in fractions and
freezing for one scored in thousands. Omit `schedule` and the annealer takes a
short random walk first, measures the uphill moves it actually meets, and picks a
range accepting a typical one 80% of the time at the start and the smallest one 1%
of the time at the end. Tens of moves; bypass it with an explicit `schedule=`.

### 8. Functions, not inheritance — but classes still work

There's no base class to subclass and no framework object holding your state.

That isn't austerity; it's what makes point 5 possible. A base class has to own the
state as `self.state`, and *that* is what forces an annealer to copy on every step
to track current, previous and best. Make state a value that moves return, and
there's nothing to copy.

If you like modelling the problem as a class, [do](#bringing-a-class) — bound
methods are functions. The functional API subsumes the class-based one; the
reverse isn't true.

---

## What to use instead

This library is narrow on purpose.

| If you're… | Use |
|---|---|
| Optimising a continuous function over ℝⁿ | **`scipy.optimize.dual_annealing`** or `basinhopping`. Mature, well-tested, built for real-valued vectors. This library targets combinatorial state — permutations, assignments, schedules — where "the gradient" isn't a thing |
| Annealing in a script or notebook: main thread, no deadline, progress on screen | **[`simanneal`](https://github.com/perrygeo/simanneal)** is a fine, long-standing choice with a progress display built in. Its constraints — a global RNG, a `SIGINT` handler installed at construction, step-count budgets, stderr output — are reasonable for interactive use, and are exactly what this library exists to avoid elsewhere |
| Needing a proven-optimal answer, not a good one | **OR-Tools CP-SAT**. Annealing offers no optimality guarantee. If your problem is small enough to solve exactly, solve it exactly |
| Tuning hyperparameters | **Optuna** or **Hyperopt**. Different problem, better tools |
| Running one long optimisation offline | Almost anything works, including this. The properties above stop mattering when nothing else shares the process |

---

## API

### `anneal(initial, *, energy, move, ...) -> Result`

| Argument | Meaning |
|---|---|
| `initial` | Any object. The annealer never inspects it |
| `energy` | `state -> float`, lower is better |
| `move` | `(state, rng) -> next_state` or `-> (next_state, delta)` |
| `max_steps` | Stop after N steps. Reproducible |
| `time_budget` | Stop after N seconds. At least one budget is required |
| `seed` / `rng` | A private generator. Never the global one |
| `schedule` | `progress(0..1) -> temperature`. Defaults to a calibrated geometric schedule |
| `stall_steps` | Give up after N steps with no new best |
| `target_energy` | Stop once this is reached |
| `copy` | Only for in-place moves. Called per improvement |
| `on_progress` / `progress_every` | Reporting hook and its cadence |
| `calibration_samples` | Moves sampled to size the default schedule |
| `clock` | Injectable time source, so deadlines are testable |

### `Result` (frozen)

`state`, `energy`, `initial_energy`, `improvement`, `steps`, `accepted`,
`improved`, `elapsed`, `stopped_by`, `delta_drift`.

### `Progress` (frozen)

`step`, `energy`, `best_energy`, `temperature`, `accepted`, `elapsed`, `fraction`.

### Schedules

`geometric(t_max, t_min)` (the usual choice), `linear(t_max, t_min=0)`,
`constant(t)` (Metropolis without annealing — useful for telling whether a
disappointing run is the schedule's fault or the move function's). A schedule is
just `Callable[[float], float]`, so your own works anywhere one is accepted.
`acceptance_probability(delta, temperature)` is exposed for the same reason.

### `calibrate(initial, *, energy, move, rng, ...) -> (t_max, t_min)`

Exposed separately so you can size a range once and reuse it across many runs
rather than paying for the sampling walk each time.

---

## Examples

| | |
|---|---|
| `examples/tsp.py` | A 20-city tour, 2-opt moves, progress through a callback. ~500k steps in 2s, 246 → 121 |
| `examples/server.py` | The case this library exists for: annealing in a request handler, stdlib only. Demonstrates both budget modes under concurrency |

---

## Development

```sh
pip install -e ".[dev]"
pytest
ruff check .
mypy --strict src/simple_annealing
```

Python 3.10+. No runtime dependencies.

## Licence

MIT.
