"""The case this library exists for: annealing inside a request handler.

Standard library only — `ThreadingHTTPServer` gives each request its own thread,
which is the same shape as a WSGI/ASGI framework's threadpool and the same shape
that breaks an optimiser holding global state.

    python examples/server.py
    curl 'localhost:8077/solve?seed=1'              # step-bounded: reproducible
    curl 'localhost:8077/solve?seed=1&budget=0.2'   # deadline-bounded: fast

Hammer the reproducible one and the answers stay identical, however many run at
once:

    seq 20 | xargs -P 20 -I{} curl -s 'localhost:8077/solve?seed=1' | sort -u | wc -l
    # 1 — twenty simultaneous runs, one answer

Do the same to the deadline-bounded one and you'll get a spread of answers, which
is not a bug and is worth seeing for yourself: twenty runs contending for the same
cores each fit a different number of steps into 0.2 seconds, so each explores a
different amount. A wall-clock bound buys a latency guarantee with exactly that
reproducibility. Pick per endpoint, and read `stopped_by` to know which bound
actually ended the run.

Two properties make the reproducible case hold at all, and neither is free:

  * the run's randomness comes from a private generator, so concurrent requests
    don't interleave draws from one global stream;
  * nothing installs a signal handler, which would raise outright off the main
    thread and take the whole handler with it.
"""
from __future__ import annotations

import json
import math
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from simple_annealing import anneal

POINTS = [(math.cos(i / 3.0) * 50 + i, math.sin(i / 2.0) * 50 - i) for i in range(40)]

# Small enough to finish comfortably inside a request, large enough to find a
# decent tour. A step count — not a duration — so the answer is the same on every
# machine and under any amount of load.
STEPS = 20_000


def tour_length(route: tuple[int, ...]) -> float:
    total = 0.0
    for a, b in zip(route, route[1:] + route[:1], strict=True):
        (ax, ay), (bx, by) = POINTS[a], POINTS[b]
        total += math.hypot(ax - bx, ay - by)
    return total


def reverse_segment(route: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    i, j = sorted((rng.randrange(len(route)), rng.randrange(len(route))))
    return route[:i] + route[i:j + 1][::-1] + route[j + 1:]


def solve(seed: int, budget: float | None) -> dict[str, object]:
    """Optimise a tour. With no `budget` the run is bounded by steps alone, so it
    is reproducible; with one, it is bounded by the clock as well, so it is
    prompt. `stopped_by` reports which bound actually ended it — worth logging,
    because a deployment that has quietly got slower shows up as a shift from
    "max_steps" to "time_budget" rather than as silence."""
    result = anneal(
        tuple(range(len(POINTS))),
        energy=tour_length,
        move=reverse_segment,
        seed=seed,
        max_steps=STEPS,
        time_budget=budget,
    )
    return {
        "length": round(result.energy, 3),
        "route": list(result.state),
        "steps": result.steps,
        "stopped_by": result.stopped_by,
        "reproducible": result.stopped_by != "time_budget",
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming
        query = parse_qs(urlparse(self.path).query)
        try:
            seed = int(query.get("seed", ["0"])[0])
            raw_budget = query.get("budget", [""])[0]
            budget = float(raw_budget) if raw_budget else None
        except ValueError:
            self.send_error(400, "seed must be an integer and budget a number of seconds")
            return
        body = json.dumps(solve(seed, budget)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        """Quiet: the point of the demo is the response, not the access log."""


if __name__ == "__main__":
    with ThreadingHTTPServer(("127.0.0.1", 8077), Handler) as server:
        print(f"listening on http://127.0.0.1:8077/solve?seed=1  ({STEPS:,} steps per request)")
        print("add &budget=0.2 to bound the run by the clock instead")
        server.serve_forever()
