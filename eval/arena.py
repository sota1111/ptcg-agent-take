"""Multi-match arena for the PTCG eval environment (SOT-1619).

Runs **N matches** of an agent-vs-agent card in one CLI invocation, building on
SOT-1618's recording single-match runner (``eval/record_match.py``): every match
is played by that recorded runner and its JSONL trace is accumulated under an
output directory. This is the statistical-evaluation substrate SOT-1620 builds on.

Design constraints (from SOT-1617's E-list):

* **E2 — one match per process.** The cabt engine keeps battle state in module
  globals (``cg.sim.Battle.battle_ptr`` / ``lib``), so two matches must never run
  concurrently in the same interpreter. We parallelise with
  ``concurrent.futures.ProcessPoolExecutor``; each worker runs exactly one match
  at a time (tasks are dispatched sequentially per worker), and a reused worker
  is safe because every match is a full start→…→``battle_finish()`` cycle.
* **E7 — no battle leak.** ``record_match`` already guarantees ``battle_finish()``
  via try/finally; the worker adds a second guard so a construction/other error
  still returns a scored-loss summary instead of killing the worker silently.
* **E1 — no engine seed.** The engine is non-deterministic and takes no seed, so
  match *outcomes* are not reproducible. Only the *agent-side* RNG is reproducible:
  each agent is seeded via ``random.Random(seed)`` with a deterministic per-match
  seed derived from the run's base seed (no global ``random.seed`` is ever set).

Side-swap pairing: for each pair the same card is played twice — once with agent A
in seat 0 and once with agent A in seat 1 — so each agent occupies seat 0 in
exactly half of the matches, removing seat/first-player bias by construction.

Abnormal matches (agent exception, illegal move → ``lib.Select`` != 0 == engine
error, timeout, worker error) are **not dropped**: they are scored as a loss for
the offending agent and counted with their failure category, matching Kaggle's
"agent error = loss" rule.

Usage:
    venv/bin/python eval/arena.py --games 100 [--deck0 deck.csv] [--deck1 deck.csv]
        [--agent-a random] [--agent-b random] [--workers K] [--seed S]
        [--level result|logs|full_obs] [--max-steps N] [--timeout SEC]
        [--out-dir eval/traces/arena_<ts>]

Run from the repo root (after scripts/setup_engine.sh has populated cg/).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # make `cg` and `eval` importable
os.chdir(REPO)            # so libcg.so & deck.csv resolve

from eval.record_match import (                     # noqa: E402
    load_deck,
    make_random_agent,
    make_raising_agent,
    record_match,
)
from eval.trace import (                            # noqa: E402
    FAIL_AGENT_EXCEPTION,
    FAIL_ENGINE_ERROR,
    FAIL_TIMEOUT,
    FAIL_WORKER_ERROR,
    RecordLevel,
)

_LEVELS = {"result": RecordLevel.RESULT, "logs": RecordLevel.LOGS, "full_obs": RecordLevel.FULL_OBS}

# Failure categories that score the offending player as the loser.
_LOSS_FAILURES = (FAIL_AGENT_EXCEPTION, FAIL_ENGINE_ERROR, FAIL_TIMEOUT, FAIL_WORKER_ERROR)


# --------------------------------------------------------------------------- #
# Agent specs (picklable) — workers rebuild the agent from a plain dict.
# --------------------------------------------------------------------------- #

def build_agent(spec: dict):
    """Rebuild an :class:`~eval.record_match.Agent` from a picklable spec dict.

    Kinds:
      ``random``  — uniform-random legal-move agent (the weakest baseline),
                    seeded with ``spec['seed']`` for reproducible agent RNG.
      ``raising`` — fault-injection agent (tests); raises after ``spec['after']``.
    """
    kind = spec["kind"]
    name = spec.get("name", kind)
    if kind == "random":
        return make_random_agent(seed=spec.get("seed"), name=name)
    if kind == "raising":
        return make_raising_agent(after=spec.get("after", 0), name=name)
    raise ValueError(f"unknown agent kind: {kind!r}")


def agent_spec(kind: str, name: Optional[str] = None, seed: Optional[int] = None, after: int = 0) -> dict:
    return {"kind": kind, "name": name or kind, "seed": seed, "after": after}


# --------------------------------------------------------------------------- #
# Match specs + side-swap pairing
# --------------------------------------------------------------------------- #

@dataclass
class MatchSpec:
    """One match to play. All fields are picklable (sent to a worker process)."""

    match_id: str
    pair_index: int
    side: int                 # 0 = agent A in seat 0; 1 = agent A in seat 1 (swapped)
    a_seat: int               # seat index (0/1) occupied by agent A this match
    deck0: list[int]          # deck for seat 0
    deck1: list[int]          # deck for seat 1
    agent0: dict              # spec for the agent in seat 0
    agent1: dict              # spec for the agent in seat 1
    out_path: str
    level: int
    max_steps: int


def derive_seed(base_seed: Optional[int], match_index: int, seat: int) -> Optional[int]:
    """Deterministic per-(match, seat) agent seed derived from the run base seed.

    Returns ``None`` (→ fresh entropy) when ``base_seed`` is ``None``. Otherwise a
    stable function of the inputs, so re-running with the same base seed assigns
    the same agent RNG seeds (agent-side reproducibility; engine stays E1-random).
    """
    if base_seed is None:
        return None
    return base_seed * 1_000_003 + match_index * 2 + seat


def build_match_specs(
    *,
    games: int,
    deck_a: list[int],
    deck_b: list[int],
    agent_a: dict,
    agent_b: dict,
    out_dir: str,
    level: RecordLevel,
    max_steps: int,
    base_seed: Optional[int],
) -> list[MatchSpec]:
    """Build ``games`` match specs as side-swap pairs.

    ``games`` is rounded **up** to an even number so every card is played as a
    complete A-first / B-first pair (a lone odd match would reintroduce seat bias).
    Match ``2k`` puts agent A in seat 0; match ``2k+1`` swaps A into seat 1.
    """
    if games <= 0:
        raise ValueError("games must be >= 1")
    n = games + (games % 2)  # round up to even → balanced pairs
    specs: list[MatchSpec] = []
    for i in range(n):
        pair_index, side = divmod(i, 2)
        a_seat = side  # side 0 → A in seat 0; side 1 → A in seat 1
        # Assign decks/agents to seats. Agent A always plays deck_a, agent B deck_b,
        # regardless of seat, so the swap changes only the seat (first-player) bias.
        if a_seat == 0:
            deck0, deck1 = deck_a, deck_b
            spec0 = {**agent_a, "seed": derive_seed(base_seed, i, 0)}
            spec1 = {**agent_b, "seed": derive_seed(base_seed, i, 1)}
        else:
            deck0, deck1 = deck_b, deck_a
            spec0 = {**agent_b, "seed": derive_seed(base_seed, i, 0)}
            spec1 = {**agent_a, "seed": derive_seed(base_seed, i, 1)}
        match_id = f"m{i:05d}"
        specs.append(
            MatchSpec(
                match_id=match_id,
                pair_index=pair_index,
                side=side,
                a_seat=a_seat,
                deck0=list(deck0),
                deck1=list(deck1),
                agent0=spec0,
                agent1=spec1,
                out_path=os.path.join(out_dir, f"{match_id}.jsonl"),
                level=int(level),
                max_steps=max_steps,
            )
        )
    return specs


# --------------------------------------------------------------------------- #
# Worker (runs in a child process — E2: one match per process at a time)
# --------------------------------------------------------------------------- #

def _run_match(spec_dict: dict) -> dict:
    """Play one match in this worker process and return an A/B-classified summary.

    ``record_match`` guarantees ``battle_finish()`` (E7); this wrapper adds a
    second guard so any *other* error (agent construction, etc.) still yields a
    scored-loss summary rather than crashing the worker.
    """
    spec = MatchSpec(**spec_dict)
    try:
        agent0 = build_agent(spec.agent0)
        agent1 = build_agent(spec.agent1)
        summary = record_match(
            spec.deck0,
            spec.deck1,
            agents=(agent0, agent1),
            out_path=spec.out_path,
            level=RecordLevel(spec.level),
            max_steps=spec.max_steps,
            trace_id=spec.match_id,
        )
    except Exception as exc:  # E7 safety net — never let a worker die silently
        summary = {
            "result": -1,
            "decisions": 0,
            "final_turn": None,
            "failure": {"player": None, "category": FAIL_WORKER_ERROR, "error": repr(exc)},
            "out_path": spec.out_path,
            "level": spec.level,
        }
    return classify(summary, spec.a_seat, match_id=spec.match_id, pair_index=spec.pair_index, side=spec.side)


# --------------------------------------------------------------------------- #
# Classification + aggregation (pure — unit-tested without the engine)
# --------------------------------------------------------------------------- #

def winner_seat(summary: dict) -> Optional[int]:
    """Seat (0/1) that won, or ``None`` for a draw / undecided match.

    Mirrors ``eval.trace.build_result``: a scored-loss failure hands the win to the
    other seat; else the engine ``result`` (0/1 = winner seat, 2 = draw, -1 = none).
    """
    failure = summary.get("failure")
    if failure and failure.get("category") in _LOSS_FAILURES and failure.get("player") in (0, 1):
        return 1 - failure["player"]
    result = summary.get("result", -1)
    if result in (0, 1):
        return result
    return None  # 2 == draw, -1 == truncated/undecided/timeout with no scored player


def classify(summary: dict, a_seat: int, **extra: Any) -> dict:
    """Annotate a match summary with the A/B outcome and failure attribution."""
    ws = winner_seat(summary)
    if ws is None:
        result = summary.get("result", -1)
        failure = summary.get("failure")
        # A draw is a real 2-2; anything else with no winner is undecided.
        outcome = "draw" if (result == 2 and not failure) else "undecided"
    elif ws == a_seat:
        outcome = "A_win"
    else:
        outcome = "B_win"

    failure = summary.get("failure")
    failed_agent = None
    failure_category = None
    if failure and failure.get("category"):
        failure_category = failure["category"]
        player = failure.get("player")
        if player in (0, 1):
            failed_agent = "A" if player == a_seat else "B"

    out = dict(summary)
    out.update(extra)
    out["a_seat"] = a_seat
    out["winner_seat"] = ws
    out["outcome"] = outcome
    out["failure_category"] = failure_category
    out["failed_agent"] = failed_agent
    return out


def aggregate(results: list[dict]) -> dict:
    """Roll per-match classified summaries up into a run report."""
    total = len(results)
    agg = {
        "total": total,
        "a_wins": 0,
        "b_wins": 0,
        "draws": 0,
        "undecided": 0,
        "a_seat0": 0,   # matches with agent A in seat 0 (should be ~half → 50:50)
        "a_seat1": 0,   # matches with agent A in seat 1
        "failures": 0,
        "failures_by_category": {},
        "failures_by_agent": {"A": 0, "B": 0},
    }
    for r in results:
        outcome = r.get("outcome")
        if outcome == "A_win":
            agg["a_wins"] += 1
        elif outcome == "B_win":
            agg["b_wins"] += 1
        elif outcome == "draw":
            agg["draws"] += 1
        else:
            agg["undecided"] += 1

        if r.get("a_seat") == 0:
            agg["a_seat0"] += 1
        elif r.get("a_seat") == 1:
            agg["a_seat1"] += 1

        cat = r.get("failure_category")
        if cat:
            agg["failures"] += 1
            agg["failures_by_category"][cat] = agg["failures_by_category"].get(cat, 0) + 1
            fa = r.get("failed_agent")
            if fa in ("A", "B"):
                agg["failures_by_agent"][fa] += 1

    decided = agg["a_wins"] + agg["b_wins"]
    agg["a_winrate"] = (agg["a_wins"] / decided) if decided else None
    # Side balance is guaranteed by construction; expose it so a caller can assert it.
    agg["side_balanced"] = agg["a_seat0"] == agg["a_seat1"]
    return agg


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_arena(
    *,
    games: int,
    deck_a: list[int],
    deck_b: list[int],
    agent_a: dict,
    agent_b: dict,
    out_dir: str,
    level: RecordLevel = RecordLevel.RESULT,
    max_steps: int = 100000,
    base_seed: Optional[int] = None,
    workers: Optional[int] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Run ``games`` side-swap matches across a process pool; return the report.

    ``timeout`` (seconds, optional) is a per-match wall-clock budget: a match that
    exceeds it is scored as a ``timeout`` loss for the seat-0 agent and its worker
    result is discarded. It is a safety net (matches are bounded by ``max_steps``);
    it is measured from submission, so it is only precise when ``workers >= games``.
    """
    os.makedirs(out_dir, exist_ok=True)
    specs = build_match_specs(
        games=games, deck_a=deck_a, deck_b=deck_b, agent_a=agent_a, agent_b=agent_b,
        out_dir=out_dir, level=level, max_steps=max_steps, base_seed=base_seed,
    )
    spec_by_id = {s.match_id: s for s in specs}
    results: list[dict] = []

    if workers is None:
        workers = min(len(specs), (os.cpu_count() or 2))
    workers = max(1, workers)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_to_spec = {ex.submit(_run_match, asdict(s)): s for s in specs}
        submitted_at = perf_counter()
        pending = set(fut_to_spec)
        timed_out: list[MatchSpec] = []
        while pending:
            poll = 0.1 if timeout else None
            done, pending = wait(pending, timeout=poll, return_when=FIRST_COMPLETED)
            for fut in done:
                results.append(fut.result())
            if timeout and pending:
                over = perf_counter() - submitted_at - timeout
                if over > 0:
                    expired = set(pending)
                    for fut in expired:
                        fut.cancel()
                        timed_out.append(fut_to_spec[fut])
                    pending -= expired
                    break

    for s in timed_out:
        summary = {
            "result": -1, "decisions": 0, "final_turn": None,
            "failure": {"player": 0, "category": FAIL_TIMEOUT,
                        "error": f"exceeded {timeout}s"},
            "out_path": s.out_path, "level": s.level,
        }
        results.append(classify(summary, s.a_seat, match_id=s.match_id,
                                pair_index=s.pair_index, side=s.side))

    results.sort(key=lambda r: r.get("match_id", ""))
    report = aggregate(results)
    report["out_dir"] = out_dir
    report["workers"] = workers
    report["matches"] = [
        {k: r.get(k) for k in ("match_id", "pair_index", "side", "a_seat", "outcome",
                               "winner_seat", "failure_category", "failed_agent",
                               "result", "final_turn", "out_path")}
        for r in results
    ]
    _ = spec_by_id  # kept for readability / future per-match lookups
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run N side-swap matches of an agent-vs-agent card.")
    p.add_argument("--games", type=int, default=100, help="number of matches (rounded up to an even number of pairs)")
    p.add_argument("--deck0", default="deck.csv", help="agent A's deck CSV")
    p.add_argument("--deck1", default=None, help="agent B's deck CSV (default: same as --deck0)")
    p.add_argument("--agent-a", default="random", choices=["random", "raising"], help="agent A kind")
    p.add_argument("--agent-b", default="random", choices=["random", "raising"], help="agent B kind")
    p.add_argument("--workers", type=int, default=None, help="process pool size (default: min(games, cpu_count))")
    p.add_argument("--seed", type=int, default=None, help="base seed for agent RNG (deterministic per-match derivation)")
    p.add_argument("--level", choices=list(_LEVELS), default="result", help="trace verbosity per match")
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--timeout", type=float, default=None, help="per-match wall-clock budget in seconds (safety net)")
    p.add_argument("--out-dir", default=None, help="trace output directory (default: eval/traces/arena_<ts>)")
    p.add_argument("--report", default=None, help="write the JSON report to this path (default: <out-dir>/report.json)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    deck_a = load_deck(args.deck0)
    deck_b = load_deck(args.deck1) if args.deck1 else deck_a

    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = os.path.join("eval", "traces", f"arena_{stamp}")

    report = run_arena(
        games=args.games,
        deck_a=deck_a,
        deck_b=deck_b,
        agent_a=agent_spec(args.agent_a, name=f"{args.agent_a}A"),
        agent_b=agent_spec(args.agent_b, name=f"{args.agent_b}B"),
        out_dir=out_dir,
        level=_LEVELS[args.level],
        max_steps=args.max_steps,
        base_seed=args.seed,
        workers=args.workers,
        timeout=args.timeout,
    )

    report_path = args.report or os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    wr = report["a_winrate"]
    wr_s = f"{wr:.3f}" if wr is not None else "n/a"
    print(
        f"ARENA DONE: games={report['total']} A_win={report['a_wins']} B_win={report['b_wins']}"
        f" draw={report['draws']} undecided={report['undecided']} A_winrate={wr_s}"
        f" side_balanced={report['side_balanced']} (A_seat0={report['a_seat0']} A_seat1={report['a_seat1']})"
        f" failures={report['failures']} by_category={report['failures_by_category']}"
        f" traces={out_dir} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
