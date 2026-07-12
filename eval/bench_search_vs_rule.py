"""Evaluation benchmark: one-ply search agent vs the rule-based agent (SOT-1659).

The 案B search agent (SOT-1657 skeleton + SOT-1658 damage-based evaluation) is
now complete; this benchmark quantifies it by playing it **head to head against
the rule-based agent** (``agents/rule_based.py``) in an N≥200 side-swap arena and
reporting three things the parent Issue asks for:

* **Win rate + Wilson 95% CI** of the search agent over *decided* games — the
  effect size of 案B against the rule baseline, with an interval so a noisy
  handful of games can't be over-read (draws / undecided / abnormal losses are
  tallied separately and never distort the rate, matching ``eval/report.py``).
* **All games complete without crashing.** The arena scores an agent exception /
  illegal move / timeout as a loss for the offender and counts it as a *failure*;
  the gate requires **zero failures** (a "Validation Episode Error" would show up
  here), so a crash can only fail the gate, never hide.
* **Per-move thinking-time statistics (mean / max)** for the *search* agent — the
  Kaggle time-limit watch value. Unlike ``eval/report.py``'s overall thinking-time
  distribution (both seats pooled), this attributes each decision to the acting
  seat via the trace's meta, so the reported times are the search agent's own —
  the slow side whose per-move budget actually matters.

This is an **evaluation** task, not a "beat the baseline" gate: the search line is
early (one-ply, best-effort hidden-info prediction), so it is not expected to
exceed the tuned rule agent. The gate therefore checks *validity and completion*
(N≥200, zero crashes, stats emitted), not that the win rate clears 0.5 — the win
rate + CI are **reported** for the parent Issue to read, not asserted.

Usage:
    venv/bin/python eval/bench_search_vs_rule.py [--games 200] [--seed S]
        [--workers K] [--z 1.96] [--time-budget 0.5] [--max-candidates 12]
        [--json report.json]

Exit code 0 iff the gate passes (N≥200 and no failed matches). Run from the repo
root (after scripts/setup_engine.sh has populated cg/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval.arena import agent_spec, run_arena           # noqa: E402
from eval.record_match import load_deck                # noqa: E402
from eval.report import _dist, find_trace_files, wilson_interval  # noqa: E402
from eval.trace import RecordLevel                     # noqa: E402

SEARCH_NAME = "search"
RULE_NAME = "rule"


def search_thinking_times(out_dir: str, search_name: str = SEARCH_NAME) -> list[float]:
    """Collect the *search* agent's per-move thinking times from the run's traces.

    Each ``decision`` record is attributed to the acting seat (``your_index``,
    falling back to ``select_player``); the trace's ``meta`` maps that seat to an
    agent name. Only decisions made by ``search_name`` contribute, so the returned
    list is the search agent's own per-move thinking time in ms — regardless of
    which seat it occupied in a given side-swapped match. Requires LOGS-level
    traces (RESULT level emits no decision records → empty list).
    """
    times: list[float] = []
    for path in find_trace_files(out_dir):
        try:
            with open(path, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        meta = next((r for r in records if r.get("kind") == "meta"), None)
        if meta is None:
            continue
        seat_name = {}
        for a in meta.get("agents") or []:
            if a.get("index") in (0, 1):
                seat_name[a["index"]] = a.get("name")
        for r in records:
            if r.get("kind") != "decision":
                continue
            seat = r.get("your_index")
            if seat not in (0, 1):
                seat = r.get("select_player")
            t = r.get("thinking_time_ms")
            if seat_name.get(seat) == search_name and isinstance(t, (int, float)):
                times.append(float(t))
    return times


def run_bench(
    games: int,
    seed: int | None,
    workers: int | None,
    z: float,
    time_budget_s: float,
    max_candidates: int,
) -> dict:
    """Run search (A) vs rule (B) and return win rate + Wilson CI + thinking stats."""
    deck = load_deck("deck.csv")
    out_dir = os.path.join("eval", "traces", f"bench_search_vs_rule_{games}")
    report = run_arena(
        games=games,
        deck_a=deck,
        deck_b=deck,
        agent_a=agent_spec(
            "search", name=SEARCH_NAME,
            time_budget_s=time_budget_s, max_candidates=max_candidates,
        ),
        agent_b=agent_spec("rule_based", name=RULE_NAME),
        # LOGS level so per-decision thinking times are recorded for the watch.
        out_dir=out_dir,
        level=RecordLevel.LOGS,
        base_seed=seed,
        workers=workers,
    )
    wins = report["a_wins"]
    decided = report["a_wins"] + report["b_wins"]
    lo, hi = wilson_interval(wins, decided, z=z)
    winrate = (wins / decided) if decided else 0.0
    thinking = search_thinking_times(out_dir)
    return {
        "games": report["total"],
        "wins": wins,
        "losses": report["b_wins"],
        "draws": report["draws"],
        "undecided": report["undecided"],
        "failures": report["failures"],
        "failures_by_category": report["failures_by_category"],
        "failures_by_agent": report["failures_by_agent"],
        "decided": decided,
        "winrate": winrate,
        "ci_low": lo,
        "ci_high": hi,
        "side_balanced": report["side_balanced"],
        "search_thinking_ms": _dist(thinking),
        "out_dir": out_dir,
        # Evaluation gate: enough games AND every match completed without a crash.
        "passed": report["total"] >= 200 and report["failures"] == 0,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search-vs-rule evaluation benchmark (win rate + Wilson CI + thinking time).")
    p.add_argument("--games", type=int, default=200, help="matches (>=200 required by the gate)")
    p.add_argument("--seed", type=int, default=None, help="base agent-RNG seed (engine stays E1-random)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--z", type=float, default=1.96, help="z for the CI (1.96 = 95%%)")
    p.add_argument("--time-budget", type=float, default=0.5, help="search agent per-decision time budget (s)")
    p.add_argument("--max-candidates", type=int, default=12, help="search agent per-decision candidate cap")
    p.add_argument("--json", default=None, help="also write the raw JSON result to this path")
    return p.parse_args(argv)


def _fmt_think(d: dict | None) -> str:
    if not d:
        return "n/a (no search decisions recorded)"
    return (f"n={d['n']} mean={d['mean']:.2f}ms median={d['median']:.2f}ms "
            f"p95={d['p95']:.2f}ms max={d['max']:.2f}ms")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    r = run_bench(args.games, args.seed, args.workers, args.z,
                  args.time_budget, args.max_candidates)
    print(
        f"BENCH search vs rule_based: games={r['games']} decided={r['decided']}"
        f" wins={r['wins']} losses={r['losses']} draws={r['draws']} undecided={r['undecided']}"
        f" failures={r['failures']} winrate={r['winrate']:.3f}"
        f" Wilson95=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        f" side_balanced={r['side_balanced']}"
    )
    print(f"SEARCH thinking/decision (ms): {_fmt_think(r['search_thinking_ms'])}")
    if r["failures"]:
        print(f"FAILURES by_category={r['failures_by_category']} by_agent={r['failures_by_agent']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=2)
        print(f"wrote JSON result -> {args.json}")
    if r["games"] < 200:
        print("GATE INVALID: need >= 200 games")
        return 2
    print(f"GATE {'PASS' if r['passed'] else 'FAIL'}: "
          f"{r['games']} games, {r['failures']} failed matches "
          f"(evaluation gate = N>=200 and no crashes; win rate is reported, not asserted)")
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
