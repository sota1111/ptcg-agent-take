"""Candidate-vs-current deck comparison arena (SOT-1661).

For each candidate deck under ``decks/`` this plays "candidate deck (agent A) vs
the current ``deck.csv`` (agent B)" with the **same rule-based agent on both
seats** (``agents/rule_based.py``), so the only thing that differs between the two
sides is the *deck* — the win rate then measures the candidate deck's strength
against the current deck, not an agent gap.

It reuses the existing evaluation substrate rather than re-implementing a match
loop:

* **Side-swap pairing + parallel execution** come straight from the multi-match
  arena (SOT-1619, ``eval/arena.run_arena``): every candidate is played over N
  side-swap games (default N=200), so seat/first-player bias is removed by
  construction and abnormal matches are scored as a loss for the offender and
  counted as failures (never silently dropped).
* **Win rate + Wilson 95% CI** come from the statistics in the aggregation report
  (SOT-1620, ``eval/report.wilson_interval``), so a noisy handful of games can't
  be over-read.

All candidates are rolled up into **one comparison summary** (text + JSON) with,
per candidate: deck name / win rate vs current / Wilson CI / decided+total games /
abnormal-termination count.

Usage:
    venv/bin/python eval/compare_decks.py [--games 200] [--seed S] [--workers K]
        [--decks-dir decks] [--current deck.csv] [--policy scoring] [--z 1.96]
        [--json summary.json]

Exit code 0 iff every candidate completed all its matches with no failures. Run
from the repo root (after scripts/setup_engine.sh has populated cg/).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # make `cg` and `eval` importable
os.chdir(REPO)            # so libcg.so & deck.csv resolve

from eval.arena import agent_spec, run_arena           # noqa: E402
from eval.record_match import load_deck                # noqa: E402
from eval.report import wilson_interval                # noqa: E402
from eval.trace import RecordLevel                     # noqa: E402

CANDIDATE_NAME = "candidate"
CURRENT_NAME = "current"


def discover_candidate_decks(decks_dir: str, current_path: Optional[str] = None) -> list[tuple[str, str]]:
    """Return ``(name, path)`` for every candidate deck CSV under ``decks_dir``.

    Sorted by path for a stable comparison order; ``name`` is the file stem
    (``decks/deck_aggro.csv`` -> ``deck_aggro``). The current deck is excluded if
    it happens to resolve to a file inside ``decks_dir``.
    """
    current_abs = os.path.abspath(current_path) if current_path else None
    out: list[tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(decks_dir, "*.csv"))):
        if current_abs is not None and os.path.abspath(path) == current_abs:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        out.append((name, path))
    return out


def deck_result(name: str, path: str, report: dict, z: float = 1.96) -> dict:
    """Turn one arena report (candidate = agent A, current = agent B) into a row.

    Pure: it only reads the aggregate fields ``run_arena`` produces, so it is
    unit-testable with a synthetic report and no engine. The win rate is over
    *decided* games (draws / undecided / abnormal losses are reported separately
    and never distort it, matching ``eval/report.py``); ``winrate`` is ``None``
    when no game was decided.
    """
    wins = report["a_wins"]
    losses = report["b_wins"]
    decided = wins + losses
    lo, hi = wilson_interval(wins, decided, z=z)
    winrate = (wins / decided) if decided else None
    return {
        "deck": name,
        "path": path,
        "games": report["total"],
        "wins": wins,
        "losses": losses,
        "draws": report["draws"],
        "undecided": report["undecided"],
        "decided": decided,
        "winrate": winrate,
        "ci_low": lo,
        "ci_high": hi,
        "failures": report["failures"],
        "failures_by_category": report.get("failures_by_category", {}),
        "failures_by_agent": report.get("failures_by_agent", {}),
        "side_balanced": report.get("side_balanced"),
        "out_dir": report.get("out_dir"),
    }


def run_compare(
    games: int,
    *,
    seed: Optional[int] = None,
    workers: Optional[int] = None,
    z: float = 1.96,
    decks_dir: str = "decks",
    current_deck: str = "deck.csv",
    policy: Optional[str] = None,
) -> dict:
    """Play every candidate deck vs the current deck and roll up one summary.

    Each candidate is a full ``run_arena`` (side-swap, parallel) of the rule-based
    agent on the candidate deck (A) against the same rule-based agent on the
    current deck (B). Returns ``{current, games, policy, candidates: [...], passed}``
    where ``passed`` is true iff every candidate completed all its matches with no
    abnormal terminations.
    """
    candidates = discover_candidate_decks(decks_dir, current_path=current_deck)
    if not candidates:
        raise FileNotFoundError(f"no candidate deck CSVs found under {decks_dir!r}")

    current = load_deck(current_deck)
    rows: list[dict] = []
    for name, path in candidates:
        deck = load_deck(path)
        out_dir = os.path.join("eval", "traces", f"compare_decks_{name}_{games}")
        report = run_arena(
            games=games,
            deck_a=deck,
            deck_b=current,
            agent_a=agent_spec("rule_based", name=CANDIDATE_NAME, policy=policy),
            agent_b=agent_spec("rule_based", name=CURRENT_NAME, policy=policy),
            out_dir=out_dir,
            level=RecordLevel.RESULT,
            base_seed=seed,
            workers=workers,
        )
        rows.append(deck_result(name, path, report, z=z))

    return {
        "current": current_deck,
        "decks_dir": decks_dir,
        "games": games,
        "policy": policy or "scoring",
        "z": z,
        "candidates": rows,
        # Completion gate: every candidate ran all games with no abnormal matches.
        "passed": all(r["failures"] == 0 and r["games"] >= games for r in rows),
    }


def _fmt_winrate(r: dict) -> str:
    if r["winrate"] is None:
        return "winrate=n/a (no decided games)"
    return (f"winrate={r['winrate']:.3f} "
            f"Wilson95=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")


def format_summary(summary: dict) -> str:
    """Render the comparison summary as a human-readable text block."""
    lines = [
        f"COMPARE DECKS vs current={summary['current']} games={summary['games']}"
        f" policy={summary['policy']} z={summary['z']}"
    ]
    for r in summary["candidates"]:
        lines.append(
            f"  {r['deck']:<16} {_fmt_winrate(r)} decided={r['decided']}"
            f" wins={r['wins']} losses={r['losses']} draws={r['draws']}"
            f" undecided={r['undecided']} games={r['games']} failures={r['failures']}"
        )
    passed = summary["passed"]
    lines.append(
        f"GATE {'PASS' if passed else 'FAIL'}: "
        f"{len(summary['candidates'])} candidate(s), "
        f"all matches completed without crashes"
        if passed else
        f"GATE FAIL: some candidate had abnormal matches / short run "
        f"({len(summary['candidates'])} candidate(s))"
    )
    return "\n".join(lines)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare candidate decks vs the current deck (rule-vs-rule, win rate + Wilson CI)."
    )
    p.add_argument("--games", type=int, default=200, help="matches per candidate (rounded up to even side-swap pairs)")
    p.add_argument("--seed", type=int, default=None, help="base agent-RNG seed (engine stays non-deterministic)")
    p.add_argument("--workers", type=int, default=None, help="process pool size (default: min(games, cpu_count))")
    p.add_argument("--z", type=float, default=1.96, help="z for the Wilson CI (1.96 = 95%%)")
    p.add_argument("--decks-dir", default="decks", help="directory of candidate deck CSVs")
    p.add_argument("--current", default="deck.csv", help="current deck CSV to compare against")
    p.add_argument("--policy", default=None, choices=["scoring", "fixed"], help="rule-based MAIN policy for both seats")
    p.add_argument("--json", default=None, help="write the JSON summary to this path (default: <traces>/compare_decks_<games>/summary.json)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    summary = run_compare(
        args.games,
        seed=args.seed,
        workers=args.workers,
        z=args.z,
        decks_dir=args.decks_dir,
        current_deck=args.current,
        policy=args.policy,
    )

    print(format_summary(summary))

    json_path = args.json or os.path.join(
        "eval", "traces", f"compare_decks_{args.games}", "summary.json"
    )
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"wrote JSON summary -> {json_path}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
