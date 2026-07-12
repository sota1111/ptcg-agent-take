"""Acceptance gate: scoring MAIN policy vs the old fixed-priority policy (SOT-1635).

R4 migrated the rule-based agent's MAIN turn from a fixed-priority ladder
(SOT-1633) to option **scoring**, adding Supporter/Item play, retreat judgement
and Bench development. This benchmark measures the improvement directly by
playing the two policies **head to head** in an N-match side-swap arena and
applying the issue's gate: the **Wilson 95% CI lower bound of the scoring
policy's win rate must exceed 0.5** — i.e. the new policy is significantly better
than the old one, not merely better on a noisy handful of games.

A head-to-head is the sharpest "improvement over the old implementation"
evidence: both agents share the same deck and the seat-swap pairing removes
first-player bias, so any edge is the policy's. Win rate is over *decided* games
(draws / undecided excluded); an agent exception / illegal move / timeout is
scored as a loss for the offender by the arena, so it can only hurt the scoring
agent — the gate stays honest.

Usage:
    venv/bin/python eval/bench_scoring_vs_fixed.py [--games 400] [--seed S]
        [--workers K] [--z 1.96]

Exit code 0 iff the gate passes. Run from the repo root.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval.arena import agent_spec, run_arena           # noqa: E402
from eval.record_match import load_deck                # noqa: E402
from eval.report import wilson_interval                # noqa: E402
from eval.trace import RecordLevel                     # noqa: E402


def run_gate(games: int, seed: int | None, workers: int | None, z: float) -> dict:
    """Run scoring (A) vs fixed (B) and return win rate + Wilson interval + pass."""
    deck = load_deck("deck.csv")
    stamp = f"bench_scoring_vs_fixed_{games}"
    report = run_arena(
        games=games,
        deck_a=deck,
        deck_b=deck,
        agent_a=agent_spec("rule_based", name="scoring", policy="scoring"),
        agent_b=agent_spec("rule_based", name="fixed", policy="fixed"),
        out_dir=os.path.join("eval", "traces", stamp),
        level=RecordLevel.RESULT,
        base_seed=seed,
        workers=workers,
    )
    wins = report["a_wins"]
    decided = report["a_wins"] + report["b_wins"]
    lo, hi = wilson_interval(wins, decided, z=z)
    winrate = (wins / decided) if decided else 0.0
    return {
        "games": report["total"],
        "wins": wins,
        "losses": report["b_wins"],
        "draws": report["draws"],
        "undecided": report["undecided"],
        "failures": report["failures"],
        "failures_by_category": report["failures_by_category"],
        "decided": decided,
        "winrate": winrate,
        "ci_low": lo,
        "ci_high": hi,
        "side_balanced": report["side_balanced"],
        "passed": lo > 0.5,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scoring-vs-fixed Wilson-CI acceptance gate.")
    p.add_argument("--games", type=int, default=400, help="matches (>=200 required by the gate)")
    p.add_argument("--seed", type=int, default=None, help="base agent-RNG seed (engine stays E1-random)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--z", type=float, default=1.96, help="z for the CI (1.96 = 95%%)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    r = run_gate(args.games, args.seed, args.workers, args.z)
    print(
        f"BENCH scoring vs fixed: games={r['games']} decided={r['decided']}"
        f" wins={r['wins']} losses={r['losses']} draws={r['draws']} undecided={r['undecided']}"
        f" failures={r['failures']} winrate={r['winrate']:.3f}"
        f" Wilson95=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        f" side_balanced={r['side_balanced']}"
    )
    if r["games"] < 200:
        print("GATE INVALID: need >= 200 games")
        return 2
    print(f"GATE {'PASS' if r['passed'] else 'FAIL'}: Wilson 95% CI lower {r['ci_low']:.4f} "
          f"{'>' if r['passed'] else '<='} 0.5")
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
