"""Win-rate evaluation: learned-policy agent vs Random (SOT-1644).

Runs an N-match side-swap arena of the learned-policy :class:`LearnedAgent`
(SOT-1644) against the uniform-random baseline and reports the win rate with a
**Wilson 95% CI**, exactly like ``eval/bench_rule_vs_random.py``. The engine is
non-deterministic (E1), so only the interval over many side-swapped games is
meaningful — never a single match.

Win rate is measured over *decided* games (draws / undecided excluded). Abnormal
matches (agent exception / illegal move / timeout) are scored as a loss for the
offending agent by the arena, so an inference bug can only ever hurt the learned
agent's rate — the report stays honest, and the fallback keeps it from happening.

Model
-----
``--model PATH`` points at a bundled policy JSON (the SOT-1643 inference format).
**Without a model the learned agent falls back to random**, so its win rate sits
near 0.5 by construction — that is the expected baseline for SOT-1644 (this
Issue delivers the inference + evaluation plumbing; a model that actually *beats*
random is SOT-1643's quality bar). The beats-random gate (Wilson lower bound >
0.5) is therefore **reported but not enforced** unless ``--require-beat-random``
is passed; the command still exits non-zero on an invalid run (< 200 games).

Usage:
    venv/bin/python eval/bench_learned_vs_random.py [--games 400] [--seed S]
        [--model agents/learned/model/policy.json] [--workers K] [--z 1.96]
        [--require-beat-random] [--json report.json]

Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from agents.learned.agent import load_model                # noqa: E402
from eval.arena import agent_spec, run_arena               # noqa: E402
from eval.record_match import load_deck                    # noqa: E402
from eval.report import wilson_interval                    # noqa: E402
from eval.trace import RecordLevel                         # noqa: E402


def run_eval(
    games: int,
    seed: int | None,
    workers: int | None,
    z: float,
    model_path: str | None,
) -> dict:
    """Run the arena (learned vs random) and return win rate + Wilson interval."""
    deck = load_deck("deck.csv")
    stamp = f"bench_learned_vs_random_{games}"
    report = run_arena(
        games=games,
        deck_a=deck,
        deck_b=deck,
        agent_a=agent_spec("learned", name="learned", model_path=model_path),
        agent_b=agent_spec("random", name="random"),
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
        "model_path": model_path,
        "model_loaded": load_model(model_path) is not None,
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
        "z": z,
        "side_balanced": report["side_balanced"],
        "beats_random": lo > 0.5,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learned-vs-Random win-rate + Wilson CI.")
    p.add_argument("--games", type=int, default=400, help="matches (>=200 required)")
    p.add_argument("--seed", type=int, default=None, help="base agent-RNG seed (engine stays E1-random)")
    p.add_argument("--model", default=None, help="learned policy JSON (absent → random fallback)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--z", type=float, default=1.96, help="z for the CI (1.96 = 95%%)")
    p.add_argument("--require-beat-random", action="store_true",
                   help="exit non-zero unless the Wilson lower bound exceeds 0.5")
    p.add_argument("--json", default=None, help="also write the report as JSON to this path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    r = run_eval(args.games, args.seed, args.workers, args.z, args.model)
    print(
        f"BENCH learned vs random: model={'yes' if r['model_loaded'] else 'none(random-fallback)'}"
        f" games={r['games']} decided={r['decided']}"
        f" wins={r['wins']} losses={r['losses']} draws={r['draws']} undecided={r['undecided']}"
        f" failures={r['failures']} winrate={r['winrate']:.3f}"
        f" Wilson95=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        f" side_balanced={r['side_balanced']}"
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=2)
        print(f"wrote {args.json}")

    if r["games"] < 200:
        print("EVAL INVALID: need >= 200 games")
        return 2
    print(
        f"beats_random(Wilson lower {r['ci_low']:.4f} > 0.5): "
        f"{'YES' if r['beats_random'] else 'NO'}"
        f"{'' if r['model_loaded'] else ' (no model — random-fallback baseline, ~0.5 expected)'}"
    )
    if args.require_beat_random and not r["beats_random"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
