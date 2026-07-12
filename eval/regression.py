"""Unified evaluation / regression harness (SOT-1636).

One command runs the standing regression suite for the rule-based agent and
prints a statistical summary, so every rule change can be re-checked the same way
— guarding against **over-fitting to Random** and watching the per-decision
thinking time (the Kaggle time-limit risk):

* Card ``rule_vs_random`` — the current agent vs the uniform-random baseline.
  Absolute-strength floor: the agent must keep beating Random (Wilson 95% CI
  lower bound > 0.5).
* Card ``new_vs_old`` — the current agent (``new``) vs a **previous version of
  itself** (``old``), head to head. Non-regression guard: ``new`` must not have
  regressed against the prior agent (Wilson 95% CI upper bound ≥ 0.5). Beating
  Random while regressing here is exactly the over-fitting this suite catches.

Each card is an N-match **side-swap arena** (``eval/arena.py``, SOT-1619) recorded
at ``LOGS`` level so per-decision thinking times are captured; win rate + Wilson
95% CI come from ``eval/report.py`` (SOT-1620) helpers and the thinking-time
distribution is aggregated from the recorded traces. The suite therefore *is*
connected to SOT-1619/1620 (acceptance: replaceable structure, dependency does
not block); if those were absent the same ``run_arena`` loop is the fallback.

Old-version reference ("旧版エージェント参照", two ways, both established here):
* **default** — the in-repo policy toggle: ``new`` = scoring MAIN policy
  (SOT-1635), ``old`` = the previous fixed-priority policy (SOT-1633). Always
  available, no external state, so the command runs anywhere.
* **``--old-ref <git-ref>``** — materialise the historical ``agents/`` package
  from any git tag / commit via ``eval/old_agent.py`` and play against it, for
  true cross-commit regression once tags exist.

Usage:
    venv/bin/python eval/regression.py [--games 100] [--seed S] [--workers K]
        [--old-ref v1.2.0] [--old-policy fixed] [--z 1.96]
        [--think-warn-ms MS] [--json report.json]

Exit code 0 iff every card's gate passes (unless ``--no-gate``). Run from the
repo root (after scripts/setup_engine.sh has populated cg/).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval import report as report_mod                 # noqa: E402
from eval.arena import agent_spec, run_arena           # noqa: E402
from eval.old_agent import materialize_agents_ref      # noqa: E402
from eval.record_match import load_deck                # noqa: E402
from eval.report import wilson_interval                # noqa: E402
from eval.trace import RecordLevel                     # noqa: E402


# --------------------------------------------------------------------------- #
# One card = one agent-A-vs-agent-B arena, summarised (pure given a report dir).
# --------------------------------------------------------------------------- #

def summarise_card(
    name: str,
    arena_report: dict,
    trace_report: Optional[dict],
    *,
    z: float,
    gate: str,
) -> dict:
    """Fold an arena report + trace report into a card summary with a gate verdict.

    ``gate`` selects the non-regression criterion:
      ``"beat"``      — Wilson CI lower bound > 0.5 (A is significantly stronger);
                        used for the absolute-strength floor (vs Random).
      ``"noregress"`` — Wilson CI upper bound ≥ 0.5 (A did *not* significantly
                        regress vs B); used for new-vs-old.
    """
    wins = arena_report["a_wins"]
    losses = arena_report["b_wins"]
    decided = wins + losses
    lo, hi = wilson_interval(wins, decided, z=z)
    winrate = (wins / decided) if decided else None

    if gate == "beat":
        passed = decided > 0 and lo > 0.5
        gate_desc = f"Wilson lower {lo:.4f} > 0.5"
    elif gate == "noregress":
        passed = decided == 0 or hi >= 0.5
        gate_desc = f"Wilson upper {hi:.4f} >= 0.5"
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown gate {gate!r}")

    think = (trace_report or {}).get("thinking_time_ms")
    return {
        "card": name,
        "gate": gate,
        "gate_desc": gate_desc,
        "passed": bool(passed),
        "games": arena_report["total"],
        "wins": wins,
        "losses": losses,
        "draws": arena_report["draws"],
        "undecided": arena_report["undecided"],
        "decided": decided,
        "winrate": winrate,
        "ci_low": lo,
        "ci_high": hi,
        "side_balanced": arena_report["side_balanced"],
        "failures": arena_report["failures"],
        "failures_by_category": arena_report["failures_by_category"],
        "thinking_time_ms": think,
        "out_dir": arena_report.get("out_dir"),
    }


def run_card(
    name: str,
    *,
    agent_a: dict,
    agent_b: dict,
    deck: list[int],
    out_dir: str,
    games: int,
    seed: Optional[int],
    workers: Optional[int],
    z: float,
    gate: str,
) -> dict:
    """Play one card's side-swap arena at LOGS level and summarise it."""
    arena_report = run_arena(
        games=games,
        deck_a=deck,
        deck_b=deck,
        agent_a=agent_a,
        agent_b=agent_b,
        out_dir=out_dir,
        level=RecordLevel.LOGS,   # LOGS → per-decision thinking_time_ms recorded
        base_seed=seed,
        workers=workers,
    )
    # Aggregate per-decision thinking time (+ reason/seat stats) from the traces.
    trace_report = report_mod.generate(out_dir, z=z)
    return summarise_card(name, arena_report, trace_report, z=z, gate=gate)


# --------------------------------------------------------------------------- #
# Suite orchestration
# --------------------------------------------------------------------------- #

def run_suite(
    *,
    games: int,
    seed: Optional[int],
    workers: Optional[int],
    z: float,
    out_root: str,
    old_ref: Optional[str] = None,
    old_policy: str = "fixed",
    new_policy: str = "scoring",
) -> dict:
    """Run the full regression suite and return the combined report.

    Returns a dict with ``cards`` (list of card summaries) and ``passed`` (all
    gates passed). ``old_ref`` — when given — plays new-vs-old against the agent
    materialised from that git ref; otherwise new-vs-old is the in-repo policy
    toggle (``new_policy`` vs ``old_policy``).
    """
    deck = load_deck("deck.csv")
    os.makedirs(out_root, exist_ok=True)
    cards: list[dict] = []

    # Card 1 — absolute strength floor vs Random.
    cards.append(
        run_card(
            "rule_vs_random",
            agent_a=agent_spec("rule_based", name="rule_new", policy=new_policy),
            agent_b=agent_spec("random", name="random"),
            deck=deck,
            out_dir=os.path.join(out_root, "rule_vs_random"),
            games=games, seed=seed, workers=workers, z=z, gate="beat",
        )
    )

    # Card 2 — new vs old (self-regression / over-fitting guard).
    import_root = None
    try:
        if old_ref:
            import_root, pkg = materialize_agents_ref(old_ref)
            old_spec = {"kind": "rule_based_ref", "name": f"old@{old_ref}",
                        "import_root": import_root, "pkg": pkg, "policy": None}
            old_label = f"git-ref {old_ref}"
        else:
            old_spec = agent_spec("rule_based", name="rule_old", policy=old_policy)
            old_label = f"policy={old_policy}"
        card = run_card(
            "new_vs_old",
            agent_a=agent_spec("rule_based", name="rule_new", policy=new_policy),
            agent_b=old_spec,
            deck=deck,
            out_dir=os.path.join(out_root, "new_vs_old"),
            games=games, seed=seed, workers=workers, z=z, gate="noregress",
        )
        card["old_reference"] = old_label
        cards.append(card)
    finally:
        if import_root:
            shutil.rmtree(import_root, ignore_errors=True)

    return {
        "out_root": out_root,
        "games": games,
        "seed": seed,
        "z": z,
        "cards": cards,
        "passed": all(c["passed"] for c in cards),
    }


# --------------------------------------------------------------------------- #
# Text formatting
# --------------------------------------------------------------------------- #

def _fmt_pct(x: Optional[float]) -> str:
    return f"{x * 100:5.1f}%" if x is not None else "  n/a"


def _fmt_think(d: Optional[dict]) -> str:
    if not d:
        return "n/a (record at LOGS level for per-decision times)"
    return (
        f"n={d['n']} mean={d['mean']:.2f}ms median={d['median']:.2f}ms "
        f"p95={d['p95']:.2f}ms max={d['max']:.2f}ms"
    )


def format_suite(suite: dict, *, think_warn_ms: Optional[float] = None) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PTCG REGRESSION SUITE (SOT-1636)  —  new vs old / vs random")
    lines.append("=" * 72)
    lines.append(f"games/card={suite['games']}  seed={suite['seed']}  z={suite['z']}")
    for c in suite["cards"]:
        lines.append("")
        ref = f"  (old = {c['old_reference']})" if c.get("old_reference") else ""
        lines.append(f"[{c['card']}]{ref}")
        lines.append(
            f"  A(new) W-L-D = {c['wins']}-{c['losses']}-{c['draws']}"
            f"  undecided={c['undecided']}  decided={c['decided']}"
            f"  side_balanced={c['side_balanced']}"
        )
        lines.append(
            f"  winrate={_fmt_pct(c['winrate'])}"
            f"  Wilson{suite['z']}=[{c['ci_low']:.4f}, {c['ci_high']:.4f}]"
        )
        lines.append(f"  think/decision : {_fmt_think(c['thinking_time_ms'])}")
        if think_warn_ms is not None and c["thinking_time_ms"]:
            p95 = c["thinking_time_ms"]["p95"]
            mx = c["thinking_time_ms"]["max"]
            if mx > think_warn_ms or p95 > think_warn_ms:
                lines.append(
                    f"  ⚠ THINK WARN: p95={p95:.1f}ms max={mx:.1f}ms exceed "
                    f"{think_warn_ms:.0f}ms budget"
                )
        if c["failures"]:
            lines.append(f"  failures={c['failures']} by_category={c['failures_by_category']}")
        lines.append(f"  GATE {'PASS' if c['passed'] else 'FAIL'}: {c['gate_desc']}")
    lines.append("")
    lines.append(f"SUITE {'PASS' if suite['passed'] else 'FAIL'}")
    lines.append("=" * 72)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified new-vs-old / vs-random regression suite.")
    p.add_argument("--games", type=int, default=100, help="matches per card (rounded up to even pairs)")
    p.add_argument("--seed", type=int, default=None, help="base agent-RNG seed (engine stays E1-random)")
    p.add_argument("--workers", type=int, default=None, help="process pool size (default: min(games, cpu))")
    p.add_argument("--z", type=float, default=1.96, help="z for the win-rate CI (1.96 = 95%%)")
    p.add_argument("--old-ref", default=None, help="git ref (tag/commit) of the OLD agent for new_vs_old")
    p.add_argument("--old-policy", default="fixed", choices=["scoring", "fixed"],
                   help="in-repo OLD MAIN policy when --old-ref is not given (default: fixed)")
    p.add_argument("--new-policy", default="scoring", choices=["scoring", "fixed"],
                   help="NEW MAIN policy under test (default: scoring)")
    p.add_argument("--think-warn-ms", type=float, default=None,
                   help="warn if per-decision p95/max thinking time exceeds this many ms")
    p.add_argument("--out-root", default=None, help="trace output root (default: eval/traces/regression_<ts>)")
    p.add_argument("--json", default=None, help="also write the raw JSON suite report to this path")
    p.add_argument("--no-gate", action="store_true", help="always exit 0 (report only, do not gate)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    out_root = args.out_root
    if out_root is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_root = os.path.join("eval", "traces", f"regression_{stamp}")

    suite = run_suite(
        games=args.games,
        seed=args.seed,
        workers=args.workers,
        z=args.z,
        out_root=out_root,
        old_ref=args.old_ref,
        old_policy=args.old_policy,
        new_policy=args.new_policy,
    )
    print(format_suite(suite, think_warn_ms=args.think_warn_ms))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(suite, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote JSON suite report -> {args.json}")
    default_json = os.path.join(out_root, "suite.json")
    with open(default_json, "w", encoding="utf-8") as fh:
        json.dump(suite, fh, ensure_ascii=False, indent=2)
    print(f"traces + suite.json -> {out_root}")

    if args.no_gate:
        return 0
    return 0 if suite["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
