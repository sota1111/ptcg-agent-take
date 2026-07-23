"""25-deck mirror rotation: current agent vs a git-ref agent (SOT-1694).

The 25-deck generalisation gate: for every deck under ``decks/rotation_baseline/`` the
**current working-tree agent** (A, archetype-adaptive) plays a side-swapped
mirror arena against the agent materialised from a git ref (B, default ``main``
= the SOT-1682 v2 champion), both on the *same* deck. Results are aggregated
overall (win rate + Wilson 95% CI), per archetype, and per deck.

Gate semantics (from the issue's acceptance criteria):
* **no degradation** — reject the change iff the overall point estimate is
  < 0.5 AND the Wilson CI upper bound is < 0.5 (a *significant* loss);
* improvement is claimed only when the CI lower bound is > 0.5.

Usage (run from the repo root):
    venv/bin/python eval/bench_25deck_rotation.py --games-per-deck 20
        [--old-ref main] [--seed 42] [--out-root eval/traces/sot1694_bench]
        [--json PATH] [--md PATH]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from agents import damage                                    # noqa: E402
from agents.archetype import classify_archetype, deck_profile  # noqa: E402
from eval.arena import agent_spec, run_arena                 # noqa: E402
from eval.old_agent import materialize_agents_ref            # noqa: E402
from eval.record_match import load_deck                      # noqa: E402
from eval.report import wilson_interval                      # noqa: E402
from eval.trace import RecordLevel                           # noqa: E402
from eval.trace_gap_report import aggregate_deck, handled_contexts, p95  # noqa: E402


def run_bench(args: argparse.Namespace) -> dict:
    deck_paths = sorted(glob.glob(args.decks_glob))
    if not deck_paths:
        raise SystemExit(f"no decks match {args.decks_glob!r}")

    cards = damage.get_card_registry()
    attacks = damage.get_attack_registry()
    handled = handled_contexts()

    import_root, pkg = materialize_agents_ref(args.old_ref)
    decks: list[dict] = []
    by_arch: dict[str, dict[str, int]] = {}
    wins = losses = draws = undecided = failures = 0
    a_failures = b_failures = 0
    think: list[float] = []
    try:
        for di, path in enumerate(deck_paths):
            name = os.path.splitext(os.path.basename(path))[0]
            deck = load_deck(path)
            arch = classify_archetype(deck_profile(deck, cards, attacks))
            out_dir = os.path.join(args.out_root, name)
            # --neutral-bands ablation: an unreadable deck_path leaves the new
            # agent's archetype adaptation at the neutral (all-zero) BandAdjust,
            # isolating the effect of the new context handlers.
            a_deck_path = "/dev/null" if args.neutral_bands else path
            rep = run_arena(
                games=args.games_per_deck,
                deck_a=deck, deck_b=deck,
                agent_a=agent_spec("rule_based", name="new", policy="scoring",
                                   deck_path=a_deck_path),
                agent_b={"kind": "rule_based_ref", "name": f"old@{args.old_ref}",
                         "import_root": import_root, "pkg": pkg, "policy": None,
                         "seed": None},
                out_dir=out_dir,
                level=RecordLevel.LOGS,
                base_seed=(args.seed + di) if args.seed is not None else None,
                workers=args.workers,
            )
            agg = aggregate_deck(out_dir, handled)
            think.extend(agg["think_ms"])

            d_wins, d_losses = rep["a_wins"], rep["b_wins"]
            lo, hi = wilson_interval(d_wins, d_wins + d_losses)
            decks.append({
                "deck": name, "archetype": arch,
                "games": rep["total"], "wins": d_wins, "losses": d_losses,
                "draws": rep["draws"], "undecided": rep["undecided"],
                "winrate": (d_wins / (d_wins + d_losses)) if (d_wins + d_losses) else None,
                "ci": [lo, hi],
                "failures": rep["failures"],
                "failures_by_agent": rep["failures_by_agent"],
                "fallback_decisions": sum(agg["ctx_fallback"].values()),
            })
            wins += d_wins
            losses += d_losses
            draws += rep["draws"]
            undecided += rep["undecided"]
            failures += rep["failures"]
            a_failures += rep["failures_by_agent"]["A"]
            b_failures += rep["failures_by_agent"]["B"]
            arch_agg = by_arch.setdefault(arch, {"wins": 0, "losses": 0})
            arch_agg["wins"] += d_wins
            arch_agg["losses"] += d_losses
            print(f"  {name} [{arch}]: {d_wins}-{d_losses}-{rep['draws']}"
                  f" wr={decks[-1]['winrate']}")
    finally:
        shutil.rmtree(import_root, ignore_errors=True)

    decided = wins + losses
    lo, hi = wilson_interval(wins, decided)
    winrate = (wins / decided) if decided else None
    for arch, agg in by_arch.items():
        d = agg["wins"] + agg["losses"]
        alo, ahi = wilson_interval(agg["wins"], d)
        agg["winrate"] = (agg["wins"] / d) if d else None
        agg["ci"] = [alo, ahi]

    degraded = decided > 0 and winrate < 0.5 and hi < 0.5
    improved = decided > 0 and lo > 0.5
    return {
        "old_ref": args.old_ref,
        "games_per_deck": args.games_per_deck,
        "seed": args.seed,
        "total_games": wins + losses + draws + undecided,
        "decided": decided,
        "wins": wins, "losses": losses, "draws": draws, "undecided": undecided,
        "winrate": winrate, "ci": [lo, hi],
        "degraded": degraded, "improved": improved,
        "failures": failures,
        "failures_by_agent": {"new": a_failures, "old": b_failures},
        "think_p95_ms": p95(think), "think_max_ms": max(think) if think else None,
        "by_archetype": by_arch,
        "decks": decks,
    }


def render_md(rep: dict) -> str:
    L = []
    L.append("# SOT-1694 25デッキ mirror rotation — 新スコアリング vs 現行v2"
             f" (old = git-ref `{rep['old_ref']}`)")
    L.append("")
    verdict = ("**改善 (CI下限>0.5)**" if rep["improved"]
               else ("**劣化 (不採用条件成立)**" if rep["degraded"] else "有意差なし (劣化なし)"))
    L.append(f"- 総合: {rep['wins']}-{rep['losses']}-{rep['draws']} "
             f"(N={rep['total_games']}, decided={rep['decided']}) "
             f"勝率 **{rep['winrate']:.3f}** Wilson95 [{rep['ci'][0]:.3f}, {rep['ci'][1]:.3f}] → {verdict}")
    L.append(f"- fault: 全体 {rep['failures']} (new={rep['failures_by_agent']['new']}, "
             f"old={rep['failures_by_agent']['old']})")
    tp = rep.get("think_p95_ms")
    tm = rep.get("think_max_ms")
    if tp is not None:
        L.append(f"- 思考時間/decision: p95={tp:.2f}ms max={tm:.2f}ms")
    L.append("")
    L.append("| archetype | W-L | 勝率 | Wilson 95% CI |")
    L.append("| --- | ---: | ---: | --- |")
    for arch, a in sorted(rep["by_archetype"].items()):
        L.append(f"| {arch} | {a['wins']}-{a['losses']} | "
                 f"{a['winrate']:.3f} | [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}] |")
    L.append("")
    L.append("| deck | archetype | W-L-D | 勝率 | CI | fault |")
    L.append("| --- | --- | --- | ---: | --- | ---: |")
    for d in rep["decks"]:
        wr = f"{d['winrate']:.2f}" if d["winrate"] is not None else "n/a"
        L.append(f"| {d['deck']} | {d['archetype']} | {d['wins']}-{d['losses']}-{d['draws']} "
                 f"| {wr} | [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}] | {d['failures']} |")
    L.append("")
    return "\n".join(L)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="25-deck mirror rotation vs a git-ref agent (SOT-1694).")
    p.add_argument("--decks-glob", default="decks/rotation_baseline/*.csv")
    p.add_argument("--games-per-deck", type=int, default=20)
    p.add_argument("--old-ref", default="main")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-root", default="eval/traces/sot1694_bench")
    p.add_argument("--json", default=None)
    p.add_argument("--md", default=None)
    p.add_argument("--neutral-bands", action="store_true",
                   help="ablation: disable archetype adaptation (handlers only)")
    p.add_argument("--kpi", nargs="?", const="", default=None, metavar="ISSUE",
                   help="append a KPI history record (eval/kpi.py, SOT-1709);"
                        " optional value = Linear issue id")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    rep = run_bench(args)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(render_md(rep))
    if args.kpi is not None:
        from eval.kpi import append_history, record_from_rotation
        print(f"KPI: appended to "
              f"{append_history(record_from_rotation(rep, issue=args.kpi or None))}")
    print(f"BENCH DONE: new-vs-old(@{args.old_ref}) {rep['wins']}-{rep['losses']}-{rep['draws']}"
          f" winrate={rep['winrate']:.4f} CI=[{rep['ci'][0]:.4f}, {rep['ci'][1]:.4f}]"
          f" degraded={rep['degraded']} improved={rep['improved']}"
          f" failures={rep['failures']} think_p95={rep['think_p95_ms']}ms")
    return 1 if rep["degraded"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
