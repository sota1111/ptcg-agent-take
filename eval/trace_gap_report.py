"""25-deck mirror trace aggregation: fallback holes + loss causes (SOT-1694).

Plays a rule-based **mirror** arena (same deck both seats, side-swapped) for every
deck under ``decks/initial/`` and aggregates the recorded traces into the two
quantities the 25-deck generalisation work needs (the SOT-1682-proven method):

(a) **random-fallback holes** — per :class:`~cg.api.SelectContext`, how many
    decisions were taken in a context with **no registered handler** in
    ``RuleBasedAgent.CONTEXT_HANDLERS`` (those decisions are legal-random picks).
    The known shape-based deferral of ``TO_HAND`` on bounce effects (options
    referencing in-play cards) is counted separately, since it is detectable from
    the recorded ``SelectData`` alone.

(b) **loss causes by archetype** — the engine result ``reason``
    (1 = prizes taken, 2 = deck-out, 3 = no Active = 場切れ, 4 = card effect)
    per deck, rolled up by the mechanically derived archetype
    (:func:`agents.archetype.classify_archetype`).

Usage (run from the repo root):
    venv/bin/python eval/trace_gap_report.py --games-per-deck 20 --seed 1694
        [--out-root eval/traces/sot1694_gap] [--skip-run]
        [--json docs/trace_gap_sot1694.json] [--md docs/trace_gap_sot1694.md]
        [--compare-json BEFORE.json]   # add a before/after fallback table

``--skip-run`` aggregates the traces already under ``--out-root`` (e.g. to
re-aggregate with a newer handler table without replaying).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from cg.api import AreaType, SelectContext          # noqa: E402
from agents import damage                            # noqa: E402
from agents.archetype import classify_archetype, deck_profile  # noqa: E402
from agents.rule_based import RuleBasedAgent         # noqa: E402
from eval.arena import agent_spec, run_arena         # noqa: E402
from eval.record_match import load_deck              # noqa: E402
from eval.trace import RecordLevel                   # noqa: E402

# Loss-reason codes from the engine RESULT log (cg/api.py).
REASONS = {1: "prize_out", 2: "deck_out", 3: "no_active", 4: "card_effect"}

_BOUNCE_AREAS = (int(AreaType.ACTIVE), int(AreaType.BENCH))


def context_name(value: Optional[int]) -> str:
    if value is None:
        return "UNKNOWN"
    try:
        return SelectContext(value).name
    except ValueError:
        return f"CTX_{value}"


def handled_contexts() -> set[int]:
    """The contexts the *current* agent handles (MAIN policies included)."""
    return {int(c) for c in RuleBasedAgent.CONTEXT_HANDLERS}


def aggregate_deck(out_dir: str, handled: set[int]) -> dict:
    """Fold one deck's mirror traces into per-context and loss-cause counts."""
    ctx_total: dict[int, int] = {}
    ctx_fallback: dict[int, int] = {}
    to_hand_bounce = 0
    losses: dict[str, int] = {}
    decided = failures = games = 0
    turns: list[int] = []
    think_ms: list[float] = []

    for path in sorted(glob.glob(os.path.join(out_dir, "m*.jsonl"))):
        games += 1
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = rec.get("kind")
                if kind == "decision":
                    sel = rec.get("select") or {}
                    ctx = sel.get("context")
                    if ctx is None:
                        continue
                    ctx_total[ctx] = ctx_total.get(ctx, 0) + 1
                    tt = rec.get("thinking_time_ms")
                    if isinstance(tt, (int, float)):
                        think_ms.append(float(tt))
                    if ctx not in handled:
                        ctx_fallback[ctx] = ctx_fallback.get(ctx, 0) + 1
                    elif ctx == int(SelectContext.TO_HAND):
                        opts = sel.get("option") or []
                        if any(o.get("area") in _BOUNCE_AREAS for o in opts):
                            to_hand_bounce += 1
                elif kind == "result":
                    if rec.get("failure"):
                        failures += 1
                    if rec.get("winner") in (0, 1):
                        decided += 1
                        reason = REASONS.get(rec.get("reason"), "other")
                        losses[reason] = losses.get(reason, 0) + 1
                    if rec.get("final_turn") is not None:
                        turns.append(rec["final_turn"])

    return {
        "games": games,
        "decided": decided,
        "failures": failures,
        "avg_turns": (sum(turns) / len(turns)) if turns else None,
        "ctx_total": ctx_total,
        "ctx_fallback": ctx_fallback,
        "to_hand_bounce_defer": to_hand_bounce,
        "losses": losses,
        "think_ms": think_ms,
    }


def _merge_counts(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def p95(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def run_and_aggregate(args: argparse.Namespace) -> dict:
    deck_paths = sorted(glob.glob(args.decks_glob))
    if not deck_paths:
        raise SystemExit(f"no decks match {args.decks_glob!r}")

    cards = damage.get_card_registry()
    attacks = damage.get_attack_registry()
    handled = handled_contexts()

    decks: list[dict] = []
    total_ctx: dict[int, int] = {}
    total_fallback: dict[int, int] = {}
    total_losses: dict[str, int] = {}
    by_arch_losses: dict[str, dict[str, int]] = {}
    all_think: list[float] = []
    total_bounce = total_failures = total_games = total_decided = 0

    for di, path in enumerate(deck_paths):
        name = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(args.out_root, name)
        deck = load_deck(path)
        if not args.skip_run:
            run_arena(
                games=args.games_per_deck,
                deck_a=deck, deck_b=deck,
                agent_a=agent_spec("rule_based", name="mirrorA", policy="scoring",
                                   deck_path=path),
                agent_b=agent_spec("rule_based", name="mirrorB", policy="scoring",
                                   deck_path=path),
                out_dir=out_dir,
                level=RecordLevel.LOGS,
                base_seed=(args.seed + di) if args.seed is not None else None,
                workers=args.workers,
            )
        profile = deck_profile(deck, cards, attacks)
        arch = classify_archetype(profile)
        agg = aggregate_deck(out_dir, handled)

        decks.append({
            "deck": name,
            "archetype": arch,
            "profile": profile.__dict__,
            "games": agg["games"],
            "decided": agg["decided"],
            "failures": agg["failures"],
            "avg_turns": agg["avg_turns"],
            "fallback_decisions": sum(agg["ctx_fallback"].values()),
            "to_hand_bounce_defer": agg["to_hand_bounce_defer"],
            "losses": agg["losses"],
        })
        _merge_counts(total_ctx, agg["ctx_total"])
        _merge_counts(total_fallback, agg["ctx_fallback"])
        _merge_counts(total_losses, agg["losses"])
        arch_losses = by_arch_losses.setdefault(arch, {})
        _merge_counts(arch_losses, agg["losses"])
        all_think.extend(agg["think_ms"])
        total_bounce += agg["to_hand_bounce_defer"]
        total_failures += agg["failures"]
        total_games += agg["games"]
        total_decided += agg["decided"]

    return {
        "games_per_deck": args.games_per_deck,
        "seed": args.seed,
        "out_root": args.out_root,
        "handled_contexts": sorted(handled),
        "total_games": total_games,
        "total_decided": total_decided,
        "total_failures": total_failures,
        "total_decisions": sum(total_ctx.values()),
        "ctx_total": {context_name(k): v for k, v in sorted(total_ctx.items())},
        "ctx_fallback": {context_name(k): v for k, v in sorted(total_fallback.items())},
        "fallback_decisions": sum(total_fallback.values()),
        "to_hand_bounce_defer": total_bounce,
        "losses": total_losses,
        "losses_by_archetype": by_arch_losses,
        "think_p95_ms": p95(all_think),
        "think_max_ms": max(all_think) if all_think else None,
        "decks": decks,
    }


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:.1f}%" if total else "n/a"


def render_md(rep: dict, compare: Optional[dict] = None) -> str:
    L: list[str] = []
    L.append("# SOT-1694 25デッキ mirror トレース集計 — fallback穴とアーキタイプ別敗因")
    L.append("")
    L.append(f"- ミラー自己対戦: 各デッキ {rep['games_per_deck']} 試合 × {len(rep['decks'])} デッキ = "
             f"{rep['total_games']} 試合 (decided {rep['total_decided']}, failures {rep['total_failures']})")
    L.append(f"- 総決定数: {rep['total_decisions']}, seed={rep['seed']}, traces: `{rep['out_root']}`")
    L.append(f"- 思考時間: p95={rep['think_p95_ms']:.2f}ms max={rep['think_max_ms']:.2f}ms / decision"
             if rep.get("think_p95_ms") is not None else "- 思考時間: n/a")
    L.append("")

    L.append("## (a) CONTEXT_HANDLERS 未登録 context への random fallback")
    L.append("")
    total_dec = rep["total_decisions"]
    L.append(f"未登録contextでの決定 = **{rep['fallback_decisions']}** / {total_dec} "
             f"({_pct(rep['fallback_decisions'], total_dec)})。"
             f"TO_HAND のバウンス効果 defer（形状検出可能な既知の穴）= {rep['to_hand_bounce_defer']} 決定。")
    L.append("")
    L.append("| context | fallback決定数 | 全決定比 |")
    L.append("| --- | ---: | ---: |")
    for name, n in sorted(rep["ctx_fallback"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {name} | {n} | {_pct(n, total_dec)} |")
    L.append("")

    if compare:
        L.append("### 改修前後比較 (before = 現行v2, after = 本Issue改修後)")
        L.append("")
        b_total, a_total = compare["total_decisions"], rep["total_decisions"]
        L.append(f"| context | before ({compare['total_games']}試合) | after ({rep['total_games']}試合) |")
        L.append("| --- | ---: | ---: |")
        names = sorted(set(compare["ctx_fallback"]) | set(rep["ctx_fallback"]),
                       key=lambda n: -compare["ctx_fallback"].get(n, 0))
        for name in names:
            L.append(f"| {name} | {compare['ctx_fallback'].get(name, 0)} | {rep['ctx_fallback'].get(name, 0)} |")
        L.append(f"| **合計 (全決定比)** | **{compare['fallback_decisions']}** ({_pct(compare['fallback_decisions'], b_total)}) "
                 f"| **{rep['fallback_decisions']}** ({_pct(rep['fallback_decisions'], a_total)}) |")
        L.append("")

    L.append("## (b) アーキタイプ別敗因分布")
    L.append("")
    L.append("敗因 = 敗者側の決着理由 (prize_out=サイド取り切り / deck_out=山札切れ / "
             "no_active=場切れ / card_effect=カード効果)。")
    L.append("")
    L.append("| archetype | 決着数 | no_active | prize_out | deck_out | card_effect | other |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for arch, losses in sorted(rep["losses_by_archetype"].items()):
        d = sum(losses.values())
        row = [f"{losses.get(k, 0)} ({_pct(losses.get(k, 0), d)})"
               for k in ("no_active", "prize_out", "deck_out", "card_effect", "other")]
        L.append(f"| {arch} | {d} | " + " | ".join(row) + " |")
    tot = rep["losses"]
    d = sum(tot.values())
    row = [f"{tot.get(k, 0)} ({_pct(tot.get(k, 0), d)})"
           for k in ("no_active", "prize_out", "deck_out", "card_effect", "other")]
    L.append(f"| **全体** | {d} | " + " | ".join(row) + " |")
    L.append("")

    L.append("## デッキ別内訳")
    L.append("")
    L.append("| deck | archetype | games | 平均turn | fallback決定 | no_active | prize_out | deck_out | fault |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dk in rep["decks"]:
        ls = dk["losses"]
        avg_t = f"{dk['avg_turns']:.1f}" if dk["avg_turns"] is not None else "n/a"
        L.append(f"| {dk['deck']} | {dk['archetype']} | {dk['games']} | {avg_t} "
                 f"| {dk['fallback_decisions']} | {ls.get('no_active', 0)} | {ls.get('prize_out', 0)} "
                 f"| {ls.get('deck_out', 0)} | {dk['failures']} |")
    L.append("")
    return "\n".join(L)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="25-deck mirror trace aggregation (SOT-1694).")
    p.add_argument("--decks-glob", default="decks/initial/*.csv")
    p.add_argument("--games-per-deck", type=int, default=20)
    p.add_argument("--seed", type=int, default=1694)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-root", default="eval/traces/sot1694_gap")
    p.add_argument("--skip-run", action="store_true",
                   help="aggregate existing traces under --out-root without replaying")
    p.add_argument("--json", default=None, help="write the raw aggregation JSON here")
    p.add_argument("--md", default=None, help="write the markdown report here")
    p.add_argument("--compare-json", default=None,
                   help="a previous run's JSON; adds a before/after fallback table to the markdown")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    rep = run_and_aggregate(args)

    compare: Optional[dict] = None
    if args.compare_json:
        with open(args.compare_json, encoding="utf-8") as fh:
            compare = json.load(fh)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(render_md(rep, compare))

    print(f"GAP REPORT: games={rep['total_games']} decisions={rep['total_decisions']}"
          f" fallback={rep['fallback_decisions']} to_hand_bounce={rep['to_hand_bounce_defer']}"
          f" failures={rep['total_failures']}"
          f" losses={rep['losses']}")
    for name, n in sorted(rep["ctx_fallback"].items(), key=lambda kv: -kv[1])[:10]:
        print(f"  fallback {name}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
