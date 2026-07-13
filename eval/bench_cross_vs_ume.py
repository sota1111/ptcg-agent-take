"""Cross-repo battle bench: take agent+deck vs ume agent+deck (SOT-1668, C1恒久化).

Permanent home of the SOT-1665 cross-battle bridge: the sibling repo
``ptcg-agent-ume`` hosts the arena (``eval/arena.run_arena``) and the engine,
and this repo's rule stack is merged in via ``agents.__path__`` extension —
the take/ume module names do not collide (take: rule_based/damage/base,
ume: rule_agent/rule_scoring/...), so both agent stacks import side by side.

Two modes:

* ``--mode mirror`` — both sides play the same deck (default: ume ``deck.csv``,
  byte-identical to ours). This is the exact SOT-1665 configuration
  (``side_swap=True``, one ``run_arena`` per seed) and doubles as the
  current-deck baseline. NOTE the engine has no seed API (its shuffles are
  non-deterministic), so "reproducing SOT-1665" means statistical consistency
  (overlapping Wilson CIs / same conclusion), never bit-equal numbers.
* ``--mode cross`` — take deck × take agent vs ume deck × ume agent.
  ``run_arena`` binds ``deck0``/``deck1`` to *seats* (side-swap moves the
  agents, not the decks), so agent-bound decks are realised as two
  ``side_swap=False`` half-runs per seed with the seat-0/seat-1 roles swapped
  together (the engine always moves seat 0 first ⇒ the pair is an exact
  先後入替 design).

Beyond win rate / Wilson CI / 先手・後攻内訳 (straight from the arena reports),
the LOGS-level traces are post-processed into a game-shape profile per side:

* step distribution of decided matches (split by which side won);
* prizes taken per match (a prize take is a ``MOVE_CARD`` log leaving
  ``AreaType.PRIZE``, deduplicated on ``(playerIndex, serial)`` so the same
  event seen from both perspectives counts once);
* prize *batch* sizes — prize takes landing in one decision's log window come
  from one KO, so batch size reads out the prize economy directly
  (single-prize KO ⇒ 1, ex ⇒ 2, Mega ex ⇒ 3);
* engine result reasons (1: prize-out, 2: deck-out, 3: no active, 4: effect).

Prize counts are exact for every event up to each side's last observation; the
winner's final take can fall after it (the match ends immediately), so winner
prize counts are reported as observed-at-least values while loser counts are
exact. Safety (illegal output / fault counts) comes from the arena reports.

Usage (from this repo's root; the ume checkout must exist as a sibling):

    venv/bin/python eval/bench_cross_vs_ume.py --mode mirror --n 400 --seeds 0,1,2
    venv/bin/python eval/bench_cross_vs_ume.py --mode cross --n 400 --seeds 0,1 \
        --take-deck decks/deck_tempo_metal.csv \
        --ume-deck /workspaces/ptcg-agent-ume/decks/deck_tank_heal_v1.csv

Writes per-run arena artifacts under ``--out-dir`` (default ``/tmp/sot1668``)
and one pooled ``summary_<mode>.json``; prints the pooled summary to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Iterable, Optional

TAKE_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_UME_REPO = os.path.join(os.path.dirname(TAKE_REPO), "ptcg-agent-ume")

PRIZE_AREA = 6      # cg.api.AreaType.PRIZE
LOG_MOVE_CARD = 6   # cg.api.LogType.MOVE_CARD
PRIZE_TOTAL = 6

TAKE_LABEL = "take_rule"
UME_LABEL = "ume_rule"


# --------------------------------------------------------------------------- #
# Pure helpers — stdlib only, unit-testable without the engine or the ume repo
# --------------------------------------------------------------------------- #

DECK_SIZE = 60


def load_deck(path: str) -> list[int]:
    """Read a deck CSV (one card id per line, blank lines ignored).

    At most the first ``DECK_SIZE`` ids are used — the same semantics as the ume
    ``eval.deck_eval.load_deck`` / submission loader, so a file with trailing
    extra lines (e.g. ume ``decks/deck_tank_heal_v1.csv``, 62 lines) is read
    exactly as its own evaluations read it.
    """
    with open(path, encoding="utf-8") as fh:
        return [int(line.strip()) for line in fh if line.strip()][:DECK_SIZE]


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for ``wins/n`` (0..1); (0.0, 1.0) when n=0."""
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, centre - half), min(1.0, centre + half))


def prize_take_batches(trace_records: Iterable[dict]) -> dict[int, list[int]]:
    """Per-player prize-take batch sizes from one match's trace records.

    A prize take is a ``MOVE_CARD`` log with ``fromArea == PRIZE``; the taker is
    its ``playerIndex``. The same take can appear in both players' log streams
    (and again in the terminal record's ``final_logs``), so events are
    deduplicated on ``(playerIndex, serial)``. Takes seen inside one record's
    log window belong to one KO and form one batch. Returns
    ``{player: [batch sizes in order]}``; total prizes taken = ``sum(batches)``.
    """
    seen: set[tuple[int, int]] = set()
    batches: dict[int, list[int]] = {0: [], 1: []}
    for rec in trace_records:
        if rec.get("kind") == "decision":
            logs = rec.get("logs") or []
        elif rec.get("kind") == "result":
            logs = rec.get("final_logs") or []
        else:
            continue
        counts = {0: 0, 1: 0}
        for log in logs:
            if not isinstance(log, dict) or log.get("type") != LOG_MOVE_CARD:
                continue
            if log.get("fromArea") != PRIZE_AREA:
                continue
            player = log.get("playerIndex")
            serial = log.get("serial")
            if player not in (0, 1) or serial is None:
                continue
            if (player, serial) in seen:
                continue
            seen.add((player, serial))
            counts[player] += 1
        for player, k in counts.items():
            if k:
                batches[player].append(k)
    return batches


def summarize_steps(steps: list[int]) -> dict:
    """Compact distribution summary (n/mean/median/p25/p75/min/max)."""
    if not steps:
        return {"n": 0}
    ordered = sorted(steps)
    q = statistics.quantiles(ordered, n=4) if len(ordered) >= 2 else [ordered[0]] * 3
    return {
        "n": len(ordered),
        "mean": round(statistics.mean(ordered), 1),
        "median": statistics.median(ordered),
        "p25": round(q[0], 1),
        "p75": round(q[2], 1),
        "min": ordered[0],
        "max": ordered[-1],
    }


def hist(values: Iterable[int]) -> dict[str, int]:
    """Sorted value → count histogram with string keys (JSON-stable)."""
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def pool_matches(matches: list[dict]) -> dict:
    """Aggregate per-match rows into the pooled cross-battle summary.

    Each row: ``take_won``/``ume_won``/``draw`` (bools), ``take_first`` (bool),
    ``steps`` (int), ``reason`` (engine reason code or None), ``take_prizes``/
    ``ume_prizes`` (ints), ``take_batches``/``ume_batches`` (lists of ints).
    """
    decided = [m for m in matches if m["take_won"] or m["ume_won"]]
    take_wins = sum(1 for m in decided if m["take_won"])
    n = len(matches)

    def seat_stats(first: bool) -> dict:
        sub = [m for m in decided if m["take_first"] == first]
        wins = sum(1 for m in sub if m["take_won"])
        lo, hi = wilson_ci(wins, len(sub))
        return {"n": len(sub), "take_wins": wins,
                "take_win_rate": round(wins / len(sub), 4) if sub else None,
                "wilson_ci95": [round(lo, 4), round(hi, 4)]}

    lo, hi = wilson_ci(take_wins, len(decided))
    return {
        "n_matches": n,
        "decided": len(decided),
        "draws": n - len(decided),
        "take_wins": take_wins,
        "ume_wins": len(decided) - take_wins,
        "take_win_rate": round(take_wins / len(decided), 4) if decided else None,
        "wilson_ci95": [round(lo, 4), round(hi, 4)],
        "take_as_first": seat_stats(True),
        "take_as_second": seat_stats(False),
        "result_reasons": hist(m["reason"] for m in decided if m["reason"] is not None),
        "steps": {
            "all_decided": summarize_steps([m["steps"] for m in decided]),
            "take_wins": summarize_steps([m["steps"] for m in decided if m["take_won"]]),
            "ume_wins": summarize_steps([m["steps"] for m in decided if m["ume_won"]]),
        },
        "prizes": {
            "take_taken_mean": round(statistics.mean([m["take_prizes"] for m in matches]), 2) if matches else None,
            "ume_taken_mean": round(statistics.mean([m["ume_prizes"] for m in matches]), 2) if matches else None,
            "take_taken_when_ume_wins": hist(m["take_prizes"] for m in decided if m["ume_won"]),
            "ume_taken_when_take_wins": hist(m["ume_prizes"] for m in decided if m["take_won"]),
            "take_batch_sizes": hist(b for m in matches for b in m["take_batches"]),
            "ume_batch_sizes": hist(b for m in matches for b in m["ume_batches"]),
        },
    }


# --------------------------------------------------------------------------- #
# Engine-bound part — imports resolve against the ume checkout at call time
# --------------------------------------------------------------------------- #

def bootstrap(ume_repo: str) -> dict:
    """Put the ume checkout first on ``sys.path``, merge this repo's agents in,
    and return the engine-bound symbols. ``chdir``s into the ume repo so its
    native engine/data files resolve (do this before touching relative paths).
    """
    ume_repo = os.path.abspath(ume_repo)
    if not os.path.isdir(os.path.join(ume_repo, "eval")):
        raise SystemExit(f"ume repo not found at {ume_repo} (--ume-repo)")
    sys.path.insert(0, ume_repo)
    os.chdir(ume_repo)

    import agents  # ume's package (first on sys.path)
    take_agents = os.path.join(TAKE_REPO, "agents")
    if take_agents not in agents.__path__:
        agents.__path__.append(take_agents)

    from agents.rule_agent import RuleAgent          # ume rule stack
    from agents.rule_based import RuleBasedAgent     # take rule stack (merged path)
    from cg.api import to_observation_class
    from eval.arena import run_arena

    class TakeRuleAdapter:
        """take ``decide(Observation)`` exposed through the ume ``act(dict)`` API."""

        name = TAKE_LABEL
        version = "scoring"

        def __init__(self, seed: Optional[int], deck_path: str):
            self.inner = RuleBasedAgent(seed=seed, deck_path=deck_path, policy="scoring")

        def act(self, obs: dict) -> Any:
            return self.inner.decide(to_observation_class(obs))

    return {
        "run_arena": run_arena,
        "RuleAgent": RuleAgent,
        "TakeRuleAdapter": TakeRuleAdapter,
        "ume_repo": ume_repo,
    }


def match_rows(report: Any, take_is_a: bool) -> list[dict]:
    """Flatten one arena report's records + traces into ``pool_matches`` rows."""
    rows = []
    for rec in report.records:
        batches: dict[int, list[int]] = {0: [], 1: []}
        reason = None
        if rec.trace_path and os.path.exists(rec.trace_path):
            with open(rec.trace_path, encoding="utf-8") as fh:
                trace = [json.loads(line) for line in fh if line.strip()]
            batches = prize_take_batches(trace)
            for r in trace:
                if r.get("kind") == "result":
                    reason = r.get("reason")
        take_seat = rec.seat_of_a if take_is_a else 1 - rec.seat_of_a
        take_won = rec.a_won if take_is_a else rec.b_won
        ume_won = rec.b_won if take_is_a else rec.a_won
        rows.append({
            "take_won": take_won,
            "ume_won": ume_won,
            "draw": rec.draw,
            "take_first": rec.first_player == take_seat,
            "steps": rec.steps,
            "reason": reason,
            "take_prizes": sum(batches[take_seat]),
            "ume_prizes": sum(batches[1 - take_seat]),
            "take_batches": batches[take_seat],
            "ume_batches": batches[1 - take_seat],
        })
    return rows


def merge_safety(reports: list[Any]) -> dict:
    """Sum the arena safety counters across runs (illegal outputs / faults)."""
    faults = 0
    undecided = 0
    categories: dict[str, int] = {}
    for rep in reports:
        s = rep.safety
        faults += (s.get("a_faults") or 0) + (s.get("b_faults") or 0)
        undecided += s.get("undecided") or 0
        for key in ("a_fault_categories", "b_fault_categories"):
            for cat, k in (s.get(key) or {}).items():
                categories[cat] = categories.get(cat, 0) + k
    return {"faults_total": faults, "undecided_total": undecided,
            "fault_categories": categories}


def run_mirror(mods: dict, deck_path: str, take_deck_path: str,
               n: int, seed: int, out_dir: str) -> Any:
    """One SOT-1665-shaped mirror run: same deck both seats, side_swap on."""
    deck = load_deck(deck_path)
    adapter = mods["TakeRuleAdapter"]
    return mods["run_arena"](
        lambda s: adapter(s, take_deck_path),
        lambda s: mods["RuleAgent"](seed=s),
        deck0=deck, n_matches=n, side_swap=True, agent_seed=seed,
        label_a=TAKE_LABEL, label_b=UME_LABEL,
        out_dir=out_dir, run_label=f"mirror-seed{seed}",
    )


def run_cross(mods: dict, take_deck_path: str, ume_deck_path: str,
              n: int, seed: int, out_dir: str) -> tuple[Any, Any]:
    """One cross-deck seed: two side_swap=False half-runs, roles swapped together.

    Half A: take (its deck) on seat 0 = 先手; half B: ume (its deck) on seat 0.
    """
    take_deck = load_deck(take_deck_path)
    ume_deck = load_deck(ume_deck_path)
    adapter = mods["TakeRuleAdapter"]
    half = max(1, n // 2)
    rep_a = mods["run_arena"](
        lambda s: adapter(s, take_deck_path),
        lambda s: mods["RuleAgent"](seed=s),
        deck0=take_deck, deck1=ume_deck, n_matches=half, side_swap=False,
        agent_seed=seed, label_a=TAKE_LABEL, label_b=UME_LABEL,
        out_dir=out_dir, run_label=f"cross-seed{seed}-takefirst",
    )
    rep_b = mods["run_arena"](
        lambda s: mods["RuleAgent"](seed=s),
        lambda s: adapter(s, take_deck_path),
        deck0=ume_deck, deck1=take_deck, n_matches=half, side_swap=False,
        agent_seed=seed, label_a=UME_LABEL, label_b=TAKE_LABEL,
        out_dir=out_dir, run_label=f"cross-seed{seed}-umefirst",
    )
    return rep_a, rep_b


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--mode", choices=("mirror", "cross"), required=True)
    p.add_argument("--n", type=int, default=400, help="matches per seed")
    p.add_argument("--seeds", default="0,1", help="comma-separated agent seeds")
    p.add_argument("--ume-repo", default=DEFAULT_UME_REPO)
    p.add_argument("--take-deck", default=None,
                   help="take-side deck CSV (default: this repo's deck.csv; "
                        "in mirror mode this is only the agent's own-deck hint)")
    p.add_argument("--ume-deck", default=None,
                   help="ume-side deck CSV (default: ume repo deck.csv)")
    p.add_argument("--mirror-deck", default=None,
                   help="mirror mode: the shared deck (default: ume repo deck.csv)")
    p.add_argument("--out-dir", default="/tmp/sot1668")
    args = p.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    out_dir = os.path.abspath(args.out_dir)
    take_deck = os.path.abspath(args.take_deck or os.path.join(TAKE_REPO, "deck.csv"))
    mods = bootstrap(args.ume_repo)  # chdir! resolve all paths before use below
    ume_deck = os.path.abspath(args.ume_deck or os.path.join(mods["ume_repo"], "deck.csv"))
    mirror_deck = os.path.abspath(args.mirror_deck) if args.mirror_deck \
        else os.path.join(mods["ume_repo"], "deck.csv")

    all_rows: list[dict] = []
    reports: list[Any] = []
    per_seed: list[dict] = []
    for seed in seeds:
        if args.mode == "mirror":
            rep = run_mirror(mods, mirror_deck, take_deck, args.n, seed, out_dir)
            seed_reports = [rep]
            rows = match_rows(rep, take_is_a=True)
        else:
            rep_a, rep_b = run_cross(mods, take_deck, ume_deck, args.n, seed, out_dir)
            seed_reports = [rep_a, rep_b]
            rows = match_rows(rep_a, take_is_a=True) + match_rows(rep_b, take_is_a=False)
        reports.extend(seed_reports)
        all_rows.extend(rows)
        seed_pool = pool_matches(rows)
        per_seed.append({"seed": seed,
                         "take_win_rate": seed_pool["take_win_rate"],
                         "wilson_ci95": seed_pool["wilson_ci95"],
                         "n": seed_pool["n_matches"]})
        print(f"[seed {seed}] take {seed_pool['take_wins']}/{seed_pool['decided']}"
              f" = {seed_pool['take_win_rate']} CI95 {seed_pool['wilson_ci95']}",
              file=sys.stderr)

    summary = {
        "mode": args.mode,
        "take_deck": take_deck,
        "ume_deck": ume_deck if args.mode == "cross" else mirror_deck,
        "n_per_seed": args.n,
        "seeds": seeds,
        "per_seed": per_seed,
        "pooled": pool_matches(all_rows),
        "safety": merge_safety(reports),
        "note": "engine has no seed API; mirror-mode reproduction of SOT-1665 "
                "is statistical (CI overlap), not bit-exact",
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"summary_{args.mode}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary written to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
