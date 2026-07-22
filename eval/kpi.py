"""Take battle KPI recording (SOT-1709).

Defines the KPI record schema (``take-kpi-v1``, see ``docs/KPI.md``), computes
KPI records from measurements or existing bench reports, and appends one line
per measurement to the committed history file ``eval/kpi_history.jsonl`` —
kept separate from the gitignored ``eval/traces/`` scratch tree so trends
survive across sessions.

Two ways to produce a record:

1. **Own measurement** (full KPI coverage) — a 25-deck side-swapped mirror
   arena of the current working-tree agent (A) vs the **pinned baseline**
   agent materialised from a fixed git ref (``BASELINE_REF``, the SOT-1694
   champion lineage). Per-match traces are recorded at LOGS level so A-side
   decisions can be attributed via ``select_player`` (fallback holes, thinking
   time) and A's losses classified by the engine's terminal ``reason``:

       venv/bin/python eval/kpi.py --measure --games-per-deck 2 --issue SOT-1709

2. **From an existing 25-deck rotation report** (win rate / faults / timing
   only; ``prize_out_loss_rate`` and ``fallback_decision_rate`` are null
   because that harness does not seat-attribute decisions):

       venv/bin/python eval/kpi.py --from-report report.json --issue SOT-1694

   ``eval/bench_25deck_rotation.py --kpi [ISSUE]`` does the same conversion
   in-process as a hook.

History and comparison display: ``eval/kpi_report.py``.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from eval.report import wilson_interval  # noqa: E402  (engine-free)

SCHEMA = "take-kpi-v1"
HISTORY_PATH = os.path.join(REPO, "eval", "kpi_history.jsonl")

# The fixed comparison baseline: the SOT-1682→SOT-1694 champion lineage as of
# the SOT-1700 merge. Pinned to a SHA (not `main`) so the KPI trend measures
# the *current* agent against a constant opponent; see docs/KPI.md for the
# baseline-update policy.
BASELINE_REF = "b51da4f9aad000f7c7bbcc1c8cc00acfa377485e"

# Improvement direction per KPI: +1 higher is better, -1 lower is better,
# 0 must stay exactly zero (any nonzero value is a regression).
KPI_DIRECTIONS = {
    "mirror_winrate_vs_baseline": 1,
    "prize_out_loss_rate": -1,
    "fallback_decision_rate": -1,
    "fault_total": 0,
    "decision_time_mean_ms": -1,
}

# Engine terminal-result reason codes (cg/api.py RESULT log).
REASONS = {1: "prize_out", 2: "deck_out", 3: "no_active", 4: "card_effect"}


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _base_record(issue: Optional[str], source: str) -> dict:
    return {
        "schema": SCHEMA,
        "ts": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "issue": issue or "unknown",
        "source": source,
    }


def build_record(stats: dict, *, issue: Optional[str],
                 source: str = "kpi-measure", baseline_ref: str = BASELINE_REF,
                 baseline_sha: Optional[str] = None,
                 deck_pool: str = "decks/rotation_baseline/*.csv",
                 n_decks: Optional[int] = None, seed=None,
                 games_per_deck: Optional[int] = None) -> dict:
    """KPI record from the flat counters accumulated by :func:`run_measure`.

    ``stats`` keys (all A = current agent, B = baseline):
    wins/losses/draws/undecided, loss_causes {cause: n} (A's losses;
    "abnormal" = failure-scored, excluded from the prize_out denominator),
    failures/failures_by_agent/failures_by_category, a_fallback/a_decisions,
    think_ms_sum/think_n/think_ms_max.
    """
    wins, losses = stats.get("wins", 0), stats.get("losses", 0)
    decided = wins + losses
    causes = dict(stats.get("loss_causes") or {})
    normal_losses = sum(n for c, n in causes.items() if c != "abnormal")
    prize_out = causes.get("prize_out", 0)
    a_dec = stats.get("a_decisions", 0)
    a_fb = stats.get("a_fallback", 0)
    think_n = stats.get("think_n", 0)
    rec = _base_record(issue, source)
    rec.update({
        "baseline_ref": baseline_ref,
        "baseline_sha": baseline_sha,
        "deck_pool": deck_pool,
        "n_decks": n_decks,
        "n_matches": (decided + stats.get("draws", 0)
                      + stats.get("undecided", 0)),
        "games_per_deck": games_per_deck,
        "seed": seed,
        "kpis": {
            "mirror_winrate_vs_baseline": {
                "value": round(wins / decided, 4) if decided else None,
                "ci95": [round(x, 4) for x in wilson_interval(wins, decided)],
                "wins": wins, "losses": losses,
                "draws": stats.get("draws", 0),
                "undecided": stats.get("undecided", 0),
            },
            "prize_out_loss_rate": {
                "value": (round(prize_out / normal_losses, 4)
                          if normal_losses else None),
                "prize_out_losses": prize_out,
                "normal_losses": normal_losses,
                "loss_causes": causes,
            },
            "fallback_decision_rate": {
                "value": round(a_fb / a_dec, 5) if a_dec else None,
                "fallback_decisions": a_fb,
                "decisions": a_dec,
            },
            "fault_total": {
                "value": stats.get("failures", 0),
                "by_agent": stats.get("failures_by_agent")
                or {"take": 0, "baseline": 0},
                "by_category": stats.get("failures_by_category") or {},
            },
            "decision_time_mean_ms": {
                "value": (round(stats.get("think_ms_sum", 0.0) / think_n, 3)
                          if think_n else None),
                "max_ms": round(stats.get("think_ms_max", 0.0), 3),
                "n_decisions": think_n,
            },
        },
    })
    return rec


def record_from_rotation(report: dict, issue: Optional[str] = None) -> dict:
    """KPI record from a ``bench_25deck_rotation.py`` report dict.

    That harness aggregates decisions without seat attribution and discards
    loss reasons, so only the win-rate / fault / timing KPIs are filled;
    ``prize_out_loss_rate`` and ``fallback_decision_rate`` stay null.
    """
    fba = report.get("failures_by_agent") or {}
    rec = _base_record(issue, "bench_25deck_rotation")
    rec.update({
        "baseline_ref": report.get("old_ref"),
        "baseline_sha": None,
        "deck_pool": "decks/rotation_baseline/*.csv",
        "n_decks": len(report.get("decks") or []) or None,
        "n_matches": report.get("total_games"),
        "games_per_deck": report.get("games_per_deck"),
        "seed": report.get("seed"),
        "kpis": {
            "mirror_winrate_vs_baseline": {
                "value": (round(report["winrate"], 4)
                          if report.get("winrate") is not None else None),
                "ci95": ([round(x, 4) for x in report["ci"]]
                         if report.get("ci") else None),
                "wins": report.get("wins", 0),
                "losses": report.get("losses", 0),
                "draws": report.get("draws", 0),
                "undecided": report.get("undecided", 0),
            },
            "prize_out_loss_rate": {
                "value": None,
                "note": "loss reasons not captured by this harness",
            },
            "fallback_decision_rate": {
                "value": None,
                "fallback_decisions": sum(
                    d.get("fallback_decisions", 0)
                    for d in report.get("decks") or []),
                "note": "decisions not seat-attributed by this harness",
            },
            "fault_total": {
                "value": report.get("failures", 0),
                "by_agent": {"take": fba.get("new", 0),
                             "baseline": fba.get("old", 0)},
                "by_category": {},
            },
            "decision_time_mean_ms": {
                "value": None,
                "p95_ms": report.get("think_p95_ms"),
                "max_ms": report.get("think_max_ms"),
                "note": "harness reports p95/max only (both seats)",
            },
        },
    })
    return rec


def append_history(record: dict, path: str = HISTORY_PATH) -> str:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_history(path: str = HISTORY_PATH) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- measurement

def scan_match_trace(path: str, a_seat: int, handled: set) -> dict:
    """A-side decision counters + terminal reason from one match's JSONL trace."""
    out = {"a_decisions": 0, "a_fallback": 0, "think_ms_sum": 0.0,
           "think_n": 0, "think_ms_max": 0.0, "reason": None}
    try:
        fh = open(path, encoding="utf-8")
    except OSError:
        return out
    with fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("kind")
            if kind == "decision" and rec.get("select_player") == a_seat:
                out["a_decisions"] += 1
                ctx = (rec.get("select") or {}).get("context")
                if ctx is not None and ctx not in handled:
                    out["a_fallback"] += 1
                tt = rec.get("thinking_time_ms")
                if isinstance(tt, (int, float)):
                    out["think_ms_sum"] += float(tt)
                    out["think_n"] += 1
                    out["think_ms_max"] = max(out["think_ms_max"], float(tt))
            elif kind == "result":
                out["reason"] = rec.get("reason")
    return out


def run_measure(args) -> int:
    """25-deck mirror arena: working-tree agent (A) vs pinned baseline (B)."""
    from eval.arena import agent_spec, run_arena
    from eval.old_agent import materialize_agents_ref, resolve_ref
    from eval.record_match import load_deck
    from eval.trace import RecordLevel
    from eval.trace_gap_report import handled_contexts

    deck_paths = sorted(glob.glob(os.path.join(REPO, args.decks_glob))
                        if not os.path.isabs(args.decks_glob)
                        else glob.glob(args.decks_glob))
    if not deck_paths:
        raise SystemExit(f"no decks match {args.decks_glob!r}")
    handled = handled_contexts()
    baseline_sha = resolve_ref(args.baseline_ref)[:8]

    out_root = args.out_root
    if out_root is None:
        stamp = datetime.datetime.now(datetime.timezone.utc)\
            .strftime("%Y%m%dT%H%M%SZ")
        out_root = os.path.join("eval", "traces", f"kpi_{stamp}")

    stats = {"wins": 0, "losses": 0, "draws": 0, "undecided": 0,
             "loss_causes": {}, "failures": 0,
             "failures_by_agent": {"take": 0, "baseline": 0},
             "failures_by_category": {}, "a_fallback": 0, "a_decisions": 0,
             "think_ms_sum": 0.0, "think_n": 0, "think_ms_max": 0.0}

    import_root, pkg = materialize_agents_ref(args.baseline_ref)
    try:
        for di, path in enumerate(deck_paths):
            name = os.path.splitext(os.path.basename(path))[0]
            deck = load_deck(path)
            rep = run_arena(
                games=args.games_per_deck,
                deck_a=deck, deck_b=deck,
                agent_a=agent_spec("rule_based", name="take",
                                   policy="scoring", deck_path=path),
                agent_b={"kind": "rule_based_ref",
                         "name": f"baseline@{baseline_sha}",
                         "import_root": import_root, "pkg": pkg,
                         "policy": None, "seed": None},
                out_dir=os.path.join(out_root, name),
                level=RecordLevel.LOGS,
                base_seed=(args.seed + di) if args.seed is not None else None,
                workers=args.workers,
            )
            stats["wins"] += rep["a_wins"]
            stats["losses"] += rep["b_wins"]
            stats["draws"] += rep["draws"]
            stats["undecided"] += rep["undecided"]
            stats["failures"] += rep["failures"]
            stats["failures_by_agent"]["take"] += rep["failures_by_agent"]["A"]
            stats["failures_by_agent"]["baseline"] += \
                rep["failures_by_agent"]["B"]
            for cat, n in rep["failures_by_category"].items():
                stats["failures_by_category"][cat] = \
                    stats["failures_by_category"].get(cat, 0) + n
            for m in rep["matches"]:
                scan = scan_match_trace(m["out_path"], m["a_seat"], handled)
                for k in ("a_decisions", "a_fallback", "think_ms_sum",
                          "think_n"):
                    stats[k] += scan[k]
                stats["think_ms_max"] = max(stats["think_ms_max"],
                                            scan["think_ms_max"])
                if m["outcome"] == "B_win":
                    cause = ("abnormal" if m.get("failure_category")
                             else REASONS.get(scan["reason"], "other"))
                    stats["loss_causes"][cause] = \
                        stats["loss_causes"].get(cause, 0) + 1
            print(f"  {name}: {rep['a_wins']}-{rep['b_wins']}-{rep['draws']}"
                  f" failures={rep['failures']}", flush=True)
    finally:
        shutil.rmtree(import_root, ignore_errors=True)

    rec = build_record(stats, issue=args.issue,
                       baseline_ref=args.baseline_ref,
                       baseline_sha=baseline_sha, deck_pool=args.decks_glob,
                       n_decks=len(deck_paths), seed=args.seed,
                       games_per_deck=args.games_per_deck)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    if args.no_append:
        print("(--no-append: history not written)")
    else:
        print(f"appended to {append_history(rec, args.history)}")
    return 0


def run_from_report(args) -> int:
    with open(args.from_report, encoding="utf-8") as fh:
        report = json.load(fh)
    rec = record_from_rotation(report, issue=args.issue)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    if args.no_append:
        print("(--no-append: history not written)")
    else:
        print(f"appended to {append_history(rec, args.history)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measure", action="store_true",
                   help="play the 25-deck mirror arena vs the pinned baseline")
    p.add_argument("--games-per-deck", type=int, default=2)
    p.add_argument("--decks-glob", default="decks/rotation_baseline/*.csv")
    p.add_argument("--baseline-ref", default=BASELINE_REF)
    p.add_argument("--seed", type=int, default=1709)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-root", default=None,
                   help="trace scratch dir (default: eval/traces/kpi_<ts>)")
    p.add_argument("--from-report", default=None,
                   help="bench_25deck_rotation JSON -> one history record")
    p.add_argument("--issue", default=None, help="Linear issue id to record")
    p.add_argument("--history", default=HISTORY_PATH)
    p.add_argument("--no-append", action="store_true",
                   help="print the record without touching the history")
    args = p.parse_args(argv)
    if args.measure:
        return run_measure(args)
    if args.from_report:
        return run_from_report(args)
    raise SystemExit("one of --measure / --from-report required")


if __name__ == "__main__":
    sys.exit(main())
