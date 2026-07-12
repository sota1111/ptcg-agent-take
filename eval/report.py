"""Trace-aggregation report for the PTCG eval environment (SOT-1620).

Turns a directory of match traces (the JSONL files produced by
``eval/record_match.py`` / ``eval/arena.py``, SOT-1618 / SOT-1619) into a
statistical summary that answers "which is stronger" and "why did it win/lose"
with more rigour than a bare win count:

* **Win rate + Wilson 95% confidence interval** per agent (agent evaluation).
  Draws (``result == 2``), truncated/undecided matches (``result == -1``) and
  **abnormal losses** (agent/engine/worker exception, timeout) are tallied
  *separately* from normal decided games so they never distort the win rate.
* **Decision-reason distribution** (1 = side taken / 2 = deck-out /
  3 = no active Pokémon / 4 = card effect).
* **Turn / decision / per-decision thinking-time** distributions — the last is
  the Kaggle time-limit watch value, and is only available when traces were
  recorded at ``--level logs`` (RESULT-level traces carry no decision records).
* **First/second-player win rate** — the side-swap pairing of the arena lets us
  read seat (first-player) bias directly.
* **Deck × deck matchup table** — win rate aggregated over agents, so deck
  strength is separated from agent strength.

The aggregation functions are pure (they take already-parsed records) so the
numbers are unit-tested against hand-computed data without needing the engine.

Usage:
    venv/bin/python eval/report.py eval/traces/arena_<ts>
    venv/bin/python eval/report.py <dir> [--json report.json] [--z 1.96]

``<dir>`` is scanned recursively for ``*.jsonl`` trace files.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from typing import Any, Optional

REASON_NAMES = {
    1: "side_taken (サイド取り切り)",
    2: "deck_out (山札切れ)",
    3: "no_active (バトル場不在)",
    4: "card_effect (カード効果)",
}

# Failure categories that score the offending player as an abnormal loss.
# Mirrors eval.trace / eval.arena so the report classifies exactly as the arena did.
LOSS_FAILURES = {"agent_exception", "engine_error", "timeout", "worker_error"}

# Outcome classes a normalized trace is bucketed into.
WIN = "win"            # a normal decided game (result 0/1, no failure)
DRAW = "draw"          # result == 2, no failure
TRUNCATED = "truncated"  # result == -1, no failure (hit max_steps / undecided)
ABNORMAL = "abnormal"  # any match carrying a failure record


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k`` successes of ``n``.

    Returns ``(low, high)`` clamped to [0, 1]. ``n == 0`` → ``(0.0, 0.0)``.
    The Wilson interval is used instead of the normal (Wald) approximation
    because it stays valid for small ``n`` and proportions near 0/1.
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _dist(values: list[float]) -> Optional[dict]:
    """Summary stats (n/mean/min/median/p95/max) for a list of numbers."""
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)

    def pct(p: float) -> float:
        # nearest-rank percentile
        if n == 1:
            return xs[0]
        idx = min(n - 1, max(0, int(math.ceil(p / 100.0 * n)) - 1))
        return xs[idx]

    return {
        "n": n,
        "mean": sum(xs) / n,
        "min": xs[0],
        "median": xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2,
        "p95": pct(95),
        "max": xs[-1],
    }


# --------------------------------------------------------------------------- #
# Trace loading + normalization
# --------------------------------------------------------------------------- #

def find_trace_files(root: str) -> list[str]:
    """Return the sorted list of ``*.jsonl`` trace files under ``root``.

    ``root`` may be a single ``.jsonl`` file or a directory (scanned
    recursively). ``report.json`` and other non-``.jsonl`` files are ignored.
    """
    if os.path.isfile(root):
        return [root] if root.endswith(".jsonl") else []
    return sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True))


def _deck_fingerprint(deck: Any) -> Optional[str]:
    """Stable short id for a deck (list of card ids). Order-sensitive."""
    if not deck:
        return None
    blob = json.dumps(deck, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def normalize_records(records: list[dict], *, source: Optional[str] = None) -> Optional[dict]:
    """Fold one trace's JSON records into a flat, classified summary.

    Needs the ``meta`` (seat agents/decks) and ``result`` (outcome/reason)
    records; ``decision`` records are optional (present only at LOGS+ level) and
    contribute the per-decision thinking times. Returns ``None`` for a trace
    with no terminal ``result`` record (partial/corrupt).
    """
    meta = next((r for r in records if r.get("kind") == "meta"), None)
    result = next((r for r in records if r.get("kind") == "result"), None)
    if result is None:
        return None

    agents = (meta or {}).get("agents") or []
    seat_names = [None, None]
    for a in agents:
        idx = a.get("index")
        if idx in (0, 1):
            seat_names[idx] = a.get("name")

    decks = (meta or {}).get("decks") or []
    deck_fps = [_deck_fingerprint(decks[i]) if i < len(decks) else None for i in (0, 1)]

    res = result.get("result", -1)
    failure = result.get("failure")
    reason = result.get("reason")
    first_player = result.get("first_player")
    if first_player not in (0, 1):
        first_player = None

    # Classify — mirrors eval.trace.build_result / eval.arena.winner_seat so the
    # report agrees with the arena's own scoring.
    winner_seat: Optional[int] = None
    failed_seat: Optional[int] = None
    failure_category: Optional[str] = None
    if failure:
        failure_category = failure.get("category")
        player = failure.get("player")
        outcome = ABNORMAL
        if player in (0, 1):
            failed_seat = player
            if failure_category in LOSS_FAILURES:
                winner_seat = 1 - player
    elif res in (0, 1):
        outcome = WIN
        winner_seat = res
    elif res == 2:
        outcome = DRAW
    else:
        outcome = TRUNCATED

    thinking = [
        r.get("thinking_time_ms")
        for r in records
        if r.get("kind") == "decision" and isinstance(r.get("thinking_time_ms"), (int, float))
    ]

    return {
        "source": source,
        "seat_names": seat_names,
        "deck_fingerprints": deck_fps,
        "first_player": first_player,
        "result": res,
        "reason": reason,
        "outcome": outcome,
        "winner_seat": winner_seat,
        "failed_seat": failed_seat,
        "failure_category": failure_category,
        "final_turn": result.get("final_turn"),
        "total_decisions": result.get("total_decisions"),
        "elapsed_ms": result.get("elapsed_ms"),
        "thinking_times": thinking,
    }


def load_trace_file(path: str) -> Optional[dict]:
    """Parse one trace JSONL file into a normalized summary (or ``None``)."""
    try:
        with open(path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    return normalize_records(records, source=os.path.basename(path))


# --------------------------------------------------------------------------- #
# Aggregation (pure — unit-tested without the engine)
# --------------------------------------------------------------------------- #

def build_report(records: list[dict], *, z: float = 1.96) -> dict:
    """Aggregate normalized trace summaries into the full statistical report."""
    total = len(records)
    counts = {WIN: 0, DRAW: 0, TRUNCATED: 0, ABNORMAL: 0}

    # Per-agent win/loss over NORMAL decided games only.
    per_agent: dict[str, dict] = {}

    def agent(name: Optional[str]) -> dict:
        key = name if name is not None else "?"
        return per_agent.setdefault(
            key, {"wins": 0, "losses": 0, "abnormal_losses": 0}
        )

    reason_counts: dict[int, int] = {}
    failures_by_category: dict[str, int] = {}
    failures_by_agent: dict[str, int] = {}

    first_wins = second_wins = seat_decided = 0
    turns: list[float] = []
    decisions: list[float] = []
    think: list[float] = []

    # Deck matchup: wins[winner_fp][loser_fp] over normal decided games.
    deck_labels: dict[str, str] = {}      # fingerprint -> D0/D1/...
    deck_wins: dict[tuple[str, str], int] = {}

    def deck_label(fp: Optional[str]) -> Optional[str]:
        if fp is None:
            return None
        if fp not in deck_labels:
            deck_labels[fp] = f"D{len(deck_labels)}"
        return deck_labels[fp]

    for r in records:
        outcome = r["outcome"]
        counts[outcome] += 1
        think.extend(r.get("thinking_times") or [])

        if isinstance(r.get("reason"), int) and r["reason"] in REASON_NAMES:
            reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1

        if outcome == ABNORMAL:
            cat = r.get("failure_category")
            if cat:
                failures_by_category[cat] = failures_by_category.get(cat, 0) + 1
            fs = r.get("failed_seat")
            if fs in (0, 1):
                loser_name = r["seat_names"][fs]
                agent(loser_name)["abnormal_losses"] += 1
                key = loser_name if loser_name is not None else "?"
                failures_by_agent[key] = failures_by_agent.get(key, 0) + 1
            continue

        if outcome != WIN:
            continue

        # --- normal decided game ---
        ws = r["winner_seat"]
        ls = 1 - ws
        agent(r["seat_names"][ws])["wins"] += 1
        agent(r["seat_names"][ls])["losses"] += 1

        if r.get("final_turn") is not None:
            turns.append(r["final_turn"])
        if r.get("total_decisions") is not None:
            decisions.append(r["total_decisions"])

        fp = r.get("first_player")
        if fp in (0, 1):
            seat_decided += 1
            if ws == fp:
                first_wins += 1
            else:
                second_wins += 1

        wfp = r["deck_fingerprints"][ws]
        lfp = r["deck_fingerprints"][ls]
        wl, ll = deck_label(wfp), deck_label(lfp)
        if wl is not None and ll is not None:
            deck_wins[(wl, ll)] = deck_wins.get((wl, ll), 0) + 1

    # Finalise per-agent win rates + Wilson CI.
    agents_out: dict[str, dict] = {}
    for name, a in sorted(per_agent.items()):
        decided = a["wins"] + a["losses"]
        lo, hi = wilson_interval(a["wins"], decided, z=z)
        agents_out[name] = {
            "wins": a["wins"],
            "losses": a["losses"],
            "decided": decided,
            "winrate": (a["wins"] / decided) if decided else None,
            "ci_low": lo if decided else None,
            "ci_high": hi if decided else None,
            "abnormal_losses": a["abnormal_losses"],
        }

    # First/second-player win rate.
    fl, fh = wilson_interval(first_wins, seat_decided, z=z)
    seat = {
        "decided": seat_decided,
        "first_wins": first_wins,
        "second_wins": second_wins,
        "first_winrate": (first_wins / seat_decided) if seat_decided else None,
        "first_ci_low": fl if seat_decided else None,
        "first_ci_high": fh if seat_decided else None,
    }

    # Deck matchup table (win rate of row-deck vs column-deck, both orientations).
    labels = sorted(deck_labels.values())
    matchup = {}
    for row in labels:
        matchup[row] = {}
        for col in labels:
            if row == col:
                matchup[row][col] = None
                continue
            w = deck_wins.get((row, col), 0)
            l = deck_wins.get((col, row), 0)
            n = w + l
            matchup[row][col] = {
                "wins": w,
                "games": n,
                "winrate": (w / n) if n else None,
            }

    return {
        "total_traces": total,
        "z": z,
        "counts": counts,
        "agents": agents_out,
        "reason_distribution": {k: reason_counts.get(k, 0) for k in sorted(reason_counts)},
        "seat": seat,
        "turns": _dist(turns),
        "decisions": _dist(decisions),
        "thinking_time_ms": _dist(think),
        "failures_by_category": failures_by_category,
        "failures_by_agent": failures_by_agent,
        "deck_labels": {label: fp for fp, label in deck_labels.items()},
        "matchup": matchup,
    }


# --------------------------------------------------------------------------- #
# Text formatting
# --------------------------------------------------------------------------- #

def _fmt_pct(x: Optional[float]) -> str:
    return f"{x * 100:5.1f}%" if x is not None else "  n/a"


def _fmt_dist(d: Optional[dict], unit: str = "") -> str:
    if not d:
        return "n/a (no data)"
    return (
        f"n={d['n']} mean={d['mean']:.2f}{unit} min={d['min']:.2f} "
        f"median={d['median']:.2f} p95={d['p95']:.2f} max={d['max']:.2f}{unit}"
    )


def format_report(report: dict) -> str:
    c = report["counts"]
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("PTCG TRACE AGGREGATION REPORT (SOT-1620)")
    lines.append("=" * 70)
    lines.append(
        f"traces={report['total_traces']}  "
        f"decided={c[WIN]}  draws={c[DRAW]}  truncated={c[TRUNCATED]}  abnormal={c[ABNORMAL]}"
    )
    z = report["z"]

    lines.append("")
    lines.append(f"-- Agent win rate (normal decided games, Wilson {z} CI) --")
    if report["agents"]:
        lines.append(f"  {'agent':<16} {'W-L':>9} {'winrate':>8}  {'95% CI':>16}  abn.loss")
        for name, a in report["agents"].items():
            ci = (
                f"[{a['ci_low']*100:4.1f},{a['ci_high']*100:5.1f}]"
                if a["ci_low"] is not None else "n/a"
            )
            lines.append(
                f"  {name:<16} {a['wins']:>4}-{a['losses']:<4} "
                f"{_fmt_pct(a['winrate'])}  {ci:>16}  {a['abnormal_losses']}"
            )
    else:
        lines.append("  (no decided games)")

    lines.append("")
    lines.append("-- Decision reason distribution --")
    total_reason = sum(report["reason_distribution"].values())
    if total_reason:
        for code, n in report["reason_distribution"].items():
            frac = n / total_reason
            lines.append(f"  {code} {REASON_NAMES.get(code, '?'):<26} {n:>5}  ({frac*100:4.1f}%)")
    else:
        lines.append("  (no reason-coded results)")

    lines.append("")
    lines.append("-- First/second-player win rate --")
    s = report["seat"]
    if s["decided"]:
        ci = f"[{s['first_ci_low']*100:4.1f},{s['first_ci_high']*100:5.1f}]"
        lines.append(
            f"  first-player: {s['first_wins']}/{s['decided']} = "
            f"{_fmt_pct(s['first_winrate'])}  95% CI {ci}   (second: {s['second_wins']})"
        )
    else:
        lines.append("  (first-player unknown in all traces)")

    lines.append("")
    lines.append("-- Timing / length --")
    lines.append(f"  turns          : {_fmt_dist(report['turns'])}")
    lines.append(f"  decisions/match: {_fmt_dist(report['decisions'])}")
    lines.append(f"  think/decision : {_fmt_dist(report['thinking_time_ms'], 'ms')}")

    lines.append("")
    lines.append("-- Deck x deck matchup (row win rate vs column) --")
    labels = sorted(report["matchup"].keys())
    if len(labels) >= 2:
        for label in labels:
            fp = report["deck_labels"].get(label)
            lines.append(f"  {label} = deck {fp}")
        header = "        " + "".join(f"{c:>10}" for c in labels)
        lines.append(header)
        for row in labels:
            cells = []
            for col in labels:
                cell = report["matchup"][row][col]
                if cell is None:
                    cells.append(f"{'--':>10}")
                elif cell["winrate"] is None:
                    cells.append(f"{'n/a':>10}")
                else:
                    cells.append(f"{cell['winrate']*100:6.1f}%({cell['games']})"[:10].rjust(10))
            lines.append(f"  {row:<6}" + "".join(cells))
    else:
        lines.append("  (need >=2 distinct decks for a matchup table)")

    if report["failures_by_category"]:
        lines.append("")
        lines.append("-- Abnormal losses --")
        lines.append(f"  by category: {report['failures_by_category']}")
        lines.append(f"  by agent   : {report['failures_by_agent']}")

    lines.append("=" * 70)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def generate(root: str, *, z: float = 1.96) -> dict:
    """Load every trace under ``root`` and build the report."""
    paths = find_trace_files(root)
    records = [rec for rec in (load_trace_file(p) for p in paths) if rec is not None]
    report = build_report(records, z=z)
    report["parsed_traces"] = len(records)
    report["scanned_files"] = len(paths)
    report["source"] = root
    return report


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate PTCG match traces into a statistical report.")
    p.add_argument("traces", help="trace directory (scanned recursively) or a single .jsonl file")
    p.add_argument("--json", default=None, help="also write the raw JSON report to this path")
    p.add_argument("--z", type=float, default=1.96, help="z for the win-rate CI (default 1.96 = 95%%)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    import sys
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not os.path.exists(args.traces):
        print(f"error: no such path: {args.traces}", file=sys.stderr)
        return 2
    report = generate(args.traces, z=args.z)
    if report["parsed_traces"] == 0:
        print(f"error: no parseable traces under {args.traces}", file=sys.stderr)
        return 1
    print(format_report(report))
    print(f"\n(scanned {report['scanned_files']} files, parsed {report['parsed_traces']} traces)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"wrote JSON report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
