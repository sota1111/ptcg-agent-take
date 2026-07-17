"""Take KPI history display + latest-vs-previous trend (SOT-1709).

Reads ``eval/kpi_history.jsonl`` (written by ``eval/kpi.py``) and prints the
time-ordered history table plus a comparison of the two most recent records:
each KPI's delta is judged 改善 (improved) / 悪化 (worsened) / 横ばい (flat)
per its improvement direction in ``kpi.KPI_DIRECTIONS`` (``fault_total`` is a
must-stay-zero gate: nonzero = NG regardless of trend).

Usage (from the repo root):
    venv/bin/python eval/kpi_report.py [--history eval/kpi_history.jsonl]
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from eval.kpi import HISTORY_PATH, KPI_DIRECTIONS, load_history  # noqa: E402

# Below this absolute delta a KPI move counts as 横ばい, not a real trend.
FLAT_EPS = {
    "mirror_winrate_vs_baseline": 0.005,
    "prize_out_loss_rate": 0.01,
    "fallback_decision_rate": 0.001,
    "fault_total": 0,
    "decision_time_mean_ms": 1.0,
}


def kpi_value(record: dict, name: str):
    return (record.get("kpis") or {}).get(name, {}).get("value")


def judge(name: str, prev, latest) -> str:
    """One KPI's trend label between two records' values."""
    direction = KPI_DIRECTIONS[name]
    if direction == 0:
        return "OK(=0)" if latest == 0 else f"NG({latest} != 0)"
    if latest is None or prev is None:
        return "n/a"
    delta = latest - prev
    if abs(delta) <= FLAT_EPS.get(name, 0):
        return "横ばい"
    improved = (delta > 0) == (direction > 0)
    return "改善" if improved else "悪化"


def compare(prev: dict, latest: dict) -> dict:
    """Per-KPI {prev, latest, delta, judgement} between two records."""
    out = {}
    for name in KPI_DIRECTIONS:
        p, l = kpi_value(prev, name), kpi_value(latest, name)
        out[name] = {
            "prev": p, "latest": l,
            "delta": round(l - p, 5) if p is not None and l is not None
            else None,
            "judgement": judge(name, p, l),
        }
    return out


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def print_history(history: list) -> None:
    cols = ("ts", "git_sha", "issue", "source", "N",
            "winrate", "ci95", "przout", "fallbk", "faults", "ms/move")
    rows = []
    for r in history:
        wr = (r.get("kpis") or {}).get("mirror_winrate_vs_baseline", {})
        ci = wr.get("ci95")
        rows.append((
            r.get("ts", "-"), r.get("git_sha", "-"), r.get("issue", "-"),
            r.get("source", "-"), fmt(r.get("n_matches")),
            fmt(wr.get("value")),
            f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "-",
            fmt(kpi_value(r, "prize_out_loss_rate")),
            fmt(kpi_value(r, "fallback_decision_rate")),
            fmt(kpi_value(r, "fault_total")),
            fmt(kpi_value(r, "decision_time_mean_ms")),
        ))
    widths = [max(len(c), *(len(row[i]) for row in rows))
              for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", default=HISTORY_PATH)
    args = p.parse_args(argv)
    history = load_history(args.history)
    if not history:
        print(f"no KPI records in {args.history}")
        return 0
    print(f"KPI history ({len(history)} records, oldest first) "
          f"— definitions: docs/KPI.md\n")
    print_history(history)
    if len(history) < 2:
        print("\n(only one record — no previous measurement to compare)")
        return 0
    prev, latest = history[-2], history[-1]
    print(f"\nlatest vs previous "
          f"({latest.get('ts')}/{latest.get('git_sha')} vs "
          f"{prev.get('ts')}/{prev.get('git_sha')}):")
    for name, c in compare(prev, latest).items():
        print(f"  {name}: {fmt(c['prev'])} -> {fmt(c['latest'])} "
              f"(Δ {fmt(c['delta'])})  {c['judgement']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
