"""Standalone tests for the search-vs-rule benchmark (SOT-1659).

No pytest dependency — run directly:
    venv/bin/python eval/test_bench_search_vs_rule.py

Covers:
  1. per-move thinking-time attribution — decisions are credited to the *search*
     agent by seat via the trace meta, regardless of which seat it occupied, and
     the other agent's / non-decision records are ignored (pure, no engine);
  2. an end-to-end small run: the search agent is registered in the arena, plays
     the rule agent over a handful of side-swap games, every match completes, and
     the search agent's per-move thinking times are collected.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval import bench_search_vs_rule as bench  # noqa: E402


def _write_trace(path, agents, decisions):
    """Write a minimal LOGS-level trace: meta + decision records + a result."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "meta", "agents": agents}) + "\n")
        for d in decisions:
            fh.write(json.dumps({"kind": "decision", **d}) + "\n")
        fh.write(json.dumps({"kind": "result", "result": 0}) + "\n")


def test_thinking_time_attribution():
    """Only the search agent's decisions count, attributed by seat both ways."""
    with tempfile.TemporaryDirectory() as tmp:
        # Trace 1: search in seat 0.
        _write_trace(
            os.path.join(tmp, "m0.jsonl"),
            agents=[{"index": 0, "name": "search"}, {"index": 1, "name": "rule"}],
            decisions=[
                {"your_index": 0, "select_player": 0, "thinking_time_ms": 5.0},
                {"your_index": 1, "select_player": 1, "thinking_time_ms": 99.0},  # rule → ignored
                {"your_index": 0, "select_player": 0, "thinking_time_ms": 7.0},
            ],
        )
        # Trace 2: search SWAPPED into seat 1 (side-swap) — still attributed to search.
        _write_trace(
            os.path.join(tmp, "m1.jsonl"),
            agents=[{"index": 0, "name": "rule"}, {"index": 1, "name": "search"}],
            decisions=[
                {"your_index": 0, "select_player": 0, "thinking_time_ms": 99.0},  # rule → ignored
                {"your_index": 1, "select_player": 1, "thinking_time_ms": 9.0},
                # your_index missing → fall back to select_player for the seat.
                {"your_index": None, "select_player": 1, "thinking_time_ms": 3.0},
            ],
        )
        times = sorted(bench.search_thinking_times(tmp))
    assert times == [3.0, 5.0, 7.0, 9.0], times
    d = bench._dist(times)
    assert d["n"] == 4 and d["max"] == 9.0 and abs(d["mean"] - 6.0) < 1e-9, d
    print("PASS test_thinking_time_attribution")


def test_bench_end_to_end_small():
    """A small real run registers+plays the search agent and records its timing."""
    r = bench.run_bench(games=4, seed=7, workers=2, z=1.96,
                        time_budget_s=0.2, max_candidates=8)
    assert r["games"] == 4, r
    assert r["wins"] + r["losses"] + r["draws"] + r["undecided"] == 4, r
    # The search agent makes at least one recorded decision → thinking stats exist.
    assert r["search_thinking_ms"] is not None, r
    assert r["search_thinking_ms"]["n"] >= 1, r
    assert r["search_thinking_ms"]["max"] >= r["search_thinking_ms"]["mean"] >= 0.0, r
    # Wilson CI is a valid sub-interval of [0, 1] bracketing nothing impossible.
    assert 0.0 <= r["ci_low"] <= r["ci_high"] <= 1.0, r
    # 4 < 200 so the validity gate must NOT pass on this tiny smoke run.
    assert r["passed"] is False, r
    print("PASS test_bench_end_to_end_small")


if __name__ == "__main__":
    test_thinking_time_attribution()
    test_bench_end_to_end_small()
    print("ALL TESTS PASSED")
