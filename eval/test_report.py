"""Standalone tests for the trace-aggregation report (SOT-1620).

No pytest dependency — run directly:
    venv/bin/python eval/test_report.py

Covers the three acceptance criteria:
  1. a trace directory yields win rate + 95% CI, reason distribution, first/
     second-player win rate and a deck x deck matchup table;
  2. draws, truncated matches and abnormal (failure) losses are tallied
     *separately* from normal decided games;
  3. the aggregate values (win rate, Wilson CI, reason distribution) match a
     hand-computed 10-game dataset.

The Wilson interval is also checked against an independent reference value.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from eval import report  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers to synthesise a trace JSONL file (meta + optional decisions + result)
# --------------------------------------------------------------------------- #

DECK_X = [1, 1, 2, 3]          # two distinct decks so the matchup table is exercised
DECK_Y = [9, 9, 8, 7]


def write_trace(
    path,
    *,
    seat0="A",
    seat1="B",
    deck0=DECK_X,
    deck1=DECK_Y,
    result=0,
    reason=None,
    first_player=None,
    failure=None,
    thinking=None,
):
    """Write a minimal but schema-shaped trace file."""
    recs = [
        {
            "kind": "meta",
            "schema_version": "1.0.0",
            "trace_id": os.path.basename(path),
            "agents": [
                {"index": 0, "name": seat0},
                {"index": 1, "name": seat1},
            ],
            "decks": [deck0, deck1],
            "first_player": first_player if first_player is not None else -1,
        }
    ]
    for i, t in enumerate(thinking or []):
        recs.append({"kind": "decision", "index": i, "thinking_time_ms": t})
    recs.append(
        {
            "kind": "result",
            "result": result,
            "reason": reason,
            "winner": result if result in (0, 1) else None,
            "first_player": first_player if first_player is not None else -1,
            "final_turn": 12,
            "total_decisions": len(thinking or []),
            "elapsed_ms": 10.0,
            "failure": failure,
        }
    )
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_wilson_reference():
    """Wilson 8/10 @ z=1.96 ≈ (0.4902, 0.9433) — independent reference value."""
    lo, hi = report.wilson_interval(8, 10, z=1.96)
    assert abs(lo - 0.490150) < 1e-4, lo
    assert abs(hi - 0.943324) < 1e-4, hi
    # Degenerate cases.
    assert report.wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = report.wilson_interval(10, 10)
    assert hi == 1.0 and 0.0 <= lo <= 1.0
    print("PASS test_wilson_reference")


def test_hand_computed_dataset():
    """A 10-game dataset whose win rate / CI / reason dist can be hand-computed.

    Construct exactly 10 NORMAL decided games: A wins 7, B wins 3. Then add
    separation cases (2 draws, 1 truncated, 1 abnormal loss by A). The win rate
    must be computed over the 10 decided games only.

    Reasons on the 10 wins: reason 1 x5, 2 x3, 3 x2  → distribution {1:5,2:3,3:2}.
    First player set so that the first player wins 6 of the 10 decided games.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # 7 A-wins. A occupies seat 0 (winner_seat 0). Vary first_player.
        # reasons: give reason 1 to five wins, 2 to three, 3 to two across all 10.
        # Track first-player wins: winner_seat==first_player.
        reasons_A = [1, 1, 1, 2, 2, 3, 3]          # 7 A-wins: r1x3,r2x2,r3x2
        # first_player for A-wins (winner seat 0): fp==0 → first-player win.
        fp_A = [0, 0, 0, 0, 1, 1, 1]               # 4 first-player wins, 3 second-player
        for i in range(7):
            write_trace(
                os.path.join(tmp, f"a{i}.jsonl"),
                seat0="A", seat1="B", result=0,
                reason=reasons_A[i], first_player=fp_A[i],
            )
        # 3 B-wins. B in seat 1 (result=1 → winner seat 1).
        reasons_B = [1, 1, 2]                       # r1x2, r2x1
        fp_B = [1, 1, 0]                            # winner seat 1: fp==1 → first-player win (2 of them)
        for i in range(3):
            write_trace(
                os.path.join(tmp, f"b{i}.jsonl"),
                seat0="A", seat1="B", result=1,
                reason=reasons_B[i], first_player=fp_B[i],
            )
        # Separation cases — must NOT count toward win/loss:
        write_trace(os.path.join(tmp, "draw0.jsonl"), result=2, reason=None)
        write_trace(os.path.join(tmp, "draw1.jsonl"), result=2, reason=None)
        write_trace(os.path.join(tmp, "trunc0.jsonl"), result=-1, reason=None)
        write_trace(
            os.path.join(tmp, "abn0.jsonl"), result=-1, reason=None,
            failure={"player": 0, "category": "agent_exception"},
        )

        rep = report.generate(tmp)

    # --- counts / separation (criterion 2) ---
    assert rep["parsed_traces"] == 14, rep["parsed_traces"]
    assert rep["counts"] == {"win": 10, "draw": 2, "truncated": 1, "abnormal": 1}, rep["counts"]

    # --- per-agent win rate + Wilson CI (criteria 1 & 3) ---
    A = rep["agents"]["A"]
    B = rep["agents"]["B"]
    assert A["wins"] == 7 and A["losses"] == 3, A
    assert B["wins"] == 3 and B["losses"] == 7, B
    assert abs(A["winrate"] - 0.7) < 1e-9, A["winrate"]
    # Wilson(7,10) reference ≈ (0.3968, 0.8922).
    assert abs(A["ci_low"] - 0.396800) < 1e-3, A["ci_low"]
    assert abs(A["ci_high"] - 0.892200) < 1e-3, A["ci_high"]
    # The abnormal loss is charged to A (seat 0 failed) and kept out of losses.
    assert A["abnormal_losses"] == 1, A
    assert B["abnormal_losses"] == 0, B

    # --- reason distribution (criterion 3): r1x5, r2x3, r3x2 ---
    assert rep["reason_distribution"] == {1: 5, 2: 3, 3: 2}, rep["reason_distribution"]

    # --- first/second-player win rate: 6 of 10 decided are first-player wins ---
    assert rep["seat"]["decided"] == 10, rep["seat"]
    assert rep["seat"]["first_wins"] == 6 and rep["seat"]["second_wins"] == 4, rep["seat"]
    assert abs(rep["seat"]["first_winrate"] - 0.6) < 1e-9

    # --- abnormal accounting ---
    assert rep["failures_by_category"] == {"agent_exception": 1}, rep["failures_by_category"]
    assert rep["failures_by_agent"] == {"A": 1}, rep["failures_by_agent"]
    print("PASS test_hand_computed_dataset")


def test_matchup_table():
    """Deck x deck matchup aggregates over agents and separates orientations."""
    with tempfile.TemporaryDirectory() as tmp:
        # DECK_X (seat0) beats DECK_Y (seat1) 3 times; DECK_Y beats DECK_X once.
        for i in range(3):
            write_trace(os.path.join(tmp, f"xy{i}.jsonl"),
                        deck0=DECK_X, deck1=DECK_Y, result=0, reason=1)
        write_trace(os.path.join(tmp, "yx0.jsonl"),
                    deck0=DECK_X, deck1=DECK_Y, result=1, reason=1)
        rep = report.generate(tmp)

    labels = sorted(rep["deck_labels"].keys())
    assert len(labels) == 2, rep["deck_labels"]
    # Identify which label maps to DECK_X.
    dx = report._deck_fingerprint(DECK_X)
    lx = next(lbl for lbl, fp in rep["deck_labels"].items() if fp == dx)
    ly = next(lbl for lbl in labels if lbl != lx)
    cell = rep["matchup"][lx][ly]
    assert cell["wins"] == 3 and cell["games"] == 4, cell
    assert abs(cell["winrate"] - 0.75) < 1e-9, cell
    # Reverse orientation is the complement.
    rev = rep["matchup"][ly][lx]
    assert rev["wins"] == 1 and rev["games"] == 4 and abs(rev["winrate"] - 0.25) < 1e-9, rev
    # Mirror diagonal is None.
    assert rep["matchup"][lx][lx] is None
    print("PASS test_matchup_table")


def test_thinking_time_distribution():
    """Per-decision thinking times are aggregated from decision records (LOGS level)."""
    with tempfile.TemporaryDirectory() as tmp:
        write_trace(os.path.join(tmp, "t0.jsonl"), result=0, reason=1,
                    thinking=[1.0, 2.0, 3.0])
        write_trace(os.path.join(tmp, "t1.jsonl"), result=1, reason=1,
                    thinking=[4.0, 5.0])
        rep = report.generate(tmp)
    d = rep["thinking_time_ms"]
    assert d is not None and d["n"] == 5, d
    assert abs(d["mean"] - 3.0) < 1e-9 and d["min"] == 1.0 and d["max"] == 5.0, d

    # RESULT-level traces (no decision records) → no thinking-time data.
    with tempfile.TemporaryDirectory() as tmp:
        write_trace(os.path.join(tmp, "r0.jsonl"), result=0, reason=1)
        rep = report.generate(tmp)
    assert rep["thinking_time_ms"] is None
    print("PASS test_thinking_time_distribution")


def test_format_and_empty():
    """format_report renders without error; empty input is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmp:
        write_trace(os.path.join(tmp, "one.jsonl"), result=0, reason=1, first_player=0)
        rep = report.generate(tmp)
    text = report.format_report(rep)
    assert "AGGREGATION REPORT" in text and "win rate" in text, text

    empty = report.build_report([])
    assert empty["counts"] == {"win": 0, "draw": 0, "truncated": 0, "abnormal": 0}
    assert report.format_report(empty)  # does not raise
    print("PASS test_format_and_empty")


if __name__ == "__main__":
    test_wilson_reference()
    test_hand_computed_dataset()
    test_matchup_table()
    test_thinking_time_distribution()
    test_format_and_empty()
    print("ALL TESTS PASSED")
