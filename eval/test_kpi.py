"""Standalone tests for the KPI recording + trend report (SOT-1709).

No pytest dependency — run directly:
    venv/bin/python eval/test_kpi.py

Engine-free: covers the pure record builders, the history append/load
round-trip, the trace scanner on a synthesised JSONL file, and the trend
judgement of ``eval/kpi_report.py``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from eval import kpi, kpi_report  # noqa: E402
from eval.report import wilson_interval  # noqa: E402


def make_stats(**over) -> dict:
    stats = {
        "wins": 30, "losses": 18, "draws": 1, "undecided": 1,
        "loss_causes": {"prize_out": 12, "deck_out": 4, "no_active": 1,
                        "abnormal": 1},
        "failures": 1,
        "failures_by_agent": {"take": 1, "baseline": 0},
        "failures_by_category": {"agent_exception": 1},
        "a_fallback": 25, "a_decisions": 1000,
        "think_ms_sum": 2500.0, "think_n": 1000, "think_ms_max": 9.5,
    }
    stats.update(over)
    return stats


def test_build_record() -> None:
    rec = kpi.build_record(make_stats(), issue="SOT-TEST", seed=7,
                           baseline_sha="abcd1234", n_decks=25,
                           games_per_deck=2)
    assert rec["schema"] == kpi.SCHEMA
    assert rec["issue"] == "SOT-TEST"
    assert rec["n_matches"] == 50
    k = rec["kpis"]
    assert k["mirror_winrate_vs_baseline"]["value"] == round(30 / 48, 4)
    lo, hi = wilson_interval(30, 48)
    assert k["mirror_winrate_vs_baseline"]["ci95"] == [round(lo, 4),
                                                       round(hi, 4)]
    # abnormal loss is excluded from the prize_out denominator (17, not 18)
    assert k["prize_out_loss_rate"]["normal_losses"] == 17
    assert k["prize_out_loss_rate"]["value"] == round(12 / 17, 4)
    assert k["fallback_decision_rate"]["value"] == round(25 / 1000, 5)
    assert k["fault_total"]["value"] == 1
    assert k["decision_time_mean_ms"]["value"] == 2.5
    assert set(k) == set(kpi.KPI_DIRECTIONS)
    print("PASS test_build_record")


def test_build_record_empty() -> None:
    rec = kpi.build_record(
        make_stats(wins=0, losses=0, draws=0, undecided=0, loss_causes={},
                   failures=0, failures_by_category={}, a_fallback=0,
                   a_decisions=0, think_ms_sum=0.0, think_n=0,
                   think_ms_max=0.0,
                   failures_by_agent={"take": 0, "baseline": 0}),
        issue=None)
    k = rec["kpis"]
    assert k["mirror_winrate_vs_baseline"]["value"] is None
    assert k["prize_out_loss_rate"]["value"] is None
    assert k["fallback_decision_rate"]["value"] is None
    assert k["decision_time_mean_ms"]["value"] is None
    assert rec["issue"] == "unknown"
    print("PASS test_build_record_empty")


def test_record_from_rotation() -> None:
    report = {
        "old_ref": "main", "games_per_deck": 20, "seed": 42,
        "total_games": 500, "wins": 260, "losses": 230, "draws": 5,
        "undecided": 5, "winrate": 260 / 490, "ci": [0.4859, 0.5748],
        "failures": 0, "failures_by_agent": {"new": 0, "old": 0},
        "think_p95_ms": 3.2, "think_max_ms": 11.0,
        "decks": [{"fallback_decisions": 3}, {"fallback_decisions": 2}],
    }
    rec = kpi.record_from_rotation(report, issue="SOT-TEST")
    k = rec["kpis"]
    assert rec["source"] == "bench_25deck_rotation"
    assert k["mirror_winrate_vs_baseline"]["value"] == round(260 / 490, 4)
    assert k["prize_out_loss_rate"]["value"] is None
    assert k["fallback_decision_rate"]["value"] is None
    assert k["fallback_decision_rate"]["fallback_decisions"] == 5
    assert k["fault_total"]["value"] == 0
    assert set(k) == set(kpi.KPI_DIRECTIONS)
    print("PASS test_record_from_rotation")


def test_history_roundtrip() -> None:
    rec = kpi.build_record(make_stats(), issue="SOT-TEST")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "kpi_history.jsonl")
        kpi.append_history(rec, path)
        kpi.append_history(rec, path)
        loaded = kpi.load_history(path)
        assert len(loaded) == 2
        assert loaded[0] == rec
        assert kpi.load_history(os.path.join(tmp, "missing.jsonl")) == []
    print("PASS test_history_roundtrip")


def test_scan_match_trace() -> None:
    records = [
        {"kind": "meta", "trace_id": "m0"},
        # A (seat 0): handled context, 2ms
        {"kind": "decision", "select_player": 0,
         "select": {"context": 1}, "thinking_time_ms": 2.0},
        # A: unhandled context -> fallback, 4ms
        {"kind": "decision", "select_player": 0,
         "select": {"context": 999}, "thinking_time_ms": 4.0},
        # B (seat 1): unhandled context — must NOT count for A
        {"kind": "decision", "select_player": 1,
         "select": {"context": 999}, "thinking_time_ms": 50.0},
        {"kind": "result", "winner": 1, "reason": 1},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m0.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        scan = kpi.scan_match_trace(path, a_seat=0, handled={1})
        assert scan["a_decisions"] == 2
        assert scan["a_fallback"] == 1
        assert scan["think_n"] == 2
        assert scan["think_ms_sum"] == 6.0
        assert scan["think_ms_max"] == 4.0
        assert scan["reason"] == 1
        missing = kpi.scan_match_trace(os.path.join(tmp, "nope.jsonl"),
                                       a_seat=0, handled={1})
        assert missing["a_decisions"] == 0 and missing["reason"] is None
    print("PASS test_scan_match_trace")


def test_judge_and_compare() -> None:
    assert kpi_report.judge("mirror_winrate_vs_baseline", 0.50, 0.56) == "改善"
    assert kpi_report.judge("mirror_winrate_vs_baseline", 0.56, 0.50) == "悪化"
    assert kpi_report.judge("mirror_winrate_vs_baseline", 0.50, 0.502) == "横ばい"
    assert kpi_report.judge("prize_out_loss_rate", 0.80, 0.60) == "改善"
    assert kpi_report.judge("fallback_decision_rate", 0.01, 0.02) == "悪化"
    assert kpi_report.judge("fault_total", 0, 0) == "OK(=0)"
    assert kpi_report.judge("fault_total", 0, 3) == "NG(3 != 0)"
    assert kpi_report.judge("decision_time_mean_ms", None, 2.0) == "n/a"

    prev = kpi.build_record(make_stats(wins=24, losses=24), issue="A")
    latest = kpi.build_record(make_stats(), issue="B")
    cmp = kpi_report.compare(prev, latest)
    assert set(cmp) == set(kpi.KPI_DIRECTIONS)
    assert cmp["mirror_winrate_vs_baseline"]["judgement"] == "改善"
    assert cmp["fault_total"]["judgement"] == "NG(1 != 0)"
    print("PASS test_judge_and_compare")


def main() -> int:
    test_build_record()
    test_build_record_empty()
    test_record_from_rotation()
    test_history_roundtrip()
    test_scan_match_trace()
    test_judge_and_compare()
    print("ALL PASS eval/test_kpi.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
