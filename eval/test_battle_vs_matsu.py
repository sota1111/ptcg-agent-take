"""Tests for the 対松 loss-cause harness pure helpers (SOT-1698).

No pytest dependency — run directly from the repo root:
    venv/bin/python eval/test_battle_vs_matsu.py

Covers only the deterministic, engine-free helpers of ``eval/battle_vs_matsu.py``
(the subprocess battle loop itself needs the sibling ``ptcg-agent-matsu`` repo and
is exercised manually — see the module docstring). Guards: Wilson CI, RESULT-log
reason extraction, mirror deck scheduling, and the LossTally record / merge /
to_dict / from_dict / aggregate round-trip that the shard aggregation relies on.
"""
from __future__ import annotations

import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval.battle_vs_matsu import (  # noqa: E402
    LossTally,
    RESULT_LOG_TYPE,
    aggregate_reports,
    build_deck_schedule,
    extract_reason,
    wilson_ci,
)


def test_wilson_ci_bounds_and_degenerate():
    lo, hi = wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 1.0), (lo, hi)
    lo, hi = wilson_ci(50, 100)
    assert 0.0 <= lo < 0.5 < hi <= 1.0, (lo, hi)
    # a lopsided sample keeps the interval inside [0, 1]
    lo, hi = wilson_ci(1, 100)
    assert 0.0 <= lo <= hi <= 1.0 and hi < 0.1, (lo, hi)
    print("PASS test_wilson_ci_bounds_and_degenerate")


def test_extract_reason_reads_result_log():
    obs = {"logs": [{"type": 1}, {"type": RESULT_LOG_TYPE, "reason": 2}]}
    assert extract_reason(obs) == 2
    assert extract_reason({"logs": []}) is None
    assert extract_reason({}) is None
    print("PASS test_extract_reason_reads_result_log")


def test_build_deck_schedule_mirrors_pairs():
    rng = random.Random(1698)
    sched = build_deck_schedule(6, ["a", "b", "c"], rng)
    assert len(sched) == 6, sched
    # each 先後 pair (2k, 2k+1) reuses one deck (mirror)
    for k in range(0, 6, 2):
        assert sched[k] == sched[k + 1], (k, sched)
    # odd n: last unpaired match still gets a deck, no crash
    assert len(build_deck_schedule(5, ["a"], rng)) == 5
    print("PASS test_build_deck_schedule_mirrors_pairs")


def test_losstally_record_and_dict():
    t = LossTally()
    t.record(take_won=False, reason=1, fault_by=None)   # 竹 loss by prize_out
    t.record(take_won=False, reason=2, fault_by=None)   # 竹 loss by deck_out
    t.record(take_won=True, reason=1, fault_by=None)    # 竹 win (松 loses by prize)
    t.record(take_won=None, reason=-2, fault_by=None)   # draw
    t.record(take_won=None, reason=None, fault_by="take")  # unfinished + fault
    assert t.decided == 3 and t.take_wins == 1 and t.matsu_wins == 2
    assert t.draws == 1 and t.unfinished == 1 and t.take_faults == 1
    assert t.take_losses == {"prize_out": 1, "deck_out": 1}
    d = t.to_dict()
    assert d["decided"] == 3 and d["take_win_rate"] == round(1 / 3, 4)
    assert d["faults"] == {"take": 1, "matsu": 0}
    # round-trip through from_dict preserves the decided-match tallies
    t2 = LossTally.from_dict(d)
    assert t2.take_wins == 1 and t2.matsu_wins == 2
    assert t2.take_losses == {"prize_out": 1, "deck_out": 1}
    print("PASS test_losstally_record_and_dict")


def test_merge_and_aggregate_reports():
    a, b = LossTally(), LossTally()
    a.record(take_won=False, reason=1, fault_by=None)
    b.record(take_won=True, reason=3, fault_by=None)
    a.merge(b)
    assert a.decided == 2 and a.take_wins == 1 and a.matsu_wins == 1

    rep = {"n": 4, "seed": 1, "decks_dir": "decks/initial", "pool_size": 25,
           "overall": a.to_dict(), "per_deck": {"x.csv": a.to_dict()}}
    agg = aggregate_reports([rep, rep])
    assert agg["n"] == 8 and agg["aggregated_from"] == 2
    assert agg["overall"]["decided"] == 4          # merged twice
    assert agg["per_deck"]["x.csv"]["decided"] == 4
    print("PASS test_merge_and_aggregate_reports")


if __name__ == "__main__":
    test_wilson_ci_bounds_and_degenerate()
    test_extract_reason_reads_result_log()
    test_build_deck_schedule_mirrors_pairs()
    test_losstally_record_and_dict()
    test_merge_and_aggregate_reports()
    print("ALL SOT-1698 HARNESS TESTS PASSED")
