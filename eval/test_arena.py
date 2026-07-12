"""Standalone tests for the multi-match arena (SOT-1619).

No pytest dependency — run directly:
    venv/bin/python eval/test_arena.py

Covers the three acceptance criteria:
  1. one CLI-equivalent call runs N side-swap matches in parallel and saves a
     trace per match (N=100 → 100 traces), with 50:50 A/B seat balance;
  2. an abnormal match (agent exception) is recorded as a failure-category loss;
  3. no battle_finish leak under parallel execution (every trace has exactly one
     result record and a finished match).

Also unit-tests the pure classification/aggregation/pairing/timeout helpers so
the failure taxonomy (incl. timeout) is exercised without a real hang.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval import arena  # noqa: E402
from eval.record_match import load_deck  # noqa: E402
from eval.trace import (  # noqa: E402
    FAIL_AGENT_EXCEPTION,
    FAIL_TIMEOUT,
    RecordLevel,
)


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _deck():
    return load_deck("deck.csv")


def test_side_swap_pairing():
    """build_match_specs rounds up to even and balances seats 50:50 per pair."""
    d = _deck()
    a = arena.agent_spec("random", name="A")
    b = arena.agent_spec("random", name="B")
    with tempfile.TemporaryDirectory() as tmp:
        # Odd request → rounded up to an even number of matches.
        specs = arena.build_match_specs(
            games=7, deck_a=d, deck_b=d, agent_a=a, agent_b=b,
            out_dir=tmp, level=RecordLevel.RESULT, max_steps=100, base_seed=123,
        )
    assert len(specs) == 8, f"7 rounded up to 8, got {len(specs)}"
    seat0 = sum(1 for s in specs if s.a_seat == 0)
    seat1 = sum(1 for s in specs if s.a_seat == 1)
    assert seat0 == seat1 == 4, f"A must take seat 0 and seat 1 equally, got {seat0}/{seat1}"
    # Each pair is one A-first + one B-first match.
    for pair in range(4):
        members = [s for s in specs if s.pair_index == pair]
        assert sorted(s.a_seat for s in members) == [0, 1], f"pair {pair} not a proper swap"
    # Deterministic seed derivation: same base seed → identical seeds.
    specs2 = arena.build_match_specs(
        games=8, deck_a=d, deck_b=d, agent_a=a, agent_b=b,
        out_dir="x", level=RecordLevel.RESULT, max_steps=100, base_seed=123,
    )
    assert [s.agent0["seed"] for s in specs] == [s.agent0["seed"] for s in specs2], "seed derivation not reproducible"
    print("PASS test_side_swap_pairing")


def test_classify_and_aggregate():
    """Pure A/B classification incl. the failure taxonomy (agent_exception + timeout)."""
    # Agent A in seat 0 wins on the engine result.
    r = arena.classify({"result": 0, "failure": None}, a_seat=0)
    assert r["outcome"] == "A_win" and r["winner_seat"] == 0

    # Agent A in seat 1; seat-0 (=B) wins.
    r = arena.classify({"result": 0, "failure": None}, a_seat=1)
    assert r["outcome"] == "B_win"

    # Agent exception by seat 0 while A is in seat 0 → A loses, category recorded.
    r = arena.classify(
        {"result": -1, "failure": {"player": 0, "category": FAIL_AGENT_EXCEPTION}}, a_seat=0
    )
    assert r["outcome"] == "B_win" and r["failure_category"] == FAIL_AGENT_EXCEPTION
    assert r["failed_agent"] == "A" and r["winner_seat"] == 1

    # Timeout scored as a loss for the offending seat.
    r = arena.classify(
        {"result": -1, "failure": {"player": 1, "category": FAIL_TIMEOUT}}, a_seat=0
    )
    assert r["outcome"] == "A_win" and r["failure_category"] == FAIL_TIMEOUT
    assert r["failed_agent"] == "B"

    # A genuine draw.
    r = arena.classify({"result": 2, "failure": None}, a_seat=0)
    assert r["outcome"] == "draw" and r["winner_seat"] is None

    agg = arena.aggregate([
        arena.classify({"result": 0, "failure": None}, a_seat=0),  # A_win, A seat0
        arena.classify({"result": 0, "failure": None}, a_seat=1),  # B_win, A seat1
        arena.classify({"result": -1, "failure": {"player": 0, "category": FAIL_TIMEOUT}}, a_seat=1),  # seat0(=B) fails → A(seat1) wins
    ])
    assert agg["total"] == 3
    assert agg["a_wins"] == 2 and agg["b_wins"] == 1
    assert agg["a_seat0"] == 1 and agg["a_seat1"] == 2
    assert agg["failures"] == 1 and agg["failures_by_category"][FAIL_TIMEOUT] == 1
    print("PASS test_classify_and_aggregate")


def test_arena_runs_parallel_and_saves_traces():
    """Criterion 1 & 3: N matches run in parallel; a trace per match; no leak."""
    d = _deck()
    n = 20  # keep CI fast; 100 is validated manually via the CLI
    with tempfile.TemporaryDirectory() as tmp:
        report = arena.run_arena(
            games=n, deck_a=d, deck_b=d,
            agent_a=arena.agent_spec("random", name="A"),
            agent_b=arena.agent_spec("random", name="B"),
            out_dir=tmp, level=RecordLevel.RESULT, max_steps=100000,
            base_seed=7, workers=8,
        )
        traces = [f for f in os.listdir(tmp) if f.endswith(".jsonl")]
        assert len(traces) == n, f"expected {n} traces, got {len(traces)}"
        assert report["total"] == n
        # Seat balance guaranteed by construction.
        assert report["side_balanced"] and report["a_seat0"] == report["a_seat1"] == n // 2
        # Every decided match has a winner; totals are consistent.
        assert report["a_wins"] + report["b_wins"] + report["draws"] + report["undecided"] == n
        # No leak: every trace ends with exactly one finished result record.
        for name in traces:
            recs = _read_jsonl(os.path.join(tmp, name))
            results = [r for r in recs if r["kind"] == "result"]
            assert len(results) == 1, f"{name}: expected 1 result record, got {len(results)}"
            res = results[0]
            # RESULT level with random agents → a real finish (0/1/2), not truncation.
            assert res["result"] in (0, 1, 2), f"{name}: unfinished match result={res['result']}"
    print("PASS test_arena_runs_parallel_and_saves_traces")


def test_arena_records_agent_exception_as_loss():
    """Criterion 2: an agent that raises is scored as a failure-category loss."""
    d = _deck()
    with tempfile.TemporaryDirectory() as tmp:
        # Agent A raises immediately; B is a normal random agent. Run several pairs
        # so A occupies both seats — the failure must always be charged to A.
        report = arena.run_arena(
            games=6, deck_a=d, deck_b=d,
            agent_a=arena.agent_spec("raising", name="boomA", after=0),
            agent_b=arena.agent_spec("random", name="B"),
            out_dir=tmp, level=RecordLevel.LOGS, max_steps=100000,
            base_seed=1, workers=4,
        )
        assert report["total"] == 6
        # Every match should be an agent_exception charged to A → all B wins.
        assert report["failures"] == 6, report
        assert report["failures_by_category"].get(FAIL_AGENT_EXCEPTION) == 6, report
        assert report["failures_by_agent"]["A"] == 6, report
        assert report["b_wins"] == 6 and report["a_wins"] == 0, report
        # The failure category is persisted in each trace's result record too.
        for m in report["matches"]:
            recs = _read_jsonl(m["out_path"])
            res = [r for r in recs if r["kind"] == "result"][0]
            assert res["failure"] and res["failure"]["category"] == FAIL_AGENT_EXCEPTION
    print("PASS test_arena_records_agent_exception_as_loss")


if __name__ == "__main__":
    test_side_swap_pairing()
    test_classify_and_aggregate()
    test_arena_runs_parallel_and_saves_traces()
    test_arena_records_agent_exception_as_loss()
    print("ALL TESTS PASSED")
