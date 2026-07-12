"""Standalone tests for the recording runner (SOT-1618).

No pytest dependency — run directly:
    venv/bin/python eval/test_record_match.py

Covers the three acceptance criteria:
  1. a match trace JSONL contains meta (engine hash + schema version), every
     decision (legal moves + choice + search_begin_input), all logs, and the
     result with reason;
  2. battle_finish() is called even when an agent raises (fault injection);
  3. the record level is switchable (RESULT vs LOGS vs FULL_OBS).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from cg import game  # noqa: E402
from eval import record_match as rm  # noqa: E402
from eval.trace import FAIL_AGENT_EXCEPTION, RecordLevel, SCHEMA_VERSION  # noqa: E402


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _decks():
    d0 = rm.load_deck("deck.csv")
    return d0, d0


def test_full_trace_contents():
    d0, d1 = _decks()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "match.jsonl")
        summary = rm.record_match(d0, d1, out_path=out, level=RecordLevel.LOGS)
        recs = _read_jsonl(out)

        meta = recs[0]
        assert meta["kind"] == "meta", meta["kind"]
        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["engine"]["sha256"] and len(meta["engine"]["sha256"]) == 64, meta["engine"]
        assert len(meta["decks"]) == 2 and len(meta["decks"][0]) == 60
        assert len(meta["agents"]) == 2

        decisions = [r for r in recs if r["kind"] == "decision"]
        result = [r for r in recs if r["kind"] == "result"]
        assert len(result) == 1, "exactly one result record"
        assert decisions, "at least one decision recorded"

        d0rec = decisions[0]
        assert d0rec["select"] is not None and "option" in d0rec["select"], "legal moves present"
        assert isinstance(d0rec["choice"], list), "choice recorded"
        assert d0rec["search_begin_input"], "search_begin_input recorded (E5)"
        assert "logs" in d0rec, "event logs present"

        res = result[0]
        # Trace decision count and result must match the runner summary (stdout parity).
        assert res["total_decisions"] == summary["decisions"] == len(decisions)
        assert res["result"] == summary["result"]
        assert res["result"] in (0, 1, 2), f"finished match, got {res['result']}"
        assert res["reason"] in (1, 2, 3, 4), f"reason present, got {res['reason']}"
        # The RESULT log (LogType 23) must be captured in the final logs.
        assert any(l.get("type") == 23 for l in res["final_logs"]), "RESULT log captured"
    print("PASS test_full_trace_contents")


def test_battle_finish_on_exception():
    d0, d1 = _decks()
    calls = {"finish": 0}
    original = game.battle_finish

    def counting_finish():
        calls["finish"] += 1
        return original()

    game.battle_finish = counting_finish
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "match.jsonl")
            agents = (rm.make_raising_agent(after=0, name="boom"), rm.make_random_agent(1))
            summary = rm.record_match(d0, d1, agents=agents, out_path=out, level=RecordLevel.LOGS)
            recs = _read_jsonl(out)
    finally:
        game.battle_finish = original

    assert calls["finish"] == 1, f"battle_finish must be called exactly once, got {calls['finish']}"
    res = [r for r in recs if r["kind"] == "result"][0]
    assert res["failure"] is not None, "failure recorded"
    assert res["failure"]["category"] == FAIL_AGENT_EXCEPTION, res["failure"]
    assert res["failure"]["player"] == 0, res["failure"]
    assert res["winner"] == 1, "non-failing player scored the win"
    assert summary["failure"]["category"] == FAIL_AGENT_EXCEPTION
    print("PASS test_battle_finish_on_exception")


def test_record_levels():
    d0, d1 = _decks()
    with tempfile.TemporaryDirectory() as tmp:
        out_r = os.path.join(tmp, "r.jsonl")
        out_l = os.path.join(tmp, "l.jsonl")
        out_f = os.path.join(tmp, "f.jsonl")
        rm.record_match(d0, d1, out_path=out_r, level=RecordLevel.RESULT)
        rm.record_match(d0, d1, out_path=out_l, level=RecordLevel.LOGS)
        rm.record_match(d0, d1, out_path=out_f, level=RecordLevel.FULL_OBS)

        r, l, f = _read_jsonl(out_r), _read_jsonl(out_l), _read_jsonl(out_f)
        # RESULT level: no decision records emitted (still meta + result).
        assert [x["kind"] for x in r] == ["meta", "result"], [x["kind"] for x in r]
        # LOGS level: decision records present, but no raw obs dump.
        l_dec = [x for x in l if x["kind"] == "decision"]
        assert l_dec and "obs" not in l_dec[0]
        # FULL_OBS level: decision records carry the full raw observation.
        f_dec = [x for x in f if x["kind"] == "decision"]
        assert f_dec and "obs" in f_dec[0] and f_dec[0]["obs"].get("current") is not None
        # RESULT still reports a real decision count even though it emits no rows.
        # (The engine takes no seed — E1 — so counts differ across separate matches;
        # assert each trace is internally consistent rather than equal across levels.)
        r_result = [x for x in r if x["kind"] == "result"][0]
        l_result = [x for x in l if x["kind"] == "result"][0]
        assert r_result["total_decisions"] > 0
        assert l_result["total_decisions"] == len(l_dec)
    print("PASS test_record_levels")


if __name__ == "__main__":
    test_full_trace_contents()
    test_battle_finish_on_exception()
    test_record_levels()
    print("ALL TESTS PASSED")
