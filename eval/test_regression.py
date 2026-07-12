"""Standalone tests for the regression suite (SOT-1636).

No pytest dependency — run directly from the repo root:
    venv/bin/python eval/test_regression.py

Covers:
  1. pure gate logic (summarise_card): "beat" vs "noregress" verdicts;
  2. old-version reference (eval/old_agent): a git ref (HEAD) materialises an
     importable, renamed copy of the agents package whose RuleBasedAgent is a
     *distinct* class from the live one (side-by-side load works);
  3. an end-to-end 2-game suite smoke: two cards run, produce traces at LOGS
     level and a per-decision thinking-time distribution, and every card carries
     a Wilson CI + gate verdict.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval import regression                              # noqa: E402
from eval import old_agent                               # noqa: E402


def _fake_arena_report(a_wins, b_wins, draws=0, undecided=0, total=None):
    return {
        "total": total if total is not None else a_wins + b_wins + draws + undecided,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "undecided": undecided,
        "side_balanced": True,
        "failures": 0,
        "failures_by_category": {},
        "out_dir": "x",
    }


def test_gate_beat():
    # Strong win → beat gate passes; near-50% → fails (lower bound not > 0.5).
    strong = regression.summarise_card(
        "c", _fake_arena_report(180, 20), {"thinking_time_ms": None}, z=1.96, gate="beat")
    assert strong["passed"] is True, strong
    assert strong["ci_low"] > 0.5

    even = regression.summarise_card(
        "c", _fake_arena_report(100, 100), {"thinking_time_ms": None}, z=1.96, gate="beat")
    assert even["passed"] is False, even
    print("ok test_gate_beat")


def test_gate_noregress():
    # No-regression: even ~50% passes (not significantly worse);
    # a clear loss (significantly below 0.5) fails.
    even = regression.summarise_card(
        "c", _fake_arena_report(100, 100), {"thinking_time_ms": None}, z=1.96, gate="noregress")
    assert even["passed"] is True, even
    assert even["ci_high"] >= 0.5

    worse = regression.summarise_card(
        "c", _fake_arena_report(20, 180), {"thinking_time_ms": None}, z=1.96, gate="noregress")
    assert worse["passed"] is False, worse
    assert worse["ci_high"] < 0.5

    # Zero decided games must never crash and is treated as non-regression.
    empty = regression.summarise_card(
        "c", _fake_arena_report(0, 0, draws=4), {"thinking_time_ms": None}, z=1.96, gate="noregress")
    assert empty["passed"] is True and empty["winrate"] is None, empty
    print("ok test_gate_noregress")


def test_old_agent_ref_load():
    # Materialise the current HEAD's agents package under a renamed package and
    # confirm its RuleBasedAgent loads and is a *distinct* class object.
    import_root, pkg = old_agent.materialize_agents_ref("HEAD")
    try:
        assert pkg.startswith("agents_"), pkg
        # Renamed package dir exists; imports were rewritten (no bare `agents.` import
        # of the package root left in rule_based.py).
        rb_src = os.path.join(import_root, pkg, "rule_based.py")
        with open(rb_src, encoding="utf-8") as fh:
            src = fh.read()
        assert f"from {pkg}.base import" in src or f"from {pkg} import" in src, "imports not rewritten"
        assert "\nfrom agents." not in src and "\nfrom agents " not in src, "stale agents import remains"

        old_cls = old_agent.load_rule_based_class(import_root, pkg)
        from agents.rule_based import RuleBasedAgent as CurrentCls
        assert old_cls is not CurrentCls, "old class must be distinct from the live one"
        # Old (HEAD) class still supports the scoring policy kwarg.
        inst = old_cls(seed=1, policy="scoring")
        assert inst.policy == "scoring"
    finally:
        shutil.rmtree(import_root, ignore_errors=True)
        # Drop the temp package from sys.modules so repeated runs stay clean.
        for name in [m for m in list(sys.modules) if m == pkg or m.startswith(pkg + ".")]:
            del sys.modules[name]
    print("ok test_old_agent_ref_load")


def test_suite_smoke():
    # End-to-end: a tiny 2-game suite runs both cards, records LOGS traces with
    # thinking times, and produces Wilson CIs + gate verdicts. Uses a seed so the
    # agent RNG is deterministic (engine outcome is still E1-random).
    out_root = tempfile.mkdtemp(prefix="regr_smoke_")
    try:
        suite = regression.run_suite(
            games=2, seed=7, workers=2, z=1.96, out_root=out_root,
        )
        assert [c["card"] for c in suite["cards"]] == ["rule_vs_random", "new_vs_old"], suite
        for c in suite["cards"]:
            assert c["games"] == 2, c
            assert "ci_low" in c and "ci_high" in c
            assert c["thinking_time_ms"] is not None and c["thinking_time_ms"]["n"] > 0, \
                "LOGS traces must yield per-decision thinking times"
            assert isinstance(c["passed"], bool)
        assert isinstance(suite["passed"], bool)
        # text render must not crash and mentions both cards
        txt = regression.format_suite(suite, think_warn_ms=1000.0)
        assert "rule_vs_random" in txt and "new_vs_old" in txt
    finally:
        shutil.rmtree(out_root, ignore_errors=True)
    print("ok test_suite_smoke")


if __name__ == "__main__":
    test_gate_beat()
    test_gate_noregress()
    test_old_agent_ref_load()
    test_suite_smoke()
    print("ALL REGRESSION TESTS PASSED")
