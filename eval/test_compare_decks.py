"""Standalone tests for the candidate-vs-current deck comparison (SOT-1661).

No pytest dependency — run directly:
    venv/bin/python eval/test_compare_decks.py

Covers:
  1. ``deck_result`` — pure win rate + Wilson CI derivation from a synthetic arena
     report (no engine), incl. the no-decided-games (winrate=None) edge case;
  2. ``discover_candidate_decks`` — stable, sorted discovery of ``decks/*.csv`` with
     the current deck excluded when it lives inside the decks dir;
  3. ``format_summary`` — the text block names every candidate and its win rate;
  4. an end-to-end small run: the real candidate decks are each played vs the
     current deck over a handful of side-swap games, every match completes, and
     each candidate gets a win rate + Wilson CI in one summary.
"""
from __future__ import annotations

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval import compare_decks as cd  # noqa: E402


def _report(a_wins, b_wins, draws=0, undecided=0, failures=0, total=None):
    total = total if total is not None else a_wins + b_wins + draws + undecided
    return {
        "total": total,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "undecided": undecided,
        "failures": failures,
        "failures_by_category": {},
        "failures_by_agent": {"A": 0, "B": 0},
        "side_balanced": True,
        "out_dir": "eval/traces/x",
    }


def test_deck_result_winrate_and_ci():
    """Win rate is over decided games; CI is a valid sub-interval of [0, 1]."""
    r = cd.deck_result("cand", "decks/cand.csv", _report(120, 80, draws=0, total=200))
    assert r["deck"] == "cand" and r["games"] == 200, r
    assert r["decided"] == 200 and abs(r["winrate"] - 0.6) < 1e-9, r
    assert 0.0 <= r["ci_low"] <= r["winrate"] <= r["ci_high"] <= 1.0, r
    # draws/undecided don't enter the decided count.
    r2 = cd.deck_result("c2", "p", _report(3, 1, draws=2, undecided=4, total=10))
    assert r2["decided"] == 4 and abs(r2["winrate"] - 0.75) < 1e-9, r2
    print("PASS test_deck_result_winrate_and_ci")


def test_deck_result_no_decided():
    """No decided games -> winrate None and CI (0, 0), not a crash."""
    r = cd.deck_result("c", "p", _report(0, 0, draws=2, undecided=2, total=4))
    assert r["winrate"] is None, r
    assert r["ci_low"] == 0.0 and r["ci_high"] == 0.0, r
    print("PASS test_deck_result_no_decided")


def test_discover_candidate_decks():
    """Sorted discovery of *.csv, with the current deck excluded when inside."""
    with tempfile.TemporaryDirectory() as tmp:
        for fn in ("deck_b.csv", "deck_a.csv", "current.csv", "notes.txt"):
            with open(os.path.join(tmp, fn), "w") as fh:
                fh.write("1\n")
        found = cd.discover_candidate_decks(tmp, current_path=os.path.join(tmp, "current.csv"))
        names = [n for n, _ in found]
        assert names == ["deck_a", "deck_b"], names           # sorted, .txt & current excluded
        # Without excluding current, it is included.
        all_names = [n for n, _ in cd.discover_candidate_decks(tmp)]
        assert all_names == ["current", "deck_a", "deck_b"], all_names
    print("PASS test_discover_candidate_decks")


def test_format_summary_lists_candidates():
    """The text summary names every candidate and shows its win rate + gate."""
    summary = {
        "current": "deck.csv", "decks_dir": "decks", "games": 200,
        "policy": "scoring", "z": 1.96,
        "candidates": [
            cd.deck_result("deck_aggro", "decks/deck_aggro.csv", _report(110, 90, total=200)),
            cd.deck_result("deck_balanced", "decks/deck_balanced.csv", _report(0, 0, undecided=4, total=4)),
        ],
        "passed": True,
    }
    text = cd.format_summary(summary)
    assert "deck_aggro" in text and "deck_balanced" in text, text
    assert "winrate=0.550" in text, text
    assert "n/a" in text, text                                # the no-decided candidate
    assert "GATE PASS" in text, text
    print("PASS test_format_summary_lists_candidates")


def test_compare_end_to_end_small():
    """A small real run plays each candidate vs the current deck and summarises."""
    summary = cd.run_compare(games=4, seed=11, workers=2)
    assert summary["candidates"], summary
    for r in summary["candidates"]:
        assert r["games"] == 4, r
        assert r["wins"] + r["losses"] + r["draws"] + r["undecided"] == 4, r
        # Each candidate has a Wilson CI (a valid sub-interval of [0, 1]).
        assert 0.0 <= r["ci_low"] <= r["ci_high"] <= 1.0, r
        if r["winrate"] is not None:
            assert 0.0 <= r["winrate"] <= 1.0, r
    print("PASS test_compare_end_to_end_small")


if __name__ == "__main__":
    test_deck_result_winrate_and_ci()
    test_deck_result_no_decided()
    test_discover_candidate_decks()
    test_format_summary_lists_candidates()
    test_compare_end_to_end_small()
    print("ALL TESTS PASSED")
