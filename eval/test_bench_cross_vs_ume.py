"""Standalone tests for the cross-repo bench (SOT-1668).

No pytest dependency — run directly:
    venv/bin/python eval/test_bench_cross_vs_ume.py

Covers the pure helpers (no engine, no ume checkout needed):
  1. ``load_deck`` — blank-line handling + first-60 truncation (ume loader parity);
  2. ``wilson_ci`` — edge cases and a known interval;
  3. ``prize_take_batches`` — batch grouping per log window, cross-perspective
     dedup on (player, serial), final_logs inclusion, non-prize/no-serial ignored;
  4. ``summarize_steps`` / ``hist``;
  5. ``pool_matches`` — win rate, 先手/後攻 split, reason & prize aggregation;
plus, when the sibling ume checkout is present, one tiny end-to-end cross run
(subprocess: the bench chdir's into the ume repo) that must complete fault-free.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bench_cross_vs_ume import (  # noqa: E402
    DEFAULT_UME_REPO,
    hist,
    load_deck,
    pool_matches,
    prize_take_batches,
    summarize_steps,
    wilson_ci,
)


def test_load_deck() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        fh.write("1\n\n2\n 3 \n")
        for i in range(70):
            fh.write(f"{100 + i}\n")
        path = fh.name
    try:
        deck = load_deck(path)
        assert deck[:3] == [1, 2, 3], deck[:3]
        assert len(deck) == 60, len(deck)  # truncated like the ume/submission loader
    finally:
        os.unlink(path)


def test_wilson_ci() -> None:
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60, (lo, hi)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi < 0.35, (lo, hi)
    lo, hi = wilson_ci(10, 10)
    assert lo > 0.65 and hi == 1.0, (lo, hi)


def _move(player: int, serial: int, from_area: int = 6, type_: int = 6) -> dict:
    return {"type": type_, "playerIndex": player, "serial": serial,
            "fromArea": from_area, "toArea": 2}


def test_prize_take_batches() -> None:
    records = [
        # one KO burst: player 0 takes 3 prizes in one log window (Mega KO)
        {"kind": "decision", "logs": [_move(0, 1), _move(0, 2), _move(0, 3)]},
        # the same events seen again from the other perspective → deduplicated
        {"kind": "decision", "logs": [_move(0, 1), _move(0, 2), _move(0, 3),
                                      _move(1, 10)]},
        # ignored: not MOVE_CARD / not from PRIZE / no serial
        {"kind": "decision", "logs": [_move(0, 4, type_=7),
                                      _move(0, 5, from_area=1),
                                      {"type": 6, "playerIndex": 1, "fromArea": 6}]},
        # tail take arrives only in the terminal record's final_logs
        {"kind": "result", "final_logs": [_move(1, 11)], "reason": 1},
    ]
    batches = prize_take_batches(records)
    assert batches[0] == [3], batches[0]
    assert batches[1] == [1, 1], batches[1]


def test_summarize_steps_and_hist() -> None:
    assert summarize_steps([]) == {"n": 0}
    s = summarize_steps([10, 20, 30, 40])
    assert s["n"] == 4 and s["mean"] == 25.0 and s["min"] == 10 and s["max"] == 40
    assert hist([3, 1, 3, 3]) == {"1": 1, "3": 3}


def _row(take_won: bool, take_first: bool, steps: int, reason: int = 3,
         take_prizes: int = 0, ume_prizes: int = 0) -> dict:
    return {"take_won": take_won, "ume_won": not take_won, "draw": False,
            "take_first": take_first, "steps": steps, "reason": reason,
            "take_prizes": take_prizes, "ume_prizes": ume_prizes,
            "take_batches": [1] * take_prizes, "ume_batches": [1] * ume_prizes}


def test_pool_matches() -> None:
    rows = [
        _row(True, True, 20, reason=1, take_prizes=6, ume_prizes=2),
        _row(True, False, 25, take_prizes=4),
        _row(False, True, 80, ume_prizes=6),
        _row(False, False, 90, ume_prizes=5),
    ]
    p = pool_matches(rows)
    assert p["n_matches"] == 4 and p["decided"] == 4 and p["draws"] == 0
    assert p["take_wins"] == 2 and p["take_win_rate"] == 0.5
    assert p["take_as_first"]["n"] == 2 and p["take_as_first"]["take_wins"] == 1
    assert p["take_as_second"]["take_wins"] == 1
    assert p["result_reasons"] == {"1": 1, "3": 3}
    assert p["steps"]["take_wins"]["mean"] == 22.5
    assert p["steps"]["ume_wins"]["mean"] == 85.0
    assert p["prizes"]["take_taken_mean"] == 2.5
    assert p["prizes"]["ume_taken_when_take_wins"] == {"0": 1, "2": 1}
    lo, hi = p["wilson_ci95"]
    assert 0 < lo < 0.5 < hi < 1


def test_end_to_end_cross_smoke() -> None:
    """Tiny real cross run through the CLI (skipped without the ume checkout)."""
    if not os.path.isdir(os.path.join(DEFAULT_UME_REPO, "eval")):
        print("  (ume checkout not found — e2e smoke skipped)")
        return
    with tempfile.TemporaryDirectory() as out_dir:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "bench_cross_vs_ume.py"),
             "--mode", "cross", "--n", "4", "--seeds", "0", "--out-dir", out_dir],
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        summary = json.load(open(os.path.join(out_dir, "summary_cross.json")))
        pooled = summary["pooled"]
        assert pooled["n_matches"] == 4, pooled
        assert summary["safety"]["faults_total"] == 0, summary["safety"]
        assert pooled["take_as_first"]["n"] + pooled["take_as_second"]["n"] \
            == pooled["decided"]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"[test] {t.__name__}")
        t()
    print(f"OK — {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
