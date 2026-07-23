"""Engine-independent board-wipe KPI regression tests for SOT-1884."""
from __future__ import annotations

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from eval.board_wipe_ab import is_board_wipe, summarise


def match(outcome: str, *, a_seat: int, wiped_seat: int | None = None) -> dict:
    players = [
        {"active": [{"id": 1}], "bench": []},
        {"active": [{"id": 2}], "bench": []},
    ]
    if wiped_seat is not None:
        players[wiped_seat] = {"active": [], "bench": []}
    return {
        "outcome": outcome, "a_seat": a_seat, "final_board": players,
        "failure_category": None,
    }


class BoardWipeKpiTest(unittest.TestCase):
    def test_classifies_both_sides_after_side_swap(self) -> None:
        self.assertTrue(is_board_wipe(match("B_win", a_seat=0, wiped_seat=0), "A"))
        self.assertTrue(is_board_wipe(match("A_win", a_seat=1, wiped_seat=0), "B"))

    def test_summary_records_loss_rate_and_avoidance(self) -> None:
        result = summarise([
            match("B_win", a_seat=0, wiped_seat=0),
            match("A_win", a_seat=1, wiped_seat=0),
            match("A_win", a_seat=0),
            match("B_win", a_seat=1),
        ], elapsed=2.0)
        self.assertEqual(result["candidate"]["board_wipe_count"], 1)
        self.assertEqual(result["champion"]["board_wipe_count"], 1)
        self.assertEqual(result["candidate"]["board_wipe_rate_in_losses"], 0.5)
        self.assertEqual(result["candidate"]["board_wipe_avoidance_rate"], 0.75)
        self.assertEqual(result["sims_per_sec"], 2.0)


if __name__ == "__main__":
    unittest.main()
