"""Small-N gated A/B evaluation for the SOT-1884 board-survival candidate."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval.arena import agent_spec, run_arena
from eval.record_match import load_deck
from eval.report import wilson_interval
from eval.trace import RecordLevel


def is_board_wipe(match: dict, side: str) -> bool:
    seat = match.get("a_seat") if side == "A" else 1 - match.get("a_seat", 0)
    board = match.get("final_board") or []
    if seat not in (0, 1) or len(board) <= seat:
        return False
    player = board[seat] or {}
    return not any(player.get("active") or []) and not any(player.get("bench") or [])


def summarise(matches: list[dict], elapsed: float) -> dict:
    completed = [m for m in matches if m.get("outcome") in ("A_win", "B_win", "draw")]
    a_wins = sum(m.get("outcome") == "A_win" for m in completed)
    b_wins = sum(m.get("outcome") == "B_win" for m in completed)
    decided = a_wins + b_wins
    lo, hi = wilson_interval(a_wins, decided)
    faults = sum(bool(m.get("failure_category")) for m in matches)

    def side_kpi(side: str, losses: int) -> dict:
        losing = "B_win" if side == "A" else "A_win"
        wipes = sum(m.get("outcome") == losing and is_board_wipe(m, side)
                    for m in completed)
        return {
            "wins": a_wins if side == "A" else b_wins,
            "losses": losses,
            "board_wipe_count": wipes,
            "board_wipe_rate_in_losses": round(wipes / losses, 4) if losses else 0.0,
            "board_wipe_avoidance_rate": round(1 - wipes / len(completed), 4)
            if completed else 0.0,
        }

    return {
        "candidate": side_kpi("A", b_wins),
        "champion": side_kpi("B", a_wins),
        "candidate_win_rate": round(a_wins / decided, 4) if decided else None,
        "candidate_wilson95": [round(lo, 4), round(hi, 4)],
        "faults": faults,
        "completed": len(completed),
        "elapsed_seconds": round(elapsed, 3),
        "sims_per_sec": round(len(matches) / elapsed, 3) if elapsed else None,
    }


def run(games_per_deck: int, deck_glob: str, out_dir: str, seed: int) -> dict:
    matches = []
    started = time.perf_counter()
    for index, path in enumerate(sorted(glob.glob(deck_glob))):
        deck = load_deck(path)
        report = run_arena(
            games=games_per_deck, deck_a=deck, deck_b=deck,
            agent_a=agent_spec("rule_based", "candidate", policy="survival", deck_path=path),
            agent_b=agent_spec("rule_based", "champion", policy="scoring", deck_path=path),
            out_dir=os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0]),
            level=RecordLevel.RESULT, base_seed=seed + index,
        )
        matches.extend(report["matches"])
    result = summarise(matches, time.perf_counter() - started)
    result.update({"games_per_deck": games_per_deck, "deck_glob": deck_glob, "seed": seed})
    lo = result["candidate_wilson95"][0]
    result["promotion_gate"] = {
        "wilson_lower_gt_0_5": lo > 0.5,
        "faults_zero": result["faults"] == 0,
        "passed": lo > 0.5 and result["faults"] == 0,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-deck", type=int, default=1)
    parser.add_argument("--decks", default="decks/rotation_baseline/*.csv")
    parser.add_argument("--out-dir", default="eval/traces/sot1884-screen")
    parser.add_argument("--json", default="artifacts/sot-1884/screen.json")
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    result = run(args.games_per_deck, args.decks, args.out_dir, args.seed)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
