"""Resumable real-runtime cross-play for the hardened Take profile (SOT-1874).

Every card uses the contestants' actual ``main.agent`` entrypoint and committed
deck, swaps seats in pairs, enforces a 600-second per-decision ceiling, and
atomically checkpoints after each pair.  The whole experiment stops cleanly at
the configured eight-hour budget and can continue with ``--resume``.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agents.profile import load_promoted_profile  # noqa: E402
from cg import game  # noqa: E402
from eval.battle_vs_matsu import extract_reason, wilson_ci  # noqa: E402

DEFAULT_OPPONENTS = {
    "sol": REPO.parent / "ptcg-agent-sol",
    "debate": REPO.parent / "ptcg-agent-debate",
    "fable": REPO.parent / "ptcg-agent-fable",
    "zero": REPO.parent / "ptcg-agent-zero",
}


def load_deck(repo: Path) -> list[int]:
    cards = [int(x) for x in (repo / "deck.csv").read_text().splitlines() if x.strip()]
    if len(cards) != 60:
        raise ValueError(f"{repo}: deck.csv has {len(cards)} cards, expected 60")
    return cards


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


@dataclass
class RuntimeAgent:
    name: str
    repo: Path
    timeout_seconds: float
    proc: subprocess.Popen | None = field(default=None, repr=False)

    def start(self) -> None:
        server = REPO / "eval" / "agent_server.py"
        interpreters = (self.repo / ".venv" / "bin" / "python",
                        self.repo / "venv" / "bin" / "python")
        python = next((path for path in interpreters if path.is_file()), Path(sys.executable))
        self.proc = subprocess.Popen(
            [str(python), str(server)], cwd=self.repo,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        assert self.proc.stderr
        ready = self.proc.stderr.readline().strip()
        if not ready.startswith("READY"):
            raise RuntimeError(f"{self.name} failed to start: {ready} {self.proc.stderr.read()}")

    def act(self, observation: dict) -> tuple[list[int], float]:
        assert self.proc and self.proc.stdin and self.proc.stdout
        started = time.perf_counter()
        self.proc.stdin.write(json.dumps(observation, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        readable, _, _ = select.select([self.proc.stdout], [], [], self.timeout_seconds)
        if not readable:
            raise TimeoutError(f"{self.name} exceeded {self.timeout_seconds}s")
        line = self.proc.stdout.readline()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not line:
            raise RuntimeError(f"{self.name} exited during decision")
        action = json.loads(line)
        if isinstance(action, dict) and "__error__" in action:
            raise RuntimeError(f"{self.name}: {action['__error__']}")
        return action, elapsed_ms

    def stop(self) -> None:
        if not self.proc:
            return
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None


def play(seat0: RuntimeAgent, seat1: RuntimeAgent, deck0: list[int], deck1: list[int]) -> dict:
    obs, started = game.battle_start(deck0, deck1)
    if obs is None:
        return {"winner": None, "fault": "engine_start", "reason": None, "think_ms": []}
    think_ms: list[dict] = []
    try:
        for _ in range(100_000):
            current = obs.get("current") or {}
            result = current.get("result", -1)
            if result != -1:
                return {"winner": result, "fault": None, "reason": extract_reason(obs),
                        "think_ms": think_ms}
            seat = current.get("yourIndex", 0)
            actor = seat0 if seat == 0 else seat1
            try:
                action, elapsed = actor.act(obs)
            except TimeoutError:
                return {"winner": 1 - seat, "fault": "timeout", "fault_seat": seat,
                        "reason": None, "think_ms": think_ms}
            except Exception as exc:  # agent fault is a scored loss
                return {"winner": 1 - seat, "fault": "agent_exception", "fault_seat": seat,
                        "detail": str(exc), "reason": None, "think_ms": think_ms}
            think_ms.append({"seat": seat, "value": round(elapsed, 3)})
            try:
                obs = game.battle_select(action)
            except Exception as exc:
                return {"winner": 1 - seat, "fault": "illegal_action", "fault_seat": seat,
                        "detail": str(exc), "reason": None, "think_ms": think_ms}
        return {"winner": None, "fault": "unfinished", "reason": None, "think_ms": think_ms}
    finally:
        game.battle_finish()


def summarise(state: dict) -> dict:
    cards = []
    all_take_times = []
    for name in state["opponents"]:
        rows = [r for r in state["matches"] if r["opponent"] == name]
        wins = sum(r["take_won"] is True for r in rows)
        losses = sum(r["take_won"] is False for r in rows)
        decided = wins + losses
        lo, hi = wilson_ci(wins, decided)
        take_times = [t["value"] for r in rows for t in r["think_ms"]
                      if t["seat"] == r["take_seat"]]
        all_take_times.extend(take_times)
        cards.append({
            "opponent": name, "games": len(rows), "wins": wins, "losses": losses,
            "winRate": round(wins / decided, 4) if decided else None,
            "wilson95": [round(lo, 4), round(hi, 4)],
            "faults": sum(r["fault"] is not None for r in rows),
            "unfinished": sum(r["fault"] == "unfinished" for r in rows),
            "illegalActions": sum(r["fault"] == "illegal_action" for r in rows),
        })
    return {
        "cards": cards,
        "leagueWinRate": round(sum(c["wins"] for c in cards) /
                               max(1, sum(c["wins"] + c["losses"] for c in cards)), 4),
        "safety": {
            "faults": sum(c["faults"] for c in cards),
            "unfinished": sum(c["unfinished"] for c in cards),
            "illegalActions": sum(c["illegalActions"] for c in cards),
            "maxTakeDecisionMs": max(all_take_times, default=0),
        },
    }


def parse_opponents(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_OPPONENTS)
    result = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep:
            raise ValueError("--opponent must be NAME=REPO")
        result[name] = Path(path).resolve()
    return result


def main(argv: list[str] | None = None) -> int:
    profile = load_promoted_profile()
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20, help="games per opponent (even)")
    parser.add_argument("--seed", type=int, default=1874)
    parser.add_argument("--opponent", action="append")
    parser.add_argument("--take-repo", type=Path, default=REPO,
                        help="Take runtime checkout (supports historical baseline A/B)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-seconds", type=float, default=profile.experiment_budget_seconds)
    args = parser.parse_args(argv)
    args.take_repo = args.take_repo.resolve()
    if args.games < 2 or args.games % 2:
        parser.error("--games must be a positive even number")
    if not (0 < args.budget_seconds <= profile.experiment_budget_seconds):
        parser.error("budget must be within the profile's 8-hour ceiling")
    opponents = parse_opponents(args.opponent)
    for name, repo in {"take": args.take_repo, **opponents}.items():
        if not (repo / "main.py").is_file():
            parser.error(f"{name} repository is not runnable: {repo}")

    if args.resume and args.checkpoint.exists():
        state = json.loads(args.checkpoint.read_text())
        if state["seed"] != args.seed or state["gamesPerOpponent"] != args.games:
            parser.error("checkpoint parameters do not match this run")
    else:
        state = {"schemaVersion": "ptcg-take-runtime-league/v1", "issue": "SOT-1874",
                 "profile": profile.profile_id, "takeRepo": str(args.take_repo), "seed": args.seed,
                 "gamesPerOpponent": args.games, "opponents": list(opponents), "matches": []}

    completed = {(r["opponent"], r["index"]) for r in state["matches"]}
    take = RuntimeAgent("take", args.take_repo, profile.competition_budget_seconds)
    started_at = time.monotonic()
    try:
        take.start()
        for name, repo in opponents.items():
            other = RuntimeAgent(name, repo, profile.competition_budget_seconds)
            try:
                other.start()
                take_deck, other_deck = load_deck(args.take_repo), load_deck(repo)
                for index in range(args.games):
                    if (name, index) in completed:
                        continue
                    if time.monotonic() - started_at >= args.budget_seconds:
                        state["status"] = "budget_exhausted"
                        atomic_json(args.checkpoint, state)
                        atomic_json(args.report, {**state, "summary": summarise(state)})
                        return 75
                    take_seat = index % 2
                    if take_seat == 0:
                        outcome = play(take, other, take_deck, other_deck)
                    else:
                        outcome = play(other, take, other_deck, take_deck)
                    winner = outcome.get("winner")
                    row = {"opponent": name, "index": index, "takeSeat": take_seat,
                           "take_seat": take_seat, "take_won": None if winner not in (0, 1)
                           else winner == take_seat, **outcome}
                    state["matches"].append(row)
                    if (index + 1) % profile.checkpoint_every_games == 0:
                        atomic_json(args.checkpoint, state)
            finally:
                other.stop()
    finally:
        take.stop()
    state["status"] = "complete"
    summary = summarise(state)
    report = {**state, "summary": summary}
    atomic_json(args.checkpoint, state)
    atomic_json(args.report, report)
    safety = summary["safety"]
    return 0 if not (safety["faults"] or safety["unfinished"] or safety["illegalActions"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
