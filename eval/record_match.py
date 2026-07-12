"""Recording single-match runner for the PTCG eval environment (SOT-1618).

Plays one full agent-vs-agent match and writes a complete JSONL trace (meta /
decision列 / event logs / result) via ``eval/trace.py``. The engine takes no seed
argument (E1), so this recording is the sole means of reproducing a match.

The original ``eval/run_match.py`` is left unchanged; this is a separate runner.

Usage:
    venv/bin/python eval/record_match.py [deck0.csv] [deck1.csv]
                    [--out PATH] [--level result|logs|full_obs]
                    [--seed N] [--max-steps N] [--inject-exception 0|1]

Run from the repo root (after scripts/setup_engine.sh has populated cg/).
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
import time
from typing import Any, Callable, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # make `cg` and `eval` importable
os.chdir(REPO)            # so libcg.so & deck.csv resolve

from cg import game                                  # noqa: E402
from cg.api import to_observation_class              # noqa: E402
from eval.trace import (                             # noqa: E402
    FAIL_AGENT_EXCEPTION,
    FAIL_ENGINE_ERROR,
    FAIL_START_ERROR,
    RecordLevel,
    TraceWriter,
)

_LEVELS = {"result": RecordLevel.RESULT, "logs": RecordLevel.LOGS, "full_obs": RecordLevel.FULL_OBS}


def load_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(x) for x in f.read().split("\n")[:60]]


class Agent:
    """A named agent: a callable ``fn(obs_dict) -> list[int]`` plus identity metadata.

    The metadata (name / version / params) is stamped into the trace's meta record.
    """

    def __init__(
        self,
        fn: Callable[[dict], Any],
        name: str = "agent",
        version: str = "0",
        params: Optional[dict] = None,
    ):
        self.fn = fn
        self.name = name
        self.version = version
        self.params = params or {}

    def act(self, obs_dict: dict) -> Any:
        return self.fn(obs_dict)

    def meta(self, index: int) -> dict:
        return {"index": index, "name": self.name, "version": self.version, "params": self.params}


def make_random_agent(seed: Optional[int] = None, name: str = "random") -> Agent:
    """A uniform-random legal-move agent (mirrors eval/run_match.py's policy)."""
    rng = random.Random(seed)

    def fn(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return []
        n = len(obs.select.option)
        k = max(obs.select.minCount, min(obs.select.maxCount, n))
        return rng.sample(range(n), k) if n else []

    return Agent(fn, name=name, version="1", params={"seed": seed})


def make_raising_agent(after: int = 0, name: str = "raising") -> Agent:
    """Test helper: raises RuntimeError on its ``after``-th call (fault injection)."""
    state = {"calls": 0}

    def fn(obs_dict: dict) -> list[int]:
        n = state["calls"]
        state["calls"] += 1
        if n >= after:
            raise RuntimeError(f"injected agent failure on call {n}")
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return []
        opt = len(obs.select.option)
        k = max(obs.select.minCount, min(obs.select.maxCount, opt))
        return list(range(k))

    return Agent(fn, name=name, version="1", params={"after": after})


def _select_player() -> Optional[int]:
    """Exact ``SerialData.selectPlayer`` for the current pending selection.

    game.py discards this field, so read it straight from the engine. Best-effort:
    returns None if unavailable (the decision's ``your_index`` still records the
    selecting player per State.yourIndex).
    """
    try:
        from cg.sim import Battle, lib  # type: ignore
        return int(lib.GetBattleData(Battle.battle_ptr).selectPlayer)
    except Exception:
        return None


def record_match(
    deck0: list[int],
    deck1: list[int],
    agents: Optional[tuple[Agent, Agent]] = None,
    out_path: str = "eval/traces/match.jsonl",
    level: RecordLevel = RecordLevel.LOGS,
    max_steps: int = 100000,
    trace_id: Optional[str] = None,
) -> dict:
    """Play one recorded match and write its trace. Returns a summary dict.

    ``battle_finish()`` is guaranteed via try/finally (E7 leak protection) whether
    the match ends normally, an agent raises, or the engine rejects a selection.
    """
    if agents is None:
        agents = (make_random_agent(0, "random0"), make_random_agent(1, "random1"))

    now = datetime.datetime.now(datetime.timezone.utc)
    trace_id = trace_id or now.strftime("%Y%m%dT%H%M%S%fZ")

    writer = TraceWriter(out_path, level)
    started = False
    result = -1
    failure: Optional[dict] = None
    final_logs: list = []
    final_turn: Optional[int] = None
    first_player: Optional[int] = None
    t0 = time.perf_counter()

    try:
        obs, start = game.battle_start(deck0, deck1)
        start_error = None
        if start.errorPlayer != -1 or start.errorType != 0:
            start_error = {"errorPlayer": start.errorPlayer, "errorType": start.errorType}

        writer.write_meta(
            trace_id=trace_id,
            created_at=now.isoformat(),
            agents=[agents[0].meta(0), agents[1].meta(1)],
            decks=[deck0, deck1],
            first_player=(obs.get("current") or {}).get("firstPlayer") if obs else None,
            start_error=start_error,
        )

        if obs is None:
            # BattleStart failed — record a failure result and stop (nothing to finish).
            elapsed = (time.perf_counter() - t0) * 1000
            failure = {
                "player": start.errorPlayer if start.errorPlayer != -1 else None,
                "category": FAIL_START_ERROR,
                "error": f"errorPlayer={start.errorPlayer} errorType={start.errorType}",
            }
            writer.write_result(
                result=-1, final_logs=[], first_player=None, final_turn=None,
                elapsed_ms=elapsed, failure=failure, start_error=start_error,
            )
            return _summary(-1, 0, None, failure, out_path, level)

        started = True
        while writer.n_decisions < max_steps:
            current = obs.get("current") or {}
            final_logs = obs.get("logs", [])
            if current.get("firstPlayer", -1) != -1:
                first_player = current.get("firstPlayer")
            final_turn = current.get("turn", final_turn)

            if current.get("result", -1) != -1:
                result = current["result"]
                break

            actor = current.get("yourIndex")
            select_player = _select_player()
            if actor is None:
                actor = select_player if select_player in (0, 1) else 0
            agent = agents[actor]

            ts = time.perf_counter()
            try:
                choice = agent.act(obs)
            except Exception as exc:  # E7: agent crash -> scored loss, still finishes
                failure = {"player": actor, "category": FAIL_AGENT_EXCEPTION, "error": repr(exc)}
                break
            thinking_ms = (time.perf_counter() - ts) * 1000

            writer.write_decision(obs, choice, select_player, thinking_ms)

            try:
                obs = game.battle_select(choice if isinstance(choice, list) else [])
            except Exception as exc:
                failure = {"player": actor, "category": FAIL_ENGINE_ERROR, "error": repr(exc)}
                break

        elapsed = (time.perf_counter() - t0) * 1000
        writer.write_result(
            result=result,
            final_logs=final_logs,
            first_player=first_player,
            final_turn=final_turn,
            elapsed_ms=elapsed,
            failure=failure,
            start_error=start_error,
        )
        return _summary(result, writer.n_decisions, final_turn, failure, out_path, level)
    finally:
        if started:
            try:
                game.battle_finish()  # E7: guaranteed cleanup
            except Exception:
                pass
        writer.close()


def _summary(result, decisions, turn, failure, out_path, level) -> dict:
    return {
        "result": result,
        "decisions": decisions,
        "final_turn": turn,
        "failure": failure,
        "out_path": out_path,
        "level": int(level),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record one PTCG match to a JSONL trace.")
    p.add_argument("deck0", nargs="?", default="deck.csv")
    p.add_argument("deck1", nargs="?", default=None)
    p.add_argument("--out", default="eval/traces/match.jsonl")
    p.add_argument("--level", choices=list(_LEVELS), default="logs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--inject-exception", type=int, default=0,
                   help="if 1, player 0 uses a fault-injecting agent (demonstrates E7 cleanup)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    deck0 = load_deck(args.deck0)
    deck1 = load_deck(args.deck1) if args.deck1 else deck0

    if args.inject_exception:
        agents = (make_raising_agent(after=0, name="raising0"), make_random_agent(args.seed + 1, "random1"))
    else:
        agents = (make_random_agent(args.seed, "random0"), make_random_agent(args.seed + 1, "random1"))

    summary = record_match(
        deck0, deck1, agents=agents, out_path=args.out,
        level=_LEVELS[args.level], max_steps=args.max_steps,
    )
    fail = summary["failure"]
    result = summary["result"]
    if fail and fail.get("player") in (0, 1):
        winner = f"player{1 - fail['player']}"  # failing player loses
    elif result in (0, 1):
        winner = f"player{result}"
    elif result == 2:
        winner = "draw"
    else:
        winner = "none"  # truncated / undecided
    print(
        f"MATCH DONE: result={result} winner={winner}"
        f" decisions={summary['decisions']} final_turn={summary['final_turn']}"
        f" failure={fail['category'] if fail else None} trace={summary['out_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
