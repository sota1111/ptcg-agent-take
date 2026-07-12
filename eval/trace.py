"""Match trace schema + JSONL writer for the PTCG eval environment (SOT-1618).

The cabt engine takes **no seed argument** (E1): re-running a match does not
reproduce it, so the *runtime recording* is the sole means of reproduction. This
module defines that recording — a JSONL trace of one match — and a writer for it.

Trace file layout (one JSON object per line):
  line 1 .. : a single ``meta`` record (schema/engine stamp, agents, decks)
  next N    : one ``decision`` record per agent decision (legal moves + choice +
              search_begin_input + the event logs seen at that decision)
  last line : one ``result`` record (outcome + reason, turn/decision counts,
              elapsed time, and any failure category)

Record verbosity is controlled by ``RecordLevel`` because dumping the full
observation JSON at every decision is IO-dominated (see the level docs below).
"""
from __future__ import annotations

import hashlib
import json
import os
from enum import IntEnum
from typing import Any, Optional, TextIO

# Bump when the trace record shape changes (E6: stamped into every trace's meta).
SCHEMA_VERSION = "1.0.0"


class RecordLevel(IntEnum):
    """How much per-decision detail to persist.

    RESULT   — meta + result only (decisions are counted but not emitted). Smallest.
    LOGS     — RESULT plus one ``decision`` record per decision, carrying the full
               SelectData (all legal moves), the chosen index/indices, thinking
               time, ``search_begin_input`` (E5), and the event logs. This is the
               default and satisfies the acceptance criteria.
    FULL_OBS — LOGS plus the full raw observation dict at each decision. IO-heavy.
    """

    RESULT = 0
    LOGS = 1
    FULL_OBS = 2


# Failure categories recorded in a ``result`` record's ``failure.category``.
FAIL_START_ERROR = "start_error"        # BattleStart reported errorPlayer/errorType
FAIL_AGENT_EXCEPTION = "agent_exception"  # the agent callable raised
FAIL_ENGINE_ERROR = "engine_error"      # game.battle_select raised (bad selection etc.)
FAIL_TRUNCATED = "truncated"            # hit max_steps without a RESULT (result == -1)
FAIL_TIMEOUT = "timeout"                # match exceeded its wall-clock budget (arena, SOT-1619)
FAIL_WORKER_ERROR = "worker_error"      # the arena worker itself raised (SOT-1619)


def engine_hash(lib_path: Optional[str] = None) -> dict:
    """Return a sha256 stamp of the loaded engine shared library (E6).

    Defaults to the exact library ``cg.sim`` loaded for the current platform.
    Best-effort: never raises — records an ``error`` field instead.
    """
    if lib_path is None:
        try:
            from cg.sim import lib_path as _resolved  # type: ignore
            lib_path = _resolved
        except Exception as exc:  # pragma: no cover - engine not importable
            return {"path": "", "sha256": None, "size": None, "error": repr(exc)}

    info: dict[str, Any] = {"path": os.path.basename(lib_path), "sha256": None, "size": None}
    try:
        digest = hashlib.sha256()
        size = 0
        with open(lib_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
                size += len(chunk)
        info["sha256"] = digest.hexdigest()
        info["size"] = size
    except Exception as exc:
        info["error"] = repr(exc)
    return info


def build_meta(
    *,
    trace_id: str,
    created_at: str,
    level: RecordLevel,
    agents: list[dict],
    decks: list[list[int]],
    first_player: Optional[int],
    start_error: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build the ``meta`` record (first line of a trace)."""
    meta = {
        "kind": "meta",
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "created_at": created_at,
        "record_level": int(level),
        "engine": engine_hash(),
        "agents": agents,
        "decks": decks,
        "first_player": first_player,
        "start_error": start_error,
    }
    if extra:
        meta["extra"] = extra
    return meta


def build_decision(
    *,
    index: int,
    obs: dict,
    choice: Any,
    select_player: Optional[int],
    thinking_time_ms: float,
    level: RecordLevel,
) -> dict:
    """Build one ``decision`` record from a raw observation dict.

    Carries the full ``SelectData`` (``option`` is the complete list of legal
    moves), the chosen index/indices, thinking time, ``search_begin_input`` (E5),
    and the event logs (LogType incl. COIN) emitted since the previous decision.
    The full observation dict is included only at ``FULL_OBS`` level.
    """
    current = obs.get("current") or {}
    record = {
        "kind": "decision",
        "index": index,
        "select_player": select_player,
        "your_index": current.get("yourIndex"),
        "turn": current.get("turn"),
        "turn_action_count": current.get("turnActionCount"),
        "select": obs.get("select"),          # full SelectData = all legal moves
        "choice": choice,
        "thinking_time_ms": round(thinking_time_ms, 3),
        "search_begin_input": obs.get("search_begin_input"),
        "logs": obs.get("logs", []),          # events since the last decision
    }
    if level >= RecordLevel.FULL_OBS:
        record["obs"] = obs
    return record


def _extract_result_log(logs: list) -> Optional[dict]:
    """Return the RESULT log (LogType 23) from an event-log list, if present."""
    for log in logs or []:
        if isinstance(log, dict) and log.get("type") == 23:  # LogType.RESULT
            return log
    return None


def build_result(
    *,
    result: int,
    final_logs: list,
    first_player: Optional[int],
    final_turn: Optional[int],
    total_decisions: int,
    elapsed_ms: float,
    failure: Optional[dict] = None,
    start_error: Optional[dict] = None,
) -> dict:
    """Build the terminal ``result`` record.

    Derives ``reason`` (1-4) and ``winner`` from the RESULT log / engine result;
    ``result == -1`` marks a truncated/aborted match (distinguished from a real
    win/draw). On an agent/engine failure the failing player is scored as the loser.
    """
    result_log = _extract_result_log(final_logs)
    reason = result_log.get("reason") if result_log else None
    truncated = result == -1 and failure is None

    if failure and failure.get("category") in (FAIL_AGENT_EXCEPTION, FAIL_ENGINE_ERROR):
        loser = failure.get("player")
        winner = (1 - loser) if loser in (0, 1) else None
    elif result in (0, 1):
        winner = result
    else:  # 2 == draw, -1 == truncated/undecided
        winner = None

    return {
        "kind": "result",
        "result": result,
        "reason": reason,
        "winner": winner,
        "truncated": truncated,
        "first_player": first_player,
        "final_turn": final_turn,
        "total_decisions": total_decisions,
        "elapsed_ms": round(elapsed_ms, 3),
        "failure": failure,
        "start_error": start_error,
        "final_logs": final_logs,
    }


class TraceWriter:
    """Streams trace records to a JSONL file, one JSON object per line.

    Flushes after every record so a partial trace survives a crash mid-match.
    """

    def __init__(self, path: str, level: RecordLevel = RecordLevel.LOGS):
        self.path = path
        self.level = RecordLevel(level)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._fh: TextIO = open(path, "w", encoding="utf-8")
        self._closed = False
        self.n_decisions = 0

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        self._fh.write("\n")
        self._fh.flush()

    def write_meta(self, **kwargs: Any) -> dict:
        rec = build_meta(level=self.level, **kwargs)
        self._write(rec)
        return rec

    def write_decision(
        self,
        obs: dict,
        choice: Any,
        select_player: Optional[int],
        thinking_time_ms: float,
    ) -> Optional[dict]:
        """Record one decision. Always counted; only emitted at LOGS or above."""
        idx = self.n_decisions
        self.n_decisions += 1
        if self.level < RecordLevel.LOGS:
            return None
        rec = build_decision(
            index=idx,
            obs=obs,
            choice=choice,
            select_player=select_player,
            thinking_time_ms=thinking_time_ms,
            level=self.level,
        )
        self._write(rec)
        return rec

    def write_result(self, **kwargs: Any) -> dict:
        rec = build_result(total_decisions=self.n_decisions, **kwargs)
        self._write(rec)
        return rec

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._fh.close()
            except Exception:
                pass

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
