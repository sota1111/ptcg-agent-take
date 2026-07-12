"""Tests for the observation featuriser (SOT-1641).

No pytest dependency — run directly from the repo root:
    venv/bin/python eval/test_features.py

Covers the two acceptance criteria:
  1. every ``decision`` record in every recorded trace under ``eval/traces/``
     featurises without raising, into fixed-length vectors;
  2. unknown SelectType / SelectContext / OptionType / enum values fall back to
     the dedicated "unknown" one-hot slot instead of raising or colliding with a
     valid class.
Plus structural checks (dimension stability, real board state is read, empty /
malformed input is tolerated).
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from agents.learned.features import (  # noqa: E402
    AREA_TYPE_DIM,
    OBSERVATION_FEATURE_DIM,
    OPTION_FEATURE_DIM,
    OPTION_TYPE_DIM,
    SELECT_CONTEXT_DIM,
    SELECT_TYPE_DIM,
    candidate_features,
    featurize,
    observation_features,
    option_features,
)

TRACES_DIR = os.path.join(REPO, "eval", "traces")


def _iter_trace_files():
    for root, _dirs, files in os.walk(TRACES_DIR):
        for name in sorted(files):
            if name.endswith(".jsonl"):
                yield os.path.join(root, name)


def _iter_decisions():
    """Yield ``(path, decision_record)`` for every decision in every trace."""
    for path in _iter_trace_files():
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and rec.get("kind") == "decision":
                    yield path, rec


def _obs_from_decision(rec: dict) -> dict:
    """Reconstruct an ``obs_dict`` from a decision record.

    FULL_OBS traces carry the raw ``obs`` verbatim; LOGS traces don't, so rebuild
    the minimal observation the featuriser consumes from the recorded columns.
    """
    if isinstance(rec.get("obs"), dict):
        return rec["obs"]
    return {
        "select": rec.get("select"),
        "current": {
            "yourIndex": rec.get("your_index"),
            "turn": rec.get("turn"),
            "turnActionCount": rec.get("turn_action_count"),
        },
        "logs": rec.get("logs", []),
    }


def _assert_vec(vec, dim, ctx):
    assert isinstance(vec, list), (ctx, type(vec))
    assert len(vec) == dim, (ctx, len(vec), dim)
    assert all(isinstance(x, float) for x in vec), ctx


def _record_fresh_full_obs_trace() -> str:
    """Record one FULL_OBS match so the test always has real data to featurise.

    The engine (``cg/``) and existing traces are license-restricted and
    git-ignored, so a fresh checkout may have neither. Recording here — exactly
    how the acceptance criterion is verified — makes this test self-contained
    while still also consuming any pre-existing traces.
    """
    from eval import record_match as rm  # noqa: E402
    from eval.trace import RecordLevel  # noqa: E402

    deck = rm.load_deck("deck.csv")
    out = os.path.join(TRACES_DIR, "_features_test.jsonl")
    rm.record_match(deck, deck, out_path=out, level=RecordLevel.FULL_OBS)
    return out


def test_all_recorded_decisions_featurise():
    """AC1: every decision in every trace vectorises without exception."""
    fresh = _record_fresh_full_obs_trace()
    try:
        n_traces = sum(1 for _ in _iter_trace_files())
        assert n_traces > 0, f"no trace files under {TRACES_DIR}"

        n_decisions = 0
        saw_full_obs_board = False
        for path, rec in _iter_decisions():
            n_decisions += 1
            obs = _obs_from_decision(rec)
            feat = featurize(obs)  # must not raise
            _assert_vec(feat.observation, OBSERVATION_FEATURE_DIM, path)
            assert feat.n_options == len(feat.candidates), path
            for cand in feat.candidates:
                _assert_vec(cand, OPTION_FEATURE_DIM, path)
            # A FULL_OBS decision carries real board state -> some feature must be
            # non-zero, proving we read state rather than emitting only zeros.
            if isinstance(rec.get("obs"), dict) and any(v != 0.0 for v in feat.observation):
                saw_full_obs_board = True

        assert n_decisions > 0, "no decision records found in traces"
        assert saw_full_obs_board, "expected a FULL_OBS decision with non-zero features"
        print(
            f"PASS test_all_recorded_decisions_featurise "
            f"({n_decisions} decisions across {n_traces} trace files)"
        )
    finally:
        try:
            os.remove(fresh)
        except OSError:
            pass


def test_unknown_enums_fall_back_without_raising():
    """AC2: unknown enum values land in the unknown slot and never raise."""
    # Unknown SelectType / SelectContext far beyond the defined ranges.
    obs = {
        "select": {
            "type": 999,
            "context": 777,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 888, "area": 999},              # unknown OptionType + area
                {"type": 3, "area": 4, "cardId": 12345},  # valid CARD in ACTIVE
                {"type": "not-an-int"},                    # garbage type
            ],
        },
        "current": {"yourIndex": 0, "turn": 3},
    }
    feat = featurize(obs)  # must not raise
    _assert_vec(feat.observation, OBSERVATION_FEATURE_DIM, "unknown-obs")
    assert feat.n_options == 3

    # SelectType unknown slot (last index of its one-hot block) must be set.
    # Layout: [select_present] + SelectType(one-hot) + ...
    st_block = feat.observation[1 : 1 + SELECT_TYPE_DIM]
    assert st_block[SELECT_TYPE_DIM - 1] == 1.0, st_block
    assert sum(st_block) == 1.0, st_block
    sc_block = feat.observation[1 + SELECT_TYPE_DIM : 1 + SELECT_TYPE_DIM + SELECT_CONTEXT_DIM]
    assert sc_block[SELECT_CONTEXT_DIM - 1] == 1.0, sc_block
    assert sum(sc_block) == 1.0, sc_block

    # Option 0: unknown OptionType -> unknown slot; unknown area -> unknown slot.
    opt0 = feat.candidates[0]
    ot_block = opt0[:OPTION_TYPE_DIM]
    assert ot_block[OPTION_TYPE_DIM - 1] == 1.0 and sum(ot_block) == 1.0, ot_block
    area_block = opt0[OPTION_TYPE_DIM : OPTION_TYPE_DIM + AREA_TYPE_DIM]
    assert area_block[AREA_TYPE_DIM - 1] == 1.0 and sum(area_block) == 1.0, area_block

    # Option 2: non-int type -> unknown slot, still fixed length.
    opt2 = feat.candidates[2]
    ot2 = opt2[:OPTION_TYPE_DIM]
    assert ot2[OPTION_TYPE_DIM - 1] == 1.0, ot2

    print("PASS test_unknown_enums_fall_back_without_raising")


def test_valid_enum_sets_correct_slot():
    """A known enum value sets exactly its own index, not the unknown slot."""
    # SelectType.CARD == 1, SelectContext.MAIN == 0, OptionType.ATTACK == 13.
    obs = {
        "select": {
            "type": 1,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 13, "attackId": 7}],
        },
        "current": {"yourIndex": 1},
    }
    feat = featurize(obs)
    st_block = feat.observation[1 : 1 + SELECT_TYPE_DIM]
    assert st_block[1] == 1.0 and sum(st_block) == 1.0, st_block
    ot_block = feat.candidates[0][:OPTION_TYPE_DIM]
    assert ot_block[13] == 1.0 and sum(ot_block) == 1.0, ot_block
    print("PASS test_valid_enum_sets_correct_slot")


def test_fixed_length_on_degenerate_input():
    """None / empty / partial inputs still yield fixed-length vectors, no raise."""
    for bad in (None, {}, {"select": None, "current": None}, {"select": "x"}, 42, []):
        vec = observation_features(bad)
        _assert_vec(vec, OBSERVATION_FEATURE_DIM, repr(bad))
        assert candidate_features(bad) == [] or isinstance(candidate_features(bad), list)
    for bad_opt in (None, {}, {"type": None}, "x", 5):
        _assert_vec(option_features(bad_opt), OPTION_FEATURE_DIM, repr(bad_opt))

    # Initial deck selection: select is None -> no candidates, obs still fixed.
    feat = featurize({"select": None, "current": None})
    assert feat.candidates == [] and feat.n_options == 0
    _assert_vec(feat.observation, OBSERVATION_FEATURE_DIM, "deck-select")
    print("PASS test_fixed_length_on_degenerate_input")


def test_dimensions_are_positive_and_stable():
    assert OBSERVATION_FEATURE_DIM == len(observation_features({})) > 0
    assert OPTION_FEATURE_DIM == len(option_features({})) > 0
    print(
        f"PASS test_dimensions_are_positive_and_stable "
        f"(obs={OBSERVATION_FEATURE_DIM}, option={OPTION_FEATURE_DIM})"
    )


if __name__ == "__main__":
    test_dimensions_are_positive_and_stable()
    test_valid_enum_sets_correct_slot()
    test_unknown_enums_fall_back_without_raising()
    test_fixed_length_on_degenerate_input()
    test_all_recorded_decisions_featurise()
    print("ALL TESTS PASSED")
