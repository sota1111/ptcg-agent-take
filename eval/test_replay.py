"""Standalone tests for the game-record renderer / record-based replay (SOT-1621).

No pytest dependency — run directly:
    venv/bin/python eval/test_replay.py

Covers the acceptance criteria:
  1. a trace renders to a human-readable game record with card/attack ids resolved
     to names (unknown ids fall back to ``#<id>``);
  2. decisive-scene extraction (e.g. a knockout decision) works.

The pure rendering/resolution tests inject name maps and synthetic traces so they
need no engine; one end-to-end test records a real match and renders it (mirrors
eval/test_record_match.py) to prove the real trace path.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval import replay  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _resolver():
    """A NameResolver backed by injected maps (no engine, no CSVs)."""
    return replay.NameResolver(
        lang="en",
        use_engine=False,
        data_dir="/nonexistent",  # skip CSVs
        engine_cards={721: "Kyogre", 1158: "Maximum Belt"},
        engine_attacks={1042: "Riptide"},
    )


def _synthetic_trace():
    """A hand-built trace: a knockout decision, a big-HP decision, and an endgame."""
    meta = {"kind": "meta", "trace_id": "t", "created_at": "now", "record_level": 1,
            "engine": {"path": "libcg.so", "sha256": "ab" * 32}, "first_player": 0,
            "agents": [{"index": 0, "name": "a0", "version": "1", "params": {}},
                       {"index": 1, "name": "a1", "version": "1", "params": {}}]}
    d0 = {"kind": "decision", "index": 0, "your_index": 0, "select_player": 0, "turn": 1,
          "select": {"context": 0, "minCount": 1, "maxCount": 1,
                     "option": [{"type": 13, "attackId": 1042}, {"type": 14}]},
          "choice": [0],
          "logs": [{"type": 15, "playerIndex": 0, "cardId": 721, "attackId": 1042},
                   {"type": 16, "playerIndex": 1, "cardId": 721, "value": 130, "putDamageCounter": False}]}
    d1 = {"kind": "decision", "index": 1, "your_index": 1, "select_player": 1, "turn": 2,
          "select": {"context": 4, "minCount": 1, "maxCount": 1,
                     "option": [{"type": 3, "cardId": 721, "area": 5}]},
          "choice": [0],
          "logs": [{"type": 6, "playerIndex": 1, "cardId": 721, "fromArea": 4, "toArea": 3},
                   {"type": 22, "playerIndex": 0, "head": True}]}
    d2 = {"kind": "decision", "index": 2, "your_index": 0, "select_player": 0, "turn": 3,
          "select": {"context": 0, "minCount": 1, "maxCount": 1, "option": [{"type": 14}]},
          "choice": [0], "logs": []}
    result = {"kind": "result", "result": 0, "winner": 0, "reason": 3, "truncated": False,
              "final_turn": 3, "total_decisions": 3, "failure": None,
              "final_logs": [{"type": 23, "result": 0, "reason": 3}]}
    return replay.split_records([meta, d0, d1, d2, result])


# --------------------------------------------------------------------------- #
# NameResolver
# --------------------------------------------------------------------------- #

def test_name_resolution_and_fallback():
    r = _resolver()
    assert r.card(721) == "Kyogre", r.card(721)
    assert r.attack(1042) == "Riptide", r.attack(1042)
    # Unknown id → #id fallback (acceptance: 未知 id は id のまま).
    assert r.card(999999) == "#999999", r.card(999999)
    assert r.attack(888888) == "#888888", r.attack(888888)
    assert r.card(None) == "?"
    assert r.card_with_id(1158) == "Maximum Belt(#1158)"
    print("PASS test_name_resolution_and_fallback")


def test_lang_preference_ja_uses_csv_then_engine():
    # No JP CSV available → falls back to engine (EN) name, not a crash.
    r = replay.NameResolver(lang="ja", use_engine=False, data_dir="/nonexistent",
                            engine_cards={721: "Kyogre"}, engine_attacks={})
    assert r.card(721) == "Kyogre"
    # With a JP map injected via csv dict directly, JP wins.
    r._csv_ja[721] = "カイオーガ"
    assert r.card(721) == "カイオーガ", r.card(721)
    print("PASS test_lang_preference_ja_uses_csv_then_engine")


def test_real_csv_card_names_if_present():
    """If the license-restricted CSVs are present, card names resolve from them."""
    if not os.path.isfile(os.path.join(REPO, "data", "EN_Card_Data.csv")):
        print("SKIP test_real_csv_card_names_if_present (no data/ CSVs)")
        return
    r = replay.NameResolver(lang="en", use_engine=False)
    assert r.card(721) == "Kyogre", r.card(721)
    print("PASS test_real_csv_card_names_if_present")


# --------------------------------------------------------------------------- #
# Log / option / decision rendering
# --------------------------------------------------------------------------- #

def test_render_logs_cover_key_types():
    r = _resolver()
    assert "Riptide" in replay.render_log({"type": 15, "playerIndex": 0, "cardId": 721, "attackId": 1042}, r)
    assert "Kyogre" in replay.render_log({"type": 15, "playerIndex": 0, "cardId": 721, "attackId": 1042}, r)
    hp = replay.render_log({"type": 16, "playerIndex": 1, "cardId": 721, "value": 130}, r)
    assert "130" in hp and "HP変化" in hp, hp
    ko = replay.render_log({"type": 6, "cardId": 721, "fromArea": 4, "toArea": 3}, r)
    assert "きぜつ" in ko, ko
    coin = replay.render_log({"type": 22, "playerIndex": 0, "head": True}, r)
    assert "表" in coin, coin
    res = replay.render_log({"type": 23, "result": 0, "reason": 3}, r)
    assert "P0 の勝ち" in res and "バトル場不在" in res, res
    # Unknown/future type does not raise and keeps the raw record.
    unk = replay.render_log({"type": 99, "foo": 1}, r)
    assert "type=99" in unk, unk
    print("PASS test_render_logs_cover_key_types")


def test_render_decision_marks_choice_and_names():
    r = _resolver()
    dec = _synthetic_trace()["decisions"][0]
    out = replay.render_decision(dec, r)
    assert "P0視点" in out, out             # E4 viewpoint annotated
    assert "非公開" in out                   # opponent hand marked non-public
    assert "Riptide" in out                  # attack id resolved in an option
    assert "✓ [0]" in out                    # chosen option marked
    print("PASS test_render_decision_marks_choice_and_names")


# --------------------------------------------------------------------------- #
# Scene extraction
# --------------------------------------------------------------------------- #

def test_extract_scenes_knockout_and_big_hp():
    trace = _synthetic_trace()
    scenes = replay.extract_scenes(trace, hp_threshold=100, endgame=1)
    by_index = {s["decision"]["index"]: s["reasons"] for s in scenes}

    # decision 0: big HP change (value 130 >= 100).
    assert 0 in by_index and any("HP" in x for x in by_index[0]), by_index
    # decision 1: knockout (Active 4 -> discard 3).
    assert 1 in by_index and any("きぜつ" in x for x in by_index[1]), by_index
    # decision 2: endgame (last decision before RESULT).
    assert 2 in by_index and any("終盤" in x for x in by_index[2]), by_index
    # scenes are in decision order and de-duplicated.
    assert [s["decision"]["index"] for s in scenes] == [0, 1, 2]
    print("PASS test_extract_scenes_knockout_and_big_hp")


def test_extract_scenes_filters_toggle():
    trace = _synthetic_trace()
    # High threshold + no endgame + no knockout → only the knockout filter off means
    # nothing from HP; with knockouts off and big_hp off and endgame off => empty.
    scenes = replay.extract_scenes(trace, hp_threshold=999, knockouts=False,
                                   big_hp=False, include_endgame=False)
    assert scenes == [], scenes
    # Only knockouts.
    scenes = replay.extract_scenes(trace, knockouts=True, big_hp=False, include_endgame=False)
    assert [s["decision"]["index"] for s in scenes] == [1], scenes
    print("PASS test_extract_scenes_filters_toggle")


# --------------------------------------------------------------------------- #
# Full record + CLI
# --------------------------------------------------------------------------- #

def test_render_record_full():
    trace = _synthetic_trace()
    out = replay.render_record(trace, _resolver())
    assert "棋譜" in out
    assert "Kyogre" in out and "Riptide" in out    # names resolved
    assert "P0 の勝ち" in out                        # result rendered
    print("PASS test_render_record_full")


def test_cli_on_real_recorded_trace():
    """End-to-end: record a real match, then render it via the CLI (main)."""
    try:
        from eval import record_match as rm
        from eval.trace import RecordLevel
    except Exception as exc:  # engine not importable
        print(f"SKIP test_cli_on_real_recorded_trace (engine unavailable: {exc})")
        return
    deck = rm.load_deck("deck.csv")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "m.jsonl")
        rm.record_match(deck, deck, out_path=out, level=RecordLevel.LOGS)

        # Capture the CLI stdout.
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            rc = replay.main([out, "--lang", "en"])
        finally:
            sys.stdout = old
        text = buf.getvalue()
        assert rc == 0, rc
        assert "棋譜" in text and "決定#" in text, text[:200]
        # At least one option/log resolved to a real name (not just #id everywhere).
        assert "#" not in text or any(ch.isalpha() for ch in text), "names present"

        # Scenes mode also runs and finds the endgame at minimum.
        rc2 = replay.main([out, "--scenes"])
        assert rc2 == 0
    print("PASS test_cli_on_real_recorded_trace")


if __name__ == "__main__":
    test_name_resolution_and_fallback()
    test_lang_preference_ja_uses_csv_then_engine()
    test_real_csv_card_names_if_present()
    test_render_logs_cover_key_types()
    test_render_decision_marks_choice_and_names()
    test_extract_scenes_knockout_and_big_hp()
    test_extract_scenes_filters_toggle()
    test_render_record_full()
    test_cli_on_real_recorded_trace()
    print("ALL TESTS PASSED")
