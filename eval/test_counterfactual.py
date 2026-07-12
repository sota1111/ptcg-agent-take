"""Standalone tests for the counterfactual analysis tool (SOT-1622).

No pytest dependency — run directly:
    venv/bin/python eval/test_counterfactual.py

Covers the acceptance criteria:
  1. from a trace + decision number, the actual choice AND an alternative option are
     each rolled out and returned for comparison;
  2. the hidden-information predictor is a swappable interface;
  3. the search session is always released (even when a branch raises).

The pure predictor / policy / parsing tests use synthetic observations so they need no
engine. The engine-backed tests record a real FULL_OBS match and analyze it; they are
skipped automatically when the cabt engine (cg/) is not importable.
"""
from __future__ import annotations

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval import counterfactual as cf  # noqa: E402
from eval.counterfactual import (  # noqa: E402
    HiddenInfo,
    HiddenInfoPredictor,
    RandomPolicy,
    UniformDeckPredictor,
    default_alternatives,
    find_decision,
    _require_full_obs,
    _visible_card_ids,
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures (no engine)
# --------------------------------------------------------------------------- #

def _pokemon(cid, *, energy=None, tools=None, pre=None):
    return {
        "id": cid, "serial": cid, "hp": 100, "maxHp": 100, "appearThisTurn": False,
        "energies": [], "energyCards": energy or [], "tools": tools or [], "preEvolution": pre or [],
    }


def _card(cid, p=0):
    return {"id": cid, "serial": cid, "playerIndex": p}


def _synth_obs(your_index=0, opp_active_facedown=False):
    """A minimal FULL_OBS observation dict for predictor tests.

    Deck compositions (below) are chosen so that
      deck == cards visible on board  +  the hidden pool that exactly fills the zones.
    Your side: deckCount 3 + prize 2 = 5 hidden. Opp side: deck 4 + prize 2 + hand 3
    (+ 1 active if face-down) hidden.
    """
    you = {
        "active": [_pokemon(100, energy=[_card(200, your_index)])],
        "bench": [], "benchMax": 5, "deckCount": 3,
        "discard": [_card(300, your_index)],
        "prize": [None, None],           # 2 face-down
        "handCount": 2,
        "hand": [_card(400, your_index), _card(401, your_index)],
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    opp_active = [None] if opp_active_facedown else [_pokemon(150)]
    opp = {
        "active": opp_active,
        "bench": [], "benchMax": 5, "deckCount": 4,
        "discard": [], "prize": [None, None], "handCount": 3, "hand": None,
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    players = [you, opp] if your_index == 0 else [opp, you]
    current = {
        "turn": 5, "turnActionCount": 0, "yourIndex": your_index, "firstPlayer": 0,
        "result": -1, "players": players, "stadium": [], "looking": None,
    }
    select = {
        "type": 0, "context": 0, "minCount": 1, "maxCount": 1,
        "remainDamageCounter": 0, "remainEnergyCost": 0,
        "option": [{"type": 14}, {"type": 7, "index": 0}, {"type": 7, "index": 1}],
        "deck": None, "contextCard": None, "effect": None,
    }
    return {"select": select, "logs": [], "current": current, "search_begin_input": "SBI"}


def _synth_meta(your_index=0, opp_active_facedown=False):
    # Deck composition = visible board cards + the hidden pool.
    your_deck = [100, 200, 300, 400, 401] + [10, 11, 12, 13, 14]        # 5 hidden
    opp_hidden = [20, 21, 22, 23, 24, 25, 26, 27, 28]                    # deck4+prize2+hand3
    if opp_active_facedown:
        opp_hidden = opp_hidden + [29]                                   # +1 for active
        opp_deck = opp_hidden[:]                                         # no face-up active card
    else:
        opp_deck = [150] + opp_hidden
    decks = [your_deck, opp_deck] if your_index == 0 else [opp_deck, your_deck]
    return {"decks": decks, "trace_id": "T", "schema_version": "1.0.0"}


# --------------------------------------------------------------------------- #
# Predictor tests (engine-free)
# --------------------------------------------------------------------------- #

def test_predictor_counts():
    for yi in (0, 1):
        obs, meta = _synth_obs(yi), _synth_meta(yi)
        hid = UniformDeckPredictor(seed=1).predict(obs, meta, your_index=yi)
        assert len(hid.your_deck) == 3, hid.your_deck
        assert len(hid.your_prize) == 2, hid.your_prize
        assert len(hid.opponent_deck) == 4, hid.opponent_deck
        assert len(hid.opponent_prize) == 2, hid.opponent_prize
        assert len(hid.opponent_hand) == 3, hid.opponent_hand
        assert hid.opponent_active == [], hid.opponent_active
        # hidden cards must come from the unseen pool, never a visible board card.
        visible = set(_visible_card_ids(obs["current"]["players"][yi], include_hand=True))
        assert not (set(hid.your_deck) | set(hid.your_prize)) & visible
    print("PASS test_predictor_counts")


def test_predictor_facedown_active():
    obs, meta = _synth_obs(0, opp_active_facedown=True), _synth_meta(0, opp_active_facedown=True)
    hid = UniformDeckPredictor(seed=2).predict(obs, meta, your_index=0)
    assert len(hid.opponent_active) == 1, hid.opponent_active
    print("PASS test_predictor_facedown_active")


def test_predictor_deterministic():
    obs, meta = _synth_obs(0), _synth_meta(0)
    a = UniformDeckPredictor(seed=7).predict(obs, meta, your_index=0)
    b = UniformDeckPredictor(seed=7).predict(obs, meta, your_index=0)
    assert a == b, (a, b)
    c = UniformDeckPredictor(seed=8).predict(obs, meta, your_index=0)
    # different seed should (very likely) permute at least one zone
    assert (a.your_deck, a.opponent_hand) != (c.your_deck, c.opponent_hand)
    print("PASS test_predictor_deterministic")


def test_predictor_requires_decks():
    obs = _synth_obs(0)
    try:
        UniformDeckPredictor().predict(obs, {"decks": None}, your_index=0)
    except ValueError as e:
        assert "decks" in str(e)
        print("PASS test_predictor_requires_decks")
        return
    raise AssertionError("expected ValueError when meta lacks decks")


def test_predictor_short_pool_topup():
    """If the deck pool is smaller than the zones need, _take tops up (never underfills)."""
    obs = _synth_obs(0)
    meta = _synth_meta(0)
    meta["decks"][0] = [100, 200, 300, 400, 401, 10]  # only 1 hidden card, need 5
    hid = UniformDeckPredictor(seed=1).predict(obs, meta, your_index=0)
    assert len(hid.your_deck) == 3 and len(hid.your_prize) == 2  # still exact counts
    print("PASS test_predictor_short_pool_topup")


def test_predictor_is_swappable():
    """A custom HiddenInfoPredictor subclass satisfies the interface analyze_decision uses."""
    class ConstantPredictor(HiddenInfoPredictor):
        def predict(self, obs, meta, *, your_index):
            return HiddenInfo(your_deck=[1], opponent_deck=[2])

    p = ConstantPredictor()
    assert isinstance(p, HiddenInfoPredictor)
    out = p.predict(_synth_obs(0), _synth_meta(0), your_index=0)
    assert out.your_deck == [1] and out.opponent_deck == [2]
    print("PASS test_predictor_is_swappable")


# --------------------------------------------------------------------------- #
# Policy tests (engine-free)
# --------------------------------------------------------------------------- #

def test_policy_fixes_coin():
    coin_select = {"context": cf._COIN_HEAD_CONTEXT, "minCount": 1, "maxCount": 1,
                   "option": [{"type": cf._OPTION_NO}, {"type": cf._OPTION_YES}]}
    assert RandomPolicy(0, coin="heads")(coin_select) == [1]  # YES option index
    assert RandomPolicy(0, coin="tails")(coin_select) == [0]  # NO option index
    print("PASS test_policy_fixes_coin")


def test_policy_count_and_determinism():
    sel = {"context": 0, "minCount": 2, "maxCount": 2, "option": [{"type": 3}] * 5}
    a = RandomPolicy(3)(sel)
    b = RandomPolicy(3)(sel)
    assert a == b and len(a) == 2 and len(set(a)) == 2 and all(0 <= i < 5 for i in a)
    print("PASS test_policy_count_and_determinism")


# --------------------------------------------------------------------------- #
# Parsing / selection helpers (engine-free)
# --------------------------------------------------------------------------- #

def test_default_alternatives_single_and_multi():
    obs = _synth_obs(0)  # 3 options, single-select
    alts = default_alternatives(obs, actual=[1])
    assert alts == [[0], [2]], alts
    # multi-select choice → no auto alternatives
    assert default_alternatives(obs, actual=[0, 1]) == []
    print("PASS test_default_alternatives_single_and_multi")


def test_find_decision_and_full_obs_guards():
    trace = {"meta": {}, "decisions": [{"index": 0, "obs": _synth_obs(0), "choice": [1]}], "result": None}
    assert find_decision(trace, 0)["index"] == 0
    try:
        find_decision(trace, 99)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing decision index")
    # a LOGS-level decision (no obs) is rejected with a clear message.
    try:
        _require_full_obs({"index": 3, "choice": [0]}, 3)
    except ValueError as e:
        assert "full_obs" in str(e)
        print("PASS test_find_decision_and_full_obs_guards")
        return
    raise AssertionError("expected ValueError for missing full observation")


# --------------------------------------------------------------------------- #
# Engine-backed end-to-end tests (skipped without cg/)
# --------------------------------------------------------------------------- #

def _engine_available():
    try:
        import cg.game  # noqa: F401
        return True
    except Exception:
        return False


def _record_full_obs_trace(path):
    from eval import record_match as rm
    d0 = rm.load_deck("deck.csv")
    from eval.trace import RecordLevel
    rm.record_match(d0, d0, out_path=path, level=RecordLevel.FULL_OBS, max_steps=400)


def _pick_decision(trace):
    """Earliest single-select decision with >1 option (full decks ⇒ shuffle-free rollout)."""
    for d in trace["decisions"]:
        obs = d.get("obs")
        if not obs:
            continue
        sel = obs.get("select") or {}
        opt = sel.get("option") or []
        if len(opt) > 1 and sel.get("minCount", 1) <= 1 <= sel.get("maxCount", 1) \
                and len(cf._choice_indices(d.get("choice"))) == 1:
            return d["index"]
    return None


def test_analyze_returns_actual_and_alternative():
    if not _engine_available():
        print("SKIP test_analyze_returns_actual_and_alternative (engine unavailable)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cf.jsonl")
        _record_full_obs_trace(path)
        trace = cf.load_trace(path)
        idx = _pick_decision(trace)
        assert idx is not None, "no suitable single-select decision found"
        report = cf.analyze_decision(trace, idx, seed=0, max_depth=30, rollouts=1)
        labels = [b["label"] for b in report["branches"]]
        assert labels[0] == "actual", labels
        assert len(labels) >= 2, labels                        # actual + >=1 alternative
        for b in report["branches"]:
            assert "selection" in b and b["n_rollouts"] == 1
        print("PASS test_analyze_returns_actual_and_alternative")


def test_reproducible_with_fixed_coin():
    """Same seed + manual_coin ⇒ identical rollout (up to the engine's seedless internal
    shuffles, C2). A bounded early-game rollout is shuffle-free and bit-reproducible."""
    if not _engine_available():
        print("SKIP test_reproducible_with_fixed_coin (engine unavailable)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cf.jsonl")
        _record_full_obs_trace(path)
        trace = cf.load_trace(path)
        idx = _pick_decision(trace)
        r1 = cf.analyze_decision(trace, idx, seed=0, max_depth=25, manual_coin=True, rollouts=1)
        r2 = cf.analyze_decision(trace, idx, seed=0, max_depth=25, manual_coin=True, rollouts=1)
        assert r1["branches"] == r2["branches"], "fixed-coin rollout was not reproducible"
        print("PASS test_reproducible_with_fixed_coin")


def test_session_always_released():
    """A branch whose rollout raises must not leak a search session: subsequent
    analyses on the same process still succeed (search_release/search_end ran)."""
    if not _engine_available():
        print("SKIP test_session_always_released (engine unavailable)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cf.jsonl")
        _record_full_obs_trace(path)
        trace = cf.load_trace(path)
        idx = _pick_decision(trace)
        # An out-of-range alternative selection forces the first search_step to raise;
        # the branch records the error but the session is released via try/finally.
        n_opt = None
        for d in trace["decisions"]:
            if d["index"] == idx:
                n_opt = len(d["obs"]["select"]["option"])
        bad = [[n_opt + 5]]  # invalid option index
        report = cf.analyze_decision(trace, idx, alt_selections=bad, seed=0, max_depth=20)
        alt = report["branches"][1]
        assert alt["errors"] == 1, alt                          # error captured
        # If the session leaked, this second analysis would fail; assert it still works.
        ok = cf.analyze_decision(trace, idx, seed=1, max_depth=20)
        assert ok["branches"][0]["label"] == "actual"
        print("PASS test_session_always_released")


if __name__ == "__main__":
    test_predictor_counts()
    test_predictor_facedown_active()
    test_predictor_deterministic()
    test_predictor_requires_decks()
    test_predictor_short_pool_topup()
    test_predictor_is_swappable()
    test_policy_fixes_coin()
    test_policy_count_and_determinism()
    test_default_alternatives_single_and_multi()
    test_find_decision_and_full_obs_guards()
    test_analyze_returns_actual_and_alternative()
    test_reproducible_with_fixed_coin()
    test_session_always_released()
    print("ALL TESTS PASSED")
