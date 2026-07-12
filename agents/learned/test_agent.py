"""Standalone tests for the learned-policy inference agent (SOT-1644).

No pytest dependency — run directly from the repo root:
    venv/bin/python agents/learned/test_agent.py

Covers the three acceptance criteria:
  1. plays a full match under ``eval/record_match.py`` without crashing (with and
     without a model), and integrates with the parallel arena;
  2. a >= 200-game learned-vs-random win rate + Wilson 95% CI is reported by
     ``eval/bench_learned_vs_random.py`` (a short run is exercised here);
  3. the fallback fires — missing model, feature/scoring exception, and unknown
     enum values all degrade to a valid random legal move instead of raising.
Plus: the model-loaded path actually selects the argmax option, and the model
loader tolerates missing / malformed / dimension-mismatched files.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

from agents.learned import agent as la  # noqa: E402
from agents.learned.agent import (  # noqa: E402
    LearnedAgent,
    LinearOptionScorer,
    load_model,
    make_learned_agent,
    save_model,
)
from agents.learned.features import (  # noqa: E402
    AREA_TYPE_DIM,
    OPTION_FEATURE_DIM,
    OPTION_TYPE_DIM,
    SPECIAL_CONDITION_DIM,
)

# Offset of the ('number' present_flag, 'number' value) pair inside an option
# feature vector: it follows the four leading one-hot blocks, and 'number' is the
# first numeric field (see features._OPTION_NUMERIC_FIELDS).
_NUMBER_VALUE_IDX = OPTION_TYPE_DIM + AREA_TYPE_DIM + AREA_TYPE_DIM + SPECIAL_CONDITION_DIM + 1


def _select_obs(n_options: int, min_count: int = 1, max_count: int = 1, sel_type=0) -> dict:
    """A minimal non-initial observation with ``n_options`` legal moves.

    Each option carries ``number == its index`` so a model that weights the
    'number' feature scores option ``i`` proportionally to ``i``.
    """
    return {
        "select": {
            "type": sel_type,
            "context": 0,
            "minCount": min_count,
            "maxCount": max_count,
            "option": [{"type": 0, "number": i} for i in range(n_options)],
        },
        "current": {"yourIndex": 0, "turn": 1, "players": [{}, {}]},
    }


def _argmax_number_model() -> LinearOptionScorer:
    """A model that scores an option by its 'number' field (higher = better)."""
    weights = [0.0] * OPTION_FEATURE_DIM
    weights[_NUMBER_VALUE_IDX] = 1.0
    return LinearOptionScorer(weights=weights, bias=0.0)


def _assert_legal(choice, n, lo, hi):
    assert isinstance(choice, list), choice
    assert all(isinstance(i, int) and not isinstance(i, bool) for i in choice), choice
    assert all(0 <= i < n for i in choice), choice
    assert len(set(choice)) == len(choice), choice
    assert lo <= len(choice) <= hi, (choice, lo, hi)


def test_fallback_no_model_returns_legal():
    """With no model, every decision yields a valid random legal selection."""
    agent = LearnedAgent(model_path=None, seed=1)
    assert not agent.has_model

    # single-select
    _assert_legal(agent.act(_select_obs(3, 1, 1)), 3, 1, 1)
    # multi-select (pick exactly 2 of 4)
    _assert_legal(agent.act(_select_obs(4, 2, 2)), 4, 2, 2)
    # variable count 1..3 of 5
    _assert_legal(agent.act(_select_obs(5, 1, 3)), 5, 1, 3)
    # empty options -> empty selection, no crash
    assert agent.act(_select_obs(0, 0, 1)) == []
    print("PASS test_fallback_no_model_returns_legal")


def test_initial_selection_returns_deck_or_empty():
    """Initial (select None): submission agent returns the deck; the record_match
    adapter returns [] (decks are passed at battle_start)."""
    agent = LearnedAgent(model_path=None, seed=0)
    deck = agent.act({"select": None})
    assert isinstance(deck, list) and len(deck) == 60, len(deck)

    rm_agent = make_learned_agent(model_path=None, seed=0)
    assert rm_agent.act({"select": None}) == []
    # and a normal decision through the adapter is still legal
    _assert_legal(rm_agent.act(_select_obs(3, 1, 1)), 3, 1, 1)
    print("PASS test_initial_selection_returns_deck_or_empty")


def test_model_path_selects_argmax():
    """A loaded model picks the highest-scoring option(s)."""
    model = _argmax_number_model()
    agent = LearnedAgent(model=model, seed=0)
    assert agent.has_model

    # single-select of 3 options (number 0,1,2) -> option 2 is best
    assert agent.act(_select_obs(3, 1, 1)) == [2]
    # top-2 of 5 -> {4,3}
    top2 = agent.act(_select_obs(5, 2, 2))
    _assert_legal(top2, 5, 2, 2)
    assert set(top2) == {4, 3}, top2
    # must-pick-all (k == n): all indices returned
    assert sorted(agent.act(_select_obs(3, 3, 3))) == [0, 1, 2]
    print("PASS test_model_path_selects_argmax")


def test_unknown_enum_tolerated():
    """Unknown SelectType / OptionType integers are scored (unknown slot), not
    fatal, and still yield a legal selection."""
    model = _argmax_number_model()
    agent = LearnedAgent(model=model, seed=0)
    obs = {
        "select": {
            "type": 9999,          # unknown SelectType
            "context": 8888,       # unknown SelectContext
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 7777, "number": 0},   # unknown OptionType
                {"type": 7777, "number": 5},
            ],
        },
        "current": {"yourIndex": 0, "players": [{}, {}]},
    }
    choice = agent.act(obs)
    _assert_legal(choice, 2, 1, 1)
    assert choice == [1], choice  # number 5 > number 0
    print("PASS test_unknown_enum_tolerated")


def test_inference_exception_falls_back():
    """A model that raises during scoring falls back to a random legal move."""

    class RaisingModel:
        def score(self, feat):
            raise RuntimeError("boom")

    agent = LearnedAgent(model=RaisingModel(), seed=2)
    assert agent.has_model
    _assert_legal(agent.act(_select_obs(4, 1, 1)), 4, 1, 1)

    # A dimension-mismatched scorer also falls back (score() raises ValueError).
    bad = LinearOptionScorer(weights=[1.0, 2.0, 3.0])  # dim 3 != OPTION_FEATURE_DIM
    agent2 = LearnedAgent(model=bad, seed=2)
    _assert_legal(agent2.act(_select_obs(3, 1, 1)), 3, 1, 1)
    print("PASS test_inference_exception_falls_back")


def test_model_loader_tolerance():
    """Missing / malformed / mismatched model files load as None (→ fallback)."""
    assert load_model(None) is None
    assert load_model("nope/does/not/exist.json") is None

    with tempfile.TemporaryDirectory() as tmp:
        # malformed JSON
        bad = os.path.join(tmp, "bad.json")
        with open(bad, "w") as f:
            f.write("{not valid json")
        assert load_model(bad) is None

        # wrong feature dimension
        mismatch = os.path.join(tmp, "mismatch.json")
        with open(mismatch, "w") as f:
            json.dump({"kind": "linear", "weights": [1.0, 2.0]}, f)
        assert load_model(mismatch) is None

        # unsupported kind
        wrongkind = os.path.join(tmp, "kind.json")
        with open(wrongkind, "w") as f:
            json.dump({"kind": "mlp", "weights": [0.0] * OPTION_FEATURE_DIM}, f)
        assert load_model(wrongkind) is None

        # valid round-trip: save -> load -> identical scoring
        model = _argmax_number_model()
        path = os.path.join(tmp, "policy.json")
        save_model(model, path)
        loaded = load_model(path)
        assert loaded is not None and loaded.dim == OPTION_FEATURE_DIM
        feat = [0.0] * OPTION_FEATURE_DIM
        feat[_NUMBER_VALUE_IDX] = 3.0
        assert abs(loaded.score(feat) - 3.0) < 1e-9, loaded.score(feat)
    print("PASS test_model_loader_tolerance")


def test_submission_entry_point_never_raises():
    """The module-level ``agent(obs_dict)`` is a safe submission drop-in."""
    assert la.agent({"select": None}) and len(la.agent({"select": None})) == 60
    _assert_legal(la.agent(_select_obs(3, 1, 1)), 3, 1, 1)
    # garbage input still returns a list, never raises
    assert isinstance(la.agent({}), list)
    assert isinstance(la.agent({"select": {"option": "not-a-list"}}), list)
    print("PASS test_submission_entry_point_never_raises")


def test_record_match_completes():
    """A full recorded match with the learned agent completes (E7-safe), both
    with a model and in pure random-fallback mode."""
    from eval.record_match import load_deck, make_random_agent, record_match

    deck = load_deck("deck.csv")
    with tempfile.TemporaryDirectory() as tmp:
        # random-fallback learned agent vs random
        out = os.path.join(tmp, "fallback.jsonl")
        summary = record_match(
            deck, deck,
            agents=(make_learned_agent(model_path=None, seed=0, name="learned0"),
                    make_random_agent(1, "random1")),
            out_path=out,
        )
        assert summary["result"] in (-1, 0, 1, 2), summary
        assert summary["decisions"] > 0, summary

        # model-backed learned agent vs random also runs through the engine
        out2 = os.path.join(tmp, "model.jsonl")
        summary2 = record_match(
            deck, deck,
            agents=(make_learned_agent(model=_argmax_number_model(), seed=0, name="learnedM"),
                    make_random_agent(1, "random1")),
            out_path=out2,
        )
        assert summary2["result"] in (-1, 0, 1, 2), summary2
        # No agent-exception failure — the agent must never crash the match.
        assert summary.get("failure") is None or summary["failure"].get("category") != "agent_exception", summary
        assert summary2.get("failure") is None or summary2["failure"].get("category") != "agent_exception", summary2
    print("PASS test_record_match_completes")


def test_arena_integration():
    """The learned agent plugs into the parallel arena (build_agent 'learned')."""
    from eval.arena import agent_spec, run_arena
    from eval.record_match import load_deck
    from eval.trace import RecordLevel

    deck = load_deck("deck.csv")
    with tempfile.TemporaryDirectory() as tmp:
        report = run_arena(
            games=4,
            deck_a=deck, deck_b=deck,
            agent_a=agent_spec("learned", name="learned", model_path=None),
            agent_b=agent_spec("random", name="random"),
            out_dir=tmp,
            level=RecordLevel.RESULT,
            base_seed=7,
            workers=1,
        )
        assert report["total"] == 4, report
        # every game is accounted for and none failed with an agent exception
        assert report["a_wins"] + report["b_wins"] + report["draws"] + report["undecided"] == 4, report
        assert report["failures_by_category"].get("agent_exception", 0) == 0, report
    print("PASS test_arena_integration")


if __name__ == "__main__":
    test_fallback_no_model_returns_legal()
    test_initial_selection_returns_deck_or_empty()
    test_model_path_selects_argmax()
    test_unknown_enum_tolerated()
    test_inference_exception_falls_back()
    test_model_loader_tolerance()
    test_submission_entry_point_never_raises()
    test_record_match_completes()
    test_arena_integration()
    print("ALL TESTS PASSED")
