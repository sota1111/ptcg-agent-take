"""Tests for the search agent skeleton (SOT-1657).

No pytest dependency — run directly from the repo root:
    venv/bin/python agents/test_search_agent.py

Covers the two acceptance criteria:
  1. SearchAgent plays as an ``eval/record_match.py`` Agent against the
     rule-based agent and the match runs to completion (no agent/engine failure);
  2. the outermost guard — an unknown / newly-appended enum, a degenerate
     observation, a raising internal step, an abnormal search state, and a
     raising evaluator all degrade to a legal random selection (never raise,
     never emit an illegal move, never crash the match).
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from cg.api import (  # noqa: E402
    Observation,
    Option,
    OptionType,
    PlayerState,
    SelectContext,
    SelectData,
    SelectType,
    State,
    to_observation_class,
)

from agents.base import is_valid_selection  # noqa: E402
from agents.search_agent import SearchAgent, default_evaluate  # noqa: E402


def _make_select(n_options: int, min_count: int, max_count: int,
                 context: object = SelectContext.SWITCH,
                 stype: object = SelectType.CARD) -> SelectData:
    """Build a synthetic single-select SelectData with generic options."""
    return SelectData(
        type=stype,
        context=context,
        minCount=min_count,
        maxCount=max_count,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=[Option(type=OptionType.CARD, index=i) for i in range(n_options)],
        deck=None,
        contextCard=None,
        effect=None,
    )


def _obs(select) -> Observation:
    # current=None and no search_begin_input: the real search cannot run, so the
    # guard must carry these to a legal random selection.
    return Observation(select=select, logs=[], current=None)


def test_initial_selection_returns_deck():
    agent = SearchAgent(seed=0)
    deck = agent.decide(_obs(None))
    assert isinstance(deck, list) and len(deck) == 60, len(deck)
    print("PASS test_initial_selection_returns_deck")


def test_single_select_falls_back_to_valid():
    # A degenerate observation (no current/search_begin_input) can't be searched;
    # the guard must still return a legal selection.
    agent = SearchAgent(seed=1)
    sel = _make_select(5, 1, 2)
    out = agent.decide(_obs(sel))
    assert is_valid_selection(out, sel), out
    print("PASS test_single_select_falls_back_to_valid")


def test_unknown_enum_returns_valid():
    # Enum values the engine may append later must miss and fall back, not crash.
    agent = SearchAgent(seed=2)
    sel = _make_select(4, 1, 1, context=9999, stype=8888)
    out = agent.decide(_obs(sel))
    assert is_valid_selection(out, sel), out
    print("PASS test_unknown_enum_returns_valid")


def test_min_count_zero_is_valid():
    # Optional selection (minCount == 0): the empty selection is a legal option.
    agent = SearchAgent(seed=3)
    sel = _make_select(3, 0, 1)
    out = agent.decide(_obs(sel))
    assert is_valid_selection(out, sel), out
    print("PASS test_min_count_zero_is_valid")


def test_guard_on_raising_internal_step():
    # If the search core itself raises, decide must still return a legal move.
    agent = SearchAgent(seed=4)
    sel = _make_select(6, 1, 3)

    def boom(_obs):
        raise RuntimeError("search core blew up")

    agent._search_decide = boom  # type: ignore[assignment]
    out = agent.decide(_obs(sel))
    assert is_valid_selection(out, sel), out
    print("PASS test_guard_on_raising_internal_step")


def test_abnormal_search_state_scores_none():
    # Scoring a candidate on an observation that is not a real agent observation
    # (no search_begin_input) must return None, not raise — so the candidate is
    # skipped and the agent falls back.
    agent = SearchAgent(seed=5)
    sel = _make_select(3, 1, 1)
    score = agent._score_candidate(_obs(sel), ([], [], [], [], [], []), [0], 0)
    assert score is None, score
    print("PASS test_abnormal_search_state_scores_none")


def _empty_player() -> PlayerState:
    return PlayerState(
        active=[], bench=[], benchMax=5, deckCount=0, discard=[], prize=[],
        handCount=0, hand=None, poisoned=False, burned=False, asleep=False,
        paralyzed=False, confused=False,
    )


def _terminal_obs(result: int) -> Observation:
    state = State(
        turn=10, turnActionCount=0, yourIndex=0, firstPlayer=0,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
        retreated=False, result=result, stadium=[], looking=None,
        players=[_empty_player(), _empty_player()],
    )
    return Observation(select=None, logs=[], current=state)


def test_default_evaluate_prefers_winning():
    # A terminal position we won scores far above a losing one.
    won = _terminal_obs(result=0)
    assert default_evaluate(won, your_index=0) > 0
    assert default_evaluate(won, your_index=1) < 0
    print("PASS test_default_evaluate_prefers_winning")


def test_search_agent_completes_vs_rulebased():
    """AC1: N full matches SearchAgent vs RuleBasedAgent, zero failures."""
    from agents.rule_based import RuleBasedAgent
    from eval import record_match as rm

    def search_fn(seed):
        a = SearchAgent(seed=seed, time_budget_s=0.05, max_candidates=6)
        return lambda obs_dict: a.decide(to_observation_class(obs_dict))

    def rule_fn(seed):
        a = RuleBasedAgent(seed=seed)
        return lambda obs_dict: a.decide(to_observation_class(obs_dict))

    deck = rm.load_deck("deck.csv")
    n_matches = 3
    for i in range(n_matches):
        agents = (
            rm.Agent(search_fn(i), name="search0", version="1"),
            rm.Agent(rule_fn(100 + i), name="rule1", version="1"),
        )
        out = f"eval/traces/_search_smoke_{i}.jsonl"
        summary = rm.record_match(deck, deck, agents=agents, out_path=out)
        assert summary["failure"] is None, (i, summary["failure"])
        assert summary["decisions"] > 0, (i, summary)
        try:
            os.remove(out)
        except OSError:
            pass
    print(f"PASS test_search_agent_completes_vs_rulebased ({n_matches} matches)")


def test_raising_evaluator_still_completes():
    """AC2: an evaluator that always raises never crashes the match (fallback)."""
    from eval import record_match as rm

    def boom(_observation, _your_index):
        raise RuntimeError("evaluator blew up")

    def search_fn(seed):
        a = SearchAgent(seed=seed, evaluate=boom, time_budget_s=0.05, max_candidates=6)
        return lambda obs_dict: a.decide(to_observation_class(obs_dict))

    deck = rm.load_deck("deck.csv")
    agents = (
        rm.Agent(search_fn(7), name="search_boom", version="1"),
        rm.Agent(rm.make_random_agent(200, "random1").fn, name="random1", version="1"),
    )
    out = "eval/traces/_search_boom_smoke.jsonl"
    summary = rm.record_match(deck, deck, agents=agents, out_path=out)
    assert summary["failure"] is None, summary["failure"]
    assert summary["decisions"] > 0, summary
    try:
        os.remove(out)
    except OSError:
        pass
    print("PASS test_raising_evaluator_still_completes")


if __name__ == "__main__":
    test_initial_selection_returns_deck()
    test_single_select_falls_back_to_valid()
    test_unknown_enum_returns_valid()
    test_min_count_zero_is_valid()
    test_guard_on_raising_internal_step()
    test_abnormal_search_state_scores_none()
    test_default_evaluate_prefers_winning()
    test_search_agent_completes_vs_rulebased()
    test_raising_evaluator_still_completes()
    print("ALL TESTS PASSED")
