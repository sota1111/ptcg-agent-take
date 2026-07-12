"""Tests for the agents package (SOT-1632).

No pytest dependency — run directly from the repo root:
    venv/bin/python agents/test_agents.py

Covers:
  1. shared helpers — legal_random_sample always returns a valid selection, and
     is_valid_selection accepts/rejects the right shapes;
  2. RandomAgent + RuleBasedAgent produce legal selections and return the deck on
     the initial selection;
  3. the rule-based safety guard — an unimplemented context, an unknown enum
     value, a raising handler, and a handler returning an invalid result all fall
     back to a legal random selection (never raise, never emit an illegal move);
  4. an integration check: rule_based (all-fallback) completes N record_match
     games with zero agent/engine exceptions, i.e. behaves like random.
"""
from __future__ import annotations

import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from cg.api import (  # noqa: E402
    Observation,
    Option,
    OptionType,
    SelectContext,
    SelectData,
    SelectType,
    to_observation_class,
)

from agents.base import is_valid_selection, legal_random_sample, read_deck_csv  # noqa: E402
from agents.random_agent import RandomAgent  # noqa: E402
from agents.rule_based import RuleBasedAgent  # noqa: E402


def _make_select(n_options: int, min_count: int, max_count: int,
                 context: object = SelectContext.SWITCH,
                 stype: object = SelectType.CARD) -> SelectData:
    """Build a synthetic SelectData with ``n_options`` generic options."""
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
    return Observation(select=select, logs=[], current=None)


def test_legal_random_sample_is_always_valid():
    rng = random.Random(0)
    # Sweep option counts and bounds, including degenerate (0 options / bounds).
    for n in range(0, 8):
        for lo in range(0, n + 2):
            for hi in range(lo, n + 2):
                sel = _make_select(n, lo, min(hi, n))
                out = legal_random_sample(sel, rng)
                assert isinstance(out, list), (n, lo, hi, out)
                assert len(set(out)) == len(out), out          # unique
                assert all(0 <= i < n for i in out), (n, out)   # in range
                # length within the clamped bounds
                assert min(lo, n) <= len(out) <= min(hi, n), (n, lo, hi, out)
    print("PASS test_legal_random_sample_is_always_valid")


def test_is_valid_selection():
    sel = _make_select(4, 1, 2)
    assert is_valid_selection([0], sel)
    assert is_valid_selection([0, 3], sel)
    assert not is_valid_selection([], sel)            # below minCount
    assert not is_valid_selection([0, 1, 2], sel)     # above maxCount
    assert not is_valid_selection([0, 0], sel)        # duplicate
    assert not is_valid_selection([4], sel)           # out of range
    assert not is_valid_selection([-1], sel)          # negative index
    assert not is_valid_selection([True], sel)        # bool is not a valid index
    assert not is_valid_selection("nope", sel)        # not a list
    assert not is_valid_selection([1.0], sel)         # not an int
    print("PASS test_is_valid_selection")


def test_random_agent_selection_and_deck():
    agent = RandomAgent(seed=1)
    sel = _make_select(5, 1, 2)
    out = agent.decide(_obs(sel))
    assert is_valid_selection(out, sel), out
    # Initial selection returns the 60-card deck.
    deck = agent.decide(_obs(None))
    assert isinstance(deck, list) and len(deck) == 60, len(deck)
    # Seed is honoured: same seed -> same sequence.
    a2 = RandomAgent(seed=1)
    assert a2.decide(_obs(sel)) == out
    print("PASS test_random_agent_selection_and_deck")


def test_rule_based_fallback_when_unimplemented():
    agent = RuleBasedAgent(seed=2)
    sel = _make_select(6, 1, 3)
    out = agent.decide(_obs(sel))
    assert is_valid_selection(out, sel), out
    # Initial selection returns the deck.
    assert len(agent.decide(_obs(None))) == 60
    print("PASS test_rule_based_fallback_when_unimplemented")


def test_rule_based_unknown_enum_values():
    # Enum values the engine may append later (E-notes: "new elements may be
    # appended"). Plain ints that are not registered must miss and fall back.
    agent = RuleBasedAgent(seed=3)
    sel = _make_select(4, 1, 1, context=9999, stype=8888)
    out = agent.decide(_obs(sel))
    assert is_valid_selection(out, sel), out
    print("PASS test_rule_based_unknown_enum_values")


def test_rule_based_guard_on_raising_handler():
    agent = RuleBasedAgent(seed=4)
    sel = _make_select(5, 1, 2)

    def boom(a, s, o):
        raise RuntimeError("handler blew up")

    # Register directly on the instance's class table for this test, then clean up.
    RuleBasedAgent.CONTEXT_HANDLERS[sel.context] = boom
    try:
        out = agent.decide(_obs(sel))
    finally:
        RuleBasedAgent.CONTEXT_HANDLERS.pop(sel.context, None)
    assert is_valid_selection(out, sel), out
    print("PASS test_rule_based_guard_on_raising_handler")


def test_rule_based_guard_on_invalid_handler_result():
    agent = RuleBasedAgent(seed=5)
    sel = _make_select(3, 1, 1)

    def too_many(a, s, o):
        return [0, 1, 2]  # exceeds maxCount -> must be rejected, fall back

    RuleBasedAgent.TYPE_HANDLERS[sel.type] = too_many
    try:
        out = agent.decide(_obs(sel))
    finally:
        RuleBasedAgent.TYPE_HANDLERS.pop(sel.type, None)
    assert is_valid_selection(out, sel), out
    print("PASS test_rule_based_guard_on_invalid_handler_result")


def test_rule_based_registered_handler_is_used():
    agent = RuleBasedAgent(seed=6)
    sel = _make_select(4, 1, 1, context=SelectContext.SWITCH)

    def pick_last(a, s, o):
        return [len(s.option) - 1]

    RuleBasedAgent.CONTEXT_HANDLERS[SelectContext.SWITCH] = pick_last
    try:
        out = agent.decide(_obs(sel))
    finally:
        RuleBasedAgent.CONTEXT_HANDLERS.pop(SelectContext.SWITCH, None)
    assert out == [3], out
    print("PASS test_rule_based_registered_handler_is_used")


def test_rule_based_completes_matches_like_random():
    """N full matches with the all-fallback rule_based agent: zero exceptions."""
    from eval import record_match as rm  # noqa: E402

    def rule_fn(seed):
        a = RuleBasedAgent(seed=seed)
        return lambda obs_dict: a.decide(to_observation_class(obs_dict))

    deck = rm.load_deck("deck.csv")
    n_matches = 5
    for i in range(n_matches):
        agents = (
            rm.Agent(rule_fn(i), name="rule0", version="1"),
            rm.Agent(rule_fn(100 + i), name="rule1", version="1"),
        )
        out = f"eval/traces/_rulebased_smoke_{i}.jsonl"
        summary = rm.record_match(deck, deck, agents=agents, out_path=out)
        # No agent crash and no illegal-move engine rejection.
        assert summary["failure"] is None, (i, summary["failure"])
        assert summary["decisions"] > 0, (i, summary)
        try:
            os.remove(out)
        except OSError:
            pass
    print(f"PASS test_rule_based_completes_matches_like_random ({n_matches} matches)")


if __name__ == "__main__":
    test_legal_random_sample_is_always_valid()
    test_is_valid_selection()
    test_random_agent_selection_and_deck()
    test_rule_based_fallback_when_unimplemented()
    test_rule_based_unknown_enum_values()
    test_rule_based_guard_on_raising_handler()
    test_rule_based_guard_on_invalid_handler_result()
    test_rule_based_registered_handler_is_used()
    test_rule_based_completes_matches_like_random()
    print("ALL TESTS PASSED")
