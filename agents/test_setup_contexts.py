"""Tests for the setup / forced-selection context handlers (SOT-1634, R3).

No pytest dependency — run directly from the repo root:
    venv/bin/python agents/test_setup_contexts.py

Each ``decide`` function is exercised on a synthetic position built from **real
engine card ids** (so ``CardData`` — HP, type, evolution line — is the engine's,
not a mock). Covers:
  1. SETUP_ACTIVE / SETUP_BENCH — prefers a Basic that heads an evolution line;
  2. TO_ACTIVE — prefers the most-Energised, then highest-HP Bench Pokémon;
  3. IS_FIRST / COIN_HEAD / MULLIGAN — fixed YesNo defaults;
  4. DISCARD / TO_DECK — sheds surplus (Basic) Energy first, keeps Pokémon;
  5. DISCARD_ENERGY — sheds Basic before Special Energy;
  6. DAMAGE_COUNTER / DAMAGE — targets the lowest-HP Pokémon (finish the KO);
  7. HEAL — prefers the Active Pokémon;
  8. every handler still produces a legal selection (or defers) and the full
     dispatch never emits an illegal move, matched by a record_match sweep that
     asserts zero exceptions across all contexts the games actually reach.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from cg.api import (  # noqa: E402
    AreaType,
    Card,
    Observation,
    Option,
    OptionType,
    Pokemon,
    PlayerState,
    SelectContext,
    SelectData,
    SelectType,
    State,
    to_observation_class,
)

from agents import damage  # noqa: E402
from agents.base import is_valid_selection  # noqa: E402
from agents.rule_based import RuleBasedAgent  # noqa: E402

# Real engine card ids used across the fixtures.
KYOGRE = 721      # Basic, HP 150, no evolution line
SNOVER = 722      # Basic, HP 90, heads an evolution line (-> Mega Abomasnow ex)
ABOMA = 723       # Stage1, HP 350 (not Basic)
BASIC_W = 3       # Basic {W} Energy
SPECIAL_E = 9     # a Special Energy
ITEM = 1077       # an Item card


def _poke(cid: int, hp: int, *, max_hp: int | None = None, n_energy: int = 0) -> Pokemon:
    """A live Pokémon carrying ``n_energy`` Basic {W} Energy cards."""
    energy_cards = [Card(id=BASIC_W, serial=900 + i, playerIndex=0) for i in range(n_energy)]
    return Pokemon(
        id=cid, serial=cid, hp=hp, maxHp=max_hp if max_hp is not None else hp,
        appearThisTurn=False, energies=[3] * n_energy, energyCards=energy_cards,
        tools=[], preEvolution=[],
    )


def _player(*, hand=None, active=None, bench=None) -> PlayerState:
    return PlayerState(
        active=list(active) if active is not None else [],
        bench=list(bench) if bench is not None else [],
        benchMax=5, deckCount=40, discard=[], prize=[], handCount=len(hand or []),
        hand=list(hand) if hand is not None else None,
        poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )


def _obs(select: SelectData, players: list[PlayerState], your_index: int = 0) -> Observation:
    state = State(
        turn=1, turnActionCount=0, yourIndex=your_index, firstPlayer=your_index,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
        retreated=False, result=-1, stadium=[], looking=None, players=players,
    )
    return Observation(select=select, logs=[], current=state)


def _select(context, options, *, stype=SelectType.CARD, min_count=1, max_count=1,
            remain_energy=0, remain_dmg=0) -> SelectData:
    return SelectData(
        type=stype, context=context, minCount=min_count, maxCount=max_count,
        remainDamageCounter=remain_dmg, remainEnergyCost=remain_energy,
        option=options, deck=None, contextCard=None, effect=None,
    )


def _card_opt(area, index, player_index=0) -> Option:
    return Option(type=OptionType.CARD, area=area, index=index, playerIndex=player_index)


# --------------------------------------------------------------------------- #
# 1. setup: place the best Basic
# --------------------------------------------------------------------------- #

def test_setup_active_prefers_evolution_line():
    agent = RuleBasedAgent(seed=0)
    hand = [Card(id=KYOGRE, serial=1, playerIndex=0),   # Basic, HP 150, no line
            Card(id=SNOVER, serial=2, playerIndex=0),   # Basic, HP 90, has line
            Card(id=BASIC_W, serial=3, playerIndex=0)]  # not a Pokémon
    sel = _select(SelectContext.SETUP_ACTIVE_POKEMON,
                  [_card_opt(AreaType.HAND, 0), _card_opt(AreaType.HAND, 1), _card_opt(AreaType.HAND, 2)])
    out = agent.decide(_obs(sel, [_player(hand=hand), _player()]))
    assert out == [1], out  # Snover — evolution line ranks above HP
    print("PASS test_setup_active_prefers_evolution_line")


def test_setup_bench_optional_still_develops():
    agent = RuleBasedAgent(seed=0)
    hand = [Card(id=KYOGRE, serial=1, playerIndex=0), Card(id=SNOVER, serial=2, playerIndex=0)]
    # minCount 0 (optional) — but we still place the best Basic to develop.
    sel = _select(SelectContext.SETUP_BENCH_POKEMON,
                  [_card_opt(AreaType.HAND, 0), _card_opt(AreaType.HAND, 1)],
                  min_count=0, max_count=1)
    out = agent.decide(_obs(sel, [_player(hand=hand), _player()]))
    assert out == [1], out  # Snover (has line)
    print("PASS test_setup_bench_optional_still_develops")


# --------------------------------------------------------------------------- #
# 2. TO_ACTIVE: readiest Bench Pokémon
# --------------------------------------------------------------------------- #

def test_to_active_prefers_energised_then_hp():
    agent = RuleBasedAgent(seed=0)
    bench = [_poke(KYOGRE, 60, n_energy=2), _poke(ABOMA, 350, n_energy=0), _poke(SNOVER, 90, n_energy=1)]
    sel = _select(SelectContext.TO_ACTIVE,
                  [_card_opt(AreaType.BENCH, 0), _card_opt(AreaType.BENCH, 1), _card_opt(AreaType.BENCH, 2)])
    out = agent.decide(_obs(sel, [_player(bench=bench), _player()]))
    assert out == [0], out  # 2 energy beats the 350-HP but energy-less one
    print("PASS test_to_active_prefers_energised_then_hp")


def test_to_active_hp_breaks_energy_tie():
    agent = RuleBasedAgent(seed=0)
    bench = [_poke(SNOVER, 90, n_energy=1), _poke(KYOGRE, 150, n_energy=1)]
    sel = _select(SelectContext.TO_ACTIVE,
                  [_card_opt(AreaType.BENCH, 0), _card_opt(AreaType.BENCH, 1)])
    out = agent.decide(_obs(sel, [_player(bench=bench), _player()]))
    assert out == [1], out  # equal energy -> higher HP (Kyogre)
    print("PASS test_to_active_hp_breaks_energy_tie")


# --------------------------------------------------------------------------- #
# 3. YesNo defaults
# --------------------------------------------------------------------------- #

def test_yesno_defaults():
    agent = RuleBasedAgent(seed=0)
    for ctx in (SelectContext.IS_FIRST, SelectContext.COIN_HEAD, SelectContext.MULLIGAN):
        # options in YES-then-NO order -> pick YES (index 0)
        sel = _select(ctx, [Option(type=OptionType.YES), Option(type=OptionType.NO)],
                      stype=SelectType.YES_NO)
        out = agent.decide(_obs(sel, [_player(), _player()]))
        assert out == [0], (ctx, out)
    # NO-then-YES order -> still pick YES (index 1), not positional
    sel = _select(SelectContext.MULLIGAN, [Option(type=OptionType.NO), Option(type=OptionType.YES)],
                  stype=SelectType.YES_NO)
    out = agent.decide(_obs(sel, [_player(), _player()]))
    assert out == [1], out
    print("PASS test_yesno_defaults")


# --------------------------------------------------------------------------- #
# 4-5. discard surplus energy first
# --------------------------------------------------------------------------- #

def test_discard_card_sheds_energy_keeps_pokemon():
    agent = RuleBasedAgent(seed=0)
    hand = [Card(id=KYOGRE, serial=1, playerIndex=0),   # Pokémon (keep)
            Card(id=BASIC_W, serial=2, playerIndex=0),  # Basic energy (shed first)
            Card(id=ITEM, serial=3, playerIndex=0)]      # item
    sel = _select(SelectContext.DISCARD,
                  [_card_opt(AreaType.HAND, 0), _card_opt(AreaType.HAND, 1), _card_opt(AreaType.HAND, 2)])
    out = agent.decide(_obs(sel, [_player(hand=hand), _player()]))
    assert out == [1], out  # the Basic energy
    print("PASS test_discard_card_sheds_energy_keeps_pokemon")


def test_discard_optional_is_noop():
    agent = RuleBasedAgent(seed=0)
    hand = [Card(id=BASIC_W, serial=1, playerIndex=0)]
    sel = _select(SelectContext.DISCARD, [_card_opt(AreaType.HAND, 0)], min_count=0, max_count=1)
    out = agent.decide(_obs(sel, [_player(hand=hand), _player()]))
    assert out == [], out  # optional discard -> select nothing
    print("PASS test_discard_optional_is_noop")


def test_discard_energy_prefers_basic_over_special():
    agent = RuleBasedAgent(seed=0)
    active = _poke(KYOGRE, 100, n_energy=0)
    active.energyCards = [Card(id=SPECIAL_E, serial=1, playerIndex=0),
                          Card(id=BASIC_W, serial=2, playerIndex=0)]
    active.energies = [0, 3]
    opts = [Option(type=OptionType.ENERGY, area=AreaType.ACTIVE, index=0, playerIndex=0, energyIndex=0, count=1),
            Option(type=OptionType.ENERGY, area=AreaType.ACTIVE, index=0, playerIndex=0, energyIndex=1, count=1)]
    sel = _select(SelectContext.DISCARD_ENERGY, opts, stype=SelectType.ENERGY, remain_energy=1)
    out = agent.decide(_obs(sel, [_player(active=[active]), _player()]))
    assert out == [1], out  # the Basic energy (energyIndex 1), not the Special
    print("PASS test_discard_energy_prefers_basic_over_special")


# --------------------------------------------------------------------------- #
# 6. damage placement finishes a KO
# --------------------------------------------------------------------------- #

def test_damage_targets_lowest_hp():
    agent = RuleBasedAgent(seed=0)
    # Opponent (player 1) is the target of DAMAGE; we (player 0) select.
    opp = _player(active=[_poke(KYOGRE, 120)], bench=[_poke(SNOVER, 30)])
    opts = [Option(type=OptionType.CARD, area=AreaType.ACTIVE, index=0, playerIndex=1),
            Option(type=OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=1)]
    sel = _select(SelectContext.DAMAGE, opts, remain_dmg=3)
    out = agent.decide(_obs(sel, [_player(), opp]))
    assert out == [1], out  # 30 HP bench Pokémon is closest to a KO
    print("PASS test_damage_targets_lowest_hp")


# --------------------------------------------------------------------------- #
# 7. heal prefers the Active
# --------------------------------------------------------------------------- #

def test_heal_prefers_active():
    agent = RuleBasedAgent(seed=0)
    me = _player(active=[_poke(KYOGRE, 40, max_hp=150)], bench=[_poke(SNOVER, 10, max_hp=90)])
    opts = [Option(type=OptionType.CARD, area=AreaType.ACTIVE, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=0)]
    sel = _select(SelectContext.HEAL, opts)
    out = agent.decide(_obs(sel, [me, _player()]))
    assert out == [0], out  # Active preferred even though the bench took more damage
    print("PASS test_heal_prefers_active")


# --------------------------------------------------------------------------- #
# 8. robustness: handlers never emit an illegal move; full sweep, zero exceptions
# --------------------------------------------------------------------------- #

def test_handlers_defer_safely_on_degenerate_obs():
    """A malformed / empty observation must never raise or emit an illegal move."""
    agent = RuleBasedAgent(seed=0)
    for ctx in RuleBasedAgent.CONTEXT_HANDLERS:
        sel = _select(ctx, [_card_opt(AreaType.HAND, 99)])  # index out of range
        out = agent.decide(Observation(select=sel, logs=[], current=None))
        assert is_valid_selection(out, sel), (ctx, out)
    print("PASS test_handlers_defer_safely_on_degenerate_obs")


def test_record_match_zero_exceptions_all_contexts():
    """N full matches: every reached context handled with zero agent/engine errors."""
    from eval import record_match as rm  # noqa: E402

    def rule_fn(seed):
        a = RuleBasedAgent(seed=seed)
        return lambda obs_dict: a.decide(to_observation_class(obs_dict))

    deck = rm.load_deck("deck.csv")
    n_matches = 12
    for i in range(n_matches):
        agents = (
            rm.Agent(rule_fn(i), name="rule0", version="1"),
            rm.Agent(rule_fn(500 + i), name="rule1", version="1"),
        )
        out = f"eval/traces/_r3_smoke_{i}.jsonl"
        summary = rm.record_match(deck, deck, agents=agents, out_path=out)
        assert summary["failure"] is None, (i, summary["failure"])
        assert summary["decisions"] > 0, (i, summary)
        try:
            os.remove(out)
        except OSError:
            pass
    print(f"PASS test_record_match_zero_exceptions_all_contexts ({n_matches} matches)")


if __name__ == "__main__":
    test_setup_active_prefers_evolution_line()
    test_setup_bench_optional_still_develops()
    test_to_active_prefers_energised_then_hp()
    test_to_active_hp_breaks_energy_tie()
    test_yesno_defaults()
    test_discard_card_sheds_energy_keeps_pokemon()
    test_discard_optional_is_noop()
    test_discard_energy_prefers_basic_over_special()
    test_damage_targets_lowest_hp()
    test_heal_prefers_active()
    test_handlers_defer_safely_on_degenerate_obs()
    test_record_match_zero_exceptions_all_contexts()
    print("ALL TESTS PASSED")
