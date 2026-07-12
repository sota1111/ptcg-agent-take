"""Tests for damage calculation and the MAIN-turn policy (SOT-1633).

No pytest dependency — run directly from the repo root:
    venv/bin/python agents/test_damage.py

Covers:
  1. ``damage.attack_damage`` — weakness (×2), resistance (−30), both, neither,
     0-damage / effect-only attacks, the >=0 clamp, and missing card data;
  2. ``damage.is_ko`` — exact / over / under current HP, unknown or non-positive
     HP, and 0-damage attacks;
  3. the engine-backed registries return the real static tables;
  4. a fixture check driven by a **SOT-1618 trace of real positions**: a full-obs
     match is recorded (into the gitignored ``eval/traces/``, since traces are
     license-restricted and never committed) and every MAIN position is replayed
     through the policy — it must always return a legal selection or defer, must
     actually engage on real positions, and ``damage`` must compute without
     raising on the real Active/attack pairs.
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from cg.api import (  # noqa: E402
    Attack,
    CardData,
    CardType,
    EnergyType,
    SelectType,
    to_observation_class,
)

from agents import damage  # noqa: E402
from agents.base import is_valid_selection  # noqa: E402
from agents.rule_based import RuleBasedAgent, main_context_handler  # noqa: E402


# --------------------------------------------------------------------------- #
# Builders for synthetic, engine-independent card/attack data.
# --------------------------------------------------------------------------- #

def _card(*, energy=EnergyType.WATER, hp=100, weakness=None, resistance=None) -> CardData:
    return CardData(
        cardId=1, name="Test", cardType=CardType.POKEMON, retreatCost=1, hp=hp,
        weakness=weakness, resistance=resistance, energyType=energy, basic=True,
        stage1=False, stage2=False, ex=False, megaEx=False, tera=False,
        aceSpec=False, evolvesFrom=None, skills=[], attacks=[],
    )


def _attack(damage_value: int) -> Attack:
    return Attack(attackId=1, name="Hit", text="", damage=damage_value, energies=[EnergyType.WATER])


# --------------------------------------------------------------------------- #
# 1. attack_damage
# --------------------------------------------------------------------------- #

def test_attack_damage_weakness_doubles():
    atk = _attack(130)
    attacker = _card(energy=EnergyType.WATER)
    defender = _card(weakness=EnergyType.WATER)   # weak to the attacker's type
    assert damage.attack_damage(atk, attacker, defender) == 260
    print("PASS test_attack_damage_weakness_doubles")


def test_attack_damage_resistance_subtracts_30():
    atk = _attack(130)
    attacker = _card(energy=EnergyType.WATER)
    defender = _card(resistance=EnergyType.WATER)
    assert damage.attack_damage(atk, attacker, defender) == 100
    print("PASS test_attack_damage_resistance_subtracts_30")


def test_attack_damage_weakness_then_resistance():
    # Both apply: ×2 first, then −30. 50 -> 100 -> 70.
    atk = _attack(50)
    attacker = _card(energy=EnergyType.WATER)
    defender = _card(weakness=EnergyType.WATER, resistance=EnergyType.WATER)
    assert damage.attack_damage(atk, attacker, defender) == 70
    print("PASS test_attack_damage_weakness_then_resistance")


def test_attack_damage_neither_matches():
    atk = _attack(90)
    attacker = _card(energy=EnergyType.WATER)
    defender = _card(weakness=EnergyType.FIRE, resistance=EnergyType.GRASS)
    assert damage.attack_damage(atk, attacker, defender) == 90
    print("PASS test_attack_damage_neither_matches")


def test_attack_damage_zero_stays_zero():
    # Effect-only / setup attacks (damage 0) are never scaled up by weakness.
    atk = _attack(0)
    attacker = _card(energy=EnergyType.WATER)
    defender = _card(weakness=EnergyType.WATER)
    assert damage.attack_damage(atk, attacker, defender) == 0
    print("PASS test_attack_damage_zero_stays_zero")


def test_attack_damage_resistance_clamped_to_zero():
    atk = _attack(20)
    attacker = _card(energy=EnergyType.WATER)
    defender = _card(resistance=EnergyType.WATER)  # 20 - 30 -> clamp 0
    assert damage.attack_damage(atk, attacker, defender) == 0
    print("PASS test_attack_damage_resistance_clamped_to_zero")


def test_attack_damage_missing_card_data_uses_base():
    atk = _attack(60)
    defender = _card(weakness=EnergyType.WATER)
    # Unknown attacker or defender -> no scaling, base damage returned.
    assert damage.attack_damage(atk, None, defender) == 60
    assert damage.attack_damage(atk, _card(), None) == 60
    print("PASS test_attack_damage_missing_card_data_uses_base")


# --------------------------------------------------------------------------- #
# 2. is_ko
# --------------------------------------------------------------------------- #

def test_is_ko_thresholds():
    attacker = _card(energy=EnergyType.WATER)
    defender = _card()
    assert damage.is_ko(_attack(100), attacker, defender, 100) is True   # exact
    assert damage.is_ko(_attack(120), attacker, defender, 100) is True   # over
    assert damage.is_ko(_attack(90), attacker, defender, 100) is False   # under
    assert damage.is_ko(_attack(100), attacker, defender, None) is False  # unknown hp
    assert damage.is_ko(_attack(100), attacker, defender, 0) is False    # already 0
    assert damage.is_ko(_attack(0), attacker, defender, 1) is False      # no damage
    print("PASS test_is_ko_thresholds")


def test_is_ko_uses_weakness():
    attacker = _card(energy=EnergyType.WATER)
    tough = _card(weakness=EnergyType.WATER, hp=150)
    # 130 base does not KO 150 HP, but weakness ×2 = 260 does.
    assert damage.is_ko(_attack(130), attacker, _card(hp=150), 150) is False
    assert damage.is_ko(_attack(130), attacker, tough, 150) is True
    print("PASS test_is_ko_uses_weakness")


# --------------------------------------------------------------------------- #
# 3. engine-backed registries
# --------------------------------------------------------------------------- #

def test_registries_expose_engine_tables():
    cards = damage.get_card_registry()
    attacks = damage.get_attack_registry()
    assert damage.get_card_registry() is cards          # cached (same object)
    assert damage.get_attack_registry() is attacks
    # Kyogre (721) is a Basic Water Pokémon with 150 HP in the tracked deck.
    kyogre = cards.get(721)
    assert kyogre is not None and kyogre.hp == 150 and kyogre.energyType == EnergyType.WATER
    # Its 'Swirling Waves' attack (1043) deals 130.
    assert attacks[1043].damage == 130
    print("PASS test_registries_expose_engine_tables")


# --------------------------------------------------------------------------- #
# 4. fixture check on real positions (SOT-1618 trace)
# --------------------------------------------------------------------------- #

def _record_fixture(path: str) -> None:
    """Record one full-obs rule-vs-random match to ``path`` (real positions)."""
    from eval import record_match as rm
    from eval.trace import RecordLevel

    a = RuleBasedAgent(seed=0)
    agents = (
        rm.Agent(lambda d: a.decide(to_observation_class(d)), name="rule", version="1"),
        rm.make_random_agent(1, "random"),
    )
    deck = rm.load_deck("deck.csv")
    summary = rm.record_match(
        deck, deck, agents=agents, out_path=path, level=RecordLevel.FULL_OBS,
        trace_id="test-sot1633",
    )
    assert summary["failure"] is None, summary["failure"]


def test_main_policy_on_recorded_positions():
    path = f"eval/traces/_test_main_{os.getpid()}.jsonl"
    _record_fixture(path)
    try:
        records = [json.loads(line) for line in open(path)]
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    agent = RuleBasedAgent(seed=0)
    main_positions = 0
    policy_engaged = 0
    for rec in records:
        if rec.get("kind") != "decision":
            continue
        obs = to_observation_class(rec["obs"])
        if obs.select is None or obs.select.type != SelectType.MAIN:
            continue
        main_positions += 1
        # The recorded choice was legal (engine accepted it).
        assert is_valid_selection(rec["choice"], obs.select), rec["choice"]
        # Replaying the policy on the real position: always legal, or a defer.
        result = main_context_handler(agent, obs.select, obs)
        assert result is None or is_valid_selection(result, obs.select), result
        if result is not None:
            policy_engaged += 1
        # damage must compute on the real Active/attack pairs without raising.
        cards = damage.get_card_registry()
        attacks = damage.get_attack_registry()
        state = obs.current
        me = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]
        atk_cd = cards.get(me.active[0].id) if me.active and me.active[0] else None
        dfn = opp.active[0] if opp.active and opp.active[0] else None
        dfn_cd = cards.get(dfn.id) if dfn else None
        for o in obs.select.option:
            if o.attackId is not None and o.attackId in attacks:
                d = damage.attack_damage(attacks[o.attackId], atk_cd, dfn_cd)
                assert isinstance(d, int) and d >= 0

    assert main_positions > 0, "fixture had no MAIN positions"
    assert policy_engaged > 0, "policy never engaged on any real MAIN position"
    print(
        f"PASS test_main_policy_on_recorded_positions "
        f"({policy_engaged}/{main_positions} MAIN positions handled)"
    )


if __name__ == "__main__":
    test_attack_damage_weakness_doubles()
    test_attack_damage_resistance_subtracts_30()
    test_attack_damage_weakness_then_resistance()
    test_attack_damage_neither_matches()
    test_attack_damage_zero_stays_zero()
    test_attack_damage_resistance_clamped_to_zero()
    test_attack_damage_missing_card_data_uses_base()
    test_is_ko_thresholds()
    test_is_ko_uses_weakness()
    test_registries_expose_engine_tables()
    test_main_policy_on_recorded_positions()
    print("ALL DAMAGE/MAIN TESTS PASSED")
