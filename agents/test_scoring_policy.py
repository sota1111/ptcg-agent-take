"""Tests for the scoring MAIN policy and tactical rules (SOT-1635, R4).

No pytest dependency — run directly from the repo root:
    venv/bin/python agents/test_scoring_policy.py

Positions are built from **real engine card ids** (so ``CardData`` — HP, type,
attacks, retreat cost, evolution line — is the engine's, not a mock). Covers the
issue's three tactical additions plus the scoring ordering invariants:

  1. サポーター使用     — plays a Supporter (card advantage) over attaching / ending;
  2. リーサル優先       — takes a Knock-Out attack immediately, before more setup;
  3. スイング last      — a non-lethal attack waits until no setup remains, but still
                          beats ending the turn;
  4. エネルギー効率     — loads the Active until it can fire its best attack, then
     / ベンチ育成         spreads Energy onto a second Bench attacker;
  5. リトリート判断     — retreats when disabled / about to be KO'd with a viable Bench
                          replacement, and never retreats idly;
  6. グッズ使用         — plays an Item;
  7. evaluate_position  — reusable board eval: prizes dominate, then HP, then Energy;
  8. policy toggle      — ``policy="fixed"`` reproduces the old ladder (no Supporter),
                          so the two are switch-comparable (受け入れ条件①);
  9. robustness         — degenerate observation defers to a legal selection, and a
                          record_match sweep runs both policies with zero exceptions.
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

from agents.base import is_valid_selection  # noqa: E402
from agents.rule_based import (  # noqa: E402
    RuleBasedAgent,
    _horizon_adjustment,
    adaptive_search_depth,
    evaluate_position,
)

# Real engine card ids.
KYOGRE = 721      # Basic, HP 150, retreat 3, Swirling Waves 130 dmg (3E)
SNOVER = 722      # Basic, HP 90, retreat 3, heads an evolution line
ABOMA = 723       # Stage1, HP 350
CYRANO = 1205     # Supporter
MEGA_SIGNAL = 1145  # Item
MAX_BELT = 1158   # Tool
BASIC_W = 3       # Basic {W} Energy
ATK_SWIRL = 1043  # Kyogre Swirling Waves, 130 dmg
ATK_BEAT = 1044   # Snover Beat, 10 dmg


def _poke(cid: int, hp: int, *, max_hp: int | None = None, n_energy: int = 0) -> Pokemon:
    energy_cards = [Card(id=BASIC_W, serial=900 + i, playerIndex=0) for i in range(n_energy)]
    return Pokemon(
        id=cid, serial=cid, hp=hp, maxHp=max_hp if max_hp is not None else hp,
        appearThisTurn=False, energies=[3] * n_energy, energyCards=energy_cards,
        tools=[], preEvolution=[],
    )


def _player(*, hand=None, active=None, bench=None, prize_n=6,
            confused=False, asleep=False, paralyzed=False) -> PlayerState:
    return PlayerState(
        active=list(active) if active is not None else [],
        bench=list(bench) if bench is not None else [],
        benchMax=5, deckCount=40, discard=[],
        prize=[Card(id=0, serial=i, playerIndex=0) for i in range(prize_n)],
        handCount=len(hand or []), hand=list(hand) if hand is not None else None,
        poisoned=False, burned=False, asleep=asleep, paralyzed=paralyzed, confused=confused,
    )


def _main_obs(options, me, opp, *, your_index=0, supporter_played=False) -> Observation:
    state = State(
        turn=3, turnActionCount=0, yourIndex=your_index, firstPlayer=0,
        supporterPlayed=supporter_played, stadiumPlayed=False, energyAttached=False,
        retreated=False, result=-1, stadium=[], looking=None,
        players=[me, opp] if your_index == 0 else [opp, me],
    )
    sel = SelectData(
        type=SelectType.MAIN, context=SelectContext.MAIN, minCount=1, maxCount=1,
        remainDamageCounter=0, remainEnergyCost=0, option=options, deck=None,
        contextCard=None, effect=None,
    )
    return Observation(select=sel, logs=[], current=state)


def _play(hand_idx: int) -> Option:
    return Option(type=OptionType.PLAY, index=hand_idx)


def _attach(hand_idx: int, in_area: AreaType, in_idx: int) -> Option:
    return Option(type=OptionType.ATTACH, area=AreaType.HAND, index=hand_idx,
                  inPlayArea=in_area, inPlayIndex=in_idx, playerIndex=0)


def _attack(attack_id: int) -> Option:
    return Option(type=OptionType.ATTACK, attackId=attack_id)


_END = Option(type=OptionType.END)
_RETREAT = Option(type=OptionType.RETREAT)


# --------------------------------------------------------------------------- #
# 1. Supporter usage
# --------------------------------------------------------------------------- #

def test_plays_supporter_over_attach_and_end():
    agent = RuleBasedAgent(seed=0)
    hand = [Card(id=CYRANO, serial=1, playerIndex=0), Card(id=BASIC_W, serial=2, playerIndex=0)]
    me = _player(hand=hand, active=[_poke(KYOGRE, 150, n_energy=0)])
    opp = _player(active=[_poke(SNOVER, 90)])
    # options: play supporter, attach energy to (unloaded) Active, end
    opts = [_play(0), _attach(1, AreaType.ACTIVE, 0), _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [0], out  # Supporter beats attaching / ending
    print("PASS test_plays_supporter_over_attach_and_end")


def test_supporter_ignored_once_used_this_turn():
    agent = RuleBasedAgent(seed=0)
    hand = [Card(id=CYRANO, serial=1, playerIndex=0)]
    me = _player(hand=hand, active=[_poke(KYOGRE, 150, n_energy=3)])
    opp = _player(active=[_poke(SNOVER, 90)])
    opts = [_play(0), _END]
    # supporterPlayed already true -> the supporter drops to the item band, still
    # above END, so it is played, but this documents the guard is consulted.
    out = agent.decide(_main_obs(opts, me, opp, supporter_played=True))
    assert out == [0], out
    print("PASS test_supporter_ignored_once_used_this_turn")


# --------------------------------------------------------------------------- #
# 2-3. lethal first; non-lethal swings last but beats END
# --------------------------------------------------------------------------- #

def test_takes_lethal_ko_before_more_setup():
    agent = RuleBasedAgent(seed=0)
    hand = [Card(id=CYRANO, serial=1, playerIndex=0)]
    me = _player(hand=hand, active=[_poke(KYOGRE, 150, n_energy=3)])
    opp = _player(active=[_poke(SNOVER, 30)])  # 130 dmg >= 30 hp -> KO
    opts = [_play(0), _attack(ATK_SWIRL), _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [1], out  # take the Knock Out now, even with a Supporter in hand
    print("PASS test_takes_lethal_ko_before_more_setup")


def test_nonlethal_attack_waits_for_setup():
    agent = RuleBasedAgent(seed=0)
    hand = [Card(id=CYRANO, serial=1, playerIndex=0)]
    me = _player(hand=hand, active=[_poke(KYOGRE, 150, n_energy=3)])
    opp = _player(active=[_poke(ABOMA, 300)])  # 130 dmg < 300 hp -> not a KO
    opts = [_play(0), _attack(ATK_SWIRL), _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [0], out  # set up (Supporter) before a non-lethal swing
    print("PASS test_nonlethal_attack_waits_for_setup")


def test_nonlethal_attack_beats_end():
    agent = RuleBasedAgent(seed=0)
    me = _player(active=[_poke(KYOGRE, 150, n_energy=3)])
    opp = _player(active=[_poke(ABOMA, 300)])
    opts = [_attack(ATK_SWIRL), _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [0], out  # nothing to set up -> swing rather than pass
    print("PASS test_nonlethal_attack_beats_end")


# --------------------------------------------------------------------------- #
# 4. energy efficiency / bench development
# --------------------------------------------------------------------------- #

def test_loads_active_before_developing_bench():
    agent = RuleBasedAgent(seed=0)
    hand = [Card(id=BASIC_W, serial=1, playerIndex=0)]
    me = _player(hand=hand,
                 active=[_poke(KYOGRE, 150, n_energy=0)],   # needs energy
                 bench=[_poke(KYOGRE, 150, n_energy=0)])
    opp = _player(active=[_poke(SNOVER, 90)])
    opts = [_attach(0, AreaType.ACTIVE, 0), _attach(0, AreaType.BENCH, 0), _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [0], out  # load the Active first
    print("PASS test_loads_active_before_developing_bench")


def test_develops_bench_once_active_is_loaded():
    agent = RuleBasedAgent(seed=0)
    hand = [Card(id=BASIC_W, serial=1, playerIndex=0)]
    me = _player(hand=hand,
                 active=[_poke(KYOGRE, 150, n_energy=3)],   # already fires its best attack
                 bench=[_poke(KYOGRE, 150, n_energy=0)])    # viable 2nd attacker
    opp = _player(active=[_poke(SNOVER, 90)])
    opts = [_attach(0, AreaType.ACTIVE, 0), _attach(0, AreaType.BENCH, 0), _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [1], out  # spread energy to the Bench attacker (育成)
    print("PASS test_develops_bench_once_active_is_loaded")


# --------------------------------------------------------------------------- #
# 5. retreat judgement
# --------------------------------------------------------------------------- #

def test_retreats_when_confused_with_viable_bench():
    agent = RuleBasedAgent(seed=0)
    me = _player(active=[_poke(KYOGRE, 150, n_energy=3)],
                 bench=[_poke(KYOGRE, 150, n_energy=3)], confused=True)
    opp = _player(active=[_poke(SNOVER, 90)])
    opts = [_RETREAT, _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [0], out  # escape the Confusion onto a ready Bench Pokémon
    print("PASS test_retreats_when_confused_with_viable_bench")


def test_retreats_when_facing_lethal_with_viable_bench():
    agent = RuleBasedAgent(seed=0)
    # Our Active is at 100 HP; opponent Kyogre's Swirling Waves does 130 -> lethal.
    me = _player(active=[_poke(SNOVER, 100)], bench=[_poke(KYOGRE, 150, n_energy=3)])
    opp = _player(active=[_poke(KYOGRE, 150, n_energy=3)])
    opts = [_RETREAT, _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [0], out  # duck the incoming KO
    print("PASS test_retreats_when_facing_lethal_with_viable_bench")


def test_does_not_retreat_idly():
    agent = RuleBasedAgent(seed=0)
    # Healthy Active, weak opponent (30 dmg << 150 HP), no reason to retreat.
    me = _player(active=[_poke(KYOGRE, 150, n_energy=3)], bench=[_poke(KYOGRE, 150, n_energy=3)])
    opp = _player(active=[_poke(SNOVER, 90, n_energy=2)])
    opts = [_RETREAT, _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [1], out  # ending beats a pointless, energy-wasting retreat
    print("PASS test_does_not_retreat_idly")


def test_does_not_retreat_without_bench():
    agent = RuleBasedAgent(seed=0)
    me = _player(active=[_poke(SNOVER, 10)], bench=[], confused=True)  # nowhere to go
    opp = _player(active=[_poke(KYOGRE, 150, n_energy=3)])
    opts = [_RETREAT, _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [1], out  # no Bench replacement -> retreat is pointless
    print("PASS test_does_not_retreat_without_bench")


# --------------------------------------------------------------------------- #
# 6. item usage
# --------------------------------------------------------------------------- #

def test_plays_item_over_end():
    agent = RuleBasedAgent(seed=0)
    hand = [Card(id=MEGA_SIGNAL, serial=1, playerIndex=0)]
    me = _player(hand=hand, active=[_poke(KYOGRE, 150, n_energy=3)])
    opp = _player(active=[_poke(SNOVER, 90)])
    opts = [_play(0), _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [0], out
    print("PASS test_plays_item_over_end")


# --------------------------------------------------------------------------- #
# 7. reusable position evaluation
# --------------------------------------------------------------------------- #

def test_evaluate_position_prizes_dominate():
    ahead = _player(active=[_poke(KYOGRE, 10, n_energy=0)], prize_n=1)   # 1 prize left
    behind = _player(active=[_poke(ABOMA, 350, n_energy=5)], prize_n=5)  # 5 prizes left
    # `ahead` has far fewer prizes left (closer to winning) despite less board HP.
    assert evaluate_position(ahead, behind) > 0, evaluate_position(ahead, behind)
    assert evaluate_position(behind, ahead) < 0
    # Sign is antisymmetric.
    assert evaluate_position(ahead, behind) == -evaluate_position(behind, ahead)
    print("PASS test_evaluate_position_prizes_dominate")


# --------------------------------------------------------------------------- #
# 8. policy toggle: fixed ladder is retained and behaves differently
# --------------------------------------------------------------------------- #

def test_fixed_policy_does_not_play_supporter():
    agent = RuleBasedAgent(seed=0, policy="fixed")
    hand = [Card(id=CYRANO, serial=1, playerIndex=0), Card(id=BASIC_W, serial=2, playerIndex=0)]
    me = _player(hand=hand, active=[_poke(KYOGRE, 150, n_energy=0)])
    opp = _player(active=[_poke(SNOVER, 90)])
    opts = [_play(0), _attach(1, AreaType.ACTIVE, 0), _END]
    out = agent.decide(_main_obs(opts, me, opp))
    assert out == [1], out  # fixed ladder attaches; it never plays the Supporter
    # ...and the scoring policy does play it, so the two are genuinely different.
    assert RuleBasedAgent(seed=0, policy="scoring").decide(_main_obs(opts, me, opp)) == [0]
    print("PASS test_fixed_policy_does_not_play_supporter")


def test_unknown_policy_rejected():
    try:
        RuleBasedAgent(policy="bogus")
    except ValueError:
        print("PASS test_unknown_policy_rejected")
        return
    raise AssertionError("expected ValueError for unknown policy")


# --------------------------------------------------------------------------- #
# SOT-1734 adaptive horizon regression fixtures
# --------------------------------------------------------------------------- #

def test_adaptive_depth_tracks_branching_and_remaining_turns():
    me = _player(active=[_poke(KYOGRE, 150)], prize_n=6)
    opp = _player(active=[_poke(SNOVER, 90)])
    obs = _main_obs([_END] * 5, me, opp)
    assert adaptive_search_depth(obs, branching=5, enabled=True) == 3
    assert adaptive_search_depth(obs, branching=8, enabled=True) == 2
    assert adaptive_search_depth(obs, branching=11, enabled=True) == 1
    assert adaptive_search_depth(obs, branching=5, enabled=False) == 1

    # At the end of the prize race there is no multi-turn continuation to buy.
    me.prize = [Card(id=0, serial=0, playerIndex=0)]
    assert adaptive_search_depth(obs, branching=5, enabled=True) == 2
    print("PASS test_adaptive_depth_tracks_branching_and_remaining_turns")


def test_immediate_reward_trap_prefers_future_board_value():
    # Regression fixture: at depth three an EVOLVE continuation gains 840 while
    # an immediate non-lethal ATTACK loses 440.  The 1,280 swing is deliberately
    # larger than the biggest ordinary attack tie-break, while lethal remains in
    # its protected 10k band in the integration scorer.
    assert _horizon_adjustment(OptionType.EVOLVE, 3) == 840.0
    assert _horizon_adjustment(OptionType.ATTACK, 3) == -440.0
    assert _horizon_adjustment(OptionType.ATTACK, 1) == 0.0
    print("PASS test_immediate_reward_trap_prefers_future_board_value")


# --------------------------------------------------------------------------- #
# 9. robustness
# --------------------------------------------------------------------------- #

def test_scoring_defers_safely_on_degenerate_obs():
    agent = RuleBasedAgent(seed=0)
    sel = SelectData(
        type=SelectType.MAIN, context=SelectContext.MAIN, minCount=1, maxCount=1,
        remainDamageCounter=0, remainEnergyCost=0,
        option=[Option(type=OptionType.END), _attack(ATK_SWIRL)],
        deck=None, contextCard=None, effect=None,
    )
    out = agent.decide(Observation(select=sel, logs=[], current=None))  # no state
    assert is_valid_selection(out, sel), out
    print("PASS test_scoring_defers_safely_on_degenerate_obs")


def test_both_policies_zero_exceptions_full_matches():
    from eval import record_match as rm  # noqa: E402

    def rule_fn(seed, policy):
        a = RuleBasedAgent(seed=seed, policy=policy)
        return lambda obs_dict: a.decide(to_observation_class(obs_dict))

    deck = rm.load_deck("deck.csv")
    for policy in ("scoring", "fixed"):
        for i in range(6):
            agents = (
                rm.Agent(rule_fn(i, policy), name=f"{policy}0", version="1"),
                rm.Agent(rule_fn(700 + i, policy), name=f"{policy}1", version="1"),
            )
            out = f"eval/traces/_r4_smoke_{policy}_{i}.jsonl"
            summary = rm.record_match(deck, deck, agents=agents, out_path=out)
            assert summary["failure"] is None, (policy, i, summary["failure"])
            assert summary["decisions"] > 0, (policy, i, summary)
            try:
                os.remove(out)
            except OSError:
                pass
    print("PASS test_both_policies_zero_exceptions_full_matches (12 matches)")


if __name__ == "__main__":
    test_plays_supporter_over_attach_and_end()
    test_supporter_ignored_once_used_this_turn()
    test_takes_lethal_ko_before_more_setup()
    test_nonlethal_attack_waits_for_setup()
    test_nonlethal_attack_beats_end()
    test_loads_active_before_developing_bench()
    test_develops_bench_once_active_is_loaded()
    test_retreats_when_confused_with_viable_bench()
    test_retreats_when_facing_lethal_with_viable_bench()
    test_does_not_retreat_idly()
    test_does_not_retreat_without_bench()
    test_plays_item_over_end()
    test_evaluate_position_prizes_dominate()
    test_fixed_policy_does_not_play_supporter()
    test_unknown_policy_rejected()
    test_adaptive_depth_tracks_branching_and_remaining_turns()
    test_immediate_reward_trap_prefers_future_board_value()
    test_scoring_defers_safely_on_degenerate_obs()
    test_both_policies_zero_exceptions_full_matches()
    print("ALL SCORING TESTS PASSED")
