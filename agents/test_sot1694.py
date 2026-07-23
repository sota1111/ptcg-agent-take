"""Tests for the SOT-1694 work: archetype adaptation + 25-deck fallback handlers.

No pytest dependency — run directly from the repo root:
    venv/bin/python agents/test_sot1694.py

Covers:
  1. deck_profile / classify_archetype over the real 25 tournament decks —
     every deck classifies, representative decks land in the expected bucket,
     and megas (which carry ``ex=False`` in the shipped registry) count as
     multi-prize bodies;
  2. band_adjustments — bounded by ±MAX_NUDGE, deterministic, and the adjusted
     scores can never leapfrog the neighbouring band (the SOT-1682 ordering);
  3. the new fallback-hole context handlers on synthetic positions built from
     real engine card ids (TO_BENCH, ACTIVATE, DRAW_COUNT deck guard, SKILL_ORDER,
     ATTACK / DISABLE_ATTACK, DETACH_FROM, DISCARD_TOOL_CARD, EVOLVES_TO);
  4. the MAIN-scoring no-active / deck-out guards (bench insurance, Supporter
     deck guard);
  5. every new handler produces a legal selection through the full dispatch.
"""
from __future__ import annotations

import glob
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
)

from agents import damage  # noqa: E402
from agents.archetype import (  # noqa: E402
    MAX_NUDGE,
    band_adjustments,
    classify_archetype,
    deck_profile,
)
from agents.base import is_valid_selection, read_deck_csv  # noqa: E402
from agents.rule_based import (  # noqa: E402
    S_EVOLVE,
    S_ITEM,
    S_PLAY_BASIC,
    S_SUPPORTER,
    RuleBasedAgent,
)

# Real engine card ids (same fixtures as agents/test_setup_contexts.py).
KYOGRE = 721      # Basic, HP 150; attacks: 1042 (0 dmg), 1043 (130 dmg)
SNOVER = 722      # Basic, HP 90, heads a line; attacks: 1044 (10), 1045 (30)
ABOMA = 723       # Mega ex, HP 350; attacks: 1046 (0), 1047 (200)
BASIC_W = 3       # Basic {W} Energy
SUPPORTER = 1182  # Boss’s Orders
ITEM = 1077       # an Item card
TOOL = 1154       # a Pokémon Tool


def _poke(cid: int, hp: int, *, max_hp: int | None = None, n_energy: int = 0) -> Pokemon:
    energy_cards = [Card(id=BASIC_W, serial=900 + i, playerIndex=0) for i in range(n_energy)]
    return Pokemon(
        id=cid, serial=cid, hp=hp, maxHp=max_hp if max_hp is not None else hp,
        appearThisTurn=False, energies=[3] * n_energy, energyCards=energy_cards,
        tools=[], preEvolution=[],
    )


def _player(*, hand=None, active=None, bench=None, deck_count: int = 40) -> PlayerState:
    return PlayerState(
        active=list(active) if active is not None else [],
        bench=list(bench) if bench is not None else [],
        benchMax=5, deckCount=deck_count, discard=[], prize=[],
        handCount=len(hand or []),
        hand=list(hand) if hand is not None else None,
        poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )


def _obs(select: SelectData, players: list[PlayerState], your_index: int = 0) -> Observation:
    state = State(
        turn=3, turnActionCount=0, yourIndex=your_index, firstPlayer=your_index,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
        retreated=False, result=-1, stadium=[], looking=None, players=players,
    )
    return Observation(select=select, current=state, logs=[], search_begin_input=None)


def _sel(context: SelectContext, options: list[Option], *, min_count=1, max_count=1,
         deck: list[Card] | None = None) -> SelectData:
    return SelectData(
        type=SelectType.CARD, context=context, minCount=min_count, maxCount=max_count,
        remainDamageCounter=0, remainEnergyCost=0, option=options, deck=deck,
        contextCard=None, effect=None,
    )


def _agent() -> RuleBasedAgent:
    return RuleBasedAgent(seed=0)


def _check_dispatch_legal(agent: RuleBasedAgent, obs: Observation) -> list[int]:
    result = agent.decide(obs)
    assert is_valid_selection(result, obs.select), f"illegal selection {result}"
    return result


# --------------------------------------------------------------------------- #
# 1+2. archetype profiles / classification / band bounds on the real 25 decks
# --------------------------------------------------------------------------- #

def test_archetype_over_25_decks() -> None:
    cards = damage.get_card_registry()
    attacks = damage.get_attack_registry()
    paths = sorted(glob.glob("decks/rotation_baseline/*.csv"))
    assert len(paths) == 25, f"expected 25 decks, found {len(paths)}"

    expected = {
        "01_dragapult": "stage2",
        "13_festival_lead": "single_prize",
        "16_crustle_mysterious_rock_inn": "control",
        "21_lillie_s_clefairy_ex_naic_champion": "ex_mega",
        # megas carry ex=False in the registry: this deck (1 ex + 3 mega) must
        # still land in ex_mega, not single_prize.
        "14_mega_lucario_ex": "ex_mega",
    }
    seen: dict[str, str] = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        profile = deck_profile(read_deck_csv(path), cards, attacks)
        arch = classify_archetype(profile)
        seen[name] = arch
        assert arch in ("stage2", "ex_mega", "single_prize", "control", "midrange")
        assert profile.pokemon > 0 and profile.energy > 0

        adj = band_adjustments(profile)
        assert adj == band_adjustments(profile), "band_adjustments must be deterministic"
        for v in (adj.evolve, adj.play_basic, adj.attach_load, adj.attach_bench):
            assert abs(v) <= MAX_NUDGE, f"{name}: nudge {v} exceeds ±{MAX_NUDGE}"
        assert adj.supporter_hand_max in (6, 8)
        # Band-ordering safety: the adjusted band + its intra-band tie-break
        # (< 100) must stay below the next band up.
        assert S_EVOLVE + adj.evolve + 100 <= S_SUPPORTER + 100  # evolve caps at 100
        assert S_EVOLVE + adj.evolve < S_SUPPORTER
        assert S_PLAY_BASIC + adj.play_basic < S_EVOLVE
        assert S_ITEM < S_PLAY_BASIC + adj.play_basic

    for name, arch in expected.items():
        assert seen[name] == arch, f"{name}: expected {arch}, got {seen[name]}"
    print(f"PASS test_archetype_over_25_decks ({len(paths)} decks: "
          f"{ {a: list(seen.values()).count(a) for a in set(seen.values())} })")


# --------------------------------------------------------------------------- #
# 3. new fallback-hole handlers
# --------------------------------------------------------------------------- #

def test_to_bench_fetches_pokemon() -> None:
    # Deck refs: an Item, a line-heading Basic (Snover), a plain Basic (Kyogre).
    deck = [Card(id=ITEM, serial=1, playerIndex=0),
            Card(id=SNOVER, serial=2, playerIndex=0),
            Card(id=KYOGRE, serial=3, playerIndex=0)]
    opts = [Option(type=OptionType.CARD, area=AreaType.DECK, index=i, playerIndex=0)
            for i in range(3)]
    sel = _sel(SelectContext.TO_BENCH, opts, min_count=0, max_count=2, deck=deck)
    me = _player(hand=[], active=[_poke(KYOGRE, 150)])
    obs = _obs(sel, [me, _player(active=[_poke(KYOGRE, 150)])])
    agent = _agent()
    result = _check_dispatch_legal(agent, obs)
    # Takes both Pokémon (maxCount=2), never the Item.
    assert result == [1, 2], f"expected the two Pokémon, got {result}"
    print("PASS test_to_bench_fetches_pokemon")


def test_activate_yes_and_deck_guard_no() -> None:
    opts = [Option(type=OptionType.YES), Option(type=OptionType.NO)]
    sel = _sel(SelectContext.ACTIVATE, opts)
    agent = _agent()

    me = _player(hand=[], active=[_poke(KYOGRE, 150)], deck_count=30)
    result = _check_dispatch_legal(agent, _obs(sel, [me, _player()]))
    assert result == [0], f"expected YES, got {result}"

    me_low = _player(hand=[], active=[_poke(KYOGRE, 150)], deck_count=4)
    result = _check_dispatch_legal(agent, _obs(sel, [me_low, _player()]))
    assert result == [1], f"expected NO under deck-out guard, got {result}"
    print("PASS test_activate_yes_and_deck_guard_no")


def test_draw_count_deck_guard() -> None:
    opts = [Option(type=OptionType.NUMBER, number=n) for n in (0, 2, 4)]
    sel = _sel(SelectContext.DRAW_COUNT, opts)
    agent = _agent()

    me = _player(hand=[], active=[_poke(KYOGRE, 150)], deck_count=30)
    assert _check_dispatch_legal(agent, _obs(sel, [me, _player()])) == [2]  # max

    me_low = _player(hand=[], active=[_poke(KYOGRE, 150)], deck_count=5)
    assert _check_dispatch_legal(agent, _obs(sel, [me_low, _player()])) == [0]  # min
    print("PASS test_draw_count_deck_guard")


def test_skill_order_identity() -> None:
    opts = [Option(type=OptionType.SKILL), Option(type=OptionType.SKILL)]
    sel = _sel(SelectContext.SKILL_ORDER, opts, min_count=2, max_count=2)
    me = _player(hand=[], active=[_poke(KYOGRE, 150)])
    result = _check_dispatch_legal(_agent(), _obs(sel, [me, _player()]))
    assert result == [0, 1], f"expected identity order, got {result}"
    print("PASS test_skill_order_identity")


def test_attack_select_prefers_ko() -> None:
    # Kyogre (130 dmg attack 1043) vs a 100 HP Snover: 1043 KOs, 1042 deals 0.
    opts = [Option(type=OptionType.ATTACK, attackId=1042),
            Option(type=OptionType.ATTACK, attackId=1043)]
    sel = _sel(SelectContext.ATTACK, opts)
    me = _player(hand=[], active=[_poke(KYOGRE, 150, n_energy=3)])
    opp = _player(active=[_poke(SNOVER, 90)])
    result = _check_dispatch_legal(_agent(), _obs(sel, [me, opp]))
    assert result == [1], f"expected the KO attack, got {result}"
    print("PASS test_attack_select_prefers_ko")


def test_disable_attack_targets_biggest_threat() -> None:
    # Opponent Mega Abomasnow: attack 1047 (200 dmg) is the threat, 1046 deals 0.
    opts = [Option(type=OptionType.ATTACK, attackId=1046),
            Option(type=OptionType.ATTACK, attackId=1047)]
    sel = _sel(SelectContext.DISABLE_ATTACK, opts)
    me = _player(hand=[], active=[_poke(KYOGRE, 150)])
    opp = _player(active=[_poke(ABOMA, 350, n_energy=3)])
    result = _check_dispatch_legal(_agent(), _obs(sel, [me, opp]))
    assert result == [1], f"expected to disable the 200-dmg attack, got {result}"
    print("PASS test_disable_attack_targets_biggest_threat")


def test_detach_from_prefers_own_non_attacker() -> None:
    # Own board: loaded Kyogre (viable attacker) vs an energyless Snover wall
    # with surplus energy… give Snover 3 energies (surplus 1 over its 2-cost
    # best attack) — still a weaker attacker but the point is the viable
    # attacker keeps its charge: Kyogre (3 energy, needs 3, surplus 0).
    kyogre = _poke(KYOGRE, 150, n_energy=3)
    snover = _poke(SNOVER, 90, n_energy=3)
    me = _player(hand=[], active=[kyogre], bench=[snover])
    opts = [Option(type=OptionType.CARD, area=AreaType.ACTIVE, index=0, playerIndex=0),
            Option(type=OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=0)]
    sel = _sel(SelectContext.DETACH_FROM, opts)
    result = _check_dispatch_legal(_agent(), _obs(sel, [me, _player()]))
    # Snover has the bigger surplus (3 attached vs a 2-energy best attack).
    assert result == [1], f"expected the surplus Snover, got {result}"
    print("PASS test_detach_from_prefers_own_non_attacker")


def test_discard_tool_prefers_opponent() -> None:
    mine = _poke(KYOGRE, 150)
    mine.tools = [Card(id=TOOL, serial=800, playerIndex=0)]
    theirs = _poke(SNOVER, 90)
    theirs.tools = [Card(id=TOOL, serial=801, playerIndex=1)]
    me = _player(hand=[], active=[mine])
    opp = _player(active=[theirs])
    opts = [Option(type=OptionType.TOOL_CARD, area=AreaType.ACTIVE, index=0,
                   playerIndex=0, toolIndex=0),
            Option(type=OptionType.TOOL_CARD, area=AreaType.ACTIVE, index=0,
                   playerIndex=1, toolIndex=0)]
    sel = _sel(SelectContext.DISCARD_TOOL_CARD, opts)
    result = _check_dispatch_legal(_agent(), _obs(sel, [me, opp]))
    assert result == [1], f"expected the opponent's tool, got {result}"
    print("PASS test_discard_tool_prefers_opponent")


def test_evolves_to_prefers_stronger() -> None:
    # Deck offers Snover (30 best) and Mega Abomasnow (200 best): take the Mega.
    deck = [Card(id=SNOVER, serial=1, playerIndex=0),
            Card(id=ABOMA, serial=2, playerIndex=0)]
    opts = [Option(type=OptionType.CARD, area=AreaType.DECK, index=i, playerIndex=0)
            for i in range(2)]
    sel = _sel(SelectContext.EVOLVES_TO, opts, min_count=0, max_count=1, deck=deck)
    me = _player(hand=[], active=[_poke(SNOVER, 90)])
    result = _check_dispatch_legal(_agent(), _obs(sel, [me, _player(active=[_poke(KYOGRE, 150)])]))
    assert result == [1], f"expected the stronger evolution, got {result}"
    print("PASS test_evolves_to_prefers_stronger")


# --------------------------------------------------------------------------- #
# 4. MAIN-scoring guards
# --------------------------------------------------------------------------- #

def _main_sel(options: list[Option]) -> SelectData:
    return SelectData(
        type=SelectType.MAIN, context=SelectContext.MAIN, minCount=1, maxCount=1,
        remainDamageCounter=0, remainEnergyCost=0, option=options, deck=None,
        contextCard=None, effect=None,
    )


def test_bench_insurance_when_bench_empty() -> None:
    # Hand: a Basic and a Supporter. Bench EMPTY → play the Basic first even
    # though a Supporter normally outranks it.
    hand = [Card(id=KYOGRE, serial=1, playerIndex=0),
            Card(id=SUPPORTER, serial=2, playerIndex=0)]
    opts = [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.PLAY, area=AreaType.HAND, index=1, playerIndex=0),
            Option(type=OptionType.END)]
    me = _player(hand=hand, active=[_poke(SNOVER, 90)], bench=[])
    obs = _obs(_main_sel(opts), [me, _player(active=[_poke(KYOGRE, 150)])])
    result = _check_dispatch_legal(_agent(), obs)
    assert result == [0], f"expected the Basic (bench insurance), got {result}"

    # With a healthy bench the Supporter goes first again.
    me2 = _player(hand=hand, active=[_poke(SNOVER, 90)], bench=[_poke(KYOGRE, 150)])
    obs2 = _obs(_main_sel(opts), [me2, _player(active=[_poke(KYOGRE, 150)])])
    result2 = _check_dispatch_legal(_agent(), obs2)
    assert result2 == [1], f"expected the Supporter with a safe bench, got {result2}"
    print("PASS test_bench_insurance_when_bench_empty")


def test_supporter_deck_guard() -> None:
    # Deck nearly empty → the Supporter scores below END and is not played.
    hand = [Card(id=SUPPORTER, serial=1, playerIndex=0)]
    opts = [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0, playerIndex=0),
            Option(type=OptionType.END)]
    me = _player(hand=hand, active=[_poke(KYOGRE, 150)], bench=[_poke(SNOVER, 90)],
                 deck_count=3)
    obs = _obs(_main_sel(opts), [me, _player(active=[_poke(KYOGRE, 150)])])
    result = _check_dispatch_legal(_agent(), obs)
    assert result == [1], f"expected END under the deck-out guard, got {result}"
    print("PASS test_supporter_deck_guard")


def main() -> int:
    test_archetype_over_25_decks()
    test_to_bench_fetches_pokemon()
    test_activate_yes_and_deck_guard_no()
    test_draw_count_deck_guard()
    test_skill_order_identity()
    test_attack_select_prefers_ko()
    test_disable_attack_targets_biggest_threat()
    test_detach_from_prefers_own_non_attacker()
    test_discard_tool_prefers_opponent()
    test_evolves_to_prefers_stronger()
    test_bench_insurance_when_bench_empty()
    test_supporter_deck_guard()
    print("ALL SOT-1694 TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
