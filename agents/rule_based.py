"""Rule-based agent skeleton (SOT-1632, R1).

This is the scaffold the later rule work (R2+) builds on. It provides:

- a **SelectContext dispatch table** (:data:`RuleBasedAgent.CONTEXT_HANDLERS`)
  mapping a :class:`~cg.api.SelectContext` to a pure handler function; and
- an **outermost safety guard** (:meth:`RuleBasedAgent.decide`) that turns any
  gap — an unimplemented context, an unknown / newly-appended enum value, an
  invalid handler result, or an internal exception — into a legal random
  selection, so the agent can never emit an illegal move or crash the match.

Adding a rule later is purely additive: write a pure function
``handler(agent, select, obs) -> list[int] | None`` and register it in
``CONTEXT_HANDLERS`` (or ``TYPE_HANDLERS`` for a whole SelectType). Returning
``None`` from a handler means "no opinion — fall back", so partial rules are
safe. No existing code needs to change and the guard keeps holding.
"""

from __future__ import annotations

import os.path
import random
import time
from typing import Callable, Optional

from cg.api import (
    AreaType,
    CardType,
    Observation,
    OptionType,
    SelectContext,
    SelectData,
    SelectType,
)

from agents import damage
from agents.archetype import BandAdjust, band_adjustments, deck_profile
from agents.base import Agent, is_valid_selection, legal_random_sample, read_deck_csv
from agents.profile import RuntimeProfile, load_promoted_profile

# A context/type handler: given the agent, the current selection, and the full
# observation, return the chosen option indices, or ``None`` to defer to the
# fallback. Handlers must be pure w.r.t. engine state (read-only on ``obs``).
Handler = Callable[["RuleBasedAgent", SelectData, Observation], Optional[list[int]]]


def main_context_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """Fixed-priority policy for the :data:`SelectContext.MAIN` selection.

    A MAIN selection offers the turn's actions (PLAY / ATTACH / EVOLVE / ATTACK /
    END, single choice). We pick the highest-priority available action:

    1. play a Basic Pokémon onto an open Bench spot,
    2. evolve a Pokémon,
    3. attach an Energy/Tool, the Active (main attacker) first,
    4. an attack that Knocks Out the Defending Pokémon (max damage among those),
    5. otherwise the highest expected-damage attack (if it deals any damage),
    6. else end the turn.

    Because attacking ends the turn, steps 1–3 run across successive MAIN calls
    (playing, evolving, attaching one action at a time) and the attack fires only
    once nothing better remains — i.e. set up first, then swing. Returns the
    chosen option index list, or ``None`` to defer to the safe fallback when the
    observation is too degenerate to reason about.
    """
    state = obs.current
    if state is None:
        return None
    you = state.yourIndex
    try:
        me = state.players[you]
        opp = state.players[1 - you]
    except (IndexError, TypeError, AttributeError):
        return None

    cards = damage.get_card_registry()
    attacks = damage.get_attack_registry()

    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None
    attacker_cd = cards.get(my_active.id) if my_active is not None else None
    defender_cd = cards.get(opp_active.id) if opp_active is not None else None
    defender_hp = opp_active.hp if opp_active is not None else None
    hand = me.hand or []
    bench_open = len(me.bench) < me.benchMax

    play_basic: list[int] = []
    evolve: list[int] = []
    attach_active: list[int] = []
    attach_other: list[int] = []
    attacks_scored: list[tuple[int, int, bool]] = []  # (option index, damage, is_ko)
    end_idx: Optional[int] = None

    for i, o in enumerate(select.option):
        t = o.type
        if t == OptionType.END:
            end_idx = i
        elif t == OptionType.PLAY:
            idx = o.index
            if idx is not None and 0 <= idx < len(hand) and bench_open:
                cd = cards.get(hand[idx].id)
                if cd is not None and cd.cardType == CardType.POKEMON and cd.basic:
                    play_basic.append(i)
        elif t == OptionType.EVOLVE:
            evolve.append(i)
        elif t == OptionType.ATTACH:
            if o.inPlayArea == AreaType.ACTIVE:
                attach_active.append(i)
            else:
                attach_other.append(i)
        elif t == OptionType.ATTACK and o.attackId is not None:
            atk = attacks.get(o.attackId)
            if atk is not None:
                dmg = damage.attack_damage(atk, attacker_cd, defender_cd)
                ko = damage.is_ko(atk, attacker_cd, defender_cd, defender_hp)
                attacks_scored.append((i, dmg, ko))

    if play_basic:  # 1. develop the board
        return [play_basic[0]]
    if evolve:  # 2. evolve
        return [evolve[0]]
    if attach_active:  # 3. power up the main attacker first
        return [attach_active[0]]
    if attach_other:
        return [attach_other[0]]
    ko_opts = [s for s in attacks_scored if s[2]]  # 4. lethal attack
    if ko_opts:
        return [max(ko_opts, key=lambda s: s[1])[0]]
    dmg_opts = [s for s in attacks_scored if s[1] > 0]  # 5. best damaging attack
    if dmg_opts:
        return [max(dmg_opts, key=lambda s: s[1])[0]]
    if end_idx is not None:  # 6. nothing productive left
        return [end_idx]
    return None


# --------------------------------------------------------------------------- #
# Shared helpers for the setup / forced-selection handlers (SOT-1634).
#
# A CARD/ENERGY option points at a card by (playerIndex, area, index) — and, for
# an attached energy, an extra energyIndex. These resolvers turn that reference
# into the live ``Pokemon`` / ``Card`` from the observation, or the card's static
# ``CardData``. They never raise: an out-of-range / malformed reference (e.g. an
# unexpected observation shape) resolves to ``None`` so the handler simply defers.
# --------------------------------------------------------------------------- #

def _player_state(obs: Observation, player_index: Optional[int]):
    """Return ``players[player_index]`` from the observation, or ``None``."""
    state = obs.current
    if state is None:
        return None
    try:
        players = state.players
        if player_index is None or not (0 <= player_index < len(players)):
            return None
        return players[player_index]
    except (TypeError, AttributeError):
        return None


def _referenced_pokemon(option, obs: Observation):
    """Return the live in-play ``Pokemon`` an option points at (Active/Bench)."""
    ps = _player_state(obs, option.playerIndex)
    if ps is None or option.index is None:
        return None
    try:
        if option.area == AreaType.ACTIVE:
            return ps.active[option.index]  # may itself be None (facedown)
        if option.area == AreaType.BENCH:
            return ps.bench[option.index]
    except (IndexError, TypeError, AttributeError):
        return None
    return None


def _referenced_card_id(option, obs: Observation) -> Optional[int]:
    """Return the ``CardData`` id of the card an option points at.

    Handles the areas the setup / discard contexts use — Hand, Discard, and the
    in-play Active / Bench Pokémon (whose ``id`` is its card id).
    """
    ps = _player_state(obs, option.playerIndex)
    if ps is None or option.index is None:
        return None
    try:
        if option.area == AreaType.HAND:
            return ps.hand[option.index].id if ps.hand else None
        if option.area == AreaType.DISCARD:
            return ps.discard[option.index].id
        if option.area == AreaType.ACTIVE:
            p = ps.active[option.index]
            return p.id if p is not None else None
        if option.area == AreaType.BENCH:
            return ps.bench[option.index].id
    except (IndexError, TypeError, AttributeError):
        return None
    return None


def _referenced_energy_card_id(option, obs: Observation) -> Optional[int]:
    """Return the card id of the attached energy an ENERGY option points at."""
    poke = _referenced_pokemon(option, obs)
    if poke is None or option.energyIndex is None:
        return None
    try:
        return poke.energyCards[option.energyIndex].id
    except (IndexError, TypeError, AttributeError):
        return None


def _top_k(scored: list[tuple[int, tuple]], k: int) -> Optional[list[int]]:
    """Return the option indices of the ``k`` highest-scoring entries.

    ``scored`` is ``(option_index, sort_key)``; ties break toward the smaller
    option index for determinism. Returns ``None`` when ``k`` distinct options
    cannot be produced (``k`` exceeds the count, or ``k`` is negative), so the
    caller defers to the safe fallback rather than emitting an illegal selection.
    """
    if k < 0 or k > len(scored):
        return None
    if k == 0:
        return []
    ordered = sorted(scored, key=lambda s: (tuple(-v for v in s[1]), s[0]))
    return sorted(idx for idx, _ in ordered[:k])


def _pick_count(select: SelectData, want_at_least_one: bool) -> int:
    """Resolve how many options to select, clamped into ``[minCount, maxCount]``.

    ``want_at_least_one`` marks contexts where doing nothing is pointless (place a
    Pokémon, promote to Active, heal): there we select one even when ``minCount``
    is 0. Discard-style contexts pass ``False`` so an optional (``minCount == 0``)
    selection stays a no-op instead of discarding needlessly.
    """
    lo = max(0, select.minCount)
    if want_at_least_one:
        lo = max(lo, 1)
    return min(lo, select.maxCount)


# --------------------------------------------------------------------------- #
# Setup / forced-selection context handlers (SOT-1634).
# --------------------------------------------------------------------------- #

def place_basic_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """SETUP_ACTIVE_POKEMON / SETUP_BENCH_POKEMON: place the best Basic.

    Prefer a Basic Pokémon that heads an evolution line (so it can grow), then
    higher HP. Options reference cards in hand; the Bench variant is optional
    (``minCount == 0``) but we still develop the board by placing one.
    """
    cards = damage.get_card_registry()
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        cid = _referenced_card_id(o, obs)
        cd = cards.get(cid) if cid is not None else None
        is_poke = 1 if (cd is not None and cd.cardType == CardType.POKEMON) else 0
        has_line = 1 if damage.has_evolution_line(cd) else 0
        atk = _best_attack(cd)
        best_dmg = atk.damage if atk is not None else 0
        hp = cd.hp if cd is not None else 0
        scored.append((i, (is_poke, has_line, best_dmg, hp)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def place_basic_active_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """SETUP_ACTIVE_POKEMON: place the Basic that survives the opening race.

    SOT-1682: matches here are short (median 5 turns) and decided by board
    attrition, so the *Active* pick is HP-first — it faces the first attacks —
    with attack damage and the evolution line as tie-breaks. The Bench pick
    (:func:`place_basic_handler`) stays line-first: benched Pokémon get the time
    to evolve.
    """
    cards = damage.get_card_registry()
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        cid = _referenced_card_id(o, obs)
        cd = cards.get(cid) if cid is not None else None
        is_poke = 1 if (cd is not None and cd.cardType == CardType.POKEMON) else 0
        has_line = 1 if damage.has_evolution_line(cd) else 0
        atk = _best_attack(cd)
        best_dmg = atk.damage if atk is not None else 0
        hp = cd.hp if cd is not None else 0
        scored.append((i, (is_poke, hp, best_dmg, has_line)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def _opponent_active_card(obs: Observation):
    """The opponent Active's :class:`~cg.api.CardData` (``None`` if unknown)."""
    state = obs.current
    if state is None:
        return None
    try:
        opp = state.players[1 - state.yourIndex]
        opp_active = opp.active[0] if opp.active else None
    except (IndexError, TypeError, AttributeError):
        return None
    if opp_active is None:
        return None
    return damage.get_card_registry().get(opp_active.id)


def _prize_value(card) -> int:
    """Prize cards the opponent takes when this Pokémon is Knocked Out (1–3)."""
    if card is None:
        return 1
    if card.megaEx:
        return 3
    if card.ex:
        return 2
    return 1


def to_active_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """TO_ACTIVE (after a Knock Out) / SWITCH: promote the best Bench Pokémon.

    Prefer a Pokémon that can fire a damaging attack *right now* (SOT-1682), then
    the better **prize trade** (SOT-1730), then the one closest to firing
    (smallest Energy deficit), then the one hitting the current defender hardest
    (weakness/resistance-aware), then the cheaper prize gift, then higher HP.

    The prize trade scores the promotion like the race it starts: Knocking Out
    the defender right now earns the defender's prize value; being Knocked Out
    by the opponent's affordable counter-attack next turn concedes the
    candidate's own prize value. Mirror-loss traces (SOT-1730) show 2-prize ex
    promoted as fodder — dying without a return KO — is the dominant conceded
    multi-prize event, so a 1-prize attacker making the same trade now wins the
    tie instead. A firing candidate whose trade *concedes* 2+ net prizes also
    loses its can-fire privilege: feeding a cheap body instead stretches the
    opponent's prize clock by a KO while the engine survives to swing later.
    """
    cards = damage.get_card_registry()
    defender_cd = _opponent_active_card(obs)
    state = obs.current
    opp_active = None
    if state is not None:
        try:
            opp = state.players[1 - state.yourIndex]
            opp_active = opp.active[0] if opp.active else None
        except (IndexError, TypeError, AttributeError):
            opp_active = None
    defender_hp = opp_active.hp if opp_active is not None else None
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        p = _referenced_pokemon(o, obs)
        pc = cards.get(p.id) if p is not None else None
        if p is not None and _is_viable_attacker(pc):
            deficit = _energy_deficit(p, pc)
        else:
            deficit = 99  # never going to attack: last resort
        can_fire = 1 if deficit <= 0 else 0
        dmg = _best_damage_vs(pc, defender_cd)
        ko_now = bool(can_fire and defender_hp is not None and dmg >= defender_hp)
        incoming = _incoming_damage(opp_active, defender_cd, pc)
        dies_next = p is not None and p.hp is not None and incoming >= p.hp
        race = (_prize_value(defender_cd) if ko_now else 0) - (
            _prize_value(pc) if dies_next else 0
        )
        # A promotion that concedes 2+ net prizes without a return KO is a
        # multi-prize gift, not an attacker — rank it with the non-firing
        # bodies so cheaper fodder soaks the hit first.
        fire_rank = can_fire if race > -2 else 0
        hp = p.hp if p is not None else 0
        scored.append((i, (fire_rank, race, -deficit, dmg, -_prize_value(pc), hp)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


# YesNo contexts: a fixed, reasonable default. Being *prompted* to mulligan almost
# always means the opening hand has no playable Basic, so redrawing is the safe
# default; going first keeps board/tempo initiative; a coin call is 50/50 so we
# pick heads deterministically.
_YESNO_DEFAULT: dict[SelectContext, OptionType] = {
    SelectContext.IS_FIRST: OptionType.YES,
    SelectContext.COIN_HEAD: OptionType.YES,
    SelectContext.MULLIGAN: OptionType.YES,
    # FIRST_EFFECT orders a card's own effects; the printed first effect is the
    # primary one, so YES is a reasonable deterministic default (SOT-1694).
    SelectContext.FIRST_EFFECT: OptionType.YES,
}


def yesno_default_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """IS_FIRST / COIN_HEAD / MULLIGAN: answer with the context's fixed default."""
    want = _YESNO_DEFAULT.get(select.context)
    if want is None or select.minCount > 1:
        return None
    for i, o in enumerate(select.option):
        if o.type == want:
            return [i]
    return None


# How discardable a card is — higher means shed it sooner. Surplus Energy goes
# first (Basic before the scarcer Special), then non-Pokémon cards, and Pokémon
# are kept (lowest priority).
_DISCARD_PRIORITY: dict[int, int] = {
    int(CardType.BASIC_ENERGY): 4,
    int(CardType.SPECIAL_ENERGY): 3,
    int(CardType.ITEM): 2,
    int(CardType.TOOL): 2,
    int(CardType.SUPPORTER): 2,
    int(CardType.STADIUM): 2,
    int(CardType.POKEMON): 0,
}


def _discard_score(cd) -> int:
    if cd is None:
        return 1  # unknown card: above Pokémon, below any energy
    return _DISCARD_PRIORITY.get(int(cd.cardType), 1)


def discard_card_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """DISCARD / TO_DECK / TO_DECK_BOTTOM: shed surplus Energy first.

    Card-selection contexts that remove a card from hand/play. We rank cards by
    :data:`_DISCARD_PRIORITY` (Basic Energy first, Pokémon last) and select the
    required number; an optional selection (``minCount == 0``) stays a no-op.
    """
    cards = damage.get_card_registry()
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        cid = _referenced_card_id(o, obs)
        cd = cards.get(cid) if cid is not None else None
        scored.append((i, (_discard_score(cd),)))
    return _top_k(scored, _pick_count(select, want_at_least_one=False))


def discard_energy_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """DISCARD_ENERGY / TO_HAND_ENERGY / TO_DECK_ENERGY / DISCARD_ENERGY_CARD.

    Attached-Energy variant of the discard family: when Energy must be paid /
    moved off a Pokémon, shed the more expendable Basic Energy before Special
    Energy, keeping the scarcer card in play.
    """
    cards = damage.get_card_registry()
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        cid = _referenced_energy_card_id(o, obs)
        cd = cards.get(cid) if cid is not None else None
        scored.append((i, (_discard_score(cd),)))
    return _top_k(scored, _pick_count(select, want_at_least_one=False))


def damage_target_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """DAMAGE_COUNTER / DAMAGE_COUNTER_ANY / DAMAGE: aim to finish a Knock Out.

    Prefer an *opponent* target, then one the remaining counters can actually
    Knock Out (``select.remainDamageCounter``, 10 damage each), then the lowest
    current HP — so placements complete a KO rather than being spread thin.
    """
    state = obs.current
    you = state.yourIndex if state is not None else None
    remain = select.remainDamageCounter or 0
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        p = _referenced_pokemon(o, obs)
        is_opp = 1 if (you is not None and o.playerIndex is not None and o.playerIndex != you) else 0
        if p is None or p.hp is None:
            scored.append((i, (is_opp, 0, -(10 ** 9))))  # unknown target: least preferred
        else:
            ko = 1 if (remain > 0 and p.hp <= remain * 10) else 0
            scored.append((i, (is_opp, ko, -p.hp)))  # lower HP -> larger key -> first
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def _option_card_id(option, select: SelectData, obs: Observation) -> Optional[int]:
    """Like :func:`_referenced_card_id` but also resolves DECK / LOOKING refs.

    A DECK option indexes into ``select.deck`` (populated when selecting cards
    from the deck); a LOOKING option indexes into ``state.looking`` (the cards an
    effect is currently revealing). Facedown / malformed refs resolve to ``None``.
    """
    try:
        if option.area == AreaType.DECK:
            if select.deck and option.index is not None and 0 <= option.index < len(select.deck):
                return select.deck[option.index].id
            return None
        if option.area == AreaType.LOOKING:
            state = obs.current
            looking = state.looking if state is not None else None
            if looking and option.index is not None and 0 <= option.index < len(looking):
                card = looking[option.index]
                return card.id if card is not None else None
            return None
    except (IndexError, TypeError, AttributeError):
        return None
    return _referenced_card_id(option, obs)


# How much we want to *gain* a card (take it to hand / attach it from an
# effect) — the mirror image of :data:`_DISCARD_PRIORITY`. The dominant loss
# mode is running out of Pokémon to promote (バトル場不在), so Basic Pokémon top
# the list as bench insurance, then evolutions, then Energy to load them.
_GAIN_PRIORITY: dict[int, int] = {
    int(CardType.POKEMON): 4,        # +1 more below if it is a Basic
    int(CardType.BASIC_ENERGY): 3,
    int(CardType.SPECIAL_ENERGY): 3,
    int(CardType.SUPPORTER): 2,
    int(CardType.ITEM): 2,
    int(CardType.TOOL): 2,
    int(CardType.STADIUM): 2,
}


def to_hand_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """TO_HAND (fetch a card from deck / discard / looking into hand).

    SOT-1682: this context used to fall back to a *random* pick. Prefer a Basic
    Pokémon (the dominant loss is having no Pokémon to promote), then any
    Pokémon, then Energy, then trainers. Options referencing in-play cards
    (a bounce effect) are deferred to the safe fallback — bouncing our own board
    away needs a smarter rule than "gaining a card is good".
    """
    cards = damage.get_card_registry()
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        if o.area in (AreaType.ACTIVE, AreaType.BENCH):
            return None  # bounce effect: no opinion
        cid = _option_card_id(o, select, obs)
        cd = cards.get(cid) if cid is not None else None
        gain = _GAIN_PRIORITY.get(int(cd.cardType), 1) if cd is not None else 0
        is_basic_poke = 1 if (
            cd is not None and cd.cardType == CardType.POKEMON and cd.basic
        ) else 0
        hp = cd.hp if cd is not None else 0
        scored.append((i, (gain, is_basic_poke, hp)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def attach_card_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """ATTACH_TO (pick the card an effect attaches, e.g. from LOOKING).

    SOT-1682: was a random pick. Prefer Energy (Basic first — the scarcer
    Special Energy stays available); an unknown / facedown card ranks last.
    """
    cards = damage.get_card_registry()
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        cid = _option_card_id(o, select, obs)
        cd = cards.get(cid) if cid is not None else None
        if cd is None:
            rank = 0
        elif cd.cardType == CardType.BASIC_ENERGY:
            rank = 3
        elif cd.cardType == CardType.SPECIAL_ENERGY:
            rank = 2
        else:
            rank = 1
        scored.append((i, (rank,)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def attach_target_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """ATTACH_FROM (pick the in-play Pokémon an effect attaches Energy to).

    SOT-1682: was a random pick. Energy acceleration goes to the attacker it
    helps most: a viable attacker still short of its best attack (smallest
    deficit first, then the biggest weakness-aware hit on the current defender),
    then an already-loaded attacker, and a Pokémon with no damaging attack last.
    """
    cards = damage.get_card_registry()
    defender_cd = _opponent_active_card(obs)
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        p = _referenced_pokemon(o, obs)
        pc = cards.get(p.id) if p is not None else None
        if p is not None and _is_viable_attacker(pc):
            deficit = _energy_deficit(p, pc)
            band = 2 if deficit > 0 else 1  # still loading > already able to fire
            scored.append((i, (band, -deficit, _best_damage_vs(pc, defender_cd))))
        else:
            hp = p.hp if p is not None else 0
            scored.append((i, (0, 0, hp)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def draw_count_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """DRAW_COUNT: draw as many cards as offered — unless the deck is nearly gone.

    SOT-1682: was a random pick; more cards is more options. SOT-1694: across the
    25 tournament decks deck-out is 26% of mirror losses, so at
    :data:`DECK_LOW_THRESHOLD` cards or fewer the preference flips to drawing as
    *few* as allowed (every remaining card is a turn of survival).
    """
    if select.minCount > 1:
        return None
    deck_low = _own_deck_low(obs)
    best_i: Optional[int] = None
    best_n: Optional[int] = None
    for i, o in enumerate(select.option):
        if o.type != OptionType.NUMBER or o.number is None:
            return None  # unexpected shape: no opinion
        better = best_n is None or (o.number < best_n if deck_low else o.number > best_n)
        if better:
            best_n = o.number
            best_i = i
    return [best_i] if best_i is not None else None


# --------------------------------------------------------------------------- #
# 25-deck fallback-hole handlers (SOT-1694).
#
# The 25-deck mirror trace aggregation (eval/trace_gap_report.py) surfaced ~1.7k
# legal-random decisions per 500 games in contexts with no registered handler.
# Each handler below covers one of those observed contexts using only generic
# card attributes (no card IDs), and defers (``None``) on any unexpected shape.
# --------------------------------------------------------------------------- #

def to_bench_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """TO_BENCH / TO_FIELD: fetch as many Pokémon onto the board as offered.

    Options reference deck cards (a Nest Ball-style search). Board width is the
    no-active safety net, so take ``maxCount`` picks, ranked like the setup Bench
    rule: Basic first, evolution line, best attack, HP. Defers when no option
    resolves to a Pokémon (facedown / unknown refs — nothing to rank).
    """
    cards = damage.get_card_registry()
    scored: list[tuple[int, tuple]] = []
    any_pokemon = False
    for i, o in enumerate(select.option):
        cid = _option_card_id(o, select, obs)
        cd = cards.get(cid) if cid is not None else None
        is_poke = cd is not None and cd.cardType == CardType.POKEMON
        any_pokemon = any_pokemon or is_poke
        atk = _best_attack(cd)
        scored.append((i, (
            1 if is_poke else 0,
            1 if (cd is not None and cd.basic) else 0,
            1 if damage.has_evolution_line(cd) else 0,
            atk.damage if atk is not None else 0,
            cd.hp if cd is not None else 0,
        )))
    if not any_pokemon:
        return None
    k = max(select.minCount, min(select.maxCount, len(scored)))
    return _top_k(scored, k)


def activate_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """ACTIVATE: use the optional effect — unless the deck is nearly gone.

    "Would you like to activate the effect?" prompts are the deciding player's
    own triggered effects (draw / search abilities), so YES is the human default.
    Many of them draw, so under the deck-out guard the answer flips to NO.
    """
    if select.minCount > 1:
        return None
    want = OptionType.NO if _own_deck_low(obs) else OptionType.YES
    for i, o in enumerate(select.option):
        if o.type == want:
            return [i]
    return None


def skill_order_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """SKILL_ORDER: resolve simultaneous effects in the offered order.

    The observed prompts order copies of the *same* effect, where the order
    cannot matter — a deterministic identity pick just removes the randomness.
    """
    k = max(select.minCount, min(select.maxCount, len(select.option)))
    return list(range(k))


def attack_select_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """ATTACK (an effect asks which of our attacks to use): the hardest hit.

    Ranks by weakness/resistance-adjusted damage against the current defender,
    with a Knock Out outranking any non-lethal hit.
    """
    state = obs.current
    if state is None or select.minCount > 1:
        return None
    try:
        me = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]
    except (IndexError, TypeError, AttributeError):
        return None
    cards = damage.get_card_registry()
    attacks = damage.get_attack_registry()
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None
    attacker_cd = cards.get(my_active.id) if my_active is not None else None
    defender_cd = cards.get(opp_active.id) if opp_active is not None else None
    defender_hp = opp_active.hp if opp_active is not None else None
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        atk = attacks.get(o.attackId) if o.attackId is not None else None
        if atk is None:
            scored.append((i, (0, 0)))
            continue
        dmg = damage.attack_damage(atk, attacker_cd, defender_cd)
        ko = 1 if damage.is_ko(atk, attacker_cd, defender_cd, defender_hp) else 0
        scored.append((i, (ko, dmg)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def disable_attack_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """DISABLE_ATTACK: disable the opponent's biggest threat to our Active."""
    state = obs.current
    if state is None or select.minCount > 1:
        return None
    try:
        me = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]
    except (IndexError, TypeError, AttributeError):
        return None
    cards = damage.get_card_registry()
    attacks = damage.get_attack_registry()
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None
    my_cd = cards.get(my_active.id) if my_active is not None else None
    opp_cd = cards.get(opp_active.id) if opp_active is not None else None
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        atk = attacks.get(o.attackId) if o.attackId is not None else None
        dmg = damage.attack_damage(atk, opp_cd, my_cd) if atk is not None else 0
        scored.append((i, (dmg,)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def evolve_context_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """EVOLVE (an effect offers evolution source+target pairs): best evolution.

    Same shape as a MAIN EVOLVE option (hand card + in-play target). Prefer the
    Active target, then the evolution hitting the current defender hardest, then
    the biggest HP body.
    """
    state = obs.current
    if state is None:
        return None
    try:
        me = state.players[state.yourIndex]
    except (IndexError, TypeError, AttributeError):
        return None
    cards = damage.get_card_registry()
    defender_cd = _opponent_active_card(obs)
    hand = me.hand or []
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        idx = o.index
        evo_cd = cards.get(hand[idx].id) if (idx is not None and 0 <= idx < len(hand)) else None
        scored.append((i, (
            1 if o.inPlayArea == AreaType.ACTIVE else 0,
            _best_damage_vs(evo_cd, defender_cd),
            evo_cd.hp if evo_cd is not None else 0,
        )))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def evolves_to_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """EVOLVES_TO (pick the evolution card, e.g. from the deck): the strongest.

    Prefer the evolution hitting the current defender hardest, then higher HP.
    """
    cards = damage.get_card_registry()
    defender_cd = _opponent_active_card(obs)
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        cid = _option_card_id(o, select, obs)
        cd = cards.get(cid) if cid is not None else None
        is_poke = 1 if (cd is not None and cd.cardType == CardType.POKEMON) else 0
        scored.append((i, (
            is_poke,
            _best_damage_vs(cd, defender_cd),
            cd.hp if cd is not None else 0,
        )))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def detach_from_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """DETACH_FROM: pick the Pokémon that loses the least by giving a card up.

    For our own board (an energy-moving / cost-paying effect): a Pokémon with no
    damaging attack first, then the biggest Energy surplus beyond its best
    attack's cost. When the options point at the *opponent's* board (a disruption
    effect), invert: strip their loaded viable attacker.
    """
    state = obs.current
    you = state.yourIndex if state is not None else None
    if you is None:
        return None
    cards = damage.get_card_registry()
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        p = _referenced_pokemon(o, obs)
        pc = cards.get(p.id) if p is not None else None
        viable = _is_viable_attacker(pc)
        atk = _best_attack(pc)
        cost = len(atk.energies or []) if atk is not None else 0
        surplus = _energy_count(p) - cost
        if o.playerIndex is not None and o.playerIndex != you:
            # Opponent's Pokémon: hurt the loaded attacker hardest.
            scored.append((i, (1, 1 if viable else 0, _energy_count(p))))
        else:
            scored.append((i, (0, 0 if viable else 1, surplus)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def discard_tool_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """DISCARD_TOOL_CARD: trash the opponent's Tool first, then any of ours."""
    state = obs.current
    you = state.yourIndex if state is not None else None
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        is_opp = 1 if (you is not None and o.playerIndex is not None and o.playerIndex != you) else 0
        scored.append((i, (is_opp,)))
    has_opp = any(s[1][0] for s in scored)
    # Optional discards stay a no-op unless an opponent Tool can be stripped.
    return _top_k(scored, _pick_count(select, want_at_least_one=has_opp))


def heal_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """HEAL: prefer the Active Pokémon, then whichever has taken the most damage."""
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        p = _referenced_pokemon(o, obs)
        is_active = 1 if o.area == AreaType.ACTIVE else 0
        if p is not None and p.maxHp is not None and p.hp is not None:
            damage_taken = max(0, p.maxHp - p.hp)
        else:
            damage_taken = 0
        scored.append((i, (is_active, damage_taken)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def remove_damage_count_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """REMOVE_DAMAGE_COUNTER_COUNT: remove as many damage counters as offered.

    SOT-1707 cycle 2: opened by the ability budget (healing abilities were never
    reached while abilities were banned) — was a random fallback. Healing is
    free value; the deck guards do not apply because no cards are drawn.
    """
    if select.minCount > 1:
        return None
    best_i: Optional[int] = None
    best_n: Optional[int] = None
    for i, o in enumerate(select.option):
        if o.type != OptionType.NUMBER or o.number is None:
            return None  # unexpected shape: no opinion
        if best_n is None or o.number > best_n:
            best_n = o.number
            best_i = i
    return [best_i] if best_i is not None else None


# --------------------------------------------------------------------------- #
# Scoring MAIN policy (SOT-1635, R4).
#
# The R2 policy (:func:`main_context_handler`) walks a *fixed priority ladder*.
# R4 replaces the ladder with **option scoring**: every legal MAIN option is
# given a numeric score and the highest wins. This is behaviour-compatible with
# the ladder on the actions it already handled (set up, then swing, KO first)
# because the score bands are ordered the same way — but it also lets *new*
# tactical options slot in at a principled rank:
#
#   * **サポーター/グッズ使用** — the fixed policy never played a Supporter/Item
#     (it only played Basic Pokémon and attached Energy), leaving card-advantage
#     Supporters unused; scoring plays them.
#   * **リトリート判断** — retreat only when it pays: escape a disabling Special
#     Condition, or duck a lethal hit onto a viable Bench attacker; never idly.
#   * **ベンチ育成 / エネルギー効率** — once the Active can already fire its best
#     attack, extra Energy goes to a second Bench attacker instead of overloading
#     the Active.
#
# Every scorer is a pure function of the observation, so the same per-option
# score — and the position value in :func:`evaluate_position` — can be reused as
# the evaluation function of the R6 (SOT-1638) one-ply search.
# --------------------------------------------------------------------------- #

# Score bands, high = preferred. They are spaced so an intra-band tie-break
# (damage, HP, energy deficit) never bleeds across a band boundary. Ordering:
# take a winning KO > develop/support the board > justified retreat > swing >
# end; wasteful actions (voluntary discard, idle retreat, optional ability) sit
# below END so they are only ever chosen when literally nothing else is offered.
S_LETHAL = 10_000.0    # + damage + prize value: an attack that Knocks Out — win tempo
S_BENCH_INSURANCE = 9_000.0  # play a Basic while the Bench is EMPTY: one KO from a
                             # no-active (場切れ) loss, only a winning attack outranks
                             # rebuilding the safety net (SOT-1694)
S_SUPPORTER = 3_000.0  # play a Supporter (draw / search): card advantage, once per turn
S_EVOLVE = 2_800.0     # evolve: more HP and stronger attacks
S_PLAY_BASIC = 2_600.0 # + evolution-line / HP tie-break: develop the Bench
S_ATTACH_LOAD = 2_400.0  # attach Energy to a Pokémon still short of its best attack
S_RETREAT_OK = 2_200.0   # a *justified* retreat (condition escape / dodge a KO)
S_ITEM = 2_000.0         # play an Item / Tool / Stadium: generically useful
S_ATTACH_BENCH = 1_600.0 # ベンチ育成: load a second Bench attacker
S_ATTACH_OVER = 900.0    # drop Energy on an already-loaded Pokémon (still > a swing)
S_ATTACK_BASE = 500.0    # + damage: a non-lethal attack ends the turn, so swing last
S_END = 0.0              # end the turn
S_ABILITY_ON = 1_800.0   # use a Pokémon ability (draw / search engines), budget-guarded
S_EVOLVE_DOOMED = 800.0  # evolve an Active that dies next turn even evolved: the
                         # evolution card is lost with it — develop elsewhere first,
                         # but still above a swing (evolving never ends the turn)
S_DECK_GUARD = -0.2      # a Supporter with the deck nearly empty: Supporters are
                         # draw-heavy and deck-out (山札切れ) is 26% of 25-deck
                         # mirror losses (SOT-1694) — stop digging, END instead
S_ATTACH_DOOMED = -0.3   # attach to an Active that dies next turn: the Energy is
                         # discarded with it — keeping the card in hand for the
                         # successor beats feeding the doomed Pokémon (SOT-1682)
S_ABILITY = -0.4         # ability once the per-turn budget is spent (or the deck is
                         # low): prefer END — the budget is what avoids no-op loops
S_RETREAT_NO = -0.5      # an *unjustified* retreat: never chosen over ending the turn
S_DISCARD = -0.6         # never voluntarily discard a card from play

# Ability budget (SOT-1707): MAIN ABILITY options used to be *never* chosen
# (S_ABILITY < S_END) because a state-preserving ability would be re-picked
# forever. Instead of banning them, cap ability uses per turn — draw/search
# engines (e.g. Dudunsparce) are exactly the card advantage the decks are built
# around. The deck-low guard still applies (most abilities dig the deck).
ABILITY_BUDGET_PER_TURN = 8

# Abilities are draw/search-heavy, so they get their own, much earlier deck
# floor: with ≤ this many cards left the budget reads as spent. Chosen by A/B
# (SOT-1707 cycle 2): floor 6 shifted losses wholesale into deck_out.
ABILITY_DECK_FLOOR = 27

# Deck-parity guard (SOT-1707 cycle 3/4): self-mill draw engines turned the
# ability budget into deck-out losses on grind decks (N's Zoroark ν/ex and
# Alakazam-Dudunsparce lost 0.30-0.35 of mirrors with 70%+ deck_out). Abilities
# pause while the own deck is thinner than the opponent's by more than this
# margin, so digging can never out-mill the opponent's natural draw rate.
# Cycle 3 measured the guard globally: it repaired the grind decks (e.g.
# 07_n_s_zoroark_n 0.300→0.500) but throttled the decks whose abilities win
# prize races (06_hydrapple 0.750→0.525, 05_dragapult_dudunsparce 0.688→0.550)
# for a flat 0.517 aggregate — so cycle 4 keys it per deck instead. Only decks
# whose cycle-3 deck_out losses causally dropped under the guard are listed;
# composition detection would misfire (05 shares Dudunsparce with 12 but is
# hurt by the guard). An unknown deck (e.g. a submission deck.csv) plays
# unguarded, i.e. the pre-cycle-3 champion behaviour.
ABILITY_PARITY_MARGIN = -3

# Deck stems (deck_path basename without .csv) whose MAIN abilities obey the
# parity guard. Cycle-3 evidence, deck_out losses per 80: 07: 42→24,
# 24: 40→37, 12: 37→33, 20: 19→10, 18: 20→15.
ABILITY_PARITY_DECKS = frozenset({
    "07_n_s_zoroark_n",
    "12_alakazam_dudunsparce",
    "18_rocket_s_honchkrow",
    "20_cynthia_s_garchomp_ex",
    "24_n_s_zoroark_ex_naic_10th",
})

# SOT-1734 baseline (25-deck mirror, seed 20260719) identified these decks as
# the bottom cohort.  They benefit from looking past the immediately available
# damage and preserving a multi-turn board; other decks retain the champion's
# proven one-step ordering.
ADAPTIVE_SEARCH_DECKS = frozenset({
    "01_dragapult",
    "03_dragapult_blaziken",
    "06_hydrapple",
    "08_ogerpon_box",
    "21_lillie_s_clefairy_ex_naic_champion",
    "23_slowking_naic_4th",
    "24_n_s_zoroark_ex_naic_10th",
})


def adaptive_search_depth(
    obs: Observation,
    branching: int,
    enabled: bool,
    profile: Optional[RuntimeProfile] = None,
) -> int:
    """Return a bounded tactical horizon from width and remaining game length.

    Wide positions stay at depth one to keep decision time predictable.  A
    low-win deck gets two plies when the option set is tractable, and three only
    in a narrow position with at least three forced draws/prize turns left.
    This is a deterministic selective-extension policy rather than an engine
    rollout, so hidden cards are never guessed and legality remains unchanged.
    """
    branch_limit = profile.max_branching_for_extension if profile else 10
    max_depth = profile.max_depth if profile else 3
    if not enabled or branching > branch_limit or max_depth <= 1:
        return 1
    state = obs.current
    if state is None:
        return 1
    try:
        me = state.players[state.yourIndex]
        turns_left = min(me.deckCount, max(1, len(me.prize)))
    except (IndexError, TypeError, AttributeError):
        return 1
    wanted = 3 if branching <= 5 and turns_left >= 3 else 2
    return min(max_depth, wanted)


def _horizon_adjustment(option_type, depth: int) -> float:
    """Approximate future board value for MAIN actions at a selective depth."""
    if depth <= 1:
        return 0.0
    future = float(depth - 1)
    if option_type == OptionType.EVOLVE:
        return 420.0 * future
    if option_type == OptionType.ATTACH:
        return 300.0 * future
    if option_type == OptionType.PLAY:
        return 180.0 * future
    if option_type == OptionType.ATTACK:
        # Lethal attacks remain protected by their 10k band.  This only stops a
        # tempting non-lethal swing from ending a turn before durable setup.
        return -220.0 * future
    return 0.0


def _ability_deck_parity_ok(obs: Observation) -> bool:
    """True iff own deck is not more than the margin thinner than opponent's."""
    state = obs.current
    if state is None:
        return True
    try:
        me = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]
        return me.deckCount + ABILITY_PARITY_MARGIN >= opp.deckCount
    except (IndexError, TypeError, AttributeError):
        return True


def _own_deck_below(obs: Observation, floor: int) -> bool:
    """True iff the deciding player's own deck has ``floor`` cards or fewer."""
    state = obs.current
    if state is None:
        return False
    try:
        return state.players[state.yourIndex].deckCount <= floor
    except (IndexError, TypeError, AttributeError):
        return False

# Deck-out guard (SOT-1694): at or below this many cards left, stop volunteering
# draws (max DRAW_COUNT, optional ACTIVATE effects, Supporters) — every turn start
# forcibly draws one card, so this is the remaining safety margin in turns.
DECK_LOW_THRESHOLD = 6


def _own_deck_low(obs: Observation) -> bool:
    """True iff the deciding player's own deck is at the deck-out guard level."""
    state = obs.current
    if state is None:
        return False
    try:
        return state.players[state.yourIndex].deckCount <= DECK_LOW_THRESHOLD
    except (IndexError, TypeError, AttributeError):
        return False


def _energy_count(poke) -> int:
    """Number of Energy units attached to ``poke`` (0 for ``None``)."""
    if poke is None:
        return 0
    return len(poke.energies) if poke.energies else 0


def _best_attack(card) -> Optional[object]:
    """The highest-damage attack of ``card`` (``None`` if it has no attacks)."""
    if card is None or not card.attacks:
        return None
    reg = damage.get_attack_registry()
    best = None
    for aid in card.attacks:
        atk = reg.get(aid)
        if atk is None:
            continue
        if best is None or atk.damage > best.damage:
            best = atk
    return best


def _energy_deficit(poke, card) -> int:
    """Energy still needed before ``poke`` can use ``card``'s best attack.

    Counts units only (the engine enforces exact colour when it offers the
    ATTACK option); ``0`` means the Pokémon can already fire its best attack, so
    further Energy is better spent developing the Bench.
    """
    atk = _best_attack(card)
    if atk is None:
        return 0
    return max(0, len(atk.energies or []) - _energy_count(poke))


def _is_viable_attacker(card) -> bool:
    """True iff ``card`` is a Pokémon with at least one damaging attack."""
    atk = _best_attack(card)
    return atk is not None and atk.damage > 0


def _attach_enables_ko(poke, card, defender_cd, defender_hp) -> bool:
    """True iff one more Energy lets ``poke`` fire an attack that KOs the defender.

    Guard for the doomed-Active rule (SOT-1682): even a dying Active should be
    fed the Energy that lets it take the Knock Out *first* this very turn.
    """
    if poke is None or card is None or not card.attacks:
        return False
    reg = damage.get_attack_registry()
    affordable = _energy_count(poke) + 1
    for aid in card.attacks:
        atk = reg.get(aid)
        if atk is None or len(atk.energies or []) > affordable:
            continue
        if damage.is_ko(atk, card, defender_cd, defender_hp):
            return True
    return False


def _best_damage_vs(card, defender_cd) -> int:
    """Max weakness/resistance-adjusted damage ``card`` can deal to ``defender_cd``.

    ``0`` when ``card`` has no attacks (or is ``None``); with an unknown defender
    the raw attack damage is used (``attack_damage`` treats ``None`` as neutral).
    """
    if card is None or not card.attacks:
        return 0
    reg = damage.get_attack_registry()
    best = 0
    for aid in card.attacks:
        atk = reg.get(aid)
        if atk is None:
            continue
        best = max(best, damage.attack_damage(atk, card, defender_cd))
    return best


def _inplay_pokemon(option, obs: Observation):
    """Resolve the in-play Pokémon an ATTACH / EVOLVE option targets.

    ATTACH/EVOLVE options point at their target via ``inPlayArea`` /
    ``inPlayIndex`` (the ``area`` / ``index`` fields address the *hand* card being
    attached). Returns the live :class:`~cg.api.Pokemon`, or ``None`` on any
    malformed / out-of-range reference so the caller can fall back safely.
    """
    ps = _player_state(obs, option.playerIndex)
    if ps is None or option.inPlayIndex is None:
        return None
    try:
        if option.inPlayArea == AreaType.ACTIVE:
            return ps.active[option.inPlayIndex]
        if option.inPlayArea == AreaType.BENCH:
            return ps.bench[option.inPlayIndex]
    except (IndexError, TypeError, AttributeError):
        return None
    return None


def _incoming_damage(opp_active, opp_card, my_active_card, *, headroom: int = 1) -> int:
    """Largest damage the opponent's Active could deal to our Active next turn.

    SOT-1682: only attacks the opponent could actually *pay for* count — cost at
    most their attached Energy plus ``headroom`` (the one manual attachment they
    get next turn). Without this the threat estimate triggers doom/retreat logic
    against attacks that are turns away from being usable.
    """
    if opp_active is None or opp_card is None or not opp_card.attacks:
        return 0
    reg = damage.get_attack_registry()
    affordable = _energy_count(opp_active) + headroom
    worst = 0
    for aid in opp_card.attacks:
        atk = reg.get(aid)
        if atk is None or len(atk.energies or []) > affordable:
            continue
        worst = max(worst, damage.attack_damage(atk, opp_card, my_active_card))
    return worst


def _should_retreat(me, opp, cards) -> bool:
    """Decide whether retreating the Active this turn actually pays.

    Retreat costs Energy (discarded to pay the retreat cost) and switching clears
    Special Conditions, so it is worth it only when the Active is a liability:

    * it is Confused / Asleep / Paralyzed (can't reliably attack), or
    * the opponent's Active would Knock it Out next turn,

    **and** there is a Bench Pokémon worth promoting (a viable attacker, or one
    with more HP than the endangered Active). With no Bench, retreat is impossible
    and pointless, so we never score it up.

    SOT-1682: additionally, a *dead wall* — an Active with no damaging attack —
    retreats for a Bench attacker that can already fire, so the board never gets
    stuck behind a Pokémon that can only pass turns.
    """
    active = me.active[0] if me.active else None
    if active is None or not me.bench:
        return False
    active_card = cards.get(active.id)

    disabled = me.confused or me.asleep or me.paralyzed
    opp_active = opp.active[0] if opp.active else None
    opp_card = cards.get(opp_active.id) if opp_active is not None else None
    incoming = _incoming_damage(opp_active, opp_card, active_card)
    dying = active.hp is not None and incoming >= active.hp
    if not (disabled or dying):
        if not _is_viable_attacker(active_card):
            # Dead wall: swap only for a Bench attacker that can fire NOW (a
            # retreat costs attached Energy, so a not-yet-loaded replacement
            # would trade tempo for nothing).
            for p in me.bench:
                pc = cards.get(p.id)
                if _is_viable_attacker(pc) and _energy_deficit(p, pc) <= 0:
                    return True
        return False

    active_hp = active.hp if active.hp is not None else 0
    for p in me.bench:
        if _is_viable_attacker(cards.get(p.id)):
            return True
        if (p.hp or 0) > active_hp:
            return True
    return False


def _score_main_option(
    o, obs: Observation, *, cards, me, opp, state, hand, bench_open,
    attacker_cd, defender_cd, defender_hp, bands: BandAdjust,
    ability_ok: bool = False,
) -> Optional[float]:
    """Score a single MAIN option (higher = better), or ``None`` to ignore it.

    ``None`` is returned for options this policy has no opinion on (e.g. a
    newly-appended, unknown :class:`~cg.api.OptionType`); if *every* option scores
    ``None`` the handler defers to the safe fallback rather than guess.
    """
    t = o.type
    if t == OptionType.END:
        return S_END
    if t == OptionType.ATTACK:
        if o.attackId is None:
            return S_ATTACK_BASE
        atk = damage.get_attack_registry().get(o.attackId)
        if atk is None:
            return S_ATTACK_BASE
        dmg = damage.attack_damage(atk, attacker_cd, defender_cd)
        if damage.is_ko(atk, attacker_cd, defender_cd, defender_hp):
            prize = 1
            if defender_cd is not None:
                if defender_cd.megaEx:
                    prize = 3
                elif defender_cd.ex:
                    prize = 2
            return S_LETHAL + dmg + 100.0 * prize
        return S_ATTACK_BASE + dmg
    if t == OptionType.PLAY:
        idx = o.index
        if idx is None or not (0 <= idx < len(hand)):
            return None
        cd = cards.get(hand[idx].id)
        if cd is None:
            return S_ITEM
        if cd.cardType == CardType.POKEMON:
            if not (cd.basic and bench_open):
                return None
            line = 1.0 if damage.has_evolution_line(cd) else 0.0
            tie = 10.0 * line + (cd.hp or 0) / 100.0
            if not me.bench:
                # 場切れ guard (SOT-1694): an empty Bench is one Knock Out from
                # a no-active loss — rebuilding it outranks everything but a
                # winning attack.
                return S_BENCH_INSURANCE + tie
            return S_PLAY_BASIC + bands.play_basic + tie
        if cd.cardType == CardType.SUPPORTER:
            if _own_deck_low(obs):
                return S_DECK_GUARD
            if state.supporterPlayed:
                return S_ITEM
            # SOT-1682: with a big hand the Supporter's card advantage is worth
            # less than spending the cards we already hold — develop first (the
            # Supporter still gets played later in the turn at the S_ITEM rank).
            # The threshold comes from BandAdjust (an archetype-adaptive variant
            # was measured and rejected in SOT-1694; it stays at the v2 value).
            return S_ITEM if len(hand) >= bands.supporter_hand_max else S_SUPPORTER
        return S_ITEM
    if t == OptionType.EVOLVE:
        # SOT-1682: prefer evolving the Active, and prefer the evolution that
        # hits the current defender hardest / adds the most HP. Tie-break stays
        # < 100 so it can never cross the band gap down to S_PLAY_BASIC.
        bonus = 50.0 if o.inPlayArea == AreaType.ACTIVE else 0.0
        idx = o.index
        evo_cd = cards.get(hand[idx].id) if (idx is not None and 0 <= idx < len(hand)) else None
        if evo_cd is not None:
            bonus += min(_best_damage_vs(evo_cd, defender_cd), 300) / 10.0
            bonus += (evo_cd.hp or 0) / 100.0
        if o.inPlayArea == AreaType.ACTIVE and evo_cd is not None:
            # Doomed-Active guard, EVOLVE extension (SOT-1694): if the Active
            # dies next turn even *after* evolving (damage counters persist, so
            # the evolved HP is evo max HP minus damage already taken) and the
            # evolution does not enable a Knock Out first, the evolution card is
            # lost with it — develop elsewhere before feeding it.
            target = _inplay_pokemon(o, obs)
            opp_active = opp.active[0] if opp.active else None
            opp_cd = cards.get(opp_active.id) if opp_active is not None else None
            incoming = _incoming_damage(opp_active, opp_cd, evo_cd)
            if target is not None and target.hp is not None and target.maxHp is not None:
                evolved_hp = (evo_cd.hp or 0) - max(0, target.maxHp - target.hp)
                if (
                    incoming >= evolved_hp
                    and not _attach_enables_ko(target, evo_cd, defender_cd, defender_hp)
                ):
                    return S_EVOLVE_DOOMED + bonus
        return S_EVOLVE + bands.evolve + bonus
    if t == OptionType.ATTACH:
        target = _inplay_pokemon(o, obs)
        card = cards.get(target.id) if target is not None else None
        if target is None:
            return S_ATTACH_OVER
        deficit = _energy_deficit(target, card)
        if deficit <= 0:
            return S_ATTACH_OVER  # already able to fire its best attack
        # SOT-1682: among attach targets, prefer the one closest to firing and
        # the one hitting the current defender hardest (weakness-aware). Bounded
        # to < 100 so the tie-break never crosses a band boundary.
        bonus = 50.0 / deficit + min(_best_damage_vs(card, defender_cd), 300) / 10.0
        if o.inPlayArea == AreaType.ACTIVE:
            # Doomed Active (SOT-1682): if the opponent can Knock it Out next
            # turn (and it cannot win the race by KO'ing first — a lethal ATTACK
            # would outscore any attach anyway), the attached Energy is lost with
            # it. Keep the card in hand for the successor instead.
            my_active = me.active[0] if me.active else None
            opp_active = opp.active[0] if opp.active else None
            opp_cd = cards.get(opp_active.id) if opp_active is not None else None
            if (
                my_active is not None and my_active.hp is not None
                and _incoming_damage(opp_active, opp_cd, attacker_cd) >= my_active.hp
                and not _attach_enables_ko(target, attacker_cd, defender_cd, defender_hp)
            ):
                return S_ATTACH_DOOMED
            return S_ATTACH_LOAD + bands.attach_load + bonus
        return (S_ATTACH_BENCH + bands.attach_bench + bonus) if _is_viable_attacker(card) else S_ATTACH_OVER
    if t == OptionType.RETREAT:
        return S_RETREAT_OK if _should_retreat(me, opp, cards) else S_RETREAT_NO
    if t == OptionType.ABILITY:
        return S_ABILITY_ON if ability_ok else S_ABILITY
    if t == OptionType.DISCARD:
        return S_DISCARD
    return None


def scoring_main_context_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """MAIN policy (SOT-1635): score every legal option, pick the highest.

    Behaviourally a superset of :func:`main_context_handler` — it sets up before
    swinging and takes a Knock Out first — but it additionally plays Supporters /
    Items, retreats when it pays, and spreads Energy onto a second Bench attacker.
    Returns the single chosen option index, or ``None`` (defer) when the position
    is too degenerate to reason about or no option is scoreable.
    """
    state = obs.current
    if state is None:
        return None
    you = state.yourIndex
    try:
        me = state.players[you]
        opp = state.players[1 - you]
    except (IndexError, TypeError, AttributeError):
        return None

    cards = damage.get_card_registry()
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None
    attacker_cd = cards.get(my_active.id) if my_active is not None else None
    defender_cd = cards.get(opp_active.id) if opp_active is not None else None
    defender_hp = opp_active.hp if opp_active is not None else None
    hand = me.hand or []
    bench_open = len(me.bench) < me.benchMax

    bands = getattr(agent, "bands", None) or BandAdjust()
    # Ability budget (SOT-1707): reset the counter when the turn changes; an
    # ability is worth choosing while budget remains and the deck is not low.
    turn = getattr(state, "turn", None)
    if getattr(agent, "_ability_turn", None) != turn:
        agent._ability_turn = turn
        agent._ability_uses = 0
    ability_ok = (
        getattr(agent, "_ability_uses", 0) < ABILITY_BUDGET_PER_TURN
        and not _own_deck_below(obs, ABILITY_DECK_FLOOR)
        and (not getattr(agent, "_ability_parity_guarded", False)
             or _ability_deck_parity_ok(obs))
    )
    search_depth = adaptive_search_depth(
        obs, len(select.option), getattr(agent, "_adaptive_search", False),
        getattr(agent, "runtime_profile", None),
    )
    best_i: Optional[int] = None
    best_s: Optional[float] = None
    for i, o in enumerate(select.option):
        s = _score_main_option(
            o, obs, cards=cards, me=me, opp=opp, state=state, hand=hand,
            bench_open=bench_open, attacker_cd=attacker_cd,
            defender_cd=defender_cd, defender_hp=defender_hp, bands=bands,
            ability_ok=ability_ok,
        )
        if s is None:
            continue
        # Keep terminal KO ordering absolute; all other choices receive the
        # bounded continuation value of the selected horizon.
        # Never let continuation value resurrect an explicitly guarded action
        # (deck-out draw, doomed attachment, idle retreat, discard).
        if S_END < s < S_LETHAL:
            s += _horizon_adjustment(o.type, search_depth)
        if best_s is None or s > best_s:  # strict '>' keeps the first (lowest) index on ties
            best_s = s
            best_i = i
    if best_i is None:
        return None
    if select.option[best_i].type == OptionType.ABILITY:
        agent._ability_uses = getattr(agent, "_ability_uses", 0) + 1
    return [best_i]


def evaluate_position(me, opp) -> float:
    """Static board value for ``me`` (positive = ``me`` is ahead).

    The reusable evaluation for the R6 (SOT-1638) one-ply search: prizes dominate
    (fewer of *your* prize cards left = closer to winning), then total board HP,
    then Energy developed on board. It reads only the observation's player states,
    so it can score any candidate position the search reaches.
    """
    def board(ps):
        pokes = [p for p in (list(ps.active) + list(ps.bench)) if p is not None]
        hp = sum(p.hp for p in pokes if p.hp is not None)
        energy = sum(_energy_count(p) for p in pokes)
        return hp, energy

    my_prizes_left = len(me.prize)
    opp_prizes_left = len(opp.prize)
    my_hp, my_energy = board(me)
    opp_hp, opp_energy = board(opp)
    return (
        1_000.0 * (opp_prizes_left - my_prizes_left)
        + 1.0 * (my_hp - opp_hp)
        + 10.0 * (my_energy - opp_energy)
    )


class RuleBasedAgent(Agent):
    """Dispatches each selection to a per-context rule, guarded by a safe fallback.

    Args:
        seed: Seed for the fallback RNG (reproducible random tie-breaks / fills).
        deck_path: Path to the deck CSV used for the initial selection.
        policy: MAIN-turn policy — ``"scoring"`` (default, SOT-1635 option
            scoring) or ``"fixed"`` (the SOT-1633 fixed-priority ladder). Only the
            MAIN context differs; all other context handlers are shared. Exposed so
            the new policy can be A/B compared against the old one.
    """

    # SelectContext -> handler. R2 (SOT-1633) registers the MAIN-turn policy;
    # R3 (SOT-1634) covers the setup / forced-selection contexts below so they no
    # longer rely on the random fallback. Any context still absent defers safely.
    CONTEXT_HANDLERS: dict[SelectContext, Handler] = {
        SelectContext.MAIN: main_context_handler,
        # Setup: Active is HP-first (survives the opening race, SOT-1682);
        # Bench stays line-first (it gets the time to evolve).
        SelectContext.SETUP_ACTIVE_POKEMON: place_basic_active_handler,
        SelectContext.SETUP_BENCH_POKEMON: place_basic_handler,
        # After a KO / on retreat: promote the readiest Bench Pokémon (Energy, HP).
        SelectContext.TO_ACTIVE: to_active_handler,
        SelectContext.SWITCH: to_active_handler,
        # Forced YesNo prompts: fixed reasonable defaults.
        SelectContext.IS_FIRST: yesno_default_handler,
        SelectContext.COIN_HEAD: yesno_default_handler,
        SelectContext.MULLIGAN: yesno_default_handler,
        # Discard / return-to-deck: shed surplus Energy first.
        SelectContext.DISCARD: discard_card_handler,
        SelectContext.TO_DECK: discard_card_handler,
        SelectContext.TO_DECK_BOTTOM: discard_card_handler,
        # Attached-Energy discard / move: Basic Energy before Special.
        SelectContext.DISCARD_ENERGY: discard_energy_handler,
        SelectContext.TO_HAND_ENERGY: discard_energy_handler,
        SelectContext.TO_DECK_ENERGY: discard_energy_handler,
        SelectContext.DISCARD_ENERGY_CARD: discard_energy_handler,
        # Damage placement: concentrate to finish a Knock Out.
        SelectContext.DAMAGE_COUNTER: damage_target_handler,
        SelectContext.DAMAGE_COUNTER_ANY: damage_target_handler,
        SelectContext.DAMAGE: damage_target_handler,
        # Healing: Active first, then the most-damaged Pokémon.
        SelectContext.HEAL: heal_handler,
        # Ability-budget fallback holes (SOT-1707 cycle 2) — healing abilities
        # opened these once MAIN ABILITY options became choosable (cycle 1).
        SelectContext.REMOVE_DAMAGE_COUNTER: heal_handler,
        SelectContext.REMOVE_DAMAGE_COUNTER_COUNT: remove_damage_count_handler,
        # Card-effect selections (SOT-1682) — these used to fall back to random.
        SelectContext.TO_HAND: to_hand_handler,
        SelectContext.ATTACH_TO: attach_card_handler,
        SelectContext.ATTACH_FROM: attach_target_handler,
        SelectContext.DRAW_COUNT: draw_count_handler,
        # 25-deck fallback holes (SOT-1694) — observed random-fallback contexts
        # from the mirror trace aggregation, largest first.
        SelectContext.TO_BENCH: to_bench_handler,
        SelectContext.TO_FIELD: to_bench_handler,
        SelectContext.ACTIVATE: activate_handler,
        SelectContext.SWITCH_ENERGY_CARD: discard_energy_handler,
        SelectContext.SWITCH_ENERGY: discard_energy_handler,
        SelectContext.SKILL_ORDER: skill_order_handler,
        SelectContext.ATTACK: attack_select_handler,
        SelectContext.DISABLE_ATTACK: disable_attack_handler,
        SelectContext.EVOLVE: evolve_context_handler,
        SelectContext.EVOLVES_TO: evolves_to_handler,
        SelectContext.DETACH_FROM: detach_from_handler,
        SelectContext.DISCARD_TOOL_CARD: discard_tool_handler,
        SelectContext.FIRST_EFFECT: yesno_default_handler,
        SelectContext.TO_PRIZE: discard_card_handler,
    }

    # SelectType -> handler. Consulted only when no CONTEXT_HANDLERS entry
    # applies, so a rule can cover a whole selection type at once. Also empty
    # in R1.
    TYPE_HANDLERS: dict[SelectType, Handler] = {}

    # MAIN-turn policy handlers, selected by the ``policy`` argument. The scoring
    # policy (SOT-1635) is the default; the fixed-priority ladder (SOT-1633) is
    # kept for A/B comparison.
    MAIN_POLICIES: dict[str, Handler] = {
        "scoring": scoring_main_context_handler,
        "fixed": main_context_handler,
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        deck_path: str = "deck.csv",
        policy: str = "scoring",
        profile_path: Optional[str] = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.deck_path = deck_path
        if policy not in self.MAIN_POLICIES:
            raise ValueError(
                f"unknown policy {policy!r}; expected one of {sorted(self.MAIN_POLICIES)}"
            )
        self.policy = policy
        # SOT-1869 promotion: the evaluated balanced adaptive-tempo profile is
        # now the production default. Loading is explicit and validated so the
        # submission cannot silently run with partial or unsafe tuning.
        self.runtime_profile = load_promoted_profile(profile_path)
        self._match_think_seconds = 0.0
        # Per-instance MAIN handler so the policy can vary without mutating the
        # shared class table; every other context still reads ``CONTEXT_HANDLERS``
        # live, so later registrations there keep taking effect.
        self._main_handler: Handler = self.MAIN_POLICIES[policy]
        # Archetype adaptation (SOT-1694): derive bounded scoring-band nudges
        # from the deck's composition, once per match. Best-effort — an
        # unreadable deck or registry leaves the neutral (all-zero) adjustment.
        # Ability budget bookkeeping (SOT-1707): per-turn use counter so that a
        # state-preserving ability cannot be re-picked forever.
        self._ability_turn: Optional[int] = None
        self._ability_uses: int = 0
        # Deck-conditional parity guard (SOT-1707 cycle 4): only known
        # deck-out-prone decks pause abilities on deck-parity deficit.
        stem = os.path.splitext(os.path.basename(deck_path))[0]
        self._ability_parity_guarded: bool = stem in ABILITY_PARITY_DECKS
        self._adaptive_search: bool = (
            self.runtime_profile.strategy == "matchup-responsive-balanced-tempo"
            or stem in ADAPTIVE_SEARCH_DECKS
        )
        self.bands = BandAdjust()
        try:
            deck = read_deck_csv(deck_path)
            self.bands = band_adjustments(
                deck_profile(deck, damage.get_card_registry(), damage.get_attack_registry())
            )
        except Exception:
            pass

    def decide(self, obs: Observation) -> list[int]:
        """Return a selection, never raising and never emitting an illegal move.

        The whole body is wrapped so that any failure — including one inside a
        handler or while parsing an unexpected observation — degrades to a legal
        random selection rather than crashing the match.
        """
        started = time.perf_counter()
        try:
            if obs.select is None:
                # The submission process can be reused across matches.  The
                # 600-second allowance is per match, so the deck-selection
                # observation is the unambiguous clock reset boundary.
                self._match_think_seconds = 0.0
            if self._match_think_seconds >= self.runtime_profile.competition_budget_seconds:
                return self._safe_fallback(obs)
            if obs.select is None:
                # Initial selection: the engine expects the 60-card deck.
                return read_deck_csv(self.deck_path)

            result = self._dispatch(obs.select, obs)
            if result is not None and is_valid_selection(result, obs.select):
                return result
            # No handler, handler deferred, or produced an invalid result.
            return legal_random_sample(obs.select, self.rng)
        except Exception:
            return self._safe_fallback(obs)
        finally:
            self._match_think_seconds += time.perf_counter() - started

    def _dispatch(
        self, select: SelectData, obs: Observation
    ) -> Optional[list[int]]:
        """Route ``select`` to a context handler, then a type handler.

        Returns the handler's result, or ``None`` if nothing handled it (the
        caller then falls back). ``.get`` on the dispatch tables makes an unknown
        or newly-appended enum value simply miss and defer, with no crash.
        """
        if select.context == SelectContext.MAIN:
            handler = self._main_handler
        else:
            handler = self.CONTEXT_HANDLERS.get(select.context)
        if handler is None:
            handler = self.TYPE_HANDLERS.get(select.type)
        if handler is None:
            return None
        return handler(self, select, obs)

    def _safe_fallback(self, obs: Observation) -> list[int]:
        """Last-resort legal selection used when :meth:`decide` hits an exception."""
        try:
            if obs.select is None:
                return read_deck_csv(self.deck_path)
            return legal_random_sample(obs.select, self.rng)
        except Exception:
            # Even the fallback failed (e.g. a malformed observation) — an empty
            # selection is the only thing guaranteed not to raise.
            return []
