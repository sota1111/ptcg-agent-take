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

import random
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
from agents.base import Agent, is_valid_selection, legal_random_sample, read_deck_csv

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
        hp = cd.hp if cd is not None else 0
        scored.append((i, (is_poke, has_line, hp)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


def to_active_handler(
    agent: "RuleBasedAgent", select: SelectData, obs: Observation
) -> Optional[list[int]]:
    """TO_ACTIVE (after a Knock Out): promote the readiest Bench Pokémon.

    Prefer the Pokémon carrying the most Energy (closest to attacking), then the
    highest current HP so the new Active survives longer.
    """
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        p = _referenced_pokemon(o, obs)
        n_energy = len(p.energies) if (p is not None and p.energies) else 0
        hp = p.hp if p is not None else 0
        scored.append((i, (n_energy, hp)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


# YesNo contexts: a fixed, reasonable default. Being *prompted* to mulligan almost
# always means the opening hand has no playable Basic, so redrawing is the safe
# default; going first keeps board/tempo initiative; a coin call is 50/50 so we
# pick heads deterministically.
_YESNO_DEFAULT: dict[SelectContext, OptionType] = {
    SelectContext.IS_FIRST: OptionType.YES,
    SelectContext.COIN_HEAD: OptionType.YES,
    SelectContext.MULLIGAN: OptionType.YES,
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

    Concentrate damage on the target with the lowest current HP — the Pokémon
    closest to being Knocked Out — so placements complete a KO rather than being
    spread thin.
    """
    scored: list[tuple[int, tuple]] = []
    for i, o in enumerate(select.option):
        p = _referenced_pokemon(o, obs)
        if p is None or p.hp is None:
            hp_key = -(10 ** 9)  # unknown target: least preferred
        else:
            hp_key = -p.hp        # lower HP -> larger key -> picked first
        scored.append((i, (hp_key,)))
    return _top_k(scored, _pick_count(select, want_at_least_one=True))


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
S_ABILITY = -0.4         # skip optional abilities (prefer END; avoids no-op loops)
S_RETREAT_NO = -0.5      # an *unjustified* retreat: never chosen over ending the turn
S_DISCARD = -0.6         # never voluntarily discard a card from play


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


def _incoming_damage(opp_active, opp_card, my_active_card) -> int:
    """Largest damage the opponent's Active could deal to our Active next turn."""
    if opp_active is None or opp_card is None or not opp_card.attacks:
        return 0
    reg = damage.get_attack_registry()
    worst = 0
    for aid in opp_card.attacks:
        atk = reg.get(aid)
        if atk is None:
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
    attacker_cd, defender_cd, defender_hp,
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
            return S_PLAY_BASIC + 10.0 * line + (cd.hp or 0) / 100.0
        if cd.cardType == CardType.SUPPORTER:
            return S_ITEM if state.supporterPlayed else S_SUPPORTER
        return S_ITEM
    if t == OptionType.EVOLVE:
        return S_EVOLVE
    if t == OptionType.ATTACH:
        target = _inplay_pokemon(o, obs)
        card = cards.get(target.id) if target is not None else None
        if target is None:
            return S_ATTACH_OVER
        if _energy_deficit(target, card) <= 0:
            return S_ATTACH_OVER  # already able to fire its best attack
        if o.inPlayArea == AreaType.ACTIVE:
            return S_ATTACH_LOAD
        return S_ATTACH_BENCH if _is_viable_attacker(card) else S_ATTACH_OVER
    if t == OptionType.RETREAT:
        return S_RETREAT_OK if _should_retreat(me, opp, cards) else S_RETREAT_NO
    if t == OptionType.ABILITY:
        return S_ABILITY
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

    best_i: Optional[int] = None
    best_s: Optional[float] = None
    for i, o in enumerate(select.option):
        s = _score_main_option(
            o, obs, cards=cards, me=me, opp=opp, state=state, hand=hand,
            bench_open=bench_open, attacker_cd=attacker_cd,
            defender_cd=defender_cd, defender_hp=defender_hp,
        )
        if s is None:
            continue
        if best_s is None or s > best_s:  # strict '>' keeps the first (lowest) index on ties
            best_s = s
            best_i = i
    if best_i is None:
        return None
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
        # Setup: place the best Basic (evolution line, then HP).
        SelectContext.SETUP_ACTIVE_POKEMON: place_basic_handler,
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
    ) -> None:
        self.rng = random.Random(seed)
        self.deck_path = deck_path
        if policy not in self.MAIN_POLICIES:
            raise ValueError(
                f"unknown policy {policy!r}; expected one of {sorted(self.MAIN_POLICIES)}"
            )
        self.policy = policy
        # Per-instance MAIN handler so the policy can vary without mutating the
        # shared class table; every other context still reads ``CONTEXT_HANDLERS``
        # live, so later registrations there keep taking effect.
        self._main_handler: Handler = self.MAIN_POLICIES[policy]

    def decide(self, obs: Observation) -> list[int]:
        """Return a selection, never raising and never emitting an illegal move.

        The whole body is wrapped so that any failure — including one inside a
        handler or while parsing an unexpected observation — degrades to a legal
        random selection rather than crashing the match.
        """
        try:
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
