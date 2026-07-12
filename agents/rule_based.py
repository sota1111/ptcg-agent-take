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


class RuleBasedAgent(Agent):
    """Dispatches each selection to a per-context rule, guarded by a safe fallback.

    Args:
        seed: Seed for the fallback RNG (reproducible random tie-breaks / fills).
        deck_path: Path to the deck CSV used for the initial selection.
    """

    # SelectContext -> handler. R2 (SOT-1633) registers the MAIN-turn policy;
    # R3 (SOT-1634) covers the setup / forced-selection contexts below so they no
    # longer rely on the random fallback. Any context still absent defers safely.
    CONTEXT_HANDLERS: dict[SelectContext, Handler] = {
        SelectContext.MAIN: main_context_handler,
        # Setup: place the best Basic (evolution line, then HP).
        SelectContext.SETUP_ACTIVE_POKEMON: place_basic_handler,
        SelectContext.SETUP_BENCH_POKEMON: place_basic_handler,
        # After a KO: promote the readiest Bench Pokémon (Energy, then HP).
        SelectContext.TO_ACTIVE: to_active_handler,
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

    def __init__(self, seed: Optional[int] = None, deck_path: str = "deck.csv") -> None:
        self.rng = random.Random(seed)
        self.deck_path = deck_path

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
