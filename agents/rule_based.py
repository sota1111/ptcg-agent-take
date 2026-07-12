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


class RuleBasedAgent(Agent):
    """Dispatches each selection to a per-context rule, guarded by a safe fallback.

    Args:
        seed: Seed for the fallback RNG (reproducible random tie-breaks / fills).
        deck_path: Path to the deck CSV used for the initial selection.
    """

    # SelectContext -> handler. R2 (SOT-1633) registers the MAIN-turn policy;
    # other contexts still defer to the safe fallback until later rules land.
    CONTEXT_HANDLERS: dict[SelectContext, Handler] = {
        SelectContext.MAIN: main_context_handler,
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
