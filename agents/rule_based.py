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

from cg.api import Observation, SelectContext, SelectData, SelectType

from agents.base import Agent, is_valid_selection, legal_random_sample, read_deck_csv

# A context/type handler: given the agent, the current selection, and the full
# observation, return the chosen option indices, or ``None`` to defer to the
# fallback. Handlers must be pure w.r.t. engine state (read-only on ``obs``).
Handler = Callable[["RuleBasedAgent", SelectData, Observation], Optional[list[int]]]


class RuleBasedAgent(Agent):
    """Dispatches each selection to a per-context rule, guarded by a safe fallback.

    Args:
        seed: Seed for the fallback RNG (reproducible random tie-breaks / fills).
        deck_path: Path to the deck CSV used for the initial selection.
    """

    # SelectContext -> handler. Empty in R1: every context currently defers to
    # the fallback. R2+ registers concrete pure functions here.
    CONTEXT_HANDLERS: dict[SelectContext, Handler] = {}

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
