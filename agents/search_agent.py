"""Search-based agent skeleton (SOT-1657, B-line R1).

This is the scaffold for the *search* line (案B): instead of a per-context rule
table, it uses the engine's official one-ply lookahead API
(``search_begin`` / ``search_step`` / ``search_end``, see ``cg/api.py``) to try
each candidate move, evaluate the resulting board with a simple perspective
score, and pick the best. It is deliberately independent of ``rule_based`` — a
separate module the later B-line work (damage-based evaluation, deeper search)
builds on.

Two properties matter most and are guaranteed here:

- **One-ply lookahead.** For a single-select decision every legal option is a
  candidate; each is played on a *search copy* of the current position, the
  resulting observation is scored (:func:`damage_based_evaluate`, a damage-based
  perspective value reusing the R4 scoring and weakness/resistance calc), and
  the highest-scoring option is chosen.
- **Outermost random fallback (the whole point of R1).** Any failure whatsoever
  — an observation that can't be reconstructed, a ``search_begin`` that rejects
  our hidden-info prediction, an unknown / newly-appended enum, a raising
  evaluator, or the time budget expiring before a single candidate is scored —
  degrades to a legal random selection. The agent never raises and never emits
  an illegal move, so it can never trigger a Validation Episode Error.

The hidden information ``search_begin`` requires (both decks, prizes, the
opponent's hand / face-down Active) is unknown in live play; a simple
best-effort predictor (:class:`UniformDeckPredictor`) samples it from our own
deck list. When a prediction is wrong the engine rejects the search and that
candidate is skipped — the fallback still holds.
"""

from __future__ import annotations

import collections
import random
import time
from typing import Callable, Optional, Sequence

from cg.api import Observation, PlayerState

from agents import damage
from agents.base import Agent, is_valid_selection, legal_random_sample, read_deck_csv
from agents.rule_based import evaluate_position

# An evaluator scores a resulting :class:`~cg.api.Observation` from the acting
# player's perspective (higher is better for ``your_index``). Kept pluggable: the
# default is :func:`damage_based_evaluate` (SOT-1658, damage-based); any callable
# with this signature can replace it without touching the search/fallback machinery.
Evaluator = Callable[[Observation, int], float]

# default_evaluate weights: winning dominates everything, then prize progress
# (fewer of our own prize cards remaining = we have taken more prizes = winning),
# then board HP as a fine tie-break.
_WIN_SCORE = 1_000_000.0
_PRIZE_WEIGHT = 1_000.0
_HP_WEIGHT = 1.0


# --------------------------------------------------------------------------- #
# Provisional evaluation function (perspective score).
# --------------------------------------------------------------------------- #

def _board_hp(ps: PlayerState) -> int:
    """Total current HP of a player's in-play Pokémon (Active + Bench)."""
    total = 0
    for pk in (ps.active or []):
        if pk is not None:
            total += pk.hp or 0
    for pk in (ps.bench or []):
        if pk is not None:
            total += pk.hp or 0
    return total


def default_evaluate(observation: Observation, your_index: int) -> float:
    """A simple perspective score for a resulting position (higher = better).

    Provisional per SOT-1657: a terminal win/loss is decisive, otherwise the
    prize-card differential (how much closer we are to taking all our prizes than
    the opponent) with board HP as a tie-break. The damage-based evaluation
    function is a later child Issue that swaps this out via the ``evaluate`` hook.
    """
    current = observation.current
    if current is None:
        return 0.0
    result = current.result
    if result in (0, 1):
        return _WIN_SCORE if result == your_index else -_WIN_SCORE
    if result == 2:  # draw
        return 0.0

    players = current.players
    me = players[your_index]
    opp = players[1 - your_index]
    # prize remaining: lower for us / higher for the opponent both mean we lead.
    prize_term = (len(opp.prize) - len(me.prize)) * _PRIZE_WEIGHT
    hp_term = (_board_hp(me) - _board_hp(opp)) * _HP_WEIGHT
    return prize_term + hp_term


# --------------------------------------------------------------------------- #
# Damage-based evaluation function (SOT-1658, B-line R2).
#
# The real one-ply evaluator that replaces the provisional ``default_evaluate``.
# It reuses two existing pieces rather than reinventing them:
#
#   * the **R4 positional value** ``rule_based.evaluate_position`` (SOT-1635):
#     prizes dominate, then board HP, then Energy developed — a symmetric
#     (me − opponent) score already tuned for this engine; and
#   * the **damage calculation** ``agents.damage`` (SOT-1633): weakness ×2 /
#     resistance −30, plus a KO test scaled by the prizes the KO would take.
#
# On top of the positional base it adds an *offensive* term — the net best-attack
# damage our Active could deal to the opponent's Active minus theirs to ours —
# so the search prefers moves that set up (or land) a hard-hitting / lethal swing,
# not just moves that passively hold HP. Every lookup is guarded: an unknown card
# id, a missing card table, a newly-appended enum, or a malformed observation all
# degrade to the neutral (0) contribution instead of raising, so the outermost
# random fallback in :class:`SearchAgent` is never needed just to evaluate.
# --------------------------------------------------------------------------- #

# Offensive-term weights. A point of net expected damage is worth a little more
# than a point of static HP (an unrealised threat), and a reachable KO is worth a
# flat bonus per prize it would take (a KO both removes an attacker and advances
# the prize race that dominates :func:`~agents.rule_based.evaluate_position`).
_DMG_WEIGHT = 2.0
_KO_BONUS = 500.0


def _offense(attacker, defender, cards, attacks) -> tuple[int, bool, int]:
    """Best single-attack damage ``attacker``'s Active could deal to ``defender``.

    Returns ``(best_damage, can_ko, prizes)`` where ``prizes`` is how many prize
    cards Knocking the defender Out would award (1, or 2/3 for ex / Mega ex).
    Any missing piece — no Active on either side, an unknown card id, a Pokémon
    with no attacks — yields ``(0, False, 1)`` (a neutral, non-threatening
    contribution) rather than raising.
    """
    if attacker is None or defender is None:
        return 0, False, 1
    attacker_cd = cards.get(attacker.id)
    defender_cd = cards.get(defender.id)
    if attacker_cd is None or not attacker_cd.attacks:
        return 0, False, 1
    best_dmg = 0
    can_ko = False
    for aid in attacker_cd.attacks:
        atk = attacks.get(aid)
        if atk is None:
            continue
        dmg = damage.attack_damage(atk, attacker_cd, defender_cd)
        if dmg > best_dmg:
            best_dmg = dmg
        if damage.is_ko(atk, attacker_cd, defender_cd, defender.hp):
            can_ko = True
    prizes = 1
    if defender_cd is not None:
        if defender_cd.megaEx:
            prizes = 3
        elif defender_cd.ex:
            prizes = 2
    return best_dmg, can_ko, prizes


def damage_based_evaluate(observation: Observation, your_index: int) -> float:
    """Damage-based perspective score for a resulting position (higher = better).

    A terminal win/loss is decisive. Otherwise the score is the R4 positional
    value (:func:`~agents.rule_based.evaluate_position`: prizes, then board HP,
    then Energy) plus an offensive damage differential computed with weakness
    (×2) / resistance (−30): the best hit our Active threatens on the opponent's
    Active minus the best hit theirs threatens on ours, with a bonus for a
    reachable KO scaled by the prizes it would take. Guarded throughout — unknown
    enums / missing fields fall back to the neutral contribution and never raise.
    """
    current = observation.current
    if current is None:
        return 0.0
    result = current.result
    if result in (0, 1):
        return _WIN_SCORE if result == your_index else -_WIN_SCORE
    if result == 2:  # draw
        return 0.0

    try:
        players = current.players
        me = players[your_index]
        opp = players[1 - your_index]
    except (IndexError, TypeError, AttributeError):
        return 0.0

    # Positional base: reuse the R4 scoring (prizes ≫ HP ≫ Energy), symmetric.
    try:
        base = evaluate_position(me, opp)
    except Exception:
        base = 0.0

    # Offensive differential: net best-attack damage + KO potential, both sides.
    # With no Active on either side there is nothing to attack, so skip the
    # (engine-backed) card/attack table lookups entirely and stay positional.
    offense = 0.0
    try:
        my_active = me.active[0] if me.active else None
        opp_active = opp.active[0] if opp.active else None
        if my_active is not None or opp_active is not None:
            cards = damage.get_card_registry()
            attacks = damage.get_attack_registry()
            my_dmg, my_ko, my_prizes = _offense(my_active, opp_active, cards, attacks)
            opp_dmg, opp_ko, opp_prizes = _offense(opp_active, my_active, cards, attacks)
            offense = _DMG_WEIGHT * (my_dmg - opp_dmg)
            if my_ko:
                offense += _KO_BONUS * my_prizes
            if opp_ko:
                offense -= _KO_BONUS * opp_prizes
    except Exception:
        offense = 0.0

    return base + offense


# --------------------------------------------------------------------------- #
# Hidden-information prediction (best-effort, pluggable).
#
# ``search_begin`` needs a full prediction of every hidden zone. In live play we
# know only our own deck list, so we sample each hidden zone from that list minus
# the cards already visible on the board — and, lacking any opponent-deck
# knowledge in R1, assume the opponent's deck has the same composition as ours.
# Wrong guesses just make ``search_begin`` reject the candidate; the guard holds.
# --------------------------------------------------------------------------- #

def _visible_card_ids(ps: PlayerState, *, include_hand: bool) -> list[int]:
    """Card IDs currently visible on the board for one player.

    Pokémon in play (Active/Bench) with their attached energy/tool/pre-evolution
    cards, the discard pile, revealed prize cards, and — only where the hand is
    visible (our own side) — the hand. These are removed from the deck multiset
    to form the hidden pool.
    """
    ids: list[int] = []

    def add_pokemon(pk) -> None:
        if pk is None:
            return
        ids.append(pk.id)
        for group in (pk.energyCards, pk.tools, pk.preEvolution):
            for card in group or []:
                ids.append(card.id)

    for pk in ps.active or []:
        add_pokemon(pk)
    for pk in ps.bench or []:
        add_pokemon(pk)
    for card in ps.discard or []:
        ids.append(card.id)
    for card in ps.prize or []:
        if card is not None:  # revealed prize
            ids.append(card.id)
    if include_hand and ps.hand is not None:
        for card in ps.hand:
            ids.append(card.id)
    return ids


class UniformDeckPredictor:
    """Sample hidden zones uniformly from a deck list minus visible cards.

    ``deck_ids`` is the assumed 60-card composition, reused for both players in
    R1 (we have no opponent-deck knowledge yet). Seeded for reproducibility.
    Swap this out for a deck-tracking predictor later without touching the agent.
    """

    def __init__(self, deck_ids: Sequence[int], rng: random.Random) -> None:
        self.deck_ids = list(deck_ids)
        self.rng = rng

    def _pool(self, ps: PlayerState, *, include_hand: bool) -> list[int]:
        remaining = collections.Counter(self.deck_ids)
        for cid in _visible_card_ids(ps, include_hand=include_hand):
            remaining[cid] -= 1
        pool: list[int] = []
        for cid, n in remaining.items():
            if n > 0:
                pool.extend([cid] * n)
        self.rng.shuffle(pool)
        return pool

    def _take(self, pool: list[int], n: int) -> list[int]:
        """Pop ``n`` cards off ``pool``; top up from the deck list if short."""
        if n <= 0:
            return []
        out = pool[:n]
        del pool[:n]
        i = 0
        while len(out) < n and self.deck_ids:
            out.append(self.deck_ids[i % len(self.deck_ids)])
            i += 1
        return out

    def predict(self, obs: Observation, your_index: int) -> tuple:
        """Return the six hidden-card lists in ``search_begin`` argument order."""
        state = obs.current
        players = state.players
        me = players[your_index]
        opp = players[1 - your_index]

        # Our side: the hand is visible, so subtract it; deck + prize are hidden.
        my_pool = self._pool(me, include_hand=True)
        your_deck = self._take(my_pool, me.deckCount)
        your_prize = self._take(my_pool, len(me.prize))

        # Opponent side: the hand is hidden too, so don't subtract it.
        opp_pool = self._pool(opp, include_hand=False)
        opponent_deck = self._take(opp_pool, opp.deckCount)
        opponent_prize = self._take(opp_pool, len(opp.prize))
        opponent_hand = self._take(opp_pool, opp.handCount)
        active = opp.active or []
        active_facedown = len(active) > 0 and active[0] is None
        opponent_active = self._take(opp_pool, 1) if active_facedown else []

        return (
            your_deck,
            your_prize,
            opponent_deck,
            opponent_prize,
            opponent_hand,
            opponent_active,
        )


# --------------------------------------------------------------------------- #
# The search agent.
# --------------------------------------------------------------------------- #

class SearchAgent(Agent):
    """One-ply search agent with an outermost random-fallback guard.

    Args:
        seed: Seed for the fallback RNG and hidden-info sampling (reproducible).
        deck_path: Deck CSV used for the initial selection and as the hidden-info
            composition prior for both players (R1 has no opponent-deck model).
        time_budget_s: Wall-clock budget per decision. Candidates are searched in
            order until the budget is spent; the best scored so far is returned
            (or a legal random move if none was scored in time).
        max_candidates: Cap on how many candidate options are searched per
            decision (bounds cost on wide MAIN selections).
        evaluate: Position evaluator (defaults to :func:`damage_based_evaluate`,
            the SOT-1658 damage-based function; pass a callable to override).
        manual_coin: Passed to ``search_begin`` (fix coin flips during lookahead).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        deck_path: str = "deck.csv",
        time_budget_s: float = 0.5,
        max_candidates: int = 12,
        evaluate: Optional[Evaluator] = None,
        manual_coin: bool = False,
    ) -> None:
        self.rng = random.Random(seed)
        self.deck_path = deck_path
        self.time_budget_s = time_budget_s
        self.max_candidates = max_candidates
        self.evaluate = evaluate or damage_based_evaluate
        self.manual_coin = manual_coin
        self._deck_ids: Optional[list[int]] = None

    # -- public API -------------------------------------------------------- #

    def decide(self, obs: Observation) -> list[int]:
        """Return a selection, never raising and never emitting an illegal move.

        The whole body is guarded: any failure — a bad observation, a rejected
        search, an unknown enum, a raising evaluator, or the budget expiring —
        degrades to a legal random selection instead of crashing the match.
        """
        try:
            if obs.select is None:
                # Initial selection: the engine expects the 60-card deck.
                return self._deck()

            best = self._search_decide(obs)
            if best is not None and is_valid_selection(best, obs.select):
                return best
            # No candidate could be scored (all rejected / budget spent) — fall back.
            return legal_random_sample(obs.select, self.rng)
        except Exception:
            return self._safe_fallback(obs)

    # -- search core ------------------------------------------------------- #

    def _search_decide(self, obs: Observation) -> Optional[list[int]]:
        """One-ply search over candidate moves; returns the best or ``None``.

        Predicts hidden info once (so every candidate is compared on the same
        sampled world), then plays each candidate on a fresh search copy and
        scores the resulting observation. Returns ``None`` when no candidate can
        be scored, so the caller falls back to a legal random move.
        """
        candidates = self._candidates(obs)
        if not candidates:
            return None

        your_index = obs.current.yourIndex if obs.current is not None else 0
        predictor = UniformDeckPredictor(self._deck(), self.rng)
        hidden = predictor.predict(obs, your_index)

        deadline = time.perf_counter() + max(0.0, self.time_budget_s)
        best_move: Optional[list[int]] = None
        best_score = float("-inf")
        scored_any = False
        for move in candidates:
            if time.perf_counter() >= deadline and scored_any:
                break  # budget spent and we already have something to return
            score = self._score_candidate(obs, hidden, move, your_index)
            if score is None:
                continue
            scored_any = True
            if score > best_score:
                best_score, best_move = score, move
        return best_move

    def _candidates(self, obs: Observation) -> list[list[int]]:
        """Enumerate candidate first-moves for a single-select decision.

        Each legal option index is one candidate ``[i]`` (plus the empty
        selection when ``minCount == 0``). Multi-select decisions are left to the
        random fallback in R1 — the search skeleton only reads single selects.
        Capped at ``max_candidates`` to bound per-decision cost.
        """
        select = obs.select
        options = select.option or []
        if not (select.minCount <= 1 <= select.maxCount):
            return []
        cands: list[list[int]] = []
        if select.minCount == 0:
            cands.append([])
        cands.extend([i] for i in range(len(options)))
        return cands[: self.max_candidates]

    def _score_candidate(
        self,
        obs: Observation,
        hidden: tuple,
        move: list[int],
        your_index: int,
    ) -> Optional[float]:
        """Play ``move`` on a fresh search copy and score the resulting position.

        The search session is ALWAYS torn down (``search_release`` + ``search_end``
        via try/finally) even if reconstruction, the step, or the evaluator
        raises. Returns ``None`` on any failure so the candidate is skipped.
        """
        # Imported lazily so the module (and its tests) load without the engine.
        from cg.api import search_begin, search_end, search_release, search_step

        if not is_valid_selection(move, obs.select):
            return None

        search_id: Optional[int] = None
        started = False
        try:
            root = search_begin(obs, *hidden, manual_coin=self.manual_coin)
            started = True
            search_id = root.searchId
            state = search_step(search_id, list(move))
            return float(self.evaluate(state.observation, your_index))
        except Exception:
            return None
        finally:
            if started and search_id is not None:
                try:
                    search_release(search_id)
                except Exception:
                    pass
            if started:
                try:
                    search_end()
                except Exception:
                    pass

    # -- helpers ----------------------------------------------------------- #

    def _deck(self) -> list[int]:
        """Load and cache the deck list (used for the deck prior and initial pick)."""
        if self._deck_ids is None:
            self._deck_ids = read_deck_csv(self.deck_path)
        return list(self._deck_ids)

    def _safe_fallback(self, obs: Observation) -> list[int]:
        """Last-resort legal selection used when :meth:`decide` hits an exception."""
        try:
            if obs.select is None:
                return self._deck()
            return legal_random_sample(obs.select, self.rng)
        except Exception:
            # Even the fallback failed (malformed observation) — an empty
            # selection is the only thing guaranteed not to raise.
            return []
