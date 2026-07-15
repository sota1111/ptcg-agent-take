"""Archetype-adaptive scoring parameters derived from the deck list (SOT-1694).

The scoring MAIN policy (SOT-1635, tuned in SOT-1682) was calibrated on a single
deck. Across the 25 tournament decks (SOT-1684) the optimal emphasis differs by
archetype: a Stage-2 line deck lives or dies by completing its evolutions, an
ex/mega deck concentrates energy on one big attacker, a single-prize deck wins by
board width and attacker redundancy. This module derives that emphasis
**mechanically from the deck's composition** — no card IDs, only the generic
:class:`~cg.api.CardData` attributes (stage flags, ex/megaEx, cardType,
energyType, attack list) — as two pure functions:

- :func:`deck_profile` — count the deck's structural features.
- :func:`band_adjustments` — turn a profile into bounded nudges of the MAIN
  scoring bands (:class:`BandAdjust`).

Adjustments are clamped to ``±MAX_NUDGE`` (150), strictly less than the 200-point
band spacing in ``agents/rule_based.py``, so an adjusted band can approach but
never leapfrog its neighbour: archetype adaptation re-weights *within* the
established priority order instead of rewriting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cg.api import CardData, CardType

# Hard bound on every band nudge. The scoring bands are spaced 200 apart, so a
# nudge < 200 preserves the band ordering by construction.
MAX_NUDGE = 150.0


@dataclass(frozen=True)
class DeckProfile:
    """Structural counts of a 60-card deck (unknown card ids count nowhere)."""

    pokemon: int              # Pokémon cards
    basics: int               # Basic Pokémon
    stage1: int               # Stage 1 Pokémon
    stage2: int               # Stage 2 Pokémon
    ex_or_mega: int           # Pokémon ex (incl. mega): 2-3 prize bodies
    mega: int                 # Mega ex: 3-prize bodies
    single_prize_attackers: int  # non-ex Pokémon with a damaging attack
    viable_attackers: int     # any Pokémon with a damaging attack
    energy: int               # Energy cards (Basic + Special)
    energy_types: int         # distinct Basic-Energy types
    supporters: int
    items: int                # Items + Tools + Stadiums (generic trainer engine)

    @property
    def evolutions(self) -> int:
        return self.stage1 + self.stage2

    @property
    def trainers(self) -> int:
        return self.supporters + self.items


@dataclass(frozen=True)
class BandAdjust:
    """Additive nudges applied to the MAIN scoring bands (all within ±MAX_NUDGE).

    ``supporter_hand_max`` is not a band but the hand-size threshold above which
    a draw Supporter is demoted to the S_ITEM rank. It is carried here as the
    adaptation hook, but stays at the v2-tuned 6 for every archetype — the
    evolution-deck variant (8) measured worse (see :func:`band_adjustments`).
    """

    evolve: float = 0.0        # S_EVOLVE
    play_basic: float = 0.0    # S_PLAY_BASIC
    attach_load: float = 0.0   # S_ATTACH_LOAD (Active still short of its attack)
    attach_bench: float = 0.0  # S_ATTACH_BENCH (loading a second attacker)
    supporter_hand_max: int = 6


def _has_damaging_attack(card: CardData, attack_registry: dict) -> bool:
    for aid in card.attacks or []:
        atk = attack_registry.get(aid)
        if atk is not None and atk.damage > 0:
            return True
    return False


def deck_profile(
    deck_ids: list[int],
    card_registry: dict[int, CardData],
    attack_registry: Optional[dict] = None,
) -> DeckProfile:
    """Count the deck's structural features. Pure; unknown ids are skipped."""
    attack_registry = attack_registry or {}
    pokemon = basics = stage1 = stage2 = 0
    ex_or_mega = mega = single_prize_attackers = viable_attackers = 0
    energy = supporters = items = 0
    basic_energy_types: set[int] = set()

    for cid in deck_ids:
        cd = card_registry.get(cid)
        if cd is None:
            continue
        ct = cd.cardType
        if ct == CardType.POKEMON:
            pokemon += 1
            if cd.basic:
                basics += 1
            if cd.stage1:
                stage1 += 1
            if cd.stage2:
                stage2 += 1
            # The API comment says ``ex`` includes megas, but in the shipped
            # registry mega cards carry ``ex=False`` — count either flag.
            multi_prize = cd.ex or cd.megaEx
            if multi_prize:
                ex_or_mega += 1
            if cd.megaEx:
                mega += 1
            damaging = _has_damaging_attack(cd, attack_registry)
            if damaging:
                viable_attackers += 1
                if not multi_prize:
                    single_prize_attackers += 1
        elif ct == CardType.BASIC_ENERGY:
            energy += 1
            basic_energy_types.add(int(cd.energyType))
        elif ct == CardType.SPECIAL_ENERGY:
            energy += 1
        elif ct == CardType.SUPPORTER:
            supporters += 1
        elif ct in (CardType.ITEM, CardType.TOOL, CardType.STADIUM):
            items += 1

    return DeckProfile(
        pokemon=pokemon,
        basics=basics,
        stage1=stage1,
        stage2=stage2,
        ex_or_mega=ex_or_mega,
        mega=mega,
        single_prize_attackers=single_prize_attackers,
        viable_attackers=viable_attackers,
        energy=energy,
        energy_types=len(basic_energy_types),
        supporters=supporters,
        items=items,
    )


def classify_archetype(profile: DeckProfile) -> str:
    """Bucket a profile into one of the four archetype axes (for reporting).

    Mechanical thresholds over composition only. Precedence: a deep Stage-2 line
    defines the deck's game plan even when its attacker is an ex (Dragapult), so
    ``stage2`` is checked first; a trainer-heavy, attacker-light list is
    ``control``; multi-prize bodies make it ``ex_mega``; a list with (almost) no
    multi-prize Pokémon is ``single_prize``; the rest are ``midrange``.
    """
    if profile.stage2 >= 2:
        return "stage2"
    if profile.pokemon <= 10 and profile.trainers >= 34:
        return "control"
    if profile.ex_or_mega >= 4:
        return "ex_mega"
    if profile.ex_or_mega <= 1:
        return "single_prize"
    return "midrange"


def band_adjustments(profile: DeckProfile) -> BandAdjust:
    """Derive the band nudges for a deck. Pure; every nudge is within ±MAX_NUDGE.

    * **Evolution emphasis** — the deeper the evolution package (Stage-2 counts
      double: the line must be completed twice over), the more an EVOLVE option
      is worth relative to its band peers, and the longer draw Supporters stay
      at full priority (digging for line pieces).
    * **Board width** — single-prize / basic-heavy decks win by attacker
      redundancy, so playing another Basic and loading a *second* Bench attacker
      score higher; a concentrated ex/mega deck instead keeps energy flowing to
      its Active (attach_load up, attach_bench down).
    """
    # Evolution package weight, 0..1: 8+ evolution slots (stage2 double) = full.
    evo_w = min(1.0, (profile.stage1 + 2 * profile.stage2) / 8.0)
    # Multi-prize concentration, 0..1: share of Pokémon that give 2-3 prizes.
    conc_w = (profile.ex_or_mega / profile.pokemon) if profile.pokemon else 0.0
    # Width weight, 0..1: single-prize attacker redundancy (4+ = full).
    width_w = min(1.0, profile.single_prize_attackers / 4.0)

    evolve = min(MAX_NUDGE, 100.0 * evo_w)
    play_basic = min(MAX_NUDGE, 60.0 * width_w + 30.0 * evo_w)
    attach_load = min(MAX_NUDGE, 80.0 * conc_w)
    attach_bench = max(-MAX_NUDGE, min(MAX_NUDGE, 80.0 * width_w - 60.0 * conc_w))

    # Measured (SOT-1694, 25-deck mirror N=1000 vs v2): raising the threshold to
    # 8 for evolution decks scored 0.497 vs 0.519 for the v2-tuned 6 — the
    # adaptive threshold was tested and rejected, so it stays at 6 for every
    # archetype (the field remains the hook for future evidence).
    return BandAdjust(
        evolve=evolve,
        play_basic=play_basic,
        attach_load=attach_load,
        attach_bench=attach_bench,
        supporter_hand_max=6,
    )
