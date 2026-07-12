"""Damage calculation and KO judgement for the rule-based agent (SOT-1633).

The MAIN-turn policy needs to answer two questions about a candidate attack:

- *how much damage does it deal* to the defending Pokémon, and
- *does that damage Knock Out* the defender.

This module answers both from the engine's static card/attack tables
(:func:`cg.api.all_card_data` / :func:`cg.api.all_attack`) plus the live
:class:`~cg.api.Pokemon` seen in the observation. The rule is intentionally
simple — enough to beat a uniform-random opponent, not a full damage engine:

    damage = attack.damage
             × 2   if the defender's weakness matches the attacker's type
             − 30  if the defender's resistance matches the attacker's type
             (never below 0)

Energy sufficiency is **not** checked here: the engine only offers an ``ATTACK``
option when its cost is already paid, so a presented attack is always usable.

Card-id specific branches are deliberately avoided: everything is derived from
the generic type / damage / HP fields, so the calculation covers new cards
without edits.
"""

from __future__ import annotations

from typing import Optional

from cg.api import Attack, CardData, all_attack, all_card_data

# Weakness doubles damage; resistance subtracts a flat 30 (this engine's rule).
WEAKNESS_MULTIPLIER = 2
RESISTANCE_REDUCTION = 30

# Lazily-built, engine-backed lookup tables (id -> static data). Cached because
# ``all_card_data()`` / ``all_attack()`` cross the ctypes boundary each call.
_CARD_REGISTRY: Optional[dict[int, CardData]] = None
_ATTACK_REGISTRY: Optional[dict[int, Attack]] = None


def get_card_registry() -> dict[int, CardData]:
    """Return ``cardId -> CardData`` for every card, built once and cached."""
    global _CARD_REGISTRY
    if _CARD_REGISTRY is None:
        _CARD_REGISTRY = {c.cardId: c for c in all_card_data()}
    return _CARD_REGISTRY


def get_attack_registry() -> dict[int, Attack]:
    """Return ``attackId -> Attack`` for every attack, built once and cached."""
    global _ATTACK_REGISTRY
    if _ATTACK_REGISTRY is None:
        _ATTACK_REGISTRY = {a.attackId: a for a in all_attack()}
    return _ATTACK_REGISTRY


def attack_damage(
    attack: Attack,
    attacker: Optional[CardData],
    defender: Optional[CardData],
) -> int:
    """Damage ``attack`` deals from ``attacker`` to ``defender``.

    Applies weakness (×2) then resistance (−30) using the attacker's
    :attr:`~cg.api.CardData.energyType` against the defender's weakness /
    resistance type. A base damage of ``0`` (setup / effect-only attacks) is
    returned unscaled. Missing attacker/defender card data disables scaling —
    the base damage is returned as-is. The result is clamped to ``>= 0``.
    """
    base = attack.damage
    if base <= 0:
        return max(0, base)
    dmg = base
    if attacker is not None and defender is not None:
        etype = attacker.energyType
        if defender.weakness is not None and defender.weakness == etype:
            dmg *= WEAKNESS_MULTIPLIER
        if defender.resistance is not None and defender.resistance == etype:
            dmg -= RESISTANCE_REDUCTION
    return max(0, dmg)


def is_ko(
    attack: Attack,
    attacker: Optional[CardData],
    defender: Optional[CardData],
    defender_current_hp: Optional[int],
) -> bool:
    """True iff ``attack`` deals enough to Knock Out the defender.

    A Knock Out needs the computed damage to be positive and to meet or exceed
    the defender's *current* HP. An unknown current HP (``None``) is treated as
    not-a-KO (we cannot prove the Knock Out), which keeps the policy conservative.
    """
    if defender_current_hp is None or defender_current_hp <= 0:
        return False
    dmg = attack_damage(attack, attacker, defender)
    return dmg > 0 and dmg >= defender_current_hp
