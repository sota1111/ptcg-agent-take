"""Deck legality validator for the PTCG AI Battle eval environment.

Checks a 60-card deck (a list of card IDs, as stored in ``deck.csv`` / ``decks/*.csv``)
against the format's construction rules, using the engine's card database
(:func:`cg.api.all_card_data`) as the source of truth for card identity and type.

Rules enforced (see :func:`validate_deck`):

1. **Size** — exactly 60 cards.
2. **Known cards** — every card ID exists in ``all_card_data()``.
3. **Copy limit** — at most 4 cards sharing the same *name*, EXCEPT Basic Energy
   cards, which may appear any number of times.
4. **Basic Pokémon** — at least 1 Basic Pokémon (you cannot start a game without one).
5. **ACE SPEC** — at most 1 ACE SPEC card in the whole deck.

Usage::

    from eval.deck_validator import validate_deck, load_deck_csv, is_valid
    errors = validate_deck(load_deck_csv("decks/deck_balanced.csv"))
    assert errors == []           # legal deck -> no error strings

Run as a script to validate ``deck.csv`` plus every ``decks/*.csv``::

    python eval/deck_validator.py [deck.csv ...]
"""
from __future__ import annotations

import glob
import os
import sys
from collections import Counter
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from cg.api import CardData, CardType, all_card_data  # noqa: E402

DECK_SIZE = 60
MAX_COPIES = 4

# Card registry keyed by card ID. Built lazily and memoised because each
# ``all_card_data()`` call crosses the ctypes boundary and decodes JSON.
_REGISTRY: Optional[Dict[int, CardData]] = None


def build_registry() -> Dict[int, CardData]:
    """Return ``{cardId: CardData}`` for every card the engine knows about."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = {c.cardId: c for c in all_card_data()}
    return _REGISTRY


def load_deck_csv(path: str) -> List[int]:
    """Read a deck CSV (one card ID per line) and return its card IDs.

    Only the first :data:`DECK_SIZE` non-empty lines are taken, mirroring the
    loaders in ``eval/run_match.py`` / ``agents/base.py`` but tolerant of a
    trailing newline.
    """
    with open(path) as f:
        rows = [r.strip() for r in f.read().split("\n") if r.strip() != ""]
    return [int(r) for r in rows[:DECK_SIZE]]


def validate_deck(
    card_ids: List[int], registry: Optional[Dict[int, CardData]] = None
) -> List[str]:
    """Validate a deck, returning a list of human-readable error strings.

    An empty list means the deck is legal. ``registry`` may be injected (e.g. in
    tests) to avoid loading the engine; it defaults to :func:`build_registry`.
    """
    if registry is None:
        registry = build_registry()

    errors: List[str] = []

    # 1. Size.
    if len(card_ids) != DECK_SIZE:
        errors.append(f"deck must contain exactly {DECK_SIZE} cards, found {len(card_ids)}")

    # 2. Every card must be known to the engine.
    unknown = sorted({cid for cid in card_ids if cid not in registry})
    if unknown:
        errors.append(f"unknown card ID(s) not in card database: {unknown}")

    # 3. Copy limit by card name; Basic Energy is exempt (unlimited).
    name_counts: Counter = Counter()
    name_is_basic_energy: Dict[str, bool] = {}
    for cid in card_ids:
        card = registry.get(cid)
        if card is None:
            continue  # already reported as unknown
        name_counts[card.name] += 1
        if card.cardType == CardType.BASIC_ENERGY:
            name_is_basic_energy[card.name] = True
    for name, count in sorted(name_counts.items()):
        if not name_is_basic_energy.get(name) and count > MAX_COPIES:
            errors.append(f"'{name}' appears {count} times (max {MAX_COPIES})")

    # 4. At least one Basic Pokémon.
    has_basic_pokemon = any(
        registry[cid].cardType == CardType.POKEMON and registry[cid].basic
        for cid in card_ids
        if cid in registry
    )
    if not has_basic_pokemon:
        errors.append("deck must contain at least 1 Basic Pokémon")

    # 5. At most one ACE SPEC card.
    ace_spec = sorted(
        {registry[cid].name for cid in card_ids if cid in registry and registry[cid].aceSpec}
    )
    ace_spec_count = sum(
        1 for cid in card_ids if cid in registry and registry[cid].aceSpec
    )
    if ace_spec_count > 1:
        errors.append(
            f"deck has {ace_spec_count} ACE SPEC cards {ace_spec} (max 1)"
        )

    return errors


def is_valid(card_ids: List[int], registry: Optional[Dict[int, CardData]] = None) -> bool:
    """Return ``True`` iff ``card_ids`` is a legal deck."""
    return not validate_deck(card_ids, registry)


def _iter_default_deck_paths() -> List[str]:
    paths = []
    root_deck = os.path.join(REPO, "deck.csv")
    if os.path.exists(root_deck):
        paths.append(root_deck)
    paths.extend(sorted(glob.glob(os.path.join(REPO, "decks", "*.csv"))))
    return paths


def main(argv: List[str]) -> int:
    paths = argv[1:] or _iter_default_deck_paths()
    registry = build_registry()
    failures = 0
    for path in paths:
        try:
            deck = load_deck_csv(path)
        except (OSError, ValueError) as exc:
            print(f"FAIL {path}: cannot read deck ({exc})")
            failures += 1
            continue
        errors = validate_deck(deck, registry)
        rel = os.path.relpath(path, REPO)
        if errors:
            failures += 1
            print(f"FAIL {rel}:")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"OK   {rel}")
    print(f"\n{len(paths) - failures}/{len(paths)} deck(s) valid")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
