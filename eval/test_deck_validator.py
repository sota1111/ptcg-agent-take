"""Tests for the deck legality validator (SOT-1660).

No pytest dependency — run directly:
    venv/bin/python eval/test_deck_validator.py
(also collectable by pytest: every check is a module-level ``test_*`` function.)

Covers:
  1. the sample ``deck.csv`` and every candidate deck in ``decks/`` pass the validator;
  2. each rule rejects an intentionally illegal deck (size / unknown ID / copy limit /
     no Basic Pokémon / >1 ACE SPEC);
  3. Basic Energy is exempt from the 4-copy limit;
  4. smoke: each candidate deck completes a full BattleStart→finish match.
"""
from __future__ import annotations

import glob
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)  # so libcg.so & deck CSVs resolve

from eval.deck_validator import (  # noqa: E402
    DECK_SIZE,
    build_registry,
    is_valid,
    load_deck_csv,
    validate_deck,
)

_REGISTRY = build_registry()

# Known-good card IDs (present in the engine's card database).
WATER_ENERGY = 3        # Basic {W} Energy  (BASIC_ENERGY, unlimited copies)
KYOGRE = 721            # Basic Pokémon
SNOVER = 722            # Basic Pokémon
ULTRA_BALL = 1121       # Item
MAXIMUM_BELT = 1158     # ACE SPEC
MASTER_BALL = 1125      # ACE SPEC
UNKNOWN_ID = 99_999_999  # not in the card database


def _candidate_decks():
    return sorted(glob.glob(os.path.join(REPO, "decks", "candidates", "*.csv")))


def test_sample_and_candidate_decks_are_valid():
    """deck.csv and every decks/*.csv are legal (>=2 candidates present)."""
    candidates = _candidate_decks()
    assert len(candidates) >= 2, f"expected >=2 candidate decks, found {len(candidates)}"

    paths = [os.path.join(REPO, "deck.csv")] + candidates
    for path in paths:
        deck = load_deck_csv(path)
        assert len(deck) == DECK_SIZE, f"{path}: {len(deck)} cards"
        errors = validate_deck(deck, _REGISTRY)
        assert errors == [], f"{path} should be valid but got: {errors}"
    print("PASS test_sample_and_candidate_decks_are_valid")


def test_reject_wrong_size():
    deck = load_deck_csv(os.path.join(REPO, "deck.csv"))[:59]
    errors = validate_deck(deck, _REGISTRY)
    assert any("exactly 60" in e for e in errors), errors
    assert not is_valid(deck, _REGISTRY)
    print("PASS test_reject_wrong_size")


def test_reject_unknown_card():
    deck = [UNKNOWN_ID] + [WATER_ENERGY] * 58 + [SNOVER]
    errors = validate_deck(deck, _REGISTRY)
    assert any("unknown card ID" in e for e in errors), errors
    print("PASS test_reject_unknown_card")


def test_reject_over_copy_limit():
    # 5 copies of a non-energy card (Snover) is illegal; rest padded with energy.
    deck = [SNOVER] * 5 + [WATER_ENERGY] * 55
    assert len(deck) == DECK_SIZE
    errors = validate_deck(deck, _REGISTRY)
    assert any("appears 5 times" in e for e in errors), errors
    print("PASS test_reject_over_copy_limit")


def test_basic_energy_exempt_from_copy_limit():
    # 56 basic energy is fine (unlimited); the 4 Snover keep a Basic Pokémon present.
    deck = [SNOVER] * 4 + [WATER_ENERGY] * 56
    assert len(deck) == DECK_SIZE
    errors = validate_deck(deck, _REGISTRY)
    assert errors == [], errors
    assert is_valid(deck, _REGISTRY)
    print("PASS test_basic_energy_exempt_from_copy_limit")


def test_reject_no_basic_pokemon():
    # 60 basic energy: legal counts, but no Pokémon at all -> cannot start.
    deck = [WATER_ENERGY] * 60
    errors = validate_deck(deck, _REGISTRY)
    assert any("Basic Pokémon" in e for e in errors), errors
    print("PASS test_reject_no_basic_pokemon")


def test_reject_multiple_ace_spec():
    # Two different ACE SPEC cards -> illegal (max 1).
    deck = [MAXIMUM_BELT, MASTER_BALL] + [SNOVER] * 4 + [WATER_ENERGY] * 54
    assert len(deck) == DECK_SIZE
    errors = validate_deck(deck, _REGISTRY)
    assert any("ACE SPEC" in e for e in errors), errors
    print("PASS test_reject_multiple_ace_spec")


def _random_agent(obs, rng):
    from cg.api import to_observation_class

    o = to_observation_class(obs)
    if o.select is None:
        return None
    n = len(o.select.option)
    if n == 0:
        return []
    k = max(o.select.minCount, min(o.select.maxCount, n))
    return rng.sample(range(n), k)


def _play_match(deck0, deck1, max_steps=100000):
    from cg import game

    obs, start = game.battle_start(deck0, deck1)
    assert obs is not None, (
        f"BattleStart failed: errorPlayer={start.errorPlayer} errorType={start.errorType}"
    )
    rng = random.Random(1234)
    steps = 0
    try:
        while steps < max_steps:
            cur = obs.get("current")
            if cur and cur.get("result", -1) != -1:
                return cur["result"], steps
            obs = game.battle_select(_random_agent(obs, rng))
            steps += 1
        return -1, steps
    finally:
        game.battle_finish()


def test_candidate_decks_complete_a_match():
    """Each candidate deck must start and finish a full mirror match."""
    for path in _candidate_decks():
        deck = load_deck_csv(path)
        result, steps = _play_match(deck, deck)
        assert result != -1, f"{path} did not complete within step budget ({steps} steps)"
        assert result in (0, 1), f"{path}: unexpected result {result}"
    print("PASS test_candidate_decks_complete_a_match")


if __name__ == "__main__":
    test_sample_and_candidate_decks_are_valid()
    test_reject_wrong_size()
    test_reject_unknown_card()
    test_reject_over_copy_limit()
    test_basic_energy_exempt_from_copy_limit()
    test_reject_no_basic_pokemon()
    test_reject_multiple_ace_spec()
    test_candidate_decks_complete_a_match()
    print("ALL TESTS PASSED")
