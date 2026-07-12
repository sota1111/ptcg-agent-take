"""Uniform-random legal-move agent.

This is the original ``main.py`` behaviour moved behind the :class:`Agent`
protocol, with an injectable seed for reproducibility. On the initial selection
it returns the deck; otherwise it picks a legal random set of option indices.
"""

from __future__ import annotations

import random
from typing import Optional

from cg.api import Observation

from agents.base import Agent, legal_random_sample, read_deck_csv


class RandomAgent(Agent):
    """Selects uniformly at random among legal options.

    Args:
        seed: Seed for the internal :class:`random.Random`. ``None`` (default)
            uses an unseeded RNG, matching the original ``main.py`` behaviour.
        deck_path: Path to the deck CSV used for the initial selection.
    """

    def __init__(self, seed: Optional[int] = None, deck_path: str = "deck.csv") -> None:
        self.rng = random.Random(seed)
        self.deck_path = deck_path

    def decide(self, obs: Observation) -> list[int]:
        if obs.select is None:
            # Initial selection: the engine expects the 60-card deck.
            return read_deck_csv(self.deck_path)
        return legal_random_sample(obs.select, self.rng)
