"""Submission agents for the PTCG AI Battle competition.

This package holds the agent implementations that back ``main.py``'s submission
entry point ``agent(obs_dict) -> list[int]``:

- ``base``          — the :class:`Agent` protocol plus shared, engine-agnostic
                      helpers (deck loading, legal-random selection, validation).
- ``random_agent``  — a uniform-random legal-move agent (the original ``main.py``
                      behaviour, moved here with an injectable seed).
- ``rule_based``    — a rule-based skeleton: a SelectContext dispatch table with
                      an outermost safety guard that always falls back to a legal
                      random selection. Rules (R2+) are added as pure per-context
                      functions without touching the guard.
"""

from agents.base import Agent, legal_random_sample, read_deck_csv, is_valid_selection
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent

__all__ = [
    "Agent",
    "RandomAgent",
    "RuleBasedAgent",
    "legal_random_sample",
    "read_deck_csv",
    "is_valid_selection",
]
