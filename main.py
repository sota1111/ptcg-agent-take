"""Kaggle submission entry point.

Thin wrapper over :class:`agents.rule_based.RuleBasedAgent`. The competition
harness calls ``agent(obs_dict) -> list[int]``; here we only parse the raw dict
into an :class:`~cg.api.Observation` and delegate the decision to the agent.
The I/O contract is unchanged: each returned index is ``0 <= i < len(option)``,
the length is within ``[minCount, maxCount]`` with no duplicates, and the initial
selection (``obs.select is None``) returns the 60-card deck.
"""

import os
from copy import deepcopy

from cg.api import Observation, to_observation_class

from agents.compatibility import CompatibilityAdapter, LegacyDeckStrategy
from agents.rule_based import RuleBasedAgent

# Two independent instances let shadow mode compare stateful decisions without
# changing the authoritative legacy path.  Legacy remains the rollback-safe
# default when the environment variable is absent.
_LEGACY_AGENT = RuleBasedAgent()
_CANDIDATE_AGENT = deepcopy(_LEGACY_AGENT)
_AGENT = CompatibilityAdapter(
    legacy=_LEGACY_AGENT,
    candidate=LegacyDeckStrategy(_CANDIDATE_AGENT),
    mode=os.environ.get("PTCG_TAKE_MIGRATION_MODE", "legacy").strip().lower(),
)


def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount
    (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    obs: Observation = to_observation_class(obs_dict)
    return _AGENT.decide(obs)
