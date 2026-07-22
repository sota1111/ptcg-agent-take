"""Kaggle submission entry point.

Thin wrapper over :class:`agents.rule_based.RuleBasedAgent`. The competition
harness calls ``agent(obs_dict) -> list[int]``; here we only parse the raw dict
into an :class:`~cg.api.Observation` and delegate the decision to the agent.
The I/O contract is unchanged: each returned index is ``0 <= i < len(option)``,
the length is within ``[minCount, maxCount]`` with no duplicates, and the initial
selection (``obs.select is None``) returns the 60-card deck.
"""

import os
import sys
from copy import deepcopy

# Kaggle executes this file with exec() (no __file__), so the submission
# directory is not necessarily on sys.path. Prefer the bundled cg/ and agents/
# packages over any unrelated module with the same top-level name.
_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent"
_SUBMISSION_DIR = (_KAGGLE_AGENT_DIR if os.path.isdir(_KAGGLE_AGENT_DIR)
                   else os.path.abspath(os.getcwd()))
if sys.path[0] != _SUBMISSION_DIR:
    sys.path.insert(0, _SUBMISSION_DIR)

from cg.api import Observation, to_observation_class  # noqa: E402

from agents.compatibility import CompatibilityAdapter, LegacyDeckStrategy  # noqa: E402
from agents.rule_based import RuleBasedAgent  # noqa: E402

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
