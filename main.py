"""Kaggle submission entry point.

Thin wrapper over :class:`agents.rule_based.RuleBasedAgent`. The competition
harness calls ``agent(obs_dict) -> list[int]``; here we only parse the raw dict
into an :class:`~cg.api.Observation` and delegate the decision to the agent.
The I/O contract is unchanged: each returned index is ``0 <= i < len(option)``,
the length is within ``[minCount, maxCount]`` with no duplicates, and the initial
selection (``obs.select is None``) returns the 60-card deck.
"""

from cg.api import Observation, to_observation_class

from agents.rule_based import RuleBasedAgent

# One agent instance reused across all decisions in a match.
_AGENT = RuleBasedAgent()


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
