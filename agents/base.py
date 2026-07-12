"""Agent protocol and shared, engine-agnostic helpers.

The submission entry point is ``agent(obs_dict) -> list[int]`` (see ``main.py``).
An :class:`Agent` here works one level up from that raw dict: it receives the
parsed :class:`~cg.api.Observation` and returns the option-index list.

Two things every concrete agent needs are provided here so they are defined once:

- :func:`read_deck_csv` — load the 60-card deck for the *initial* selection
  (when ``obs.select is None`` the submission I/F must return the deck).
- :func:`legal_random_sample` / :func:`is_valid_selection` — produce and check a
  selection that satisfies the engine's contract
  (``minCount <= len <= maxCount``, no duplicates, every index in range).
"""

from __future__ import annotations

import os
import random
from typing import Protocol, runtime_checkable

from cg.api import Observation, SelectData

# Number of cards a legal deck must contain (competition rule).
DECK_SIZE = 60

# Kaggle mounts the submission bundle here; files resolve relative to it at runtime.
_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent/"


@runtime_checkable
class Agent(Protocol):
    """A submission agent.

    ``decide`` receives the parsed :class:`~cg.api.Observation` and returns the
    list of chosen option indices. On the *initial* selection (``obs.select is
    None``) it must return the 60-card deck instead — this keeps the concrete
    agents drop-in compatible with the ``agent(obs_dict) -> list[int]`` I/F that
    ``main.py`` exposes to the competition harness.
    """

    def decide(self, obs: Observation) -> list[int]:  # pragma: no cover - protocol
        ...


def read_deck_csv(path: str = "deck.csv") -> list[int]:
    """Read the deck definition, returning its 60 card IDs.

    Mirrors the original ``main.py`` loader: if ``path`` is not present in the
    current directory, fall back to the Kaggle submission mount point.
    """
    file_path = path
    if not os.path.exists(file_path):
        file_path = _KAGGLE_AGENT_DIR + path
    with open(file_path, "r") as file:
        rows = file.read().split("\n")
    return [int(rows[i]) for i in range(DECK_SIZE)]


def legal_random_sample(select: SelectData, rng: random.Random) -> list[int]:
    """Return a valid random selection for ``select``.

    Guarantees the engine's contract: the result length is clamped into
    ``[minCount, min(maxCount, len(option))]``, indices are unique and within
    range. An empty option list yields ``[]`` (nothing selectable).
    """
    n = len(select.option)
    if n == 0:
        return []
    lo = max(0, select.minCount)
    hi = min(select.maxCount, n)
    if hi < lo:
        # Degenerate bounds — respect the lower bound but never exceed n.
        hi = lo if lo <= n else n
    k = hi
    if k <= 0:
        return []
    return rng.sample(range(n), k)


def is_valid_selection(result: object, select: SelectData) -> bool:
    """Check that ``result`` satisfies the engine's selection contract.

    Valid iff it is a list of unique ints, each ``0 <= i < len(option)``, whose
    length lies in ``[minCount, maxCount]`` (``maxCount`` never exceeds
    ``len(option)`` per the API).
    """
    if not isinstance(result, list):
        return False
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in result):
        return False
    n = len(select.option)
    if any(i < 0 or i >= n for i in result):
        return False
    if len(set(result)) != len(result):
        return False
    return select.minCount <= len(result) <= select.maxCount
