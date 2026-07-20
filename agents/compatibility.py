"""Compatibility boundary for the staged ptcg-agent-core migration.

The JavaScript core publishes ``ptcg-deck-strategy/v1`` and
``ptcg-agent-adapter/v1`` as independent compatibility versions.  The Python
submission cannot import that package directly, so this module mirrors only the
small adapter boundary: a versioned strategy receives the engine observation
and returns the existing Kaggle option-index action.

``legacy`` remains the default.  ``shadow`` executes both paths but returns the
legacy result, and ``core`` makes the versioned strategy authoritative.  This
keeps rollback a configuration change rather than a source rollback.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Protocol, TextIO

from cg.api import Observation

from agents.base import Agent

DECK_STRATEGY_API_VERSION = "ptcg-deck-strategy/v1"
ADAPTER_API_VERSION = "ptcg-agent-adapter/v1"
MIGRATION_MODES = frozenset({"legacy", "shadow", "core"})


class DeckStrategy(Protocol):
    """Minimal Python side of the common deck strategy contract."""

    api_version: str
    implementation_version: str
    compatible_adapter_apis: tuple[str, ...]

    def decide(self, observation: Observation) -> list[int]: ...


@dataclass
class LegacyDeckStrategy:
    """Expose an existing agent through the versioned strategy boundary."""

    agent: Agent
    implementation_version: str = "take-rule-based/v1"
    api_version: str = DECK_STRATEGY_API_VERSION
    compatible_adapter_apis: tuple[str, ...] = (ADAPTER_API_VERSION,)

    def decide(self, observation: Observation) -> list[int]:
        return self.agent.decide(observation)


@dataclass(frozen=True)
class ShadowComparison:
    sequence: int
    matched: bool
    legacy: tuple[int, ...]
    candidate: tuple[int, ...]


@dataclass
class CompatibilityAdapter:
    """Route decisions through legacy, shadow, or authoritative-core mode."""

    legacy: Agent
    candidate: DeckStrategy
    mode: str = "legacy"
    shadow_sink: Callable[[ShadowComparison], None] | None = None
    _sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.mode not in MIGRATION_MODES:
            raise ValueError(
                f"unknown migration mode {self.mode!r}; expected one of {sorted(MIGRATION_MODES)}"
            )
        if self.candidate.api_version != DECK_STRATEGY_API_VERSION:
            raise ValueError(
                f"incompatible strategy API {self.candidate.api_version!r}; "
                f"expected {DECK_STRATEGY_API_VERSION!r}"
            )
        if ADAPTER_API_VERSION not in self.candidate.compatible_adapter_apis:
            raise ValueError(
                f"strategy {self.candidate.implementation_version!r} is incompatible "
                f"with {ADAPTER_API_VERSION!r}"
            )

    def decide(self, observation: Observation) -> list[int]:
        if self.mode == "legacy":
            return self.legacy.decide(observation)
        if self.mode == "core":
            return self.candidate.decide(observation)

        legacy_result = self.legacy.decide(observation)
        candidate_result = self.candidate.decide(observation)
        comparison = ShadowComparison(
            sequence=self._sequence,
            matched=legacy_result == candidate_result,
            legacy=tuple(legacy_result),
            candidate=tuple(candidate_result),
        )
        self._sequence += 1
        (self.shadow_sink or stderr_shadow_sink)(comparison)
        return legacy_result


def stderr_shadow_sink(comparison: ShadowComparison, stream: TextIO = sys.stderr) -> None:
    """Emit one machine-readable comparison without contaminating stdout."""
    stream.write(
        json.dumps(
            {
                "event": "ptcg_take_shadow_comparison",
                "sequence": comparison.sequence,
                "matched": comparison.matched,
                "legacy": list(comparison.legacy),
                "candidate": list(comparison.candidate),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    stream.flush()
