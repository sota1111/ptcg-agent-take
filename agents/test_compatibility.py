"""Regression tests for the staged common-core compatibility adapter."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.compatibility import (
    ADAPTER_API_VERSION,
    DECK_STRATEGY_API_VERSION,
    CompatibilityAdapter,
    LegacyDeckStrategy,
    stderr_shadow_sink,
)


class FixedAgent:
    def __init__(self, result: list[int]) -> None:
        self.result = result
        self.calls = 0

    def decide(self, _observation) -> list[int]:
        self.calls += 1
        return list(self.result)


@dataclass
class FixedStrategy:
    result: list[int]
    api_version: str = DECK_STRATEGY_API_VERSION
    implementation_version: str = "fixture/v1"
    compatible_adapter_apis: tuple[str, ...] = (ADAPTER_API_VERSION,)

    def decide(self, _observation) -> list[int]:
        return list(self.result)


class CompatibilityAdapterTest(unittest.TestCase):
    def test_legacy_is_default_and_does_not_execute_candidate(self):
        legacy = FixedAgent([1])
        candidate_agent = FixedAgent([0])
        adapter = CompatibilityAdapter(legacy, LegacyDeckStrategy(candidate_agent))
        self.assertEqual(adapter.decide(None), [1])
        self.assertEqual(legacy.calls, 1)
        self.assertEqual(candidate_agent.calls, 0)

    def test_shadow_compares_both_paths_but_keeps_legacy_authoritative(self):
        comparisons = []
        adapter = CompatibilityAdapter(
            FixedAgent([1]), FixedStrategy([0]), mode="shadow", shadow_sink=comparisons.append
        )
        self.assertEqual(adapter.decide(None), [1])
        self.assertFalse(comparisons[0].matched)
        self.assertEqual(comparisons[0].legacy, (1,))
        self.assertEqual(comparisons[0].candidate, (0,))

    def test_core_mode_switch_and_rollback_to_legacy(self):
        legacy = FixedAgent([1])
        candidate = FixedStrategy([0])
        self.assertEqual(CompatibilityAdapter(legacy, candidate, mode="core").decide(None), [0])
        self.assertEqual(CompatibilityAdapter(legacy, candidate, mode="legacy").decide(None), [1])

    def test_incompatible_contract_versions_fail_closed(self):
        cases = [
            (FixedStrategy([], api_version="ptcg-deck-strategy/v2"), "incompatible strategy API"),
            (FixedStrategy([], compatible_adapter_apis=()), "is incompatible"),
        ]
        for strategy, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                CompatibilityAdapter(FixedAgent([]), strategy)

    def test_shadow_log_is_machine_readable(self):
        output = io.StringIO()
        adapter = CompatibilityAdapter(
            FixedAgent([1]),
            FixedStrategy([1]),
            mode="shadow",
            shadow_sink=lambda comparison: stderr_shadow_sink(comparison, output),
        )
        adapter.decide(None)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "event": "ptcg_take_shadow_comparison",
                "sequence": 0,
                "matched": True,
                "legacy": [1],
                "candidate": [1],
            },
        )

    def test_submission_starts_in_every_mode(self):
        command = [sys.executable, "-c", "import main; print(type(main._AGENT).__name__)"]
        for mode in ("legacy", "shadow", "core"):
            with self.subTest(mode=mode):
                env = {**os.environ, "PTCG_TAKE_MIGRATION_MODE": mode}
                completed = subprocess.run(
                    command, env=env, text=True, capture_output=True, check=True
                )
                self.assertEqual(completed.stdout.strip(), "CompatibilityAdapter")
                self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
