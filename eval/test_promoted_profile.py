"""Runtime promotion contract tests for SOT-1869."""

import json
import os
import tempfile
import unittest
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from agents.profile import load_promoted_profile
from agents.rule_based import RuleBasedAgent


class PromotedProfileTest(unittest.TestCase):
    def test_bundled_profile_is_the_production_default(self):
        agent = RuleBasedAgent(seed=1869)
        self.assertEqual(agent.runtime_profile.profile_id, "take-adaptive-tempo-v1")
        self.assertEqual(agent.runtime_profile.risk_profile, "balanced")
        self.assertTrue(agent._adaptive_search)
        self.assertEqual(agent.runtime_profile.competition_budget_seconds, 600)
        self.assertEqual(agent.runtime_profile.experiment_budget_seconds, 8 * 60 * 60)
        self.assertEqual(agent.runtime_profile.checkpoint_every_games, 2)

    def test_invalid_profile_fails_closed(self):
        raw = {
            "schemaVersion": "ptcg-take-runtime-profile/v1", "id": "bad",
            "strategy": "matchup-responsive-balanced-tempo", "riskProfile": "bad",
            "adaptationWeight": 0.32, "riskFloor": 0.9, "riskCeiling": 0.65,
            "searchBudgetMs": 250, "maxDepth": 3,
            "maxBranchingForExtension": 10,
            "illegalActionFallback": "highest-value-legal"
            , "competitionBudgetSeconds": 600
            , "experimentBudgetSeconds": 28800
            , "checkpointEveryGames": 2
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(raw, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_promoted_profile(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
