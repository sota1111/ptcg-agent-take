"""Pure contract tests for the SOT-1874 runtime league."""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from eval.runtime_league import atomic_json, parse_opponents, summarise


def test_summary_and_safety():
    state = {"opponents": ["sol"], "matches": [
        {"opponent": "sol", "take_won": True, "fault": None,
         "take_seat": 0, "think_ms": [{"seat": 0, "value": 1.25}]},
        {"opponent": "sol", "take_won": False, "fault": None,
         "take_seat": 1, "think_ms": [{"seat": 1, "value": 2.5}]},
    ]}
    report = summarise(state)
    assert report["leagueWinRate"] == 0.5
    assert report["safety"]["maxTakeDecisionMs"] == 2.5
    assert report["safety"]["faults"] == 0


def test_atomic_checkpoint_and_specs():
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "checkpoint.json"
        atomic_json(path, {"ok": True})
        assert json.loads(path.read_text()) == {"ok": True}
        assert not path.with_suffix(".json.tmp").exists()
    parsed = parse_opponents(["old=/tmp/old"])
    assert parsed["old"] == Path("/tmp/old")


if __name__ == "__main__":
    test_summary_and_safety()
    test_atomic_checkpoint_and_specs()
    print("ALL SOT-1874 RUNTIME LEAGUE TESTS PASSED")
