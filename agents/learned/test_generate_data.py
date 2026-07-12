"""Standalone tests for the self-play data pipeline (SOT-1642).

No pytest dependency — run directly from the repo root:
    venv/bin/python agents/learned/test_generate_data.py

Covers the three acceptance criteria:
  1. N=10 self-play data generation completes and writes decision samples;
  2. the generated data connects to the SOT-1641 featuriser — every sample
     vectorises without exception into fixed-length vectors;
  3. seeding makes the agent-side policy reproducible (the engine takes no seed,
     E1, so match trajectories are not; we assert what *is* reproducible).
Plus win-label consistency (all decisions in a match share the match result;
each decision's per-actor label agrees with the winner) and agent config.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

from agents.learned import generate_data as gd  # noqa: E402
from agents.learned.features import (  # noqa: E402
    OBSERVATION_FEATURE_DIM,
    OPTION_FEATURE_DIM,
)
from cg.api import to_observation_class  # noqa: E402
from eval import record_match as rm  # noqa: E402


def _read_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_generate_completes_and_writes_samples():
    """AC1: N=10 self-play completes and writes one sample per decision."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "selfplay.jsonl")
        stats = gd.generate(10, out_path=out, agent0="random", agent1="random", seed=123)

        assert stats["matches"] == 10, stats
        assert stats["samples"] > 0, "expected decision samples"
        assert os.path.exists(out)

        samples = _read_samples(out)
        assert len(samples) == stats["samples"], (len(samples), stats["samples"])

        # winner_counts + label bookkeeping is internally consistent.
        wc = stats["winner_counts"]
        assert wc["0"] + wc["1"] + wc["draw"] + wc["none"] == 10, wc
        lc = stats["label_counts"]
        assert lc["win"] + lc["loss"] + lc["draw"] + lc["none"] == stats["samples"], lc

        s0 = samples[0]
        assert s0["kind"] == "sample" and s0["schema_version"], s0
        assert isinstance(s0["choice"], list), "choice recorded"
        assert isinstance(s0["obs"], dict) and "select" in s0["obs"], "raw obs stored"
        assert set(s0["match_id"] for s in [s0]) <= set(range(10))
    print("PASS test_generate_completes_and_writes_samples")


def test_samples_featurise_via_sot1641():
    """AC2: every generated sample vectorises through the SOT-1641 featuriser."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "selfplay.jsonl")
        gd.generate(10, out_path=out, seed=7)

        n = 0
        saw_nonzero_board = False
        for sample in gd.iter_samples(out):
            feat = gd.featurize_sample(sample)  # must not raise
            assert isinstance(feat.observation, list)
            assert len(feat.observation) == OBSERVATION_FEATURE_DIM, len(feat.observation)
            assert feat.n_options == len(feat.candidates)
            # n_options recorded on the sample matches the featuriser's count.
            assert sample["n_options"] == feat.n_options, (sample["n_options"], feat.n_options)
            for cand in feat.candidates:
                assert len(cand) == OPTION_FEATURE_DIM, len(cand)
            if any(v != 0.0 for v in feat.observation):
                saw_nonzero_board = True
            n += 1

        assert n > 0, "no samples produced"
        assert saw_nonzero_board, "expected real (non-zero) board features from stored obs"
    print(f"PASS test_samples_featurise_via_sot1641 ({n} samples)")


def test_win_label_consistency():
    """Win/loss labels are consistent: match-level result shared by all its
    decisions; each decision's per-actor label agrees with the winner."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "selfplay.jsonl")
        gd.generate(10, out_path=out, seed=555)
        samples = _read_samples(out)

        by_match: dict[int, list[dict]] = {}
        for s in samples:
            by_match.setdefault(s["match_id"], []).append(s)

        for mid, group in by_match.items():
            winners = {s["winner"] for s in group}
            results = {s["result"] for s in group}
            assert len(winners) == 1, f"match {mid}: mixed winners {winners}"
            assert len(results) == 1, f"match {mid}: mixed results {results}"

            winner = group[0]["winner"]
            result = group[0]["result"]
            for s in group:
                win = s["win"]
                actor = s["actor"]
                if actor in (0, 1) and winner in (0, 1):
                    expected = 1.0 if actor == winner else 0.0
                    assert win == expected, (mid, actor, winner, win)
                elif result == 2:
                    assert win == 0.5, (mid, result, win)
                else:
                    assert win is None, (mid, result, winner, win)
    print("PASS test_win_label_consistency")


def test_seed_makes_agent_policy_reproducible():
    """AC3: the engine has no seed (E1) so trajectories are not reproducible,
    but a seeded agent's policy IS: two agents built with the same seed choose
    identically on the same observation sequence."""
    # Harvest a real observation sequence from a generated dataset.
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "selfplay.jsonl")
        gd.generate(3, out_path=out, seed=99)
        obs_seq = [s["obs"] for s in _read_samples(out)]

    assert obs_seq, "need observations to replay"

    a1 = rm.make_random_agent(2024, "a")
    a2 = rm.make_random_agent(2024, "b")
    for obs in obs_seq:
        assert a1.act(obs) == a2.act(obs), "same seed must yield the same choice"

    # A different seed diverges on at least one decision (sanity: seed matters).
    b1 = rm.make_random_agent(1, "x")
    b2 = rm.make_random_agent(2, "y")
    diverged = any(b1.act(o) != b2.act(o) for o in obs_seq)
    assert diverged, "different seeds should diverge somewhere"
    print("PASS test_seed_makes_agent_policy_reproducible")


def test_agent_config_selectable():
    """The agent match-up is configurable; rule_based is registered and usable."""
    assert set(gd.AGENT_FACTORIES) >= {"random", "rule_based"}, gd.AGENT_FACTORIES

    # Build a rule_based-vs-random pair and confirm both satisfy the harness I/F.
    a0, a1 = gd.agents_from_config("rule_based", "random", seed=42, match_index=0)
    assert callable(a0.fn) and callable(a1.fn)
    assert a0.name.startswith("rule_based") and a1.name.startswith("random")

    # A tiny rule_based-vs-random run completes without raising.
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "mixed.jsonl")
        stats = gd.generate(2, out_path=out, agent0="rule_based", agent1="random", seed=3)
        assert stats["matches"] == 2 and stats["samples"] > 0, stats
        assert stats["agents"] == ["rule_based", "random"], stats

    # Unknown agent name is rejected clearly.
    try:
        gd.agents_from_config("does_not_exist", "random", seed=0, match_index=0)
        raise AssertionError("expected ValueError for unknown agent")
    except ValueError:
        pass
    print("PASS test_agent_config_selectable")


if __name__ == "__main__":
    test_generate_completes_and_writes_samples()
    test_samples_featurise_via_sot1641()
    test_win_label_consistency()
    test_seed_makes_agent_policy_reproducible()
    test_agent_config_selectable()
    print("ALL TESTS PASSED")
