"""Standalone tests for the policy trainer (SOT-1643).

No pytest dependency — run directly from the repo root:
    venv/bin/python agents/learned/test_train.py

Covers the acceptance criteria:
  1. data generation -> training runs end-to-end and writes a model file of a
     bundleable (committable) size;
  2. the held-out choice-agreement beats the random baseline on learnable
     (rule-based self-play) data;
  3. the saved model is the inference-only JSON format that
     ``agents.learned.agent`` reads with the standard library alone (no learning
     dependencies), and it drops into ``LearnedAgent`` / a real match.
Plus unit checks for the labelling, standardiser, split, and the honest
no-signal (random self-play) case.
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
from agents.learned import train as tr  # noqa: E402
from agents.learned.agent import (  # noqa: E402
    LearnedAgent,
    LinearOptionScorer,
    load_model,
    make_learned_agent,
)
from agents.learned.features import OPTION_FEATURE_DIM  # noqa: E402


def _gen(tmp: str, n: int, a0: str, a1: str, seed: int) -> str:
    out = os.path.join(tmp, f"{a0}_{a1}.jsonl")
    gd.generate(n, out_path=out, agent0=a0, agent1=a1, seed=seed)
    return out


def test_valid_choice_guards():
    """Only well-formed, non-exhaustive choices become labels."""
    assert tr._valid_choice([0], 3) == {0}
    assert tr._valid_choice([0, 2], 3) == {0, 2}
    assert tr._valid_choice([], 3) is None          # empty
    assert tr._valid_choice(None, 3) is None         # missing
    assert tr._valid_choice([3], 3) is None          # out of range
    assert tr._valid_choice([0, 0], 3) is None       # duplicate
    assert tr._valid_choice([0, 1, 2], 3) is None    # all options chosen
    assert tr._valid_choice([True], 3) is None       # bool is not a real index
    print("PASS test_valid_choice_guards")


def test_standardiser_matches_inference():
    """The stored mean/std reproduce the agent's own standardisation, and a
    constant feature gets std 1.0 (no divide-by-zero)."""
    rows = [[0.0, 5.0, 2.0], [2.0, 5.0, 4.0], [4.0, 5.0, 6.0]]
    mean, std = tr.fit_standardiser(rows, 3)
    assert abs(mean[0] - 2.0) < 1e-9 and abs(mean[1] - 5.0) < 1e-9, mean
    assert std[1] == 1.0, std  # constant column -> guarded to 1.0
    # A LinearOptionScorer with the same mean/std standardises identically.
    m = LinearOptionScorer(weights=[1.0, 1.0, 1.0], bias=0.0, mean=mean, std=std)
    manual = sum(tr._standardise(rows[0], mean, std))
    assert abs(m.score(rows[0]) - manual) < 1e-9
    print("PASS test_standardiser_matches_inference")


def test_split_is_decision_level_and_seeded():
    """Split covers every decision exactly once and is reproducible by seed."""
    decs = [tr.Decision([[0.0] * OPTION_FEATURE_DIM] * 2, {0}) for _ in range(10)]
    tr_a, ho_a = tr.split_decisions(decs, 0.2, seed=1)
    tr_b, ho_b = tr.split_decisions(decs, 0.2, seed=1)
    assert len(tr_a) + len(ho_a) == 10 and len(ho_a) == 2
    assert len(tr_a) == len(tr_b) and len(ho_a) == len(ho_b)  # deterministic
    # never empty on either side for a non-trivial dataset
    assert tr_a and ho_a
    print("PASS test_split_is_decision_level_and_seeded")


def test_end_to_end_beats_random_and_saves_bundleable_model():
    """AC1+AC2+AC3: gen -> train writes a small model whose holdout agreement
    beats the random baseline, in the dependency-free inference format."""
    with tempfile.TemporaryDirectory() as tmp:
        data = _gen(tmp, 60, "rule_based", "rule_based", seed=11)
        model_out = os.path.join(tmp, "model", "policy.json")
        report = tr.train(
            data_path=data, model_out=model_out, regenerate=False,
            holdout=0.25, split_seed=0, epochs=200,
        )
        # AC1: model file exists and is bundleable-small.
        assert os.path.exists(model_out), report
        assert 0 < report["model_bytes"] < 200_000, report["model_bytes"]
        assert report["n_train"] > 0 and report["n_holdout"] > 0, report
        # AC2: holdout beats the random baseline.
        he = report["holdout_eval"]
        assert he["decisions"] > 0, he
        assert he["model_agreement"] > he["random_baseline"], he
        assert he["beats_random"] is True, he
        # AC3: saved as the inference JSON format, loadable with the stdlib only.
        with open(model_out, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        assert raw["format"] == "ptcg-learned-policy" and raw["kind"] == "linear"
        assert raw["option_feature_dim"] == OPTION_FEATURE_DIM
        assert len(raw["weights"]) == OPTION_FEATURE_DIM
        assert len(raw["mean"]) == OPTION_FEATURE_DIM and len(raw["std"]) == OPTION_FEATURE_DIM
        model = load_model(model_out)
        assert model is not None and model.dim == OPTION_FEATURE_DIM
    print("PASS test_end_to_end_beats_random_and_saves_bundleable_model")


def test_trained_model_drops_into_agent_and_match():
    """The trained model loads into LearnedAgent and plays a real match without
    crashing (integration with the SOT-1644 inference agent)."""
    from eval.record_match import record_match, make_random_agent, load_deck

    with tempfile.TemporaryDirectory() as tmp:
        data = _gen(tmp, 40, "rule_based", "rule_based", seed=21)
        model_out = os.path.join(tmp, "policy.json")
        tr.train(data_path=data, model_out=model_out, regenerate=False, epochs=120)

        agent = LearnedAgent(model_path=model_out, seed=1)
        assert agent.has_model, "trained model should load"

        deck = load_deck("deck.csv")
        la = make_learned_agent(model_path=model_out, seed=2)
        ra = make_random_agent(3, name="rand")
        out = os.path.join(tmp, "match.jsonl")
        res = record_match(deck, deck, agents=(la, ra), out_path=out)
        assert isinstance(res, dict) and "result" in res, res
    print("PASS test_trained_model_drops_into_agent_and_match")


def test_random_selfplay_has_no_learnable_signal():
    """Honesty check: cloning random-vs-random winners does NOT beat the random
    baseline (the winner's move is itself uniform) — motivating rule-based data."""
    with tempfile.TemporaryDirectory() as tmp:
        data = _gen(tmp, 40, "random", "random", seed=31)
        model_out = os.path.join(tmp, "policy.json")
        report = tr.train(data_path=data, model_out=model_out, regenerate=False,
                           holdout=0.3, epochs=150)
        he = report["holdout_eval"]
        # No meaningful edge over the baseline (allow a tiny sampling wobble).
        assert he["model_agreement"] <= he["random_baseline"] + 0.06, he
    print("PASS test_random_selfplay_has_no_learnable_signal")


if __name__ == "__main__":
    test_valid_choice_guards()
    test_standardiser_matches_inference()
    test_split_is_decision_level_and_seeded()
    test_end_to_end_beats_random_and_saves_bundleable_model()
    test_trained_model_drops_into_agent_and_match()
    test_random_selfplay_has_no_learnable_signal()
    print("ALL TESTS PASSED")
