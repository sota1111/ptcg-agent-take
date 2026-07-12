"""Self-play data generation pipeline for the learning-based agent (SOT-1642).

Plays ``N`` self-play matches back-to-back and accumulates one **learning sample
per decision** into a JSONL dataset. Each sample carries the raw engine
observation, the legal-move candidates (inside the observation's ``select``), the
move the acting agent actually chose, and a win/loss label derived from the final
match result — exactly the ingredients a policy learner (SOT-1643) needs.

Design
------
* **Built on ``eval/record_match.py``.** Each match is recorded at ``FULL_OBS``
  level to a scratch trace, then its ``decision`` / ``result`` records are read
  back. This reuses the engine loop, the guaranteed ``battle_finish()`` cleanup
  (E7), and the winner-scoring logic (agent/engine crash ⇒ scored loss) rather
  than re-implementing them here.
* **Connectable to the featuriser (SOT-1641).** Every sample stores the raw
  ``obs`` (``select`` + ``current``) so ``agents.learned.features.featurize`` can
  turn it into fixed-length vectors without exception. :func:`featurize_sample`
  is the one-call bridge.
* **Win-label consistency.** Every decision in one match shares the same
  match-level ``result`` / ``winner``; each decision additionally gets a
  per-actor ``win`` label (1.0 win / 0.0 loss / 0.5 draw / ``None`` undecided)
  from the perspective of the player who made that decision.
* **Configurable match-up.** The agent pair is chosen from :data:`AGENT_FACTORIES`
  (currently ``random`` and ``rule_based``); the default is ``random`` vs
  ``random``. Swapping in the rule-based agent once it is ready is a CLI flag,
  no code change.
* **Seeding & reproducibility (E1 caveat).** Each match seeds its agents
  deterministically from ``seed`` (match ``i`` ⇒ agent seeds ``seed + 2*i + j``)
  and records that seed on every sample. The cabt engine takes **no seed
  argument** (E1), so re-running does not reproduce a match's trajectory
  byte-for-byte; reproducibility here means the *agent policy* is deterministic
  given a seed and an observation (see :func:`agents_from_config`), and each
  sample's provenance seed is persisted.

No third-party dependencies (matches the repo's zero-pip-deps policy).

Usage:
    venv/bin/python agents/learned/generate_data.py --n 10 --seed 42 \
        --agent0 random --agent1 random --out agents/learned/data/selfplay.jsonl

Run from the repo root (after scripts/setup_engine.sh has populated cg/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Callable, Iterator, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)  # make `cg`, `eval`, `agents` importable
os.chdir(REPO)            # so libcg.so & deck.csv resolve

from agents.learned.features import FeaturizedDecision, featurize  # noqa: E402
from eval import record_match as rm  # noqa: E402
from eval.trace import RecordLevel  # noqa: E402

# Bump when the sample record shape changes.
SAMPLE_SCHEMA_VERSION = "1.0.0"

DEFAULT_OUT = "agents/learned/data/selfplay.jsonl"


# ---------------------------------------------------------------------------
# Configurable agent match-up.
# ---------------------------------------------------------------------------
# A factory builds a ``record_match.Agent`` (a ``fn(obs_dict) -> list[int]`` plus
# identity metadata) for the given seed and player index. New agents register
# here so the pipeline's opponent set can grow without touching the loop.
AgentFactory = Callable[[Optional[int], int], rm.Agent]


def _random_factory(seed: Optional[int], index: int) -> rm.Agent:
    """Uniform-random legal-move agent (the proven ``record_match`` policy)."""
    return rm.make_random_agent(seed, name=f"random{index}")


def _package_agent_factory(
    make: Callable[[Optional[int]], Any], base_name: str
) -> AgentFactory:
    """Adapt an ``agents/`` package agent (``.decide(Observation)``) to the
    ``record_match`` harness (``fn(obs_dict) -> list[int]``).

    The package agents return the deck on the initial selection; under the
    ``record_match`` harness the deck is passed at ``battle_start`` and the
    initial ``select is None`` decision expects an empty list, so we mirror
    ``make_random_agent`` and return ``[]`` there.
    """
    from cg.api import to_observation_class  # noqa: E402

    def factory(seed: Optional[int], index: int) -> rm.Agent:
        agent_obj = make(seed)

        def fn(obs_dict: dict) -> list[int]:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return []
            return agent_obj.decide(obs)

        return rm.Agent(fn, name=f"{base_name}{index}", version="1", params={"seed": seed})

    return factory


def _rule_based_factory() -> AgentFactory:
    from agents.rule_based import RuleBasedAgent  # noqa: E402

    return _package_agent_factory(lambda seed: RuleBasedAgent(seed=seed), "rule_based")


AGENT_FACTORIES: dict[str, AgentFactory] = {
    "random": _random_factory,
    "rule_based": _rule_based_factory(),
}


def agents_from_config(
    agent0: str, agent1: str, seed: Optional[int], match_index: int
) -> tuple[rm.Agent, rm.Agent]:
    """Build the seeded agent pair for match ``match_index``.

    Each agent gets a distinct, deterministic seed derived from ``seed`` so the
    match-up is reproducible at the policy level (see the module E1 caveat).
    """
    if agent0 not in AGENT_FACTORIES:
        raise ValueError(f"unknown agent {agent0!r}; choose from {sorted(AGENT_FACTORIES)}")
    if agent1 not in AGENT_FACTORIES:
        raise ValueError(f"unknown agent {agent1!r}; choose from {sorted(AGENT_FACTORIES)}")
    if seed is None:
        s0: Optional[int] = None
        s1: Optional[int] = None
    else:
        s0 = seed + 2 * match_index
        s1 = seed + 2 * match_index + 1
    return AGENT_FACTORIES[agent0](s0, 0), AGENT_FACTORIES[agent1](s1, 1)


# ---------------------------------------------------------------------------
# Trace → samples.
# ---------------------------------------------------------------------------
def _read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _win_label(actor: Optional[int], winner: Optional[int], result: int) -> Optional[float]:
    """Per-actor win/loss/draw label.

    1.0 = the deciding player won, 0.0 = lost, 0.5 = draw (result 2),
    ``None`` = undecided (truncated / unknown actor).
    """
    if actor in (0, 1) and winner in (0, 1):
        return 1.0 if actor == winner else 0.0
    if result == 2:  # draw
        return 0.5
    return None


def _samples_from_trace(records: list[dict], match_id: int, seed: Optional[int]) -> Iterator[dict]:
    """Turn one match's trace records into per-decision learning samples."""
    result_rec = next((r for r in records if r.get("kind") == "result"), {})
    winner = result_rec.get("winner")
    result = result_rec.get("result", -1)

    for rec in records:
        if rec.get("kind") != "decision":
            continue
        obs = rec.get("obs")
        if not isinstance(obs, dict):
            # FULL_OBS is required for a self-contained, featurisable sample.
            continue
        your_index = rec.get("your_index")
        select_player = rec.get("select_player")
        actor = your_index if your_index in (0, 1) else (
            select_player if select_player in (0, 1) else None
        )
        select = obs.get("select") if isinstance(obs.get("select"), dict) else {}
        options = select.get("option")
        n_options = len(options) if isinstance(options, (list, tuple)) else 0

        yield {
            "kind": "sample",
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "match_id": match_id,
            "seed": seed,
            "decision_index": rec.get("index"),
            "actor": actor,
            "turn": rec.get("turn"),
            "select_type": select.get("type"),
            "select_context": select.get("context"),
            "n_options": n_options,
            "choice": rec.get("choice"),
            "obs": obs,
            # Match-level outcome — identical across every decision of this match.
            "result": result,
            "winner": winner,
            # Per-actor win/loss/draw label for the deciding player.
            "win": _win_label(actor, winner, result),
        }


# ---------------------------------------------------------------------------
# Generation driver.
# ---------------------------------------------------------------------------
def generate(
    n_matches: int,
    out_path: str = DEFAULT_OUT,
    *,
    deck0: Optional[list[int]] = None,
    deck1: Optional[list[int]] = None,
    agent0: str = "random",
    agent1: str = "random",
    seed: Optional[int] = 42,
    max_steps: int = 100000,
) -> dict:
    """Generate ``n_matches`` self-play matches into a JSONL dataset.

    Returns a stats dict (matches / samples / label + winner breakdown / timing).
    One line per decision is written to ``out_path``.
    """
    if n_matches <= 0:
        raise ValueError(f"n_matches must be positive, got {n_matches}")
    if deck0 is None:
        deck0 = rm.load_deck("deck.csv")
    if deck1 is None:
        deck1 = deck0

    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)

    stats = {
        "matches": 0,
        "samples": 0,
        "decisions_skipped": 0,
        "winner_counts": {"0": 0, "1": 0, "draw": 0, "none": 0},
        "label_counts": {"win": 0, "loss": 0, "draw": 0, "none": 0},
        "agents": [agent0, agent1],
        "seed": seed,
        "out_path": out_path,
        "schema_version": SAMPLE_SCHEMA_VERSION,
    }

    t0 = time.perf_counter()
    scratch_dir = tempfile.mkdtemp(prefix="selfplay_")
    scratch = os.path.join(scratch_dir, "match.jsonl")
    try:
        with open(out_path, "w", encoding="utf-8") as sink:
            for i in range(n_matches):
                agents = agents_from_config(agent0, agent1, seed, i)
                rm.record_match(
                    deck0, deck1, agents=agents, out_path=scratch,
                    level=RecordLevel.FULL_OBS, max_steps=max_steps,
                )
                records = _read_jsonl(scratch)
                result_rec = next((r for r in records if r.get("kind") == "result"), {})
                _tally_winner(stats, result_rec)

                match_samples = 0
                for sample in _samples_from_trace(records, match_id=i, seed=seed):
                    sink.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
                    sink.write("\n")
                    _tally_label(stats, sample["win"])
                    match_samples += 1
                # decisions counted by record_match but dropped (no FULL_OBS obs).
                n_decisions = result_rec.get("total_decisions", match_samples) or 0
                stats["decisions_skipped"] += max(0, n_decisions - match_samples)
                stats["samples"] += match_samples
                stats["matches"] += 1
    finally:
        try:
            os.remove(scratch)
        except OSError:
            pass
        try:
            os.rmdir(scratch_dir)
        except OSError:
            pass

    stats["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    return stats


def _tally_winner(stats: dict, result_rec: dict) -> None:
    winner = result_rec.get("winner")
    result = result_rec.get("result", -1)
    if winner == 0:
        stats["winner_counts"]["0"] += 1
    elif winner == 1:
        stats["winner_counts"]["1"] += 1
    elif result == 2:
        stats["winner_counts"]["draw"] += 1
    else:
        stats["winner_counts"]["none"] += 1


def _tally_label(stats: dict, win: Optional[float]) -> None:
    if win == 1.0:
        stats["label_counts"]["win"] += 1
    elif win == 0.0:
        stats["label_counts"]["loss"] += 1
    elif win == 0.5:
        stats["label_counts"]["draw"] += 1
    else:
        stats["label_counts"]["none"] += 1


# ---------------------------------------------------------------------------
# Dataset consumption bridge (SOT-1641 featuriser).
# ---------------------------------------------------------------------------
def iter_samples(path: str) -> Iterator[dict]:
    """Yield each learning sample from a generated JSONL dataset."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def featurize_sample(sample: dict) -> FeaturizedDecision:
    """Featurise one generated sample via the SOT-1641 featuriser.

    The one-call bridge from a stored sample to model-ready vectors; never
    raises (the featuriser tolerates missing / malformed observations).
    """
    return featurize(sample.get("obs"))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate self-play learning data (SOT-1642).")
    p.add_argument("--n", type=int, default=10, help="number of matches to play")
    p.add_argument("--out", default=DEFAULT_OUT, help="output JSONL path")
    p.add_argument("--seed", type=int, default=42, help="base seed (None-like -1 ⇒ unseeded)")
    p.add_argument("--agent0", default="random", choices=sorted(AGENT_FACTORIES))
    p.add_argument("--agent1", default="random", choices=sorted(AGENT_FACTORIES))
    p.add_argument("--deck0", default="deck.csv")
    p.add_argument("--deck1", default=None)
    p.add_argument("--max-steps", type=int, default=100000)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    deck0 = rm.load_deck(args.deck0)
    deck1 = rm.load_deck(args.deck1) if args.deck1 else deck0
    seed = None if args.seed < 0 else args.seed

    stats = generate(
        args.n, out_path=args.out, deck0=deck0, deck1=deck1,
        agent0=args.agent0, agent1=args.agent1, seed=seed, max_steps=args.max_steps,
    )
    print(
        f"DATAGEN DONE: matches={stats['matches']} samples={stats['samples']}"
        f" winners={stats['winner_counts']} labels={stats['label_counts']}"
        f" skipped={stats['decisions_skipped']} elapsed_ms={stats['elapsed_ms']}"
        f" out={stats['out_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
