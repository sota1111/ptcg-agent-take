"""Learned-policy inference agent with random-legal fallback (SOT-1644).

Picks a move with a **trained policy model** and, whenever anything about that
path is unavailable or goes wrong, falls back to a **uniform-random legal move**
so the agent never crashes and always returns a valid selection.

Contract
--------
The agent conforms to the ``eval/record_match.py`` ``Agent`` contract —
``act(obs_dict) -> list[int]`` returns option indices that satisfy the engine's
selection rules (``minCount <= len <= maxCount``, unique, each ``0 <= i < len``).
It is also a drop-in Kaggle submission agent: on the *initial* selection
(``obs_dict['select'] is None``) :meth:`LearnedAgent.act` returns the 60-card
deck, exactly like :class:`agents.random_agent.RandomAgent`. The eval adapter
:func:`make_learned_agent` instead returns ``[]`` on that initial step, because
``record_match`` passes the decks at ``battle_start`` (mirroring the random
agent under that harness).

Model format (inference-only, dependency-free)
----------------------------------------------
The model is a small JSON file readable with the standard library alone (no
numpy / scikit-learn at inference time), so it can be bundled with a submission
that must run without extra dependencies or network access::

    {
      "format": "ptcg-learned-policy",
      "version": "1.0.0",
      "kind": "linear",                 # linear option-scorer
      "option_feature_dim": <int>,      # must equal features.OPTION_FEATURE_DIM
      "weights": [float, ...],          # length option_feature_dim
      "bias": <float>,                  # optional, default 0.0
      "mean": [float, ...],             # optional standardisation (len == dim)
      "std":  [float, ...]              # optional standardisation (len == dim)
    }

Scoring: ``s(option) = w · standardise(option_features(option)) + bias``; the
top-scoring option(s) are selected (``k`` of them for a multi-select, see
:func:`_legal_k`). The featuriser (SOT-1641) never raises and maps unknown enum
values to a dedicated slot, so unseen ``SelectType`` / ``OptionType`` values are
scored, not fatal. The trainer (SOT-1643) writes this format; a model whose
shape does not match the current feature dimension is treated as *absent* and
the agent falls back to random — so a format drift degrades gracefully instead
of crashing.

No third-party dependencies (matches the repo's zero-pip-deps policy).
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Optional

from agents.base import read_deck_csv
from agents.learned.features import OPTION_FEATURE_DIM, candidate_features

# Default location a bundled model is looked up at (absent by default → random).
DEFAULT_MODEL_PATH = "agents/learned/model/policy.json"

MODEL_FORMAT = "ptcg-learned-policy"


# --------------------------------------------------------------------------- #
# Small, exception-proof numeric helpers.
# --------------------------------------------------------------------------- #
def _int(value: Any, default: int = 0) -> int:
    """Best-effort int; ``None`` / bad values → ``default``."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Inference model: a linear scorer over the per-option feature vector.
# --------------------------------------------------------------------------- #
class LinearOptionScorer:
    """Scores a single option's feature vector with ``w · x + b``.

    Optional per-feature standardisation ``(x - mean) / std`` is applied first
    when ``mean`` / ``std`` are provided (a zero std is treated as 1 to avoid a
    divide-by-zero). ``score`` raises :class:`ValueError` on a dimension
    mismatch; callers catch it and fall back to random.
    """

    def __init__(
        self,
        weights: list[float],
        bias: float = 0.0,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
    ) -> None:
        self.weights = [float(w) for w in weights]
        self.dim = len(self.weights)
        self.bias = float(bias)
        self.mean = [float(m) for m in mean] if mean is not None else None
        # Replace non-positive std entries with 1.0 (no scaling) for safety.
        self.std = [float(s) if float(s) != 0.0 else 1.0 for s in std] if std is not None else None
        if self.mean is not None and len(self.mean) != self.dim:
            raise ValueError("mean length does not match weights")
        if self.std is not None and len(self.std) != self.dim:
            raise ValueError("std length does not match weights")

    def score(self, feat: list[float]) -> float:
        if len(feat) != self.dim:
            raise ValueError(f"feature dim {len(feat)} != model dim {self.dim}")
        total = self.bias
        mean = self.mean
        std = self.std
        w = self.weights
        for i, f in enumerate(feat):
            x = f
            if mean is not None:
                x -= mean[i]
            if std is not None:
                x /= std[i]
            total += w[i] * x
        return total

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "format": MODEL_FORMAT,
            "version": "1.0.0",
            "kind": "linear",
            "option_feature_dim": self.dim,
            "weights": list(self.weights),
            "bias": self.bias,
        }
        if self.mean is not None:
            d["mean"] = list(self.mean)
        if self.std is not None:
            d["std"] = list(self.std)
        return d

    @classmethod
    def from_dict(cls, data: Any) -> "LinearOptionScorer":
        """Build a scorer from a parsed JSON model, validating its shape.

        Raises ``ValueError`` if the payload is not a usable linear model whose
        feature dimension matches the current featuriser output; the loader
        turns that into a graceful "no model" (random fallback).
        """
        if not isinstance(data, dict):
            raise ValueError("model payload is not an object")
        kind = data.get("kind", "linear")
        if kind != "linear":
            raise ValueError(f"unsupported model kind {kind!r}")
        weights = data.get("weights")
        if not isinstance(weights, (list, tuple)) or not weights:
            raise ValueError("model has no weights")
        dim = _int(data.get("option_feature_dim"), len(weights))
        if dim != len(weights):
            raise ValueError("option_feature_dim disagrees with weights length")
        if dim != OPTION_FEATURE_DIM:
            # A model trained against a different feature layout cannot score the
            # current candidates; treat it as absent rather than mis-scoring.
            raise ValueError(
                f"model dim {dim} != current OPTION_FEATURE_DIM {OPTION_FEATURE_DIM}"
            )
        mean = data.get("mean")
        std = data.get("std")
        return cls(
            weights=list(weights),
            bias=float(data.get("bias", 0.0)),
            mean=list(mean) if isinstance(mean, (list, tuple)) else None,
            std=list(std) if isinstance(std, (list, tuple)) else None,
        )


def load_model(path: Optional[str]) -> Optional[LinearOptionScorer]:
    """Load a policy model from ``path``; return ``None`` on any problem.

    A missing file, unreadable/garbled JSON, an unsupported ``kind``, or a
    feature-dimension mismatch all yield ``None`` (→ random fallback) rather than
    an exception, so inference is robust to a missing or drifted model.
    """
    if not path:
        return None
    candidate = path
    if not os.path.exists(candidate):
        # Mirror read_deck_csv: also try the Kaggle submission mount point.
        alt = os.path.join("/kaggle_simulations/agent/", path)
        if not os.path.exists(alt):
            return None
        candidate = alt
    try:
        with open(candidate, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return LinearOptionScorer.from_dict(data)
    except Exception:
        return None


def save_model(model: LinearOptionScorer, path: str) -> str:
    """Write ``model`` as the inference JSON format (used by the trainer/tests)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, ensure_ascii=False)
    return path


# --------------------------------------------------------------------------- #
# Legal-selection helpers (dict-native, mirror agents.base.legal_random_sample).
# --------------------------------------------------------------------------- #
def _legal_k(select: dict, n: int) -> int:
    """Number of options to pick for ``select`` given ``n`` legal options.

    Clamps ``[minCount, min(maxCount, n)]`` exactly like
    :func:`agents.base.legal_random_sample`, so every produced selection length
    is engine-valid.
    """
    lo = max(0, _int(select.get("minCount"), 0))
    hi = min(_int(select.get("maxCount"), max(lo, 1)), n)
    if hi < lo:
        hi = lo if lo <= n else n
    return hi


def _options(select: Any) -> int:
    if not isinstance(select, dict):
        return 0
    opts = select.get("option")
    return len(opts) if isinstance(opts, (list, tuple)) else 0


def _random_legal(select: Any, rng: random.Random) -> list[int]:
    """Uniform-random valid selection for ``select`` (the safe fallback)."""
    if not isinstance(select, dict):
        return []
    n = _options(select)
    if n == 0:
        return []
    k = _legal_k(select, n)
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    return rng.sample(range(n), k)


def _model_legal(obs_dict: dict, select: dict, model: LinearOptionScorer) -> list[int]:
    """Top-``k`` selection by model score. Raises on any inconsistency so the
    caller can fall back to random."""
    n = _options(select)
    if n == 0:
        return []
    k = _legal_k(select, n)
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    cand = candidate_features(obs_dict)
    if len(cand) != n:
        raise ValueError(f"candidate feature count {len(cand)} != options {n}")
    scores = [model.score(c) for c in cand]
    # Highest score first; ties broken by lower option index for determinism.
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    return order[:k]


# --------------------------------------------------------------------------- #
# The agent.
# --------------------------------------------------------------------------- #
class LearnedAgent:
    """Learned-policy agent that never crashes.

    Args:
        model_path: JSON model to load (``None`` / missing → always random).
        model: a pre-built :class:`LinearOptionScorer` (overrides ``model_path``;
            handy for tests).
        seed: RNG seed for the random fallback (agent-side reproducibility).
        deck_path: deck CSV for the initial (deck) selection.
    """

    def __init__(
        self,
        model_path: Optional[str] = DEFAULT_MODEL_PATH,
        model: Optional[LinearOptionScorer] = None,
        seed: Optional[int] = None,
        deck_path: str = "deck.csv",
    ) -> None:
        self.rng = random.Random(seed)
        self.deck_path = deck_path
        self.model = model if model is not None else load_model(model_path)

    @property
    def has_model(self) -> bool:
        return self.model is not None

    def act(self, obs_dict: Any) -> list[int]:
        """Return a legal selection for ``obs_dict``; never raises.

        On the initial selection (``select is None``) returns the 60-card deck
        (Kaggle submission I/F). Otherwise scores the legal moves with the model
        and picks the best; any failure (no model, feature/scoring error, unknown
        value the model can't handle) falls back to a random legal move.
        """
        try:
            select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
            if select is None:
                try:
                    return read_deck_csv(self.deck_path)
                except Exception:
                    return []
            if self.model is not None:
                try:
                    return _model_legal(obs_dict, select, self.model)
                except Exception:
                    pass  # fall through to the random fallback
            return _random_legal(select, self.rng)
        except Exception:
            # Absolute last resort — a valid, engine-safe empty selection.
            try:
                return _random_legal((obs_dict or {}).get("select"), self.rng)
            except Exception:
                return []


# --------------------------------------------------------------------------- #
# Adapters.
# --------------------------------------------------------------------------- #
def make_learned_agent(
    model_path: Optional[str] = DEFAULT_MODEL_PATH,
    model: Optional[LinearOptionScorer] = None,
    seed: Optional[int] = None,
    name: str = "learned",
):
    """Build an ``eval.record_match.Agent`` wrapping :class:`LearnedAgent`.

    On the initial selection this returns ``[]`` (the decks are passed at
    ``battle_start`` under the ``record_match`` harness), mirroring
    :func:`eval.record_match.make_random_agent`.
    """
    from eval.record_match import Agent  # local import: engine-only dependency

    la = LearnedAgent(model_path=model_path, model=model, seed=seed)

    def fn(obs_dict: dict) -> list[int]:
        select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        if select is None:
            return []
        return la.act(obs_dict)

    return Agent(
        fn,
        name=name,
        version="1",
        params={"model_path": model_path, "has_model": la.has_model, "seed": seed},
    )


# Module-level submission entry point (a learned drop-in for main.py's `agent`).
_DEFAULT_AGENT: Optional[LearnedAgent] = None


def agent(obs_dict: dict) -> list[int]:
    """Kaggle submission entry point using the learned policy (random fallback)."""
    global _DEFAULT_AGENT
    if _DEFAULT_AGENT is None:
        _DEFAULT_AGENT = LearnedAgent()
    return _DEFAULT_AGENT.act(obs_dict)
