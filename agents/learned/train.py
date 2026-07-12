"""Policy-model training via winner-move imitation learning (SOT-1643).

Trains the small **linear option-scorer** that :mod:`agents.learned.agent`
(SOT-1644) consumes at inference time, using **behavioral cloning of the
winner's moves**: from a self-play dataset (SOT-1642) every decision made by the
*winning* player becomes a training instance whose legal options are labelled
``1`` for the option the winner actually chose and ``0`` for every other option.
A per-option logistic regression then learns to score the chosen option above
the rest — exactly how :class:`agents.learned.agent.LinearOptionScorer` picks its
move (score every option, take the top-``k``).

Why imitate a *strong* policy's winners
----------------------------------------
In ``random`` vs ``random`` self-play the winner's move is itself uniform among
the legal options, so nothing in the option features predicts it — imitation has
no signal to learn and cannot beat the random baseline. The dataset is therefore
generated (by default) from **rule-based self-play**: the winner's moves reflect
the rule policy, which *is* a function of the option features, so cloning it
produces a model whose held-out choice-agreement clears the random baseline.

Inference-only model format (dependency-free)
---------------------------------------------
The trained model is written with :func:`agents.learned.agent.save_model` in the
JSON format documented on :class:`~agents.learned.agent.LinearOptionScorer`
(``format="ptcg-learned-policy"``, ``kind="linear"``, ``weights`` / ``bias`` /
``mean`` / ``std``). Reading it back needs only the standard library — no numpy /
scikit-learn — so the model bundles with a submission. **All learning-time work
here is pure Python too** (no third-party deps), honouring the repo's
zero-pip-deps policy; the ``mean`` / ``std`` standardisation computed on the
training split is stored in the model so inference standardises identically.

Reported metric
---------------
On a decision-level held-out split, for each decision the model scores every
legal option and takes the top-``k`` (``k`` = how many the winner chose); the
**agreement** is the overlap fraction ``|predicted ∩ winner| / k``. The
**random baseline** for the same decision is ``k / n`` (the expected overlap of a
random top-``k`` pick among ``n`` options — the multi-select generalisation of
``1 / n_options``). Training passes the acceptance bar when the mean model
agreement exceeds the mean random baseline.

Usage
-----
Run from the repo root (after ``scripts/setup_engine.sh`` has populated ``cg/``)::

    venv/bin/python agents/learned/train.py            # auto-generates data, trains, saves
    venv/bin/python agents/learned/train.py --data agents/learned/data/selfplay.jsonl
    venv/bin/python agents/learned/train.py --n 200 --agent0 rule_based --agent1 rule_based

No third-party dependencies (matches the repo's zero-pip-deps policy).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Iterable, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)  # make `cg`, `eval`, `agents` importable
os.chdir(REPO)            # so libcg.so & deck.csv resolve

from agents.learned.agent import LinearOptionScorer, save_model  # noqa: E402
from agents.learned.features import OPTION_FEATURE_DIM, option_features  # noqa: E402
from agents.learned import generate_data as gd  # noqa: E402

# Where the trained, inference-ready model is written (loaded by LearnedAgent).
DEFAULT_MODEL_OUT = "agents/learned/model/policy.json"
DEFAULT_DATA = gd.DEFAULT_OUT

# Bump when the training procedure / model contents change meaningfully.
TRAIN_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Dataset → labelled decisions.
# --------------------------------------------------------------------------- #
class Decision:
    """One winner decision turned into labelled per-option feature vectors.

    Attributes:
        features: one ``option_features`` vector per legal option.
        chosen: set of option indices the winner actually selected.
        n: number of legal options (``len(features)``).
    """

    __slots__ = ("features", "chosen", "n")

    def __init__(self, features: list[list[float]], chosen: set[int]) -> None:
        self.features = features
        self.chosen = chosen
        self.n = len(features)


def _valid_choice(choice: Any, n: int) -> Optional[set[int]]:
    """Return the chosen-index set if it is a usable label, else ``None``.

    Guards against malformed samples: the choice must be a non-empty list of
    distinct in-range integers that does not cover *every* option (a decision
    where the winner "chose" all options carries no ranking signal).
    """
    if not isinstance(choice, (list, tuple)) or not choice:
        return None
    idxs: set[int] = set()
    for c in choice:
        if isinstance(c, bool) or not isinstance(c, int):
            return None
        if not (0 <= c < n):
            return None
        idxs.add(c)
    if len(idxs) != len(choice):
        return None  # duplicate indices — malformed
    if len(idxs) >= n:
        return None  # all options chosen — nothing to rank against
    return idxs


def decisions_from_dataset(
    path: str, *, winners_only: bool = True, min_options: int = 2
) -> list[Decision]:
    """Load winner decisions from a self-play JSONL dataset (SOT-1642).

    Keeps a sample when it has a real choice among ``>= min_options`` legal
    options and — when ``winners_only`` — the deciding player won (``win == 1.0``).
    The per-option feature vectors come from the SOT-1641 featuriser, which never
    raises; a sample whose featurised option count disagrees with its recorded
    ``n_options`` (or whose choice is malformed) is skipped, not fatal.
    """
    out: list[Decision] = []
    for sample in gd.iter_samples(path):
        if winners_only and sample.get("win") != 1.0:
            continue
        n_opts = sample.get("n_options")
        if not isinstance(n_opts, int) or n_opts < min_options:
            continue
        fd = gd.featurize_sample(sample)
        feats = fd.candidates
        if len(feats) != n_opts:
            continue  # featuriser / recorded option count disagree — skip
        chosen = _valid_choice(sample.get("choice"), n_opts)
        if chosen is None:
            continue
        out.append(Decision(feats, chosen))
    return out


# --------------------------------------------------------------------------- #
# Standardisation (pure Python).
# --------------------------------------------------------------------------- #
def fit_standardiser(rows: Iterable[list[float]], dim: int) -> tuple[list[float], list[float]]:
    """Feature-wise mean/std over ``rows``; a zero std is stored as ``1.0``.

    Storing 1.0 for a constant feature keeps ``(x - mean) / std`` finite and
    matches :class:`LinearOptionScorer`'s own zero-std guard, so training and
    inference standardise byte-for-byte identically.
    """
    n = 0
    mean = [0.0] * dim
    m2 = [0.0] * dim
    for row in rows:  # Welford's online mean/variance, one pass.
        n += 1
        for i in range(dim):
            d = row[i] - mean[i]
            mean[i] += d / n
            m2[i] += d * (row[i] - mean[i])
    if n < 2:
        return mean, [1.0] * dim
    std = [math.sqrt(m2[i] / (n - 1)) for i in range(dim)]
    std = [s if s > 1e-12 else 1.0 for s in std]
    return mean, std


def _standardise(x: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(x[i] - mean[i]) / std[i] for i in range(len(x))]


# --------------------------------------------------------------------------- #
# Logistic-regression trainer (pure Python, full-batch gradient descent).
# --------------------------------------------------------------------------- #
def _sigmoid(z: float) -> float:
    # Numerically stable logistic.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def train_logreg(
    decisions: list[Decision],
    mean: list[float],
    std: list[float],
    *,
    dim: int,
    epochs: int = 300,
    lr: float = 0.5,
    l2: float = 1e-4,
    seed: int = 0,
) -> tuple[list[float], float]:
    """Fit ``w``/``b`` on per-option (chosen vs not) labels via balanced logistic
    regression.

    Every legal option across the training decisions is one example: label ``1``
    for a winner-chosen option, ``0`` otherwise. Positive/negative example weights
    are class-balanced (the pooled positive weight equals the pooled negative
    weight) so the ~1-of-``n`` positive rate does not collapse the model to
    "always negative". Uses full-batch gradient descent with L2 regularisation.
    """
    # Pre-standardise every option vector once.
    examples: list[tuple[list[float], float]] = []
    pos = 0
    for dec in decisions:
        for i, feat in enumerate(dec.features):
            examples.append((_standardise(feat, mean, std), 1.0 if i in dec.chosen else 0.0))
            pos += 1 if i in dec.chosen else 0
    total = len(examples)
    neg = total - pos
    if total == 0 or pos == 0 or neg == 0:
        # Degenerate — no separable signal; return a zero model (agent still
        # scores, but uniformly; caller reports the empty-training case).
        return [0.0] * dim, 0.0

    # Class-balanced weights: each class contributes total weight `total/2`.
    w_pos = total / (2.0 * pos)
    w_neg = total / (2.0 * neg)

    rng = random.Random(seed)
    w = [0.0] * dim
    b = 0.0
    order = list(range(total))
    for _ in range(epochs):
        rng.shuffle(order)  # order-independence for the (deterministic) full-batch step
        gw = [0.0] * dim
        gb = 0.0
        wsum = 0.0
        for idx in order:
            x, y = examples[idx]
            sw = w_pos if y == 1.0 else w_neg
            z = b
            for j in range(dim):
                z += w[j] * x[j]
            err = (_sigmoid(z) - y) * sw
            for j in range(dim):
                gw[j] += err * x[j]
            gb += err
            wsum += sw
        inv = 1.0 / wsum
        for j in range(dim):
            w[j] -= lr * (gw[j] * inv + l2 * w[j])
        b -= lr * (gb * inv)
    return w, b


# --------------------------------------------------------------------------- #
# Evaluation: held-out top-k choice agreement vs the random baseline.
# --------------------------------------------------------------------------- #
def _topk(scores: list[float], k: int) -> list[int]:
    """Indices of the top-``k`` scores; ties broken by lower index (matches
    :func:`agents.learned.agent._model_legal`)."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def evaluate(
    decisions: list[Decision],
    model: LinearOptionScorer,
) -> dict:
    """Mean top-``k`` agreement of ``model`` vs the random baseline on ``decisions``.

    For each decision: ``k = |chosen|``; the model's top-``k`` overlap fraction
    with the winner's choice is the agreement; the random baseline is ``k / n``.
    ``top1`` additionally reports how often the single highest-scored option is a
    winner-chosen one.
    """
    if not decisions:
        return {"decisions": 0, "model_agreement": 0.0, "random_baseline": 0.0,
                "top1_accuracy": 0.0, "beats_random": False}
    agree_sum = 0.0
    base_sum = 0.0
    top1_hits = 0
    for dec in decisions:
        k = len(dec.chosen)
        scores = [model.score(f) for f in dec.features]
        pred = set(_topk(scores, k))
        agree_sum += len(pred & dec.chosen) / k
        base_sum += k / dec.n
        best = _topk(scores, 1)[0]
        top1_hits += 1 if best in dec.chosen else 0
    n = len(decisions)
    model_ag = agree_sum / n
    base = base_sum / n
    return {
        "decisions": n,
        "model_agreement": round(model_ag, 6),
        "random_baseline": round(base, 6),
        "top1_accuracy": round(top1_hits / n, 6),
        "beats_random": model_ag > base,
    }


# --------------------------------------------------------------------------- #
# Train/holdout split (decision-level, seeded).
# --------------------------------------------------------------------------- #
def split_decisions(
    decisions: list[Decision], holdout: float, seed: int
) -> tuple[list[Decision], list[Decision]]:
    """Shuffle and split at the *decision* level so a decision's options never
    straddle the train/holdout boundary."""
    rng = random.Random(seed)
    idx = list(range(len(decisions)))
    rng.shuffle(idx)
    n_hold = int(round(len(decisions) * holdout))
    n_hold = min(max(n_hold, 1 if len(decisions) > 1 else 0), max(0, len(decisions) - 1))
    hold_ids = set(idx[:n_hold])
    train = [decisions[i] for i in idx[n_hold:]]
    hold = [decisions[i] for i in idx[:n_hold]]
    return train, hold


# --------------------------------------------------------------------------- #
# End-to-end training driver.
# --------------------------------------------------------------------------- #
def train(
    *,
    data_path: str = DEFAULT_DATA,
    model_out: str = DEFAULT_MODEL_OUT,
    n_matches: int = 120,
    agent0: str = "rule_based",
    agent1: str = "rule_based",
    gen_seed: int = 42,
    regenerate: bool = False,
    holdout: float = 0.2,
    split_seed: int = 0,
    epochs: int = 300,
    lr: float = 0.5,
    l2: float = 1e-4,
    winners_only: bool = True,
) -> dict:
    """Generate (if needed) → clone winner moves → evaluate → save the model.

    Returns a report dict with dataset / split sizes, the held-out agreement vs
    random baseline, and the model path. Raises ``ValueError`` only when the
    dataset yields no usable winner decision.
    """
    t0 = time.perf_counter()
    gen_stats: Optional[dict] = None
    if regenerate or not os.path.exists(data_path):
        gen_stats = gd.generate(
            n_matches, out_path=data_path,
            agent0=agent0, agent1=agent1, seed=gen_seed,
        )

    decisions = decisions_from_dataset(data_path, winners_only=winners_only)
    if not decisions:
        raise ValueError(
            f"no usable winner decisions in {data_path!r} "
            f"(need win==1.0 samples with >=2 options and a valid choice)"
        )

    train_dec, hold_dec = split_decisions(decisions, holdout, split_seed)

    # Standardise on the *training* options only (no holdout leakage).
    mean, std = fit_standardiser(
        (f for d in train_dec for f in d.features), OPTION_FEATURE_DIM
    )
    w, b = train_logreg(
        train_dec, mean, std, dim=OPTION_FEATURE_DIM,
        epochs=epochs, lr=lr, l2=l2, seed=split_seed,
    )
    model = LinearOptionScorer(weights=w, bias=b, mean=mean, std=std)

    train_eval = evaluate(train_dec, model)
    hold_eval = evaluate(hold_dec, model)

    path = save_model(model, model_out)
    size = os.path.getsize(path)

    return {
        "train_version": TRAIN_VERSION,
        "data_path": data_path,
        "regenerated": gen_stats is not None,
        "gen_stats": gen_stats,
        "option_feature_dim": OPTION_FEATURE_DIM,
        "n_decisions": len(decisions),
        "n_train": len(train_dec),
        "n_holdout": len(hold_dec),
        "winners_only": winners_only,
        "train_eval": train_eval,
        "holdout_eval": hold_eval,
        "model_path": path,
        "model_bytes": size,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the policy model by winner-move imitation (SOT-1643).")
    p.add_argument("--data", default=DEFAULT_DATA, help="self-play JSONL dataset (SOT-1642)")
    p.add_argument("--out", default=DEFAULT_MODEL_OUT, help="output model JSON path")
    p.add_argument("--n", type=int, default=120, help="matches to generate if data is absent/--regenerate")
    p.add_argument("--agent0", default="rule_based", choices=sorted(gd.AGENT_FACTORIES))
    p.add_argument("--agent1", default="rule_based", choices=sorted(gd.AGENT_FACTORIES))
    p.add_argument("--gen-seed", type=int, default=42, help="data-generation base seed")
    p.add_argument("--regenerate", action="store_true", help="regenerate the dataset even if it exists")
    p.add_argument("--holdout", type=float, default=0.2, help="held-out fraction (decision-level)")
    p.add_argument("--split-seed", type=int, default=0, help="train/holdout split + trainer seed")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--all-actors", action="store_true",
                   help="clone every decision, not just the winner's (default: winners only)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    report = train(
        data_path=args.data, model_out=args.out, n_matches=args.n,
        agent0=args.agent0, agent1=args.agent1, gen_seed=args.gen_seed,
        regenerate=args.regenerate, holdout=args.holdout, split_seed=args.split_seed,
        epochs=args.epochs, lr=args.lr, l2=args.l2, winners_only=not args.all_actors,
    )
    he = report["holdout_eval"]
    print(
        f"TRAIN DONE: decisions={report['n_decisions']}"
        f" train={report['n_train']} holdout={report['n_holdout']}"
        f" | holdout agreement={he['model_agreement']:.4f}"
        f" vs random={he['random_baseline']:.4f}"
        f" top1={he['top1_accuracy']:.4f} beats_random={he['beats_random']}"
        f" | model={report['model_path']} ({report['model_bytes']} bytes)"
        f" elapsed_ms={report['elapsed_ms']}"
    )
    # Non-zero exit if the trained model fails to beat the random baseline on the
    # holdout, so a CI/gate can catch a regression in the learning signal.
    return 0 if he["beats_random"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
