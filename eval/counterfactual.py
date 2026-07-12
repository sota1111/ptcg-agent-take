"""Counterfactual analysis of recorded positions via the engine search API (SOT-1622).

"What if I had played a different move?" — the final stage of decision auditing (E5).

The cabt engine exposes ``search_begin`` / ``search_step`` / ``search_end`` /
``search_release`` (``cg/api.py``). Keyed by the ``search_begin_input`` string that
``eval/record_match.py`` stores in every ``decision`` record, we can **reconstruct any
recorded position** and roll it forward under a chosen first move. This module compares
the *actual* choice against one or more *alternative* options from the same position and
reports simple outcome indicators (winner / prize差 / terminal reason).

Two moving parts are pluggable:

* **Hidden information** — ``search_begin`` needs a full prediction of everything you
  can't see (your own deck/prize, the opponent's deck/prize/hand, a face-down active).
  A :class:`HiddenInfoPredictor` supplies it; the default :class:`UniformDeckPredictor`
  uniformly samples the cards *not visible on the board* from the known 60-card deck
  composition recorded in the trace ``meta``. Swap in any predictor for stronger priors.
* **Rollout policy** — how both branches choose their subsequent moves. The default
  :class:`RandomPolicy` is seeded (reproducible). With ``manual_coin=True`` the policy
  fixes every coin flip (default: heads), so luck is held constant for a fair A/B and the
  whole rollout is deterministic given a seed — same input → same 展開 (a check the issue
  requires).

Reconstruction needs the **full observation** at the decision, so the trace must be
recorded at ``--level full_obs`` (``eval/record_match.py``). A LOGS-only trace omits the
full ``State`` and cannot be replayed; this tool raises a clear error in that case.

The search session is **always released** (``search_release`` + ``search_end``) via
try/finally, even if the rollout raises — no leaked search state (acceptance criterion).

Usage (run from the repo root, after scripts/setup_engine.sh has populated cg/):
    venv/bin/python eval/counterfactual.py TRACE.jsonl --decision N
                    [--alt-option K ...] [--alt-selection "i,j"]
                    [--seed S] [--max-depth D] [--rollouts R]
                    [--manual-coin 0|1] [--coin heads|tails|random] [--json]

As a library:
    from eval.counterfactual import analyze_decision, UniformDeckPredictor
    report = analyze_decision(trace_or_path, decision_index=35)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # make `cg` and `eval` importable

# Pure trace helpers (no engine import at module load) — reused from replay.py.
from eval.replay import _choice_indices, load_trace, split_records  # noqa: E402

# Engine constants we key off of (kept local so the predictor stays engine-free).
_COIN_HEAD_CONTEXT = 46   # SelectContext.COIN_HEAD (a YES_NO coin flip)
_OPTION_YES = 1           # OptionType.YES
_OPTION_NO = 2            # OptionType.NO
_LOG_RESULT = 23          # LogType.RESULT
_PRIZE_TOTAL = 6          # prize cards per player at game start


# --------------------------------------------------------------------------- #
# Hidden-information prediction (pluggable)
# --------------------------------------------------------------------------- #

@dataclass
class HiddenInfo:
    """The six hidden-card predictions ``search_begin`` requires.

    Each list is the predicted Card IDs for a hidden zone; lengths must cover the
    counts in the observation (``search_begin`` takes as many as it needs).
    ``opponent_active`` is non-empty only when the opponent's Active Spot is
    face-down (``active[0] is None``).
    """

    your_deck: list[int] = field(default_factory=list)
    your_prize: list[int] = field(default_factory=list)
    opponent_deck: list[int] = field(default_factory=list)
    opponent_prize: list[int] = field(default_factory=list)
    opponent_hand: list[int] = field(default_factory=list)
    opponent_active: list[int] = field(default_factory=list)

    def as_search_args(self) -> tuple:
        return (
            self.your_deck,
            self.your_prize,
            self.opponent_deck,
            self.opponent_prize,
            self.opponent_hand,
            self.opponent_active,
        )


class HiddenInfoPredictor(ABC):
    """Supplies the hidden-card predictions for a recorded position.

    Swap the default out for a stronger prior (deck-tracking, opponent modelling …);
    ``analyze_decision`` only depends on this interface.
    """

    @abstractmethod
    def predict(self, obs: dict, meta: dict, *, your_index: int) -> HiddenInfo:
        """Return a :class:`HiddenInfo` for ``obs`` (the full observation dict)."""
        raise NotImplementedError


def _visible_card_ids(player: dict, *, include_hand: bool) -> list[int]:
    """Card IDs currently visible on the board for one player.

    Counts Pokémon in play (active/bench) with their attached energy/tool/pre-evolution
    cards, the discard pile, revealed prize cards, and — only for the player whose hand
    we can see — the hand. These are subtracted from the deck to form the hidden pool.
    """
    ids: list[int] = []

    def add_pokemon(pk: Optional[dict]) -> None:
        if not pk:
            return
        ids.append(pk["id"])
        for group in ("energyCards", "tools", "preEvolution"):
            for card in pk.get(group) or []:
                ids.append(card["id"])

    for pk in player.get("active") or []:
        add_pokemon(pk)
    for pk in player.get("bench") or []:
        add_pokemon(pk)
    for card in player.get("discard") or []:
        ids.append(card["id"])
    for card in player.get("prize") or []:
        if card is not None:           # revealed prize
            ids.append(card["id"])
    if include_hand and player.get("hand") is not None:
        for card in player["hand"]:
            ids.append(card["id"])
    return ids


class UniformDeckPredictor(HiddenInfoPredictor):
    """Uniformly sample hidden cards from the deck cards not visible on the board.

    Uses the known 60-card deck composition recorded in ``meta['decks']``: for each
    player, ``hidden pool = deck multiset − cards visible on the board``, then draws
    (without replacement) to fill the hidden zones. If the pool falls short of the
    required count (double-counted board card, unusual mid-game state) it tops up from
    the full deck list as a best-effort fallback so ``search_begin`` never fails on a
    length check.

    Seeded → reproducible. This is the simple default the issue calls for; replace it
    with a smarter predictor via the :class:`HiddenInfoPredictor` interface.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)

    def _pool(self, deck: Sequence[int], player: dict, *, include_hand: bool) -> list[int]:
        remaining = collections.Counter(deck)
        for cid in _visible_card_ids(player, include_hand=include_hand):
            remaining[cid] -= 1
        pool: list[int] = []
        for cid, n in remaining.items():
            if n > 0:
                pool.extend([cid] * n)
        self.rng.shuffle(pool)
        return pool

    @staticmethod
    def _take(pool: list[int], n: int, fallback: Sequence[int]) -> list[int]:
        """Pop ``n`` cards off ``pool``; top up from ``fallback`` if the pool is short."""
        if n <= 0:
            return []
        out = pool[:n]
        del pool[:n]
        i = 0
        while len(out) < n and fallback:
            out.append(fallback[i % len(fallback)])
            i += 1
        return out

    def predict(self, obs: dict, meta: dict, *, your_index: int) -> HiddenInfo:
        decks = (meta or {}).get("decks")
        if not decks or len(decks) < 2:
            raise ValueError(
                "UniformDeckPredictor needs meta['decks'] (both deck lists) to sample "
                "hidden cards; supply a custom HiddenInfoPredictor for this trace."
            )
        state = obs["current"]
        players = state["players"]
        you = players[your_index]
        opp_index = 1 - your_index
        opp = players[opp_index]
        your_deck_list = decks[your_index]
        opp_deck_list = decks[opp_index]

        # Your side: hand is visible, so subtract it; deck + face-down prize are hidden.
        your_pool = self._pool(your_deck_list, you, include_hand=True)
        your_deck = self._take(your_pool, you["deckCount"], your_deck_list)
        your_prize = self._take(your_pool, len(you["prize"]), your_deck_list)

        # Opponent side: hand is hidden too, so don't subtract it from the pool.
        opp_pool = self._pool(opp_deck_list, opp, include_hand=False)
        opponent_deck = self._take(opp_pool, opp["deckCount"], opp_deck_list)
        opponent_prize = self._take(opp_pool, len(opp["prize"]), opp_deck_list)
        opponent_hand = self._take(opp_pool, opp["handCount"], opp_deck_list)
        active = opp.get("active") or []
        active_facedown = len(active) > 0 and active[0] is None
        opponent_active = self._take(opp_pool, 1, opp_deck_list) if active_facedown else []

        return HiddenInfo(
            your_deck=your_deck,
            your_prize=your_prize,
            opponent_deck=opponent_deck,
            opponent_prize=opponent_prize,
            opponent_hand=opponent_hand,
            opponent_active=opponent_active,
        )


# --------------------------------------------------------------------------- #
# Rollout policy (pluggable)
# --------------------------------------------------------------------------- #

# A policy maps a SelectData dict (the pending selection) to a list of option indices.
Policy = Callable[[dict], list[int]]


class RandomPolicy:
    """Seeded uniform-random legal-move policy for rolling a position forward.

    With ``coin`` set ('heads'/'tails') a coin-flip selection (context COIN_HEAD) is
    forced to that face instead of sampled, so ``manual_coin`` rollouts hold luck fixed.
    Given the same seed and hidden info the whole rollout is deterministic.
    """

    def __init__(self, seed: Optional[int] = None, coin: str = "heads"):
        self.rng = random.Random(seed)
        self.coin = coin

    def __call__(self, select: dict) -> list[int]:
        options = select.get("option") or []
        n = len(options)
        if n == 0:
            return []
        if self.coin in ("heads", "tails") and select.get("context") == _COIN_HEAD_CONTEXT:
            want = _OPTION_YES if self.coin == "heads" else _OPTION_NO
            for i, opt in enumerate(options):
                if opt.get("type") == want:
                    return [i]
            # fall through to random if the expected option is absent
        k = max(select.get("minCount", 1), min(select.get("maxCount", 1), n))
        return self.rng.sample(range(n), k)


# --------------------------------------------------------------------------- #
# Rollout + metrics
# --------------------------------------------------------------------------- #

@dataclass
class RolloutResult:
    """Outcome of rolling one branch forward from the reconstructed position."""

    label: str                        # "actual" or e.g. "alt[3]"
    selection: list[int]              # the first move taken from the position
    terminal: bool                    # did the match reach a result?
    truncated: bool                   # hit max_depth without a result
    depth: int                        # search_step calls made after the first move
    result: Optional[int]             # engine result: 0/1 winner, 2 draw, -1 undecided
    winner: Optional[int]             # player index that won (None if draw/undecided)
    reason: Optional[int]             # RESULT-log reason (1-4) if terminal
    prize_remaining: Optional[list[int]]  # [p0, p1] prize cards still unclaimed
    error: Optional[str] = None       # set if the branch raised (search still released)


def _result_reason(logs: list) -> Optional[int]:
    for log in reversed(logs or []):
        if isinstance(log, dict) and log.get("type") == _LOG_RESULT:
            return log.get("reason")
    return None


def _rollout_branch(
    obs: dict,
    hidden: HiddenInfo,
    first_move: list[int],
    *,
    label: str,
    policy: Policy,
    max_depth: int,
    manual_coin: bool,
) -> RolloutResult:
    """Reconstruct the position, apply ``first_move``, then roll out under ``policy``.

    The search session (search_begin) is ALWAYS torn down (search_release + search_end)
    via try/finally, even if reconstruction or a step raises (E7 / acceptance).
    """
    from cg.api import (  # imported here so the predictor stays engine-free
        search_begin,
        search_end,
        search_release,
        search_step,
        to_observation_class,
    )

    observation = to_observation_class(obs)
    search_id: Optional[int] = None
    started = False
    depth = 0
    state = None
    error: Optional[str] = None
    try:
        root = search_begin(observation, *hidden.as_search_args(), manual_coin=manual_coin)
        started = True
        search_id = root.searchId
        state = search_step(search_id, list(first_move))
        for _ in range(max_depth):
            select = state.observation.select
            if select is None:  # position resolved
                break
            move = policy(_select_to_dict(select))
            state = search_step(search_id, move)
            depth += 1
    except Exception as exc:  # pragma: no cover - defensive; still cleans up below
        error = repr(exc)
    finally:
        if started and search_id is not None:
            try:
                search_release(search_id)
            except Exception:
                pass
        if started:
            try:
                search_end()
            except Exception:
                pass

    return _finish_result(label, first_move, state, depth, error)


def _select_to_dict(select: Any) -> dict:
    """Best-effort SelectData → dict (RandomPolicy reads option/context/min/max)."""
    if isinstance(select, dict):
        return select
    options = []
    for opt in getattr(select, "option", None) or []:
        if isinstance(opt, dict):
            options.append(opt)
        else:
            options.append({"type": getattr(opt, "type", None)})
    return {
        "option": options,
        "context": getattr(select, "context", None),
        "minCount": getattr(select, "minCount", 1),
        "maxCount": getattr(select, "maxCount", 1),
    }


def _finish_result(label, first_move, state, depth, error) -> RolloutResult:
    if state is None:
        return RolloutResult(
            label=label, selection=list(first_move), terminal=False, truncated=False,
            depth=depth, result=None, winner=None, reason=None, prize_remaining=None,
            error=error,
        )
    current = state.observation.current
    select = state.observation.select
    result = current.result if current is not None else -1
    terminal = select is None or (current is not None and result != -1)
    winner = result if result in (0, 1) else None
    prize = (
        [len(current.players[p].prize) for p in (0, 1)]
        if current is not None else None
    )
    reason = _result_reason(getattr(state.observation, "logs", None)) if terminal else None
    return RolloutResult(
        label=label,
        selection=list(first_move),
        terminal=terminal,
        truncated=not terminal and error is None,
        depth=depth,
        result=result,
        winner=winner,
        reason=reason,
        prize_remaining=prize,
        error=error,
    )


# --------------------------------------------------------------------------- #
# Decision selection helpers
# --------------------------------------------------------------------------- #

def find_decision(trace: dict, decision_index: int) -> dict:
    """Return the ``decision`` record with the given index, or raise ValueError."""
    for dec in trace.get("decisions") or []:
        if dec.get("index") == decision_index:
            return dec
    available = [d.get("index") for d in trace.get("decisions") or []]
    raise ValueError(
        f"decision index {decision_index} not found in trace "
        f"(available: {available[:1]}..{available[-1:]}, n={len(available)})"
    )


def _require_full_obs(decision: dict, decision_index: int) -> dict:
    obs = decision.get("obs")
    if not obs or "current" not in obs or obs.get("search_begin_input") is None:
        raise ValueError(
            f"decision {decision_index} has no full observation with search_begin_input; "
            "record the trace at --level full_obs so positions can be reconstructed."
        )
    return obs


def default_alternatives(obs: dict, actual: list[int]) -> list[list[int]]:
    """Auto-pick alternative selections for a single-select decision.

    For a single-index decision (minCount ≤ 1 ≤ maxCount, one chosen option) every other
    legal option index becomes an alternative. Returns [] for multi-select decisions —
    the caller must then pass an explicit ``--alt-selection``.
    """
    select = obs.get("select") or {}
    options = select.get("option") or []
    min_c = select.get("minCount", 1)
    max_c = select.get("maxCount", 1)
    if len(actual) != 1 or not (min_c <= 1 <= max_c):
        return []
    return [[j] for j in range(len(options)) if [j] != actual]


# --------------------------------------------------------------------------- #
# Top-level analysis
# --------------------------------------------------------------------------- #

def analyze_decision(
    trace: "str | dict",
    decision_index: int,
    *,
    alt_selections: Optional[Sequence[Sequence[int]]] = None,
    predictor: Optional[HiddenInfoPredictor] = None,
    policy_factory: Optional[Callable[[int], Policy]] = None,
    seed: int = 0,
    max_depth: int = 400,
    manual_coin: bool = True,
    coin: str = "heads",
    rollouts: int = 1,
    max_auto_alternatives: int = 4,
) -> dict:
    """Compare the actual choice against alternatives from a recorded position.

    Args:
        trace: a trace path (JSONL) or an already-parsed ``{meta,decisions,result}`` dict.
        decision_index: which ``decision`` record to branch from.
        alt_selections: explicit alternative selections (each a list of option indices).
            If None, single-select decisions auto-enumerate other legal options (capped
            at ``max_auto_alternatives``).
        predictor: hidden-information predictor (default :class:`UniformDeckPredictor`).
        policy_factory: ``seed -> Policy`` for the rollout (default seeded RandomPolicy
            with the given ``coin``).
        seed: base RNG seed (predictor + policy). Fixed seed + manual_coin ⇒ deterministic.
        max_depth: max search_step calls after the first move before truncating.
        manual_coin: fix coin flips (luck held constant across branches).
        coin: which face to force for coin flips when manual_coin ('heads'/'tails'/'random').
        rollouts: Monte-Carlo repeats; each re-samples hidden info with a fresh seed and
            aggregates win/draw/prize stats. rollouts=1 (default) is the deterministic case.

    Returns a JSON-serializable report dict: meta about the decision, and per-branch
    ('actual' + each alternative) aggregated rollout stats.
    """
    parsed = load_trace(trace) if isinstance(trace, str) else trace
    if "decisions" not in parsed:  # a raw record list
        parsed = split_records(parsed)  # type: ignore[arg-type]
    meta = parsed.get("meta") or {}

    decision = find_decision(parsed, decision_index)
    obs = _require_full_obs(decision, decision_index)
    your_index = obs["current"]["yourIndex"]
    actual = _choice_indices(decision.get("choice"))

    if alt_selections is None:
        alts = default_alternatives(obs, actual)[:max_auto_alternatives]
        if not alts:
            raise ValueError(
                f"decision {decision_index} is a multi-select "
                f"(choice={decision.get('choice')}); pass explicit alt_selections."
            )
    else:
        alts = [list(s) for s in alt_selections]

    branches: list[tuple[str, list[int]]] = [("actual", actual)]
    branches += [(f"alt{s}", list(s)) for s in alts]

    if predictor is None:
        predictor_factory: Callable[[int], HiddenInfoPredictor] = lambda s: UniformDeckPredictor(s)
    else:
        predictor_factory = lambda s: predictor  # caller-supplied, reused across rollouts
    if policy_factory is None:
        policy_factory = lambda s: RandomPolicy(s, coin=coin)

    branch_reports: list[dict] = []
    for label, first_move in branches:
        runs: list[RolloutResult] = []
        for r in range(max(1, rollouts)):
            run_seed = seed + r * 1_000_003  # distinct per rollout, deterministic
            hidden = predictor_factory(run_seed).predict(obs, meta, your_index=your_index)
            policy = policy_factory(run_seed)
            res = _rollout_branch(
                obs, hidden, first_move,
                label=label, policy=policy, max_depth=max_depth, manual_coin=manual_coin,
            )
            runs.append(res)
        branch_reports.append(_aggregate_branch(label, first_move, runs, your_index))

    return {
        "trace_id": meta.get("trace_id"),
        "schema_version": meta.get("schema_version"),
        "decision_index": decision_index,
        "your_index": your_index,
        "turn": decision.get("turn"),
        "select_type": (obs.get("select") or {}).get("type"),
        "select_context": (obs.get("select") or {}).get("context"),
        "n_options": len((obs.get("select") or {}).get("option") or []),
        "actual_choice": actual,
        "rollouts": max(1, rollouts),
        "max_depth": max_depth,
        "manual_coin": manual_coin,
        "coin": coin,
        "seed": seed,
        "branches": branch_reports,
    }


def _aggregate_branch(
    label: str, first_move: list[int], runs: list[RolloutResult], your_index: int
) -> dict:
    n = len(runs)
    terminal = [r for r in runs if r.terminal]
    your_wins = sum(1 for r in terminal if r.winner == your_index)
    opp_wins = sum(1 for r in terminal if r.winner == (1 - your_index))
    draws = sum(1 for r in terminal if r.result == 2)
    truncated = sum(1 for r in runs if r.truncated)
    errors = sum(1 for r in runs if r.error)

    # Prize差 from your_index's view: opponent remaining − your remaining
    # (positive ⇒ you claimed more prizes / are ahead). Averaged over rollouts that
    # produced a state.
    advs = [
        r.prize_remaining[1 - your_index] - r.prize_remaining[your_index]
        for r in runs if r.prize_remaining is not None
    ]
    avg_prize_adv = round(sum(advs) / len(advs), 3) if advs else None

    report = {
        "label": label,
        "selection": list(first_move),
        "n_rollouts": n,
        "terminal": len(terminal),
        "truncated": truncated,
        "errors": errors,
        "your_wins": your_wins,
        "opp_wins": opp_wins,
        "draws": draws,
        "your_win_rate": round(your_wins / n, 3) if n else None,
        "avg_prize_advantage": avg_prize_adv,
    }
    if n == 1:  # expose the single deterministic rollout in full
        report["rollout"] = asdict(runs[0])
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def format_report(report: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"Counterfactual @ decision {report['decision_index']} "
        f"(turn {report['turn']}, your_index={report['your_index']}, "
        f"{report['n_options']} options)  trace={report['trace_id']}"
    )
    lines.append(
        f"  rollouts={report['rollouts']} max_depth={report['max_depth']} "
        f"manual_coin={report['manual_coin']} coin={report['coin']} seed={report['seed']}"
    )
    lines.append(
        f"  {'branch':<16} {'sel':<10} {'term':>4} {'trunc':>5} "
        f"{'win%':>6} {'prizeAdv':>8}  outcome"
    )
    for b in report["branches"]:
        outcome = ""
        run = b.get("rollout")
        if run:
            if run["terminal"]:
                w = run["winner"]
                who = "draw" if run["result"] == 2 else (
                    "YOU" if w == report["your_index"] else "OPP" if w is not None else "?")
                outcome = f"{who} (reason={run['reason']}, prize={run['prize_remaining']})"
            elif run["error"]:
                outcome = f"ERROR {run['error']}"
            else:
                outcome = f"undecided@depth{run['depth']} prize={run['prize_remaining']}"
        else:
            outcome = f"{b['your_wins']}W-{b['opp_wins']}L-{b['draws']}D"
        wr = "-" if b["your_win_rate"] is None else f"{b['your_win_rate']:.2f}"
        pa = "-" if b["avg_prize_advantage"] is None else f"{b['avg_prize_advantage']:+.2f}"
        lines.append(
            f"  {b['label']:<16} {str(b['selection']):<10} {b['terminal']:>4} "
            f"{b['truncated']:>5} {wr:>6} {pa:>8}  {outcome}"
        )
    return "\n".join(lines)


def _parse_selection(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x != ""]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Counterfactual analysis of a recorded position via the engine search API."
    )
    p.add_argument("trace", help="path to a full_obs trace JSONL (eval/record_match.py)")
    p.add_argument("--decision", type=int, required=True, help="decision index to branch from")
    p.add_argument("--alt-option", type=int, action="append", default=None,
                   help="alternative single-option index (repeatable)")
    p.add_argument("--alt-selection", action="append", default=None,
                   help='explicit alternative selection, e.g. "0,2" (repeatable)')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=int, default=400)
    p.add_argument("--rollouts", type=int, default=1)
    p.add_argument("--manual-coin", type=int, choices=(0, 1), default=1)
    p.add_argument("--coin", choices=("heads", "tails", "random"), default="heads")
    p.add_argument("--json", action="store_true", help="emit the raw report as JSON")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    alt_selections: Optional[list[list[int]]] = None
    if args.alt_option or args.alt_selection:
        alt_selections = [[k] for k in (args.alt_option or [])]
        alt_selections += [_parse_selection(s) for s in (args.alt_selection or [])]

    report = analyze_decision(
        args.trace,
        args.decision,
        alt_selections=alt_selections,
        seed=args.seed,
        max_depth=args.max_depth,
        manual_coin=bool(args.manual_coin),
        coin=args.coin,
        rollouts=args.rollouts,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
