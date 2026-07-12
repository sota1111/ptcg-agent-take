"""Observation featurisation for the learning-based agent (SOT-1641).

Turns a raw engine observation (the ``obs_dict`` an agent's ``act`` receives) and
its legal-move candidates into **fixed-length numeric feature vectors** suitable
as model input. Two things are produced:

- an **observation vector** — global context for the current decision
  (``SelectType`` / ``SelectContext`` one-hots, selection scalars, turn / board
  state for both players); and
- one **candidate vector per legal option** — the per-move features
  (``OptionType`` one-hot, target area, and the option's numeric fields).

Design goals
------------
* **Never raises.** Every field is read defensively (missing / ``None`` /
  wrong-typed values fall back to zeros). The engine note warns that new enum
  members may be *appended during the competition*, so any unknown
  ``SelectType`` / ``SelectContext`` / ``OptionType`` / ``AreaType`` /
  ``SpecialConditionType`` integer lands in a dedicated **unknown** one-hot slot
  instead of throwing or silently colliding with a valid class.
* **Fixed length.** ``observation_features`` and ``option_features`` always emit
  exactly :data:`OBSERVATION_FEATURE_DIM` / :data:`OPTION_FEATURE_DIM` values
  regardless of input, so a batch of decisions stacks into a rectangular matrix.

No third-party dependencies (no numpy): vectors are plain ``list[float]``. A
downstream trainer (SOT-1643) can convert to whatever tensor library it uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# ---------------------------------------------------------------------------
# One-hot dimensions. Each enum one-hot reserves its LAST index as the
# "unknown value" fallback slot; the preceding indices map 1:1 to the enum's
# integer value (index i == enum value i). A ``None``/absent value yields an
# all-zero block (distinct from "unknown", which sets the fallback slot).
# ---------------------------------------------------------------------------
# SelectType: values 0..10 (11 members) + unknown -> 12.
SELECT_TYPE_DIM = 12
# SelectContext: values 0..48 (49 members) + unknown -> 50.
SELECT_CONTEXT_DIM = 50
# OptionType: values 0..16 (17 members) + unknown -> 18.
OPTION_TYPE_DIM = 18
# AreaType: values 1..12 (index 0 unused) + unknown -> 14.
AREA_TYPE_DIM = 14
# SpecialConditionType: values 0..4 (5 members) + unknown -> 6.
SPECIAL_CONDITION_DIM = 6

# Option numeric fields encoded as (present_flag, value) pairs, in this order.
_OPTION_NUMERIC_FIELDS = (
    "number",
    "index",
    "playerIndex",
    "toolIndex",
    "energyIndex",
    "count",
    "inPlayIndex",
    "attackId",
    "cardId",
    "serial",
)


# ---------------------------------------------------------------------------
# Low-level, exception-proof primitives.
# ---------------------------------------------------------------------------
def _num(value: Any) -> float:
    """Best-effort float. ``None`` / non-numeric / bool-as-flag safe -> 0.0."""
    if value is None or isinstance(value, bool):
        # Booleans are handled by _flag; treat a stray bool here as 0 to avoid
        # conflating True with the number 1 in a numeric slot by accident.
        return float(value) if isinstance(value, bool) else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _flag(value: Any) -> float:
    """Truthiness as 1.0 / 0.0, safe for any input."""
    return 1.0 if value else 0.0


def _one_hot(value: Any, size: int) -> list[float]:
    """One-hot of ``value`` into ``size`` slots; last slot = unknown fallback.

    - ``None`` -> all zeros (absent).
    - valid int in ``[0, size-1)`` -> that index set.
    - anything else (out-of-range int, non-int) -> the unknown slot (``size-1``).
    """
    vec = [0.0] * size
    if value is None:
        return vec
    try:
        iv = int(value)
    except (TypeError, ValueError):
        vec[size - 1] = 1.0
        return vec
    if 0 <= iv < size - 1:
        vec[iv] = 1.0
    else:
        vec[size - 1] = 1.0
    return vec


def _as_dict(value: Any) -> dict:
    """Return ``value`` if it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    """Return ``value`` if it is a list/tuple, else an empty list."""
    return list(value) if isinstance(value, (list, tuple)) else []


# ---------------------------------------------------------------------------
# Option (candidate) features.
# ---------------------------------------------------------------------------
def option_features(option: Any, select: Any = None) -> list[float]:
    """Fixed-length feature vector for a single legal-move option.

    ``select`` is accepted for future context-dependent option features but is
    currently unused; passing it keeps the call site stable.
    """
    opt = _as_dict(option)
    vec: list[float] = []

    # Which kind of move (PLAY / ATTACH / ATTACK / CARD / YES / NO / ...).
    vec += _one_hot(opt.get("type"), OPTION_TYPE_DIM)
    # Target board areas.
    vec += _one_hot(opt.get("area"), AREA_TYPE_DIM)
    vec += _one_hot(opt.get("inPlayArea"), AREA_TYPE_DIM)
    # Special-condition target (only set for SPECIAL_CONDITION options).
    vec += _one_hot(opt.get("specialConditionType"), SPECIAL_CONDITION_DIM)

    # Numeric fields, each as (present_flag, value). The present flag lets the
    # model distinguish "field absent" from a genuine 0 value.
    for field in _OPTION_NUMERIC_FIELDS:
        raw = opt.get(field)
        vec.append(1.0 if raw is not None else 0.0)
        vec.append(_num(raw))

    return vec


def candidate_features(obs_dict: Any) -> list[list[float]]:
    """Per-option feature vectors for every legal move in ``obs_dict``.

    Returns an empty list on the initial deck selection (``select is None``) or
    when there are no options, so callers can special-case "nothing to choose".
    """
    obs = _as_dict(obs_dict)
    select = obs.get("select")
    if not isinstance(select, dict):
        return []
    return [option_features(opt, select) for opt in _as_list(select.get("option"))]


# ---------------------------------------------------------------------------
# Observation (global context) features.
# ---------------------------------------------------------------------------
def _player_features(player: Any) -> list[float]:
    """Fixed 17-length board summary for one player's state (or zeros)."""
    p = _as_dict(player)

    active_list = _as_list(p.get("active"))
    active = active_list[0] if active_list and isinstance(active_list[0], dict) else None
    a = _as_dict(active)

    return [
        1.0 if active is not None else 0.0,          # active present
        _num(a.get("hp")),                           # active current HP
        _num(a.get("maxHp")),                        # active max HP
        float(len(_as_list(a.get("energies")))),     # attached energy units
        float(len(_as_list(a.get("energyCards")))),  # attached energy cards
        float(len(_as_list(a.get("tools")))),        # attached tools
        float(len(_as_list(p.get("bench")))),        # bench size
        _num(p.get("benchMax")),                     # bench capacity
        _num(p.get("deckCount")),                    # cards left in deck
        _num(p.get("handCount")),                    # cards in hand
        float(len(_as_list(p.get("discard")))),      # discard pile size
        float(len(_as_list(p.get("prize")))),        # prize cards remaining
        _flag(p.get("poisoned")),
        _flag(p.get("burned")),
        _flag(p.get("asleep")),
        _flag(p.get("paralyzed")),
        _flag(p.get("confused")),
    ]


def observation_features(obs_dict: Any) -> list[float]:
    """Fixed-length global feature vector for one decision.

    Encodes the pending selection (type/context/scalars) and the current board
    state for both players, ordered self-first (the selecting player, per
    ``current.yourIndex``) then opponent. Always emits
    :data:`OBSERVATION_FEATURE_DIM` values; ``None`` / partial observations (e.g.
    a LOGS-level trace decision with no full ``current``) degrade to zeros.
    """
    obs = _as_dict(obs_dict)
    select = obs.get("select")
    current = obs.get("current")
    sel = _as_dict(select)
    cur = _as_dict(current)

    vec: list[float] = []

    # --- selection block ---
    vec.append(1.0 if isinstance(select, dict) else 0.0)  # select present
    vec += _one_hot(sel.get("type"), SELECT_TYPE_DIM)
    vec += _one_hot(sel.get("context"), SELECT_CONTEXT_DIM)
    vec += [
        _num(sel.get("minCount")),
        _num(sel.get("maxCount")),
        _num(sel.get("remainDamageCounter")),
        _num(sel.get("remainEnergyCost")),
        float(len(_as_list(sel.get("option")))),   # number of legal moves
        _flag(sel.get("deck")),                     # selecting from a deck list?
        _flag(sel.get("contextCard")),
        _flag(sel.get("effect")),
    ]

    # --- current-state block ---
    vec.append(1.0 if isinstance(current, dict) else 0.0)  # current present
    your_index = cur.get("yourIndex")
    vec += [
        _num(cur.get("turn")),
        _num(cur.get("turnActionCount")),
        _num(your_index),
        _num(cur.get("firstPlayer")),
        _flag(cur.get("supporterPlayed")),
        _flag(cur.get("stadiumPlayed")),
        _flag(cur.get("energyAttached")),
        _flag(cur.get("retreated")),
        _num(cur.get("result")),
        _flag(_as_list(cur.get("stadium"))),       # a stadium is in play?
    ]

    # --- per-player board block (self first, then opponent) ---
    players = _as_list(cur.get("players"))
    self_idx = your_index if your_index in (0, 1) else 0
    opp_idx = 1 - self_idx

    def _player_at(i: int) -> Any:
        return players[i] if 0 <= i < len(players) else None

    vec += _player_features(_player_at(self_idx))
    vec += _player_features(_player_at(opp_idx))

    return vec


# ---------------------------------------------------------------------------
# Combined featurisation.
# ---------------------------------------------------------------------------
@dataclass
class FeaturizedDecision:
    """A decision turned into model-ready features.

    Attributes:
        observation: the global context vector (:data:`OBSERVATION_FEATURE_DIM`).
        candidates: one vector per legal option (each :data:`OPTION_FEATURE_DIM`);
            empty on the initial deck selection.
        n_options: number of legal options (``len(candidates)``).
    """

    observation: list[float]
    candidates: list[list[float]]
    n_options: int


def featurize(obs_dict: Any) -> FeaturizedDecision:
    """Featurise a full decision: observation vector + per-candidate vectors."""
    observation = observation_features(obs_dict)
    candidates = candidate_features(obs_dict)
    return FeaturizedDecision(
        observation=observation,
        candidates=candidates,
        n_options=len(candidates),
    )


# ---------------------------------------------------------------------------
# Feature dimensions, derived once from a reference featurisation so the
# published constants can never drift from the actual output length.
# ---------------------------------------------------------------------------
OBSERVATION_FEATURE_DIM: int = len(observation_features(None))
OPTION_FEATURE_DIM: int = len(option_features(None))
