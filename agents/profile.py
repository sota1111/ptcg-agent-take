"""Validated production profile for the promoted Take runtime (SOT-1869)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    strategy: str
    risk_profile: str
    adaptation_weight: float
    risk_floor: float
    risk_ceiling: float
    search_budget_ms: int
    max_depth: int
    max_branching_for_extension: int
    illegal_action_fallback: str


def load_promoted_profile(path: str | None = None) -> RuntimeProfile:
    """Load and validate the bundled profile; fail closed on malformed tuning."""
    path = path or os.path.join(os.path.dirname(__file__), "promoted_profile.json")
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schemaVersion") != "ptcg-take-runtime-profile/v1":
        raise ValueError("unsupported Take runtime profile schema")
    floor = float(raw["riskFloor"])
    ceiling = float(raw["riskCeiling"])
    weight = float(raw["adaptationWeight"])
    depth = int(raw["maxDepth"])
    branching = int(raw["maxBranchingForExtension"])
    budget = int(raw["searchBudgetMs"])
    if not (0.0 <= floor <= ceiling <= 1.0 and 0.0 <= weight <= 1.0):
        raise ValueError("profile weights and risk bounds must be normalized")
    if not (1 <= depth <= 3 and 1 <= branching <= 32 and 1 <= budget <= 600_000):
        raise ValueError("profile search limits are outside runtime-safe bounds")
    fallback = str(raw["illegalActionFallback"])
    if fallback != "highest-value-legal":
        raise ValueError("unsupported illegal-action fallback")
    return RuntimeProfile(
        profile_id=str(raw["id"]), strategy=str(raw["strategy"]),
        risk_profile=str(raw["riskProfile"]), adaptation_weight=weight,
        risk_floor=floor, risk_ceiling=ceiling, search_budget_ms=budget,
        max_depth=depth, max_branching_for_extension=branching,
        illegal_action_fallback=fallback,
    )
