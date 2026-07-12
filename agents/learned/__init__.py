"""Learning-based agent scaffolding (SOT-1639 案D).

This package is the foundation for the learning-based agent line. It currently
holds only the observation featuriser (SOT-1641); the self-play data pipeline
(SOT-1642), policy learning (SOT-1643) and the inference agent (SOT-1644) build
on top of it.

``features`` turns a raw engine observation (``obs_dict``) and its legal-move
candidates into fixed-length numeric feature vectors, tolerating unknown
``SelectType`` / ``SelectContext`` / enum values via dedicated fallback slots so
that it never raises on real or malformed input.
"""

from agents.learned.features import (
    FeaturizedDecision,
    OBSERVATION_FEATURE_DIM,
    OPTION_FEATURE_DIM,
    candidate_features,
    featurize,
    observation_features,
    option_features,
)

__all__ = [
    "FeaturizedDecision",
    "OBSERVATION_FEATURE_DIM",
    "OPTION_FEATURE_DIM",
    "candidate_features",
    "featurize",
    "observation_features",
    "option_features",
]
