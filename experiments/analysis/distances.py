"""Distance combination with pooled z-scoring.

Feature statistics are computed **once** across the entire campaign and then
applied everywhere. Per-run normalization would make cells incomparable, which
defeats the purpose -- and getting this wrong silently invalidates every
cross-cell comparison, so the pooled statistics are persisted to disk and
reloaded rather than recomputed per analysis job.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .extract_features import FEATURE_NAMES, extract_features, feature_vector

# Layer weights for the combined distance. Reported alongside the per-layer
# distances, never instead of them: disagreement between layers is a finding.
DEFAULT_LAYER_WEIGHTS = {
    "structural": 0.3,
    "tree": 0.2,
    "dataflow": 0.3,
    "prompt": 0.1,
    "behavioral": 0.1,
}

# Within the structural vector, the design-semantic features carry more weight
# than the raw size features.
DEFAULT_FEATURE_WEIGHTS = {
    "n_agent_sites": 1.0,
    "n_agents_static": 1.0,
    "n_call_sites": 1.0,
    "max_loop_depth": 1.2,
    "n_branches": 1.0,
    "has_dynamic_routing": 1.5,
    "n_distinct_roles": 1.2,
    "n_output_fields": 0.8,
    "has_aggregation": 1.5,
    "has_context_accumulation": 1.2,
    "n_temperatures": 0.6,
    "n_statements": 0.5,
}


class PooledStats:
    """Mean and standard deviation per feature, over the whole campaign."""

    def __init__(self, mean: Sequence[float], std: Sequence[float],
                 feature_names: Sequence[str] = FEATURE_NAMES, n: int = 0) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.std[self.std < 1e-9] = 1.0  # constant features contribute nothing
        self.feature_names = list(feature_names)
        self.n = n

    @classmethod
    def fit(cls, codes: Sequence[str]) -> "PooledStats":
        vectors = []
        for code in codes:
            try:
                vectors.append(feature_vector(extract_features(code)))
            except SyntaxError:
                continue
        if not vectors:
            raise ValueError("no parseable designs to fit pooled statistics on")
        matrix = np.asarray(vectors, dtype=float)
        return cls(matrix.mean(axis=0), matrix.std(axis=0), n=len(vectors))

    def z(self, vector: Sequence[float]) -> np.ndarray:
        return (np.asarray(vector, dtype=float) - self.mean) / self.std

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist(),
                       "feature_names": self.feature_names, "n": self.n}, handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "PooledStats":
        with open(path, "r") as handle:
            payload = json.load(handle)
        return cls(payload["mean"], payload["std"], payload["feature_names"], payload.get("n", 0))


def structural_distance(code_a: str, code_b: str, stats: PooledStats,
                        weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted Euclidean distance over pooled-z-scored features."""
    weights = weights or DEFAULT_FEATURE_WEIGHTS
    w = np.asarray([weights.get(name, 1.0) for name in FEATURE_NAMES], dtype=float)
    za = stats.z(feature_vector(extract_features(code_a)))
    zb = stats.z(feature_vector(extract_features(code_b)))
    return float(np.sqrt(np.sum(w * (za - zb) ** 2)))


def combine(layer_distances: Dict[str, float],
            weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Weighted combination over whichever layers were computed.

    Layers that could not be computed (no runtime log, no embeddings available)
    are dropped and the remaining weights renormalized, so a partial analysis is
    still comparable in rank order.
    """
    weights = weights or DEFAULT_LAYER_WEIGHTS
    available = {k: v for k, v in layer_distances.items() if v is not None}
    if not available:
        return {"combined": None, "layers": {}, "n_layers": 0}
    total_weight = sum(weights.get(k, 0.0) for k in available) or 1.0
    combined = sum(weights.get(k, 0.0) * v for k, v in available.items()) / total_weight
    return {"combined": combined, "layers": available, "n_layers": len(available)}


def novelty_against(code: str, reference_codes: Sequence[str], stats: PooledStats) -> float:
    """Minimum structural distance to any reference design.

    Reported separately per layer. Layer 1's canonicalization is what makes this
    honest: it is precisely what defeats rename-only innovation.
    """
    if not reference_codes:
        return float("nan")
    distances = []
    for reference in reference_codes:
        try:
            distances.append(structural_distance(code, reference, stats))
        except SyntaxError:
            continue
    return min(distances) if distances else float("nan")


def pairwise_structural(codes: Sequence[str], stats: PooledStats) -> np.ndarray:
    n = len(codes)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                distance = structural_distance(codes[i], codes[j], stats)
            except SyntaxError:
                distance = float("nan")
            matrix[i, j] = matrix[j, i] = distance
    return matrix


def seed_pairwise_percentile(seed_codes: Sequence[str], stats: PooledStats,
                             percentile: float = 10.0) -> float:
    """The degeneracy threshold: the 10th percentile of seed-to-seed distances.

    A cell whose median novelty falls below this is degenerate -- kept and
    reported, never discarded.
    """
    matrix = pairwise_structural(seed_codes, stats)
    values = [matrix[i, j] for i in range(len(seed_codes)) for j in range(i + 1, len(seed_codes))
              if not math.isnan(matrix[i, j])]
    return float(np.percentile(values, percentile)) if values else float("nan")
