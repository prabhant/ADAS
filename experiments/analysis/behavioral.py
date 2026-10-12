"""Layer 4: behavioural distance over per-question outcome vectors.

Two agents can look entirely different and behave identically. This layer checks
the other three and costs the least, because the search loop now persists the raw
binary correctness vector instead of collapsing it straight into a bootstrap CI.

If agents discovered under a 4B worker and under a 27B worker differ structurally
but share near-identical error profiles, the structural difference is cosmetic,
and reporting it as a design difference would be misleading.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import numpy as np


def error_set(vector: Sequence[int]) -> set:
    return {i for i, correct in enumerate(vector) if not correct}


def hamming_distance(vector_a: Sequence[int], vector_b: Sequence[int]) -> float:
    n = min(len(vector_a), len(vector_b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if vector_a[i] != vector_b[i]) / n


def jaccard_distance(vector_a: Sequence[int], vector_b: Sequence[int]) -> float:
    """Distance over *error* sets: do the two agents fail on the same questions?"""
    errors_a = error_set(vector_a)
    errors_b = error_set(vector_b)
    union = errors_a | errors_b
    if not union:
        return 0.0
    return 1.0 - len(errors_a & errors_b) / len(union)


def behavioral_distance(vector_a: Sequence[int], vector_b: Sequence[int],
                        metric: str = "jaccard") -> float:
    if metric == "hamming":
        return hamming_distance(vector_a, vector_b)
    return jaccard_distance(vector_a, vector_b)


def pairwise_matrix(vectors: Sequence[Sequence[int]], metric: str = "jaccard") -> np.ndarray:
    n = len(vectors)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = behavioral_distance(vectors[i], vectors[j], metric)
    return matrix


def bootstrap_median_ci(vector: Sequence[int], n_resamples: int = 10000,
                        alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Median with a 95% bootstrap interval, the ADAS convention used verbatim."""
    if not len(vector):
        return {"median": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    rng = random.Random(seed)
    data = list(vector)
    n = len(data)
    medians = []
    for _ in range(n_resamples):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        medians.append(float(np.mean(sample)))
    medians.sort()
    low = medians[int(alpha / 2 * n_resamples)]
    high = medians[int((1 - alpha / 2) * n_resamples) - 1]
    # "median" is the median of the bootstrap means, as in ADAS's own helper --
    # never np.median of the raw 0/1 vector, which is a 0-or-1 step function.
    return {"median": float(np.median(medians)), "mean": float(np.mean(data)),
            "low": low, "high": high, "n": n}


def tied_with_best(entries: List[Dict[str, float]]) -> List[int]:
    """Indices tied with the best entry.

    ADAS convention: an entry is tied when its median falls inside the confidence
    interval of the best. With two seeds this is descriptive, not inferential.
    """
    if not entries:
        return []
    best_index = max(range(len(entries)), key=lambda i: entries[i].get("mean", entries[i]["median"]))
    best = entries[best_index]
    return [i for i, entry in enumerate(entries)
            if best["low"] <= entry.get("mean", entry["median"]) <= best["high"]]


def complementarity(vector_a: Sequence[int], vector_b: Sequence[int]) -> Dict[str, float]:
    """How the two agents' successes overlap. Useful when reading Q7 examples."""
    n = min(len(vector_a), len(vector_b))
    both = sum(1 for i in range(n) if vector_a[i] and vector_b[i])
    only_a = sum(1 for i in range(n) if vector_a[i] and not vector_b[i])
    only_b = sum(1 for i in range(n) if vector_b[i] and not vector_a[i])
    neither = n - both - only_a - only_b
    return {"n": n, "both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
            "oracle_union": (both + only_a + only_b) / n if n else 0.0}


def collect_vectors(archive: Sequence[dict], key: str = "valid_per_question") -> Dict[str, List[int]]:
    return {
        f"{entry.get('generation')}:{entry.get('name')}": entry[key]
        for entry in archive if entry.get(key)
    }
