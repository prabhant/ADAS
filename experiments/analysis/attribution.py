"""Q9 attribution: what makes an agent perform well.

Answered from the pooled campaign rather than from any single cell, at no extra
compute cost.

Two statistical cautions are enforced in code rather than left to the write-up:

* Archive entries are genealogically related, because later designs build on
  earlier ones. Treating them as independent overstates confidence badly, so
  every fit reports the lineage-clustered variant alongside the naive one.
* Curves are drawn per worker tier and never pooled. The complexity that suits a
  4B worker is probably not the complexity that suits a 27B one, and that
  conditionality *is* the finding.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .extract_features import FEATURE_NAMES, extract_features, feature_vector
from .lineage import MOTIFS, detect_motifs

# Complexity is not one thing. Report against at least two measures: an agent
# with one deep loop and an agent with five parallel calls are both "complex" in
# ways that need not behave alike.
COMPLEXITY_MEASURES = {
    "static_agents": lambda record, entry: record["features"]["n_agents_static"],
    "runtime_calls": lambda record, entry: (entry.get("resource") or {}).get("calls_per_question", 0.0),
    "structural_depth": lambda record, entry: (record["features"]["max_loop_depth"]
                                               + record["features"]["n_branches"]),
    "n_statements": lambda record, entry: record["features"]["n_statements"],
}


def build_design_table(entries: Sequence[dict]) -> List[dict]:
    """One row per design: features, motifs, complexity, cost and fitness."""
    rows: List[dict] = []
    for entry in entries:
        code = entry.get("code")
        if not code:
            continue
        try:
            record = extract_features(code)
        except SyntaxError:
            continue
        resource = entry.get("resource") or {}
        row = {
            "cell_id": entry.get("cell_id"),
            "meta_model": entry.get("meta_model"),
            "meta_params_total": entry.get("meta_params_total"),
            "meta_params_active": entry.get("meta_params_active"),
            "worker_model": entry.get("worker_model"),
            "generation": entry.get("generation"),
            "name": entry.get("name"),
            "abandoned": bool(entry.get("abandoned")),
            "fitness": entry.get("raw_median_accuracy"),
            "test_fitness": entry.get("test_median_accuracy"),
            "generalization_gap": entry.get("generalization_gap"),
            "tokens_per_question": resource.get("tokens_per_question", 0.0),
            "calls_per_question": resource.get("calls_per_question", 0.0),
        }
        row.update({f"f_{name}": record["features"][name] for name in FEATURE_NAMES})
        row.update({f"m_{motif}": value for motif, value in detect_motifs(code).items()})
        row.update({f"c_{name}": fn(record, entry) for name, fn in COMPLEXITY_MEASURES.items()})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Complexity-performance curve
# ---------------------------------------------------------------------------

def complexity_curve(rows: Sequence[dict], complexity_key: str = "c_static_agents",
                     n_bins: int = 6, cost_matched: bool = False) -> List[dict]:
    """Binned fitness against complexity, for ONE worker tier.

    Never call this on rows pooled across workers: the shape by tier is the
    finding. Pass ``cost_matched=True`` to residualize fitness on token spend
    first -- complexity and spend are close to collinear, so an uncontrolled
    curve may show nothing more than that more compute buys more accuracy.
    """
    usable = [row for row in rows if row.get("fitness") is not None]
    if not usable:
        return []

    fitness = np.asarray([row["fitness"] for row in usable], dtype=float)
    if cost_matched:
        cost = np.asarray([row.get("tokens_per_question", 0.0) for row in usable], dtype=float)
        fitness = _residualize(fitness, cost)

    values = np.asarray([row.get(complexity_key, 0.0) for row in usable], dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return []
    bins = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)

    curve: List[dict] = []
    for index in range(len(edges) - 1):
        mask = bins == index
        if not mask.any():
            continue
        curve.append({
            "bin": index,
            "complexity_low": float(edges[index]),
            "complexity_high": float(edges[index + 1]),
            "complexity_mean": float(values[mask].mean()),
            "fitness_mean": float(fitness[mask].mean()),
            "fitness_median": float(np.median(fitness[mask])),
            "fitness_sem": float(fitness[mask].std(ddof=1) / math.sqrt(mask.sum())) if mask.sum() > 1 else 0.0,
            "n": int(mask.sum()),
        })
    return curve


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.std() < 1e-12:
        return y
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)


def classify_curve_shape(curve: Sequence[dict], tolerance: float = 0.005) -> str:
    """Monotonic, plateau, or inverted U.

    An inverted U would be the most interesting outcome and is plausible: every
    added hop introduces error accumulation, context dilution, and another place
    to fail.
    """
    if len(curve) < 3:
        return "insufficient_data"
    values = [point["fitness_mean"] for point in curve]
    peak = int(np.argmax(values))
    rises = values[peak] - values[0]
    falls = values[peak] - values[-1]

    if peak == len(values) - 1 and rises > tolerance:
        return "monotonic"
    if falls > tolerance and rises > tolerance:
        return "inverted_u"
    if abs(values[-1] - values[peak]) <= tolerance and rises > tolerance:
        return "plateau"
    if rises <= tolerance and falls <= tolerance:
        return "flat"
    return "monotonic_decreasing" if values[-1] < values[0] else "plateau"


# ---------------------------------------------------------------------------
# Component attribution: three methods, reported side by side
# ---------------------------------------------------------------------------

def frequency_contrast(rows: Sequence[dict], quantile: float = 0.25) -> Dict[str, dict]:
    """Motif frequency in top versus bottom quartile.

    Simple and it communicates well, so keep it for the figure -- but distrust it
    for the conclusion. Motifs co-occur (ensembling and final-decision
    aggregation nearly always travel together), so each looks important alone.
    """
    usable = [row for row in rows if row.get("fitness") is not None]
    if len(usable) < 8:
        return {}
    fitness = np.asarray([row["fitness"] for row in usable])
    low, high = np.quantile(fitness, [quantile, 1 - quantile])
    bottom = [row for row in usable if row["fitness"] <= low]
    top = [row for row in usable if row["fitness"] >= high]

    out: Dict[str, dict] = {}
    for motif in MOTIFS:
        key = f"m_{motif}"
        top_rate = float(np.mean([row.get(key, 0) for row in top])) if top else 0.0
        bottom_rate = float(np.mean([row.get(key, 0) for row in bottom])) if bottom else 0.0
        out[motif] = {"top_quartile_rate": top_rate, "bottom_quartile_rate": bottom_rate,
                      "delta": top_rate - bottom_rate, "n_top": len(top), "n_bottom": len(bottom)}
    return out


def regularized_fit(rows: Sequence[dict], feature_keys: Optional[Sequence[str]] = None,
                    method: str = "l1_logistic") -> Dict[str, float]:
    """Multivariate fit that handles motif co-occurrence.

    L1 logistic regression on top-versus-bottom half, or gradient-boosted trees
    with SHAP values if requested.
    """
    usable = [row for row in rows if row.get("fitness") is not None]
    if len(usable) < 20:
        return {"error": "too few designs for a multivariate fit", "n": len(usable)}

    feature_keys = list(feature_keys or ([f"f_{n}" for n in FEATURE_NAMES]
                                         + [f"m_{m}" for m in MOTIFS]))
    X = np.asarray([[float(row.get(key, 0)) for key in feature_keys] for row in usable])
    fitness = np.asarray([row["fitness"] for row in usable])
    y = (fitness > np.median(fitness)).astype(int)

    if method == "gbt_shap":
        from sklearn.ensemble import GradientBoostingClassifier
        import shap

        model = GradientBoostingClassifier(random_state=0).fit(X, y)
        values = shap.TreeExplainer(model).shap_values(X)
        importance = np.abs(values).mean(axis=0)
        return {key: float(value) for key, value in zip(feature_keys, importance)}

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    model = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000).fit(Xs, y)
    return {key: float(value) for key, value in zip(feature_keys, model.coef_[0])}


def paired_delta(lineage: Sequence[dict], rows_by_index: Dict[int, dict],
                 feature_keys: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    """Regress change in features on change in fitness, over parent-child pairs.

    The strongest of the three methods. Comparing genealogically related designs
    controls for most between-design confounding, the way a within-subjects design
    does. This reconstructs what AFlow's tree-structured experience log captures
    and ADAS does not.
    """
    feature_keys = list(feature_keys or ([f"f_{n}" for n in FEATURE_NAMES]
                                         + [f"m_{m}" for m in MOTIFS]))
    deltas: List[List[float]] = []
    fitness_deltas: List[float] = []
    pairs: List[dict] = []

    for node in lineage:
        child = rows_by_index.get(node["index"])
        if not child or not node.get("parents"):
            continue
        parent = rows_by_index.get(node["parents"][0]["index"])
        if not parent:
            continue
        if child.get("fitness") is None or parent.get("fitness") is None:
            continue
        deltas.append([float(child.get(k, 0)) - float(parent.get(k, 0)) for k in feature_keys])
        fitness_deltas.append(child["fitness"] - parent["fitness"])
        pairs.append({
            "parent": parent.get("name"), "child": child.get("name"),
            "parent_generation": parent.get("generation"), "child_generation": child.get("generation"),
            "fitness_delta": child["fitness"] - parent["fitness"],
            "changed": [k for k, d in zip(feature_keys, deltas[-1]) if abs(d) > 1e-9],
        })

    if len(deltas) < 5:
        return {"error": "too few parent-child pairs", "n_pairs": len(deltas), "pairs": pairs}

    X = np.asarray(deltas)
    y = np.asarray(fitness_deltas)
    keep = X.std(axis=0) > 1e-9
    if not keep.any():
        return {"error": "no feature varied between parent and child", "n_pairs": len(deltas)}

    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0).fit(X[:, keep], y)
    coefficients = {key: float(value) for key, value
                    in zip([k for k, k_ok in zip(feature_keys, keep) if k_ok], model.coef_)}
    return {"n_pairs": len(deltas), "coefficients": coefficients, "pairs": pairs}


def attribution_table(rows: Sequence[dict], lineage: Sequence[dict],
                      rows_by_index: Dict[int, dict]) -> List[dict]:
    """All three methods in adjacent columns, so disagreement is visible."""
    contrast = frequency_contrast(rows)
    fit = regularized_fit(rows)
    paired = paired_delta(lineage, rows_by_index)
    paired_coefficients = paired.get("coefficients", {})

    table: List[dict] = []
    for motif in MOTIFS:
        key = f"m_{motif}"
        table.append({
            "component": motif,
            "frequency_contrast": contrast.get(motif, {}).get("delta"),
            "l1_logistic_coefficient": fit.get(key) if isinstance(fit, dict) else None,
            "paired_delta_coefficient": paired_coefficients.get(key),
        })
    return table


# ---------------------------------------------------------------------------
# Caveats that have to be checked before fitting anything
# ---------------------------------------------------------------------------

def variance_check(rows: Sequence[dict]) -> dict:
    """Compressed fitness range makes attribution unstable and curves unreadable.

    A 27B worker on MGSM may sit near the top of the scale; check before fitting.
    """
    fitness = [row["fitness"] for row in rows if row.get("fitness") is not None]
    if not fitness:
        return {"n": 0, "usable": False}
    array = np.asarray(fitness)
    spread = float(array.max() - array.min())
    return {
        "n": len(fitness),
        "min": float(array.min()), "max": float(array.max()),
        "std": float(array.std()), "range": spread,
        "ceiling_risk": float(array.mean()) > 0.9,
        "usable": spread > 0.05,
    }


def lineage_clusters(lineage: Sequence[dict]) -> Dict[int, int]:
    """Connected components of the genealogy, for clustered standard errors."""
    parent_of: Dict[int, int] = {}
    for node in lineage:
        if node.get("parents"):
            parent_of[node["index"]] = node["parents"][0]["index"]

    def root(index: int) -> int:
        seen = set()
        while index in parent_of and index not in seen:
            seen.add(index)
            index = parent_of[index]
        return index

    return {node["index"]: root(node["index"]) for node in lineage}


def group_by_worker(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("worker_model") or "unknown"].append(row)
    return dict(grouped)


def pareto_front(rows: Sequence[dict], cost_key: str = "tokens_per_question",
                 value_key: str = "fitness") -> List[dict]:
    """Accuracy-versus-cost Pareto front. The headline artifact.

    Cost is reported in tokens rather than dollars, since the models are
    self-hosted.
    """
    usable = [row for row in rows if row.get(value_key) is not None]
    ordered = sorted(usable, key=lambda row: (row.get(cost_key, 0.0), -row[value_key]))
    front: List[dict] = []
    best = -float("inf")
    for row in ordered:
        if row[value_key] > best:
            best = row[value_key]
            front.append(row)
    return front


def accuracy_at_matched_budget(rows: Sequence[dict], budget: float,
                               cost_key: str = "calls_per_question") -> Optional[float]:
    """Best fitness achievable within a call budget, reported beside raw accuracy."""
    eligible = [row["fitness"] for row in rows
                if row.get("fitness") is not None and row.get(cost_key, 0.0) <= budget]
    return max(eligible) if eligible else None
