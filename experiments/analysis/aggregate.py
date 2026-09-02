"""Campaign-level aggregation: one row per cell, one table per question.

Everything here consumes artifacts written by the search runs. Nothing calls a
model except the optional judge- and embedding-backed layers, which are opt-in.

Two rules are enforced rather than documented:

* Novelty is measured from the *code*, never from the ``thought`` field. A model
  that describes its design as novel is not evidence that it is.
* Feature statistics are pooled across the whole campaign before z-scoring, so
  cells stay comparable.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import behavioral, distances
from .attribution import (
    COMPLEXITY_MEASURES,
    accuracy_at_matched_budget,
    build_design_table,
    classify_curve_shape,
    complexity_curve,
    group_by_worker,
    pareto_front,
    variance_check,
)
from .designer_signature import signature
from .extract_features import extract_features, iter_cell_archives, load_abandoned
from .lineage import build_lineage, motif_first_appearance, stepping_stone_score

POOLED_STATS_FILENAME = "pooled_feature_stats.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_campaign(results_root: str, cells_config: Optional[dict] = None) -> Dict[str, List[dict]]:
    """All cell archives, annotated with their meta/worker models."""
    by_cell_config = {}
    if cells_config:
        by_cell_config = {cell["id"]: cell for cell in cells_config.get("cells", [])}

    campaign: Dict[str, List[dict]] = {}
    for cell_id, archive in iter_cell_archives(results_root):
        config = by_cell_config.get(cell_id, {})
        for entry in archive:
            entry.setdefault("cell_id", cell_id)
            entry.setdefault("meta_model", config.get("meta"))
            entry.setdefault("worker_model", config.get("worker"))
            entry.setdefault("stage", config.get("stage"))
        campaign[cell_id] = archive
    return campaign


def fit_pooled_stats(campaign: Dict[str, List[dict]], seed_codes: Sequence[str] = (),
                     cache_path: Optional[str] = None) -> distances.PooledStats:
    """Fit ONCE over every design in the campaign, then reuse everywhere."""
    if cache_path and os.path.exists(cache_path):
        return distances.PooledStats.load(cache_path)
    codes = [entry["code"] for archive in campaign.values() for entry in archive if entry.get("code")]
    codes.extend(seed_codes)
    stats = distances.PooledStats.fit(codes)
    if cache_path:
        stats.save(cache_path)
    return stats


# ---------------------------------------------------------------------------
# Novelty and diversity
# ---------------------------------------------------------------------------

def novelty_table(archive: Sequence[dict], seed_codes: Sequence[str],
                  stats: distances.PooledStats,
                  layers: Sequence[str] = ("structural", "tree")) -> List[dict]:
    """Per-design minimum distance to any seed agent, per layer.

    Reported per layer rather than collapsed, because a design can be
    structurally novel and behaviourally identical, and that combination is
    itself worth reporting.
    """
    rows: List[dict] = []
    for entry in archive:
        code = entry.get("code")
        if not code:
            continue
        row = {"name": entry.get("name"), "generation": entry.get("generation"),
               "cell_id": entry.get("cell_id"), "fitness": entry.get("raw_median_accuracy")}
        if "structural" in layers:
            row["novelty_structural"] = distances.novelty_against(code, seed_codes, stats)
        if "tree" in layers:
            from .tree_distance import tree_edit_distance
            values = []
            for seed in seed_codes:
                try:
                    values.append(tree_edit_distance(code, seed))
                except SyntaxError:
                    continue
            row["novelty_tree"] = min(values) if values else float("nan")
        rows.append(row)
    return rows


def within_run_diversity(archive: Sequence[dict], stats: distances.PooledStats) -> dict:
    """Mean pairwise distance among a run's designs.

    A run can hill-climb accuracy while collapsing to a single design family;
    diversity is what makes that visible.
    """
    codes = [entry["code"] for entry in archive if entry.get("code")]
    if len(codes) < 2:
        return {"n": len(codes), "mean_pairwise": None, "median_pairwise": None}
    matrix = distances.pairwise_structural(codes, stats)
    upper = [matrix[i, j] for i in range(len(codes)) for j in range(i + 1, len(codes))]
    finite = [value for value in upper if np.isfinite(value)]
    return {
        "n": len(codes),
        "mean_pairwise": float(np.mean(finite)) if finite else None,
        "median_pairwise": float(np.median(finite)) if finite else None,
    }


def complexity_summary(archive: Sequence[dict]) -> dict:
    """At least two complexity measures, never one."""
    rows = build_design_table(archive)
    summary: Dict[str, Optional[float]] = {}
    for name in COMPLEXITY_MEASURES:
        values = [row[f"c_{name}"] for row in rows if row.get(f"c_{name}") is not None]
        summary[name] = float(np.median(values)) if values else None
        summary[f"{name}_max"] = float(np.max(values)) if values else None
    summary["n"] = len(rows)
    return summary


# ---------------------------------------------------------------------------
# Per-cell row
# ---------------------------------------------------------------------------

def cell_row(cell_id: str, archive: Sequence[dict], seed_codes: Sequence[str],
             stats: distances.PooledStats, abandoned: Sequence[dict] = (),
             degeneracy_threshold: Optional[float] = None) -> dict:
    designs = [entry for entry in archive if entry.get("code")]
    validation = [entry.get("raw_median_accuracy") for entry in designs
                  if entry.get("raw_median_accuracy") is not None]
    test = [entry.get("test_median_accuracy") for entry in designs
            if entry.get("test_median_accuracy") is not None]

    novelty = novelty_table(designs, seed_codes, stats, layers=("structural",))
    novelty_values = [row["novelty_structural"] for row in novelty
                      if row.get("novelty_structural") is not None
                      and np.isfinite(row["novelty_structural"])]
    median_novelty = float(np.median(novelty_values)) if novelty_values else None

    # Best agent is chosen on validation fitness alone. Test scores are read out
    # afterwards and never used for selection.
    ranked = [{"index": i, "fitness": entry.get("raw_median_accuracy", 0.0),
               "ci_low": (entry.get("fitness_ci") or [None, None])[0]}
              for i, entry in enumerate(designs)]
    tied = behavioral.tied_with_best(ranked) if ranked else []
    best_index = tied[0] if tied else None
    best = designs[best_index] if best_index is not None else None

    lineage = build_lineage(designs, lambda a, b: _safe_structural(a, b, stats))

    row = {
        "cell_id": cell_id,
        "meta_model": designs[0].get("meta_model") if designs else None,
        "worker_model": designs[0].get("worker_model") if designs else None,
        "stage": designs[0].get("stage") if designs else None,
        "n_designs": len(designs),
        "n_abandoned": len(abandoned),
        "validation_median": float(np.median(validation)) if validation else None,
        "validation_best": float(np.max(validation)) if validation else None,
        "test_of_best": best.get("test_median_accuracy") if best else None,
        "generalization_gap_of_best": best.get("generalization_gap") if best else None,
        "n_tied_for_best": len(tied),
        "best_agent": best.get("name") if best else None,
        "median_novelty_structural": median_novelty,
        "diversity": within_run_diversity(designs, stats),
        "complexity": complexity_summary(designs),
        "stepping_stones": stepping_stone_score(lineage),
        "motif_first_appearance": motif_first_appearance(lineage),
        "variance": variance_check(build_design_table(designs)),
    }

    # Degenerate cells are reported, not discarded. A meta model that cannot
    # produce working code is a result about that model.
    row["degenerate"] = _is_degenerate(row, len(archive), degeneracy_threshold)
    return row


def _safe_structural(entry_a: dict, entry_b: dict, stats: distances.PooledStats) -> float:
    try:
        return distances.structural_distance(entry_a["code"], entry_b["code"], stats)
    except (SyntaxError, KeyError):
        return float("inf")


def _is_degenerate(row: dict, n_attempted: int, threshold: Optional[float]) -> dict:
    reasons = []
    if n_attempted and row["n_designs"] / max(n_attempted, 1) < 0.5:
        reasons.append("fewer than half of generations produced running code")
    if row["validation_median"] is not None and row["validation_median"] <= 0.0:
        reasons.append("median validation accuracy at zero")
    if threshold is not None and row["median_novelty_structural"] is not None \
            and row["median_novelty_structural"] < threshold:
        reasons.append("median novelty below the seed 10th-percentile distance")
    return {"is_degenerate": bool(reasons), "reasons": reasons}


# ---------------------------------------------------------------------------
# Campaign roll-up
# ---------------------------------------------------------------------------

def aggregate(results_root: str, seed_codes: Sequence[str],
              cells_config: Optional[dict] = None,
              output_path: Optional[str] = None) -> dict:
    campaign = load_campaign(results_root, cells_config)
    abandoned = load_abandoned(results_root)
    abandoned_by_cell: Dict[str, List[dict]] = defaultdict(list)
    for entry in abandoned:
        abandoned_by_cell[entry.get("cell_id", "unknown")].append(entry)

    cache_path = os.path.join(results_root, POOLED_STATS_FILENAME)
    stats = fit_pooled_stats(campaign, seed_codes, cache_path)
    threshold = distances.seed_pairwise_percentile(seed_codes, stats) if len(seed_codes) > 1 else None

    rows = [cell_row(cell_id, archive, seed_codes, stats,
                     abandoned_by_cell.get(cell_id, ()), threshold)
            for cell_id, archive in sorted(campaign.items())]

    all_designs = [entry for archive in campaign.values() for entry in archive]
    design_table = build_design_table(all_designs)

    payload = {
        "results_root": results_root,
        "degeneracy_threshold": threshold,
        "pooled_stats_n": stats.n,
        "cells": rows,
        "per_worker_curves": {
            worker: {
                complexity: {
                    "raw": complexity_curve(worker_rows, f"c_{complexity}"),
                    "cost_matched": complexity_curve(worker_rows, f"c_{complexity}", cost_matched=True),
                    "shape": classify_curve_shape(complexity_curve(worker_rows, f"c_{complexity}")),
                }
                for complexity in ("static_agents", "runtime_calls", "structural_depth")
            }
            for worker, worker_rows in group_by_worker(design_table).items()
        },
        "pareto_front": [
            {"cell_id": row["cell_id"], "name": row["name"], "fitness": row["fitness"],
             "tokens_per_question": row["tokens_per_question"]}
            for row in pareto_front(design_table)
        ],
        "matched_budget_accuracy": {
            worker: {str(budget): accuracy_at_matched_budget(worker_rows, budget)
                     for budget in (1, 3, 5, 10)}
            for worker, worker_rows in group_by_worker(design_table).items()
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
    return payload


def designer_signatures(results_root: str, seed_codes: Sequence[str],
                        cells_config: Optional[dict] = None,
                        judge_call=None) -> Dict[str, dict]:
    campaign = load_campaign(results_root, cells_config)
    abandoned = load_abandoned(results_root)
    abandoned_by_cell: Dict[str, List[dict]] = defaultdict(list)
    for entry in abandoned:
        abandoned_by_cell[entry.get("cell_id", "unknown")].append(entry)

    stats = fit_pooled_stats(campaign, seed_codes,
                             os.path.join(results_root, POOLED_STATS_FILENAME))
    return {
        cell_id: signature(archive, lambda a, b: _safe_structural(a, b, stats),
                           abandoned_by_cell.get(cell_id, ()), judge_call)
        for cell_id, archive in sorted(campaign.items())
    }


def crosseval_matrix(results_root: str) -> List[dict]:
    """Best agent from every cell, run against every worker model.

    Separates "this design suits this worker" from "this design is good". Only
    the best agent per cell is transferred, since the full matrix would cost more
    than the search itself.
    """
    rows: List[dict] = []
    path = os.path.join(results_root, "crosseval")
    if not os.path.isdir(path):
        return rows
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(path, filename), "r") as handle:
            payload = json.load(handle)
        rows.extend(payload if isinstance(payload, list) else [payload])
    return rows
