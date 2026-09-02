"""Figures.

Six plots carry the study. Each is drawn from the aggregate payload, so figures
never touch a model and can be regenerated on a CPU partition for free.

The accuracy-versus-cost Pareto front is the headline. If a simple agent on a
strong worker dominates a complex agent on a weak one, that is the most useful
thing this study can say, and it is a finding regardless of which direction it
falls.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless compute nodes
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .lineage import MOTIFS  # noqa: E402

FAILURE_ORDER = (
    "json_parse_error",
    "python_syntax_error",
    "runtime_exception",
    "context_overflow",
    "wallclock_timeout",
    "all_zero_accuracy",
    "semantically_wrong_but_running",
)


def _save(fig, output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Accuracy vs cost Pareto front (headline)
# ---------------------------------------------------------------------------

def pareto_plot(design_rows: Sequence[dict], front: Sequence[dict], output_dir: str,
                cost_key: str = "tokens_per_question") -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    workers = sorted({row.get("worker_model") or "unknown" for row in design_rows})
    colours = plt.cm.viridis(np.linspace(0, 0.85, max(len(workers), 1)))

    for worker, colour in zip(workers, colours):
        points = [row for row in design_rows
                  if (row.get("worker_model") or "unknown") == worker
                  and row.get("fitness") is not None]
        ax.scatter([row.get(cost_key, 0.0) for row in points],
                   [row["fitness"] for row in points],
                   s=22, alpha=0.55, color=colour, label=worker)

    ordered = sorted(front, key=lambda row: row.get(cost_key, 0.0))
    ax.plot([row.get(cost_key, 0.0) for row in ordered],
            [row["fitness"] for row in ordered],
            color="black", linewidth=1.4, marker="o", markersize=4, label="Pareto front")

    ax.set_xlabel("tokens per question")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Accuracy vs cost across the campaign")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linestyle=":")
    return _save(fig, output_dir, "pareto_accuracy_cost.png")


# ---------------------------------------------------------------------------
# 2. Complexity vs worker capability, one line per meta tier
# ---------------------------------------------------------------------------

def complexity_vs_worker(cell_rows: Sequence[dict], output_dir: str,
                         complexity_key: str = "static_agents",
                         worker_order: Optional[Sequence[str]] = None,
                         reasoning_effort: Optional[Dict[str, float]] = None) -> str:
    """Answers Q1 directly: does a stronger worker change what gets designed?

    The reasoning-effort overlay is what separates "the meta model designs
    differently" from "the meta model thinks longer".
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    workers = list(worker_order or sorted({row["worker_model"] for row in cell_rows
                                           if row.get("worker_model")}))
    positions = {worker: index for index, worker in enumerate(workers)}

    by_meta: Dict[str, List[dict]] = {}
    for row in cell_rows:
        by_meta.setdefault(row.get("meta_model") or "unknown", []).append(row)

    for meta, rows in sorted(by_meta.items()):
        points = [(positions[row["worker_model"]], row["complexity"].get(complexity_key))
                  for row in rows
                  if row.get("worker_model") in positions
                  and row.get("complexity", {}).get(complexity_key) is not None]
        if not points:
            continue
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points],
                marker="o", label=f"meta: {meta}")

    if reasoning_effort:
        twin = ax.twinx()
        effort = [(positions[w], v) for w, v in reasoning_effort.items() if w in positions]
        effort.sort()
        twin.plot([p[0] for p in effort], [p[1] for p in effort],
                  linestyle="--", color="grey", alpha=0.7)
        twin.set_ylabel("meta reasoning tokens (overlay)", color="grey", fontsize=9)

    ax.set_xticks(range(len(workers)))
    ax.set_xticklabels(workers, rotation=15, fontsize=8)
    ax.set_xlabel("worker model (increasing capability)")
    ax.set_ylabel(f"median complexity ({complexity_key})")
    ax.set_title("Designed complexity against worker capability")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linestyle=":")
    return _save(fig, output_dir, f"complexity_vs_worker_{complexity_key}.png")


# ---------------------------------------------------------------------------
# 3. Cross-evaluation heatmap
# ---------------------------------------------------------------------------

def crosseval_heatmap(rows: Sequence[dict], output_dir: str) -> Optional[str]:
    """Designed-for worker on one axis, evaluated-on worker on the other.

    Off-diagonal structure is the evidence that a design is worker-specific
    rather than simply better.
    """
    if not rows:
        return None
    designed = sorted({row["designed_for"] for row in rows if row.get("designed_for")})
    evaluated = sorted({row["evaluated_on"] for row in rows if row.get("evaluated_on")})
    if not designed or not evaluated:
        return None

    matrix = np.full((len(designed), len(evaluated)), np.nan)
    for row in rows:
        if row.get("designed_for") in designed and row.get("evaluated_on") in evaluated:
            matrix[designed.index(row["designed_for"]), evaluated.index(row["evaluated_on"])] = \
                row.get("accuracy", np.nan)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(evaluated)))
    ax.set_xticklabels(evaluated, rotation=25, fontsize=8, ha="right")
    ax.set_yticks(range(len(designed)))
    ax.set_yticklabels(designed, fontsize=8)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("designed for")
    ax.set_title("Cross-evaluation of best agents")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="accuracy")
    return _save(fig, output_dir, "crosseval_heatmap.png")


# ---------------------------------------------------------------------------
# 4. Failure taxonomy
# ---------------------------------------------------------------------------

def failure_distribution(histograms: Dict[str, Dict[str, int]], output_dir: str) -> str:
    """Stacked failure categories per cell.

    The taxonomy is a result, not just debugging telemetry: a weak meta model
    failing on JSON formatting is a different claim from one failing on semantics.
    """
    cells = sorted(histograms)
    fig, ax = plt.subplots(figsize=(max(7, len(cells) * 0.8), 5))
    bottom = np.zeros(len(cells))
    colours = plt.cm.tab10(np.linspace(0, 1, len(FAILURE_ORDER)))

    for category, colour in zip(FAILURE_ORDER, colours):
        values = np.asarray([histograms[cell].get(category, 0) for cell in cells], dtype=float)
        ax.bar(cells, values, bottom=bottom, label=category, color=colour)
        bottom += values

    ax.set_ylabel("abandoned generations")
    ax.set_title("Failure taxonomy by cell")
    ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    return _save(fig, output_dir, "failure_taxonomy.png")


# ---------------------------------------------------------------------------
# 5. Designer signature
# ---------------------------------------------------------------------------

def designer_signature_bars(signatures: Dict[str, dict], output_dir: str) -> str:
    """Motif repertoire composition per meta tier.

    Design behaviour only. Coding-competence panels are drawn separately so the
    two are not read as one axis.
    """
    cells = sorted(signatures)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bottom = np.zeros(len(cells))
    colours = plt.cm.tab20(np.linspace(0, 1, len(MOTIFS)))
    for motif, colour in zip(MOTIFS, colours):
        values = np.asarray([signatures[cell]["motif_repertoire"]["composition"].get(motif, 0.0)
                             for cell in cells], dtype=float)
        axes[0].bar(cells, values, bottom=bottom, label=motif, color=colour)
        bottom += values
    axes[0].set_title("Motif repertoire (design behaviour)")
    axes[0].set_ylabel("share of motif occurrences")
    axes[0].set_xticklabels(cells, rotation=45, ha="right", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False, ncol=2)

    debug = [signatures[cell]["debug_competence"]["mean_debug_attempts"] for cell in cells]
    hallucination = [signatures[cell]["hallucinated_api_rate"] for cell in cells]
    x = np.arange(len(cells))
    axes[1].bar(x - 0.2, debug, width=0.4, label="mean debug attempts")
    axes[1].bar(x + 0.2, hallucination, width=0.4, label="hallucinated APIs per design")
    axes[1].set_title("Coding competence (kept separate)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cells, rotation=45, ha="right", fontsize=8)
    axes[1].legend(fontsize=8, frameon=False)

    return _save(fig, output_dir, "designer_signature.png")


# ---------------------------------------------------------------------------
# 6. Complexity-performance curves, one panel per worker tier
# ---------------------------------------------------------------------------

def complexity_performance_curves(curves: Dict[str, dict], output_dir: str,
                                  complexity: str = "static_agents") -> str:
    """Never pooled across workers -- the conditionality is the finding."""
    workers = sorted(curves)
    fig, axes = plt.subplots(1, max(len(workers), 1), figsize=(4.5 * max(len(workers), 1), 4),
                             squeeze=False)

    for ax, worker in zip(axes[0], workers):
        entry = curves[worker].get(complexity, {})
        for label, style in (("raw", "-"), ("cost_matched", "--")):
            points = entry.get(label) or []
            if not points:
                continue
            ax.errorbar([p["complexity_mean"] for p in points],
                        [p["fitness_mean"] for p in points],
                        yerr=[p["fitness_sem"] for p in points],
                        linestyle=style, marker="o", capsize=3, label=label)
        ax.set_title(f"{worker}\nshape: {entry.get('shape', 'n/a')}", fontsize=9)
        ax.set_xlabel(complexity)
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle("Complexity-performance curves by worker tier", fontsize=11)
    return _save(fig, output_dir, f"complexity_performance_{complexity}.png")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render_all(aggregate_path: str, output_dir: str,
               design_rows: Optional[Sequence[dict]] = None,
               signatures: Optional[Dict[str, dict]] = None,
               failure_histograms: Optional[Dict[str, Dict[str, int]]] = None,
               crosseval_rows: Sequence[dict] = ()) -> List[str]:
    with open(aggregate_path, "r") as handle:
        payload = json.load(handle)

    written: List[str] = []
    if design_rows:
        written.append(pareto_plot(design_rows, payload.get("pareto_front", []), output_dir))
    written.append(complexity_vs_worker(payload.get("cells", []), output_dir))
    heatmap = crosseval_heatmap(crosseval_rows, output_dir)
    if heatmap:
        written.append(heatmap)
    if failure_histograms:
        written.append(failure_distribution(failure_histograms, output_dir))
    if signatures:
        written.append(designer_signature_bars(signatures, output_dir))
    if payload.get("per_worker_curves"):
        written.append(complexity_performance_curves(payload["per_worker_curves"], output_dir))
    return written
