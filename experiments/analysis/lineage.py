"""Layer 5: stepping-stone lineage.

The four distance layers describe *pairs* of agents. Lineage describes the search
*trajectory*, and it is where ADAS found its most interesting result: a pattern
appearing at iteration 3 became a stepping stone that later designs reused, and
the best ARC agent combined three ideas first surfacing at iterations 5, 11, 12.

A motif that appears once and is never reused is noise. A motif that appears and
then propagates is a stepping stone. The cross-cell question is whether stronger
meta agents produce more reusable stepping stones or merely more one-off designs
-- which is closer to what people actually want to know about ADAS than any
accuracy delta, and which no published work measures systematically.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

from .extract_features import canonicalize, extract_features

# The recurring building blocks worth tracking. Detection is deliberately
# conservative: a missed motif is better than a spurious lineage edge.
MOTIFS = (
    "ensembling",
    "majority_voting",
    "critic_loop",
    "role_specialization",
    "decomposition",
    "routing",
    "verification",
    "early_exit",
)

_TEXT_MARKERS = {
    "decomposition": (r"\bsub-?problem", r"\bdecompos", r"\bbreak (it|the problem) down", r"\bsub-?task"),
    "verification": (r"\bverif", r"\bcheck (the|your) (answer|work|solution)", r"\bvalidate"),
    "critic_loop": (r"\bcritique", r"\bfeedback", r"\breflect", r"\brefine"),
}


def detect_motifs(code: str) -> Dict[str, int]:
    """Which building blocks a design uses. Structure first, text as backup."""
    present = {motif: 0 for motif in MOTIFS}
    try:
        tree = canonicalize(code)
        record = extract_features(code)
    except SyntaxError:
        return present

    features = record["features"]

    if features["n_agents_static"] > 1 or features["max_loop_depth"] >= 1:
        present["ensembling"] = int(features["n_agents_static"] > 1)
    present["majority_voting"] = features["has_aggregation"]
    present["role_specialization"] = int(features["n_distinct_roles"] >= 2)
    present["routing"] = features["has_dynamic_routing"]
    present["early_exit"] = int(any(
        isinstance(node, ast.If) and any(isinstance(sub, (ast.Break, ast.Return))
                                         for sub in ast.walk(node))
        for node in ast.walk(tree)
    ) and features["max_loop_depth"] >= 1)

    # A loop that feeds an agent's own output back to it is a critic loop.
    if features["max_loop_depth"] >= 1 and features["has_context_accumulation"]:
        present["critic_loop"] = 1

    lowered = code.lower()
    for motif, patterns in _TEXT_MARKERS.items():
        if present.get(motif):
            continue
        if any(re.search(pattern, lowered) for pattern in patterns):
            present[motif] = 1

    return present


def build_lineage(archive: Sequence[dict], distance_fn: Callable[[dict, dict], float],
                  n_parents: int = 1) -> List[dict]:
    """Directed genealogy: each entry linked to its nearest predecessors.

    Predecessors are earlier archive entries only, so the graph is acyclic by
    construction and reflects what the meta agent could actually have seen.
    """
    entries = [entry for entry in archive if entry.get("code")]
    lineage: List[dict] = []
    for index, entry in enumerate(entries):
        candidates = entries[:index]
        parents = []
        if candidates:
            scored = sorted(
                ((distance_fn(parent, entry), i) for i, parent in enumerate(candidates)),
                key=lambda pair: pair[0],
            )
            parents = [
                {"index": i, "name": candidates[i].get("name"),
                 "generation": candidates[i].get("generation"), "distance": distance}
                for distance, i in scored[:n_parents]
            ]
        lineage.append({
            "index": index,
            "name": entry.get("name"),
            "generation": entry.get("generation"),
            "fitness": entry.get("raw_median_accuracy"),
            "parents": parents,
            "nearest_predecessor_distance": parents[0]["distance"] if parents else None,
            "motifs": detect_motifs(entry["code"]),
        })
    return lineage


def motif_first_appearance(lineage: Sequence[dict]) -> Dict[str, dict]:
    """Where each motif first shows up, and how often it is reused afterwards."""
    report: Dict[str, dict] = {}
    for motif in MOTIFS:
        first = None
        reuse = 0
        for node in lineage:
            if not node["motifs"].get(motif):
                continue
            if first is None:
                first = node
            else:
                reuse += 1
        if first is None:
            report[motif] = {"first_generation": None, "reuse_count": 0, "is_stepping_stone": False}
        else:
            report[motif] = {
                "first_generation": first["generation"],
                "first_name": first["name"],
                "reuse_count": reuse,
                # Appears and then propagates, rather than appearing once.
                "is_stepping_stone": reuse >= 2,
            }
    return report


def stepping_stone_score(lineage: Sequence[dict]) -> Dict[str, float]:
    report = motif_first_appearance(lineage)
    stones = [motif for motif, entry in report.items() if entry["is_stepping_stone"]]
    one_offs = [motif for motif, entry in report.items()
                if entry["first_generation"] is not None and entry["reuse_count"] == 0]
    return {
        "n_stepping_stones": len(stones),
        "n_one_offs": len(one_offs),
        "stepping_stones": stones,
        "one_offs": one_offs,
        "reuse_ratio": len(stones) / max(len(stones) + len(one_offs), 1),
    }


def archive_utilization(lineage: Sequence[dict], archive: Sequence[dict]) -> Dict[str, float]:
    """Does the designer build on the archive, or wander?

    AFlow argues that dumping every past workflow into the prompt makes the ADAS
    meta agent behave "more like an explorer of infinite possibilities than a
    designer seeking the optimal workflow". This measures it directly.
    """
    distances = [node["nearest_predecessor_distance"] for node in lineage
                 if node["nearest_predecessor_distance"] is not None]
    names = {str(entry.get("name", "")).lower() for entry in archive if entry.get("name")}
    references = 0
    considered = 0
    for entry in archive:
        thought = str(entry.get("thought", "")).lower()
        if not thought:
            continue
        considered += 1
        own = str(entry.get("name", "")).lower()
        if any(name and name != own and name in thought for name in names):
            references += 1

    mean_distance = sum(distances) / len(distances) if distances else None
    return {
        "mean_distance_to_nearest_predecessor": mean_distance,
        "explicit_reference_rate": references / considered if considered else 0.0,
        "n_considered": considered,
    }


def trajectory_shape(lineage: Sequence[dict], window: int = 5) -> List[dict]:
    """Within-run diversity against generation index.

    A converging designer narrows toward a design family; a wandering one does
    not, and the distinction is invisible in any endpoint metric.
    """
    points: List[dict] = []
    for index, node in enumerate(lineage):
        window_nodes = lineage[max(0, index - window + 1): index + 1]
        distances = [n["nearest_predecessor_distance"] for n in window_nodes
                     if n["nearest_predecessor_distance"] is not None]
        points.append({
            "generation": node["generation"],
            "rolling_novelty": sum(distances) / len(distances) if distances else None,
            "fitness": node.get("fitness"),
        })
    return points


def motif_propagation_graph(lineage: Sequence[dict]) -> Dict[str, List[dict]]:
    """For each motif, the chain of entries that carry it."""
    chains: Dict[str, List[dict]] = defaultdict(list)
    for node in lineage:
        for motif, present in node["motifs"].items():
            if present:
                chains[motif].append({"generation": node["generation"], "name": node["name"],
                                      "fitness": node.get("fitness")})
    return dict(chains)
