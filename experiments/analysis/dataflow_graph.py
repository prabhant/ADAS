"""Layer 2: runtime dataflow graph reconstruction.

This is the strongest available signal and it is nearly free, because the
codebase already carries provenance: ``Info`` is
``('name', 'author', 'content', 'iteration_idx')`` and ``LLMAgentBase.__repr__``
returns a unique agent identifier, so every piece of information records its
producer. The search loop now persists that as a call log.

Static code does not determine this graph. Reflexion breaks early when its critic
returns True, and dynamic role assignment routes to one of several experts, so
the realized topology depends on the input. Only a runtime trace captures it.

Because each agent runs over hundreds of questions you get a *distribution* of
graphs, not one. Entropy over the Weisfeiler-Lehman hashes gives an adaptivity
score that separates fixed scaffolds from input-conditional ones.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

try:
    import networkx as nx
except ImportError:  # analysis-only dependency
    nx = None

USER_NODE = "User"


def load_call_log(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_question(call_log: Iterable[dict], generation=None, phase=None) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for record in call_log:
        if generation is not None and record.get("generation") != generation:
            continue
        if phase is not None and record.get("phase") != phase:
            continue
        grouped[record.get("question_idx")].append(record)
    return grouped


def _node_label(record: dict) -> str:
    """Role plus sorted output fields, so a critic never matches a debater."""
    fields = ",".join(sorted(record.get("output_fields", [])))
    return f"{record.get('role', '?')}|{fields}"


def build_graph(records: List[dict]):
    """One directed graph per question: nodes are LLM calls, edges are info flow."""
    if nx is None:
        raise ImportError("networkx is required for the dataflow layer")

    graph = nx.DiGraph()
    graph.add_node(USER_NODE, label="user|task")

    # Map an agent's repr -> the call ids it produced, in order.
    producer_calls: Dict[str, List[str]] = defaultdict(list)
    for record in sorted(records, key=lambda r: r.get("t", 0)):
        call_id = record["call_id"]
        graph.add_node(call_id, label=_node_label(record),
                       agent_name=record.get("agent_name"),
                       iteration_idx=record.get("iteration_idx"))
        agent_repr = f"{record.get('agent_name')} {record.get('agent_id')}"
        for author in record.get("input_authors", []):
            if author in ("User", "user"):
                graph.add_edge(USER_NODE, call_id)
                continue
            # Edge from the most recent call made by that author.
            upstream = producer_calls.get(author)
            if upstream:
                graph.add_edge(upstream[-1], call_id)
        producer_calls[agent_repr].append(call_id)
    return graph


def graph_hash(graph, iterations: int = 3) -> str:
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="label", iterations=iterations)


def graph_edit_distance(graph_a, graph_b, timeout: float = 10.0) -> Optional[float]:
    """Continuous measure, only worth running on small comparisons."""
    if nx is None:
        raise ImportError("networkx is required for the dataflow layer")
    distances = nx.optimize_graph_edit_distance(graph_a, graph_b)
    best = None
    for value in distances:
        best = value
        break
    if best is None:
        return None
    denominator = graph_a.number_of_nodes() + graph_b.number_of_nodes()
    return float(best) / denominator if denominator else 0.0


def agent_profile(call_log: Iterable[dict], generation=None, phase=None) -> dict:
    """Graph distribution for one agent, plus its adaptivity score."""
    grouped = group_by_question(call_log, generation=generation, phase=phase)
    hashes: List[str] = []
    graphs = {}
    node_counts: List[int] = []
    for question_idx, records in grouped.items():
        if not records:
            continue
        graph = build_graph(records)
        digest = graph_hash(graph)
        hashes.append(digest)
        graphs.setdefault(digest, graph)
        node_counts.append(graph.number_of_nodes() - 1)  # exclude the User node

    counts = Counter(hashes)
    total = sum(counts.values())
    entropy = 0.0
    if total:
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)

    return {
        "n_questions": total,
        "n_distinct_topologies": len(counts),
        "modal_hash": counts.most_common(1)[0][0] if counts else None,
        "modal_share": counts.most_common(1)[0][1] / total if total else 0.0,
        # Self-Consistency produces an identical graph every time; Reflexion does
        # not. High entropy means an input-conditional scaffold, which is a
        # different thing from a complex one.
        "topology_entropy": entropy,
        "runtime_calls_mean": sum(node_counts) / len(node_counts) if node_counts else 0.0,
        "runtime_calls_max": max(node_counts) if node_counts else 0,
        "hash_distribution": dict(counts),
        "graphs": graphs,
    }


def dataflow_distance(profile_a: dict, profile_b: dict) -> float:
    """Distance between two agents' topology distributions.

    Total variation over the WL-hash distributions: 0 when two agents realize
    exactly the same topologies in the same proportions, 1 when they share none.
    """
    dist_a = profile_a.get("hash_distribution", {})
    dist_b = profile_b.get("hash_distribution", {})
    total_a = sum(dist_a.values()) or 1
    total_b = sum(dist_b.values()) or 1
    keys = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(k, 0) / total_a - dist_b.get(k, 0) / total_b) for k in keys)


def dead_code_gap(static_agent_count: int, profile: dict) -> float:
    """Static agent count minus agents actually invoked at runtime.

    Models generate agents they never call. The gap is itself a quality signal,
    and weaker meta agents should show a larger one.
    """
    return static_agent_count - profile.get("runtime_calls_max", 0)


def profiles_for_run(run_dir: str, phase: str = "search") -> Dict[str, dict]:
    """Per-generation dataflow profiles from a run's persisted call log."""
    profiles: Dict[str, dict] = {}
    for filename in os.listdir(run_dir):
        if not filename.endswith("call_log.jsonl"):
            continue
        call_log = load_call_log(os.path.join(run_dir, filename))
        generations = {record.get("generation") for record in call_log}
        for generation in generations:
            profiles[str(generation)] = agent_profile(call_log, generation=generation, phase=phase)
    return profiles
