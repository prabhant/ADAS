"""Contamination: the cheap version.

The seed archive is seven named published methods, every ladder model has read
those papers, ADAS is on GitHub, its discovered agents are printed in the paper's
appendix, and this repository ships prior archives under ``results/``. A model can
reproduce them from memory, and newer models memorized more.

Because this study makes no novelty claim, contamination degrades from a confound
into a caveat. You still want to know whether a design was recalled or
constructed, since that changes how you read it, but no finding hinges on the
distinction -- so this stays cheap. Formal recall-versus-discovery scoring is
deliberately not implemented unless something surprising turns up.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Dict, List, Optional, Sequence

COLD_PROBE = """List the LLM agent design patterns and multi-agent architectures you know of
for improving accuracy on reasoning benchmarks. For each, give its common name and
one sentence on how it works. Do not invent names. Reply as JSON:
{"patterns": [{"name": "...", "description": "..."}]}
"""

# Reference list of published architectures. Discovered designs are checked
# against this, nothing more.
PUBLISHED_DESIGNS = {
    "chain-of-thought": ["chain of thought", "chain-of-thought", "cot", "step by step"],
    "self-consistency": ["self-consistency", "self consistency", "cot-sc", "majority vote sampling"],
    "reflexion": ["reflexion", "self-refine", "self refine", "verbal reinforcement"],
    "llm-debate": ["debate", "multiagent debate", "society of minds"],
    "step-back": ["step-back", "take a step back", "abstraction prompting"],
    "quality-diversity": ["quality-diversity", "quality diversity", "diverse solutions archive"],
    "role-assignment": ["role assignment", "dynamic role", "expert routing", "persona"],
    "tree-of-thoughts": ["tree of thoughts", "tree-of-thought", "tot"],
    "graph-of-thoughts": ["graph of thoughts", "got"],
    "least-to-most": ["least-to-most", "least to most"],
    "program-of-thoughts": ["program of thoughts", "pot", "program-aided"],
    "self-discover": ["self-discover", "self discover"],
    "plan-and-solve": ["plan-and-solve", "plan and solve"],
    "react": ["react ", "reason and act", "reasoning and acting"],
    "medprompt": ["medprompt"],
    "aflow": ["aflow"],
    "agentsquare": ["agentsquare"],
    "gptswarm": ["gptswarm"],
    "maas": ["maas", "multi-agent architecture search"],
}


def cold_recall_probe(meta_model: str, chat_call: Callable[[str], dict]) -> dict:
    """Ask each meta model, with no archive, what design patterns it knows.

    ``chat_call`` takes a prompt and returns parsed JSON.
    """
    try:
        response = chat_call(COLD_PROBE)
    except Exception as exc:
        return {"model": meta_model, "error": f"{type(exc).__name__}: {exc}"}

    patterns = response.get("patterns", []) if isinstance(response, dict) else []
    names = [str(entry.get("name", "")).strip() for entry in patterns if isinstance(entry, dict)]
    matched = sorted({key for name in names for key in _match_published(name)})
    return {
        "model": meta_model,
        "n_reported": len(names),
        "reported": names,
        "matched_published": matched,
        "recall_coverage": len(matched) / len(PUBLISHED_DESIGNS),
    }


def _match_published(text: str) -> List[str]:
    lowered = f" {text.lower()} "
    return [key for key, aliases in PUBLISHED_DESIGNS.items()
            if any(alias in lowered for alias in aliases)]


def match_design(entry: dict) -> dict:
    """Does a discovered design name or describe a published architecture?"""
    haystack = " ".join([
        str(entry.get("name", "")),
        str(entry.get("thought", "")),
    ]).lower()
    matches = _match_published(haystack)
    return {
        "name": entry.get("name"),
        "generation": entry.get("generation"),
        "matched_published": matches,
        "looks_recalled": bool(matches),
    }


def scan_archive(archive: Sequence[dict]) -> dict:
    results = [match_design(entry) for entry in archive if entry.get("code")]
    recalled = [r for r in results if r["looks_recalled"]]
    return {
        "n_designs": len(results),
        "n_matching_published": len(recalled),
        "match_rate": len(recalled) / len(results) if results else 0.0,
        "matches": recalled,
    }


def prior_archive_overlap(archive: Sequence[dict], prior_results_dir: str) -> dict:
    """This repository ships prior ADAS archives under ``results/``.

    Exact name collisions with those are the clearest possible contamination
    signal and cost nothing to check.
    """
    prior_names = set()
    if os.path.isdir(prior_results_dir):
        for filename in os.listdir(prior_results_dir):
            if not filename.endswith(".json"):
                continue
            try:
                with open(os.path.join(prior_results_dir, filename), "r") as handle:
                    for entry in json.load(handle):
                        if isinstance(entry, dict) and entry.get("name"):
                            prior_names.add(_normalize(entry["name"]))
            except (json.JSONDecodeError, OSError):
                continue

    collisions = [entry.get("name") for entry in archive
                  if entry.get("name") and _normalize(entry["name"]) in prior_names]
    return {
        "n_prior_names": len(prior_names),
        "n_collisions": len(collisions),
        "collisions": collisions,
    }


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())
