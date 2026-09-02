"""Layer 6: designer signature -- what answers Q8.

Layers 1 to 5 characterize the *designs*. This layer characterizes the
*designer*. All five measurements are computed from artifacts already saved.

One caution carried through the code below: debug attempts and hallucinated APIs
measure coding competence, while archive utilization and motif repertoire measure
design behaviour. They are kept separate. A model that writes buggy code is not
necessarily a poor designer, and conflating the two would produce a tidier but
less honest story.
"""

from __future__ import annotations

import ast
import os
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

import yaml

from .extract_features import canonical_source, extract_features
from .lineage import MOTIFS, archive_utilization, build_lineage, detect_motifs, trajectory_shape

_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "config", "framework_api.yaml")


def load_framework_api(path: str = _CONFIG) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _known_names(api: dict) -> Dict[str, set]:
    agent_methods = set(api["classes"]["LLMAgentBase"]["methods"])
    agent_attrs = set(api["classes"]["LLMAgentBase"]["attributes"])
    self_methods = set(api.get("self_methods", []))
    module_functions = set(api.get("module_functions", []))
    for functions in (api.get("domain_functions") or {}).values():
        module_functions |= set(functions)
    info_fields = set(api["namedtuples"]["Info"]["fields"])
    ctor_args = set(api["classes"]["LLMAgentBase"]["constructor_args"])
    return {
        "agent_methods": agent_methods,
        "agent_attrs": agent_attrs,
        "self_methods": self_methods,
        "module_functions": module_functions,
        "info_fields": info_fields,
        "ctor_args": ctor_args,
    }


def hallucinated_apis(code: str, api: Optional[dict] = None) -> Dict[str, list]:
    """Calls to methods that do not exist in the provided framework.

    A pure capability signal, cleanly separable from design quality. The ADAS
    meta prompt carries a "WRONG Implementation examples" section precisely
    because models invent APIs, so this failure mode is expected and worth
    quantifying rather than merely debugging away.
    """
    api = api or load_framework_api()
    known = _known_names(api)
    allowed_modules = set(api.get("allowed_modules", []))

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"self_methods": [], "agent_methods": [], "info_fields": [],
                "constructor_args": [], "unknown_imports": [], "total": 0,
                "parse_error": True}

    record = extract_features(code)
    agent_names = set(record["agent_names"])

    bad_self: List[str] = []
    bad_agent: List[str] = []
    bad_info: List[str] = []
    bad_ctor: List[str] = []
    bad_imports: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            value = node.value
            if isinstance(value, ast.Name):
                if value.id == "self" and node.attr not in known["self_methods"]:
                    bad_self.append(node.attr)
                elif value.id in agent_names and node.attr not in (known["agent_methods"] | known["agent_attrs"]):
                    bad_agent.append(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "LLMAgentBase":
            for keyword in node.keywords:
                if keyword.arg and keyword.arg not in known["ctor_args"]:
                    bad_ctor.append(keyword.arg)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            names = [module] if module else [alias.name for alias in node.names]
            for name in names:
                root = (name or "").split(".")[0]
                if root and root not in allowed_modules and root != "utils":
                    bad_imports.append(root)

    # Attribute access on an Info result that is not one of its four fields.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr not in known["info_fields"] \
                and isinstance(node.value, ast.Name) and node.value.id not in agent_names \
                and node.value.id not in ("self", "np", "numpy", "json", "os", "re", "math",
                                          "random", "copy", "collections", "itertools", "functools"):
            if node.attr in ("answer", "result", "output", "text", "value", "response"):
                bad_info.append(node.attr)

    total = len(bad_self) + len(bad_agent) + len(bad_info) + len(bad_ctor) + len(bad_imports)
    return {
        "self_methods": bad_self,
        "agent_methods": bad_agent,
        "info_fields": bad_info,
        "constructor_args": bad_ctor,
        "unknown_imports": bad_imports,
        "total": total,
        "parse_error": False,
    }


THOUGHT_CODE_JUDGE_PROMPT = """You are grading whether a piece of code implements a described design.

You will be shown a natural-language DESIGN DESCRIPTION and a canonicalized
IMPLEMENTATION. Variable names and string literals have been normalized, so judge
structure and control flow, not wording.

Score the implementation on this scale:
5 - implements the described design faithfully and completely
4 - implements it, with a minor component missing or simplified
3 - implements a recognisable but materially simpler version
2 - implements something related but structurally different
1 - implements something unrelated to the description

Reply with JSON: {{"score": <1-5>, "justification": "<one sentence>"}}

DESIGN DESCRIPTION:
{thought}

IMPLEMENTATION:
{code}
"""


def thought_code_alignment(entry: dict, judge_call: Callable[[str], dict]) -> dict:
    """Does the code implement the design the thought describes?

    The sharpest signal in this layer. A weaker designer frequently writes a
    plausible, even sophisticated thought and then implements something simpler.
    This measures whether a model can execute its own stated plan, which is
    distinct from whether the plan was any good.

    ``judge_call`` must use a held-out model, and the code is canonicalized and
    name-stripped first so the judge cannot key on phrasing.
    """
    thought = entry.get("thought", "")
    code = entry.get("code", "")
    if not thought or not code:
        return {"score": None, "reason": "missing thought or code"}
    try:
        blinded = canonical_source(code)
    except SyntaxError:
        return {"score": None, "reason": "unparseable code"}

    verdict = judge_call(THOUGHT_CODE_JUDGE_PROMPT.format(thought=thought, code=blinded))
    return {"score": verdict.get("score"), "justification": verdict.get("justification")}


def motif_repertoire(archive: Sequence[dict]) -> dict:
    """The set of distinct building blocks a tier reaches for across a whole run.

    A designer that only ever ensembles is qualitatively different from one that
    routes and decomposes, even at identical accuracy.
    """
    counts = Counter()
    for entry in archive:
        if not entry.get("code"):
            continue
        for motif, present in detect_motifs(entry["code"]).items():
            if present:
                counts[motif] += 1
    total = sum(counts.values()) or 1
    return {
        "repertoire_size": len(counts),
        "counts": dict(counts),
        "composition": {motif: counts.get(motif, 0) / total for motif in MOTIFS},
        "dominant_motif": counts.most_common(1)[0][0] if counts else None,
        "concentration": max(counts.values()) / total if counts else 0.0,
    }


def debug_competence(archive: Sequence[dict], abandoned: Sequence[dict] = ()) -> dict:
    """Coding competence, reported separately from the design signals."""
    attempts = [entry.get("debug_attempts", 0) for entry in archive if "debug_attempts" in entry]
    abandoned_attempts = [entry.get("debug_attempts", 0) for entry in abandoned]
    all_attempts = attempts + abandoned_attempts
    terminal = Counter(entry.get("terminal_failure") for entry in abandoned)
    return {
        "n_successful": len(attempts),
        "n_abandoned": len(abandoned),
        "mean_debug_attempts": sum(all_attempts) / len(all_attempts) if all_attempts else 0.0,
        "clean_first_try_rate": sum(1 for a in attempts if a == 0) / len(attempts) if attempts else 0.0,
        "terminal_failure_histogram": {k: v for k, v in terminal.items() if k},
    }


def signature(archive: Sequence[dict], distance_fn, abandoned: Sequence[dict] = (),
              judge_call: Optional[Callable[[str], dict]] = None,
              api: Optional[dict] = None) -> dict:
    """The full per-tier designer signature row."""
    api = api or load_framework_api()
    designs = [entry for entry in archive if entry.get("code")]

    hallucination = [hallucinated_apis(entry["code"], api) for entry in designs]
    lineage = build_lineage(designs, distance_fn)

    row = {
        "n_designs": len(designs),
        # --- design behaviour ---
        "archive_utilization": archive_utilization(lineage, designs),
        "motif_repertoire": motif_repertoire(designs),
        "trajectory": trajectory_shape(lineage),
        # --- coding competence, kept separate ---
        "hallucinated_api_rate": (sum(h["total"] for h in hallucination) / len(hallucination)
                                  if hallucination else 0.0),
        "hallucinated_api_detail": Counter(
            name for h in hallucination for name in
            (h["self_methods"] + h["agent_methods"] + h["constructor_args"])
        ),
        "debug_competence": debug_competence(designs, abandoned),
    }

    if judge_call is not None:
        scores = [thought_code_alignment(entry, judge_call) for entry in designs]
        valid = [s["score"] for s in scores if s.get("score") is not None]
        row["thought_code_alignment"] = {
            "scores": valid,
            "mean": sum(valid) / len(valid) if valid else None,
            "distribution": dict(Counter(valid)),
        }
    return row
