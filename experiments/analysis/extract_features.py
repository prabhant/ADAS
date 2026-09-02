"""Layer 1: AST canonicalization and structural feature extraction.

Normalization is what decides whether this layer measures anything real. A weak
meta agent can rename ``Chain-of-Thought`` to ``Adaptive Reasoning Synthesizer``,
rewrite every comment and change nothing functional. So before any feature is
read off, the code is:

* parsed with ``ast``, which discards comments for free
* stripped of docstrings explicitly
* alpha-renamed: local variables become ``v0, v1, ...`` in first-appearance order
* string-literal-blanked: every string becomes ``STR`` (they are handled in
  Layer 3 instead)

Numeric literals are *retained*, because ``N = 3`` versus ``N = 5`` is a genuine
design difference.
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

STR_PLACEHOLDER = "STR"

# Names that carry meaning and must survive canonicalization.
RESERVED_NAMES = {
    "self", "taskInfo", "LLMAgentBase", "Info", "AgentSystem", "range", "len",
    "enumerate", "zip", "list", "dict", "set", "str", "int", "float", "bool",
    "sum", "max", "min", "sorted", "Counter", "defaultdict", "print", "np",
    "numpy", "json", "os", "re", "math", "random", "copy", "itertools",
    "functools", "collections", "True", "False", "None",
}

FEATURE_NAMES = (
    "n_agent_sites",
    "n_agents_static",
    "n_call_sites",
    "max_loop_depth",
    "n_branches",
    "has_dynamic_routing",
    "n_distinct_roles",
    "n_output_fields",
    "has_aggregation",
    "has_context_accumulation",
    "n_temperatures",
    "n_statements",
)

AGGREGATION_MARKERS = {"Counter", "most_common", "majority", "mode", "bincount", "argmax"}


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return tree


class _Canonicalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.mapping: Dict[str, str] = {}

    def _rename(self, name: str) -> str:
        if name in RESERVED_NAMES or name.startswith("__"):
            return name
        if name not in self.mapping:
            self.mapping[name] = f"v{len(self.mapping)}"
        return self.mapping[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._rename(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._rename(node.arg)
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # The entry point is always `forward`; inner helper names are noise.
        if node.name != "forward":
            node.name = self._rename(node.name)
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, str):
            return ast.Constant(value=STR_PLACEHOLDER)
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.keyword:
        # Keyword *names* are API surface, not local naming, so leave them alone.
        node.value = self.visit(node.value)
        return node


def canonicalize(code: str) -> ast.AST:
    tree = ast.parse(code)
    tree = _strip_docstrings(tree)
    tree = _Canonicalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def canonical_source(code: str) -> str:
    """Canonical code as text. Used for blind LLM-judge comparisons."""
    return ast.unparse(canonicalize(code))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _literal_int_env(tree: ast.AST) -> Dict[str, int]:
    """Names bound once to an integer literal, so comprehension widths resolve."""
    env: Dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            env[node.targets[0].id] = node.value.value
    return env


def _resolve_repeat(node: ast.AST, env: Dict[str, int]) -> Optional[int]:
    """How many times a comprehension generator repeats, if statically knowable."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        if len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                return arg.value
            if isinstance(arg, ast.Name):
                return env.get(arg.id)
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    return None


def _is_agent_construction(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "LLMAgentBase")


def _reads_content(node: ast.AST) -> bool:
    return any(isinstance(sub, ast.Attribute) and sub.attr == "content" for sub in ast.walk(node))


def extract_features(code: str) -> Dict[str, Any]:
    """Structural feature vector plus the sets the other layers need."""
    tree = canonicalize(code)
    env = _literal_int_env(tree)

    agent_names: Set[str] = set()
    agent_sites = 0
    agents_static = 0
    roles: Set[str] = set()
    output_fields: Set[str] = set()
    temperatures: Set[float] = set()
    unknown_multiplicity = False

    # Pass 1: agent construction sites and the names they bind to.
    for node in ast.walk(tree):
        if _is_agent_construction(node):
            agent_sites += 1
        if isinstance(node, ast.Assign) and _is_agent_construction(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    agent_names.add(target.id)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            if not _is_agent_construction(node.elt):
                continue
            # A comprehension is one AST node but N agents at runtime.
            repeat = 1
            for generator in node.generators:
                count = _resolve_repeat(generator.iter, env)
                if count is None:
                    unknown_multiplicity = True
                    repeat = None
                    break
                repeat *= count
            if repeat is None:
                continue
            agents_static += repeat - 1  # the site itself is counted below
            if isinstance(node, ast.ListComp):
                pass
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.ListComp, ast.GeneratorExp)):
            if _is_agent_construction(node.value.elt):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        agent_names.add(target.id)

    agents_static += agent_sites

    # Pass 2: everything read off the raw (uncanonicalized) tree, because roles
    # and output fields are string literals that canonicalization blanked.
    raw = _strip_docstrings(ast.parse(code))
    for node in ast.walk(raw):
        if not _is_agent_construction(node):
            continue
        if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
            for element in node.args[0].elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    output_fields.add(element.value)
        for keyword in node.keywords:
            if keyword.arg == "output_fields" and isinstance(keyword.value, (ast.List, ast.Tuple)):
                for element in keyword.value.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        output_fields.add(element.value)
            if keyword.arg == "role" and isinstance(keyword.value, ast.Constant):
                roles.add(str(keyword.value.value))
            if keyword.arg == "role" and isinstance(keyword.value, (ast.List, ast.Tuple)):
                for element in keyword.value.elts:
                    if isinstance(element, ast.Constant):
                        roles.add(str(element.value))
            if keyword.arg == "temperature" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, (int, float)):
                    temperatures.add(float(keyword.value.value))
    # Role lists declared separately from the constructor, e.g. a roles= list.
    for node in ast.walk(raw):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any("role" in name.lower() for name in target_names):
                for element in node.value.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        roles.add(element.value)

    # Structural counts.
    n_call_sites = 0
    n_branches = 0
    has_dynamic_routing = False
    has_aggregation = False
    has_context_accumulation = False
    n_statements = sum(1 for node in ast.walk(tree) if isinstance(node, ast.stmt))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in agent_names:
                n_call_sites += 1
            elif isinstance(func, ast.Attribute) and func.attr in ("query", "__call__"):
                n_call_sites += 1
            elif isinstance(func, ast.Subscript):
                n_call_sites += 1  # agents[i](...) inside a loop
            elif isinstance(func, ast.Name) and func.id in AGGREGATION_MARKERS:
                has_aggregation = True
            elif isinstance(func, ast.Attribute):
                if func.attr in AGGREGATION_MARKERS:
                    has_aggregation = True
                if func.attr in ("extend", "append"):
                    has_context_accumulation = True
        if isinstance(node, ast.If):
            n_branches += 1
            if _reads_content(node.test):
                has_dynamic_routing = True
        if isinstance(node, ast.Name) and node.id in AGGREGATION_MARKERS:
            has_aggregation = True

    max_loop_depth = _max_loop_depth(tree)

    features = {
        "n_agent_sites": agent_sites,
        "n_agents_static": agents_static,
        "n_call_sites": n_call_sites,
        "max_loop_depth": max_loop_depth,
        "n_branches": n_branches,
        "has_dynamic_routing": int(has_dynamic_routing),
        "n_distinct_roles": len(roles),
        "n_output_fields": len(output_fields),
        "has_aggregation": int(has_aggregation),
        "has_context_accumulation": int(has_context_accumulation),
        "n_temperatures": len(temperatures),
        # Never raw line count: comment verbosity varies enormously between
        # models and would otherwise dominate the complexity metric.
        "n_statements": n_statements,
    }
    return {
        "features": features,
        "output_field_set": sorted(output_fields),
        "role_set": sorted(roles),
        "temperatures": sorted(temperatures),
        "agent_names": sorted(agent_names),
        "unknown_multiplicity": unknown_multiplicity,
    }


def _max_loop_depth(tree: ast.AST) -> int:
    def walk(node: ast.AST, depth: int) -> int:
        best = depth
        for child in ast.iter_child_nodes(node):
            child_depth = depth + 1 if isinstance(child, (ast.For, ast.While, ast.AsyncFor)) else depth
            best = max(best, walk(child, child_depth))
        return best

    return walk(tree, 0)


def feature_vector(record: Dict[str, Any]) -> List[float]:
    return [float(record["features"][name]) for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Archive loading
# ---------------------------------------------------------------------------

def load_archive(path: str) -> List[dict]:
    with open(path, "r") as handle:
        return json.load(handle)


def iter_cell_archives(results_root: str) -> Iterator[tuple]:
    """Yield ``(cell_id, archive)`` for every ``*_run_archive.json`` found."""
    for dirpath, _dirnames, filenames in os.walk(results_root):
        for filename in sorted(filenames):
            if not filename.endswith("_run_archive.json"):
                continue
            cell_id = os.path.basename(dirpath) or filename[: -len("_run_archive.json")]
            if cell_id in ("results", os.path.basename(results_root)):
                cell_id = filename[: -len("_run_archive.json")]
            yield cell_id, load_archive(os.path.join(dirpath, filename))


def load_abandoned(results_root: str) -> List[dict]:
    """Abandoned generations, which the search loop now persists.

    These matter for attribution: search moves away from poor designs, so the
    archive under-samples exactly the region the analysis needs.
    """
    records: List[dict] = []
    for dirpath, _dirnames, filenames in os.walk(results_root):
        for filename in filenames:
            if not filename.endswith("abandoned_generations.json"):
                continue
            with open(os.path.join(dirpath, filename), "r") as handle:
                for entry in json.load(handle):
                    entry["cell_id"] = os.path.basename(dirpath)
                    entry["abandoned"] = True
                    records.append(entry)
    return records


def describe(designs: Iterable[dict]) -> List[dict]:
    """Feature records for a set of archive entries, skipping unparseable code."""
    out: List[dict] = []
    for design in designs:
        code = design.get("code")
        if not code:
            continue
        try:
            record = extract_features(code)
        except SyntaxError:
            continue
        record["name"] = design.get("name")
        record["generation"] = design.get("generation")
        record["cell_id"] = design.get("cell_id")
        out.append(record)
    return out
