"""Random recombination: the reference condition that samples design space
without optimization pressure.

This is the only condition that produces unbiased design/fitness pairs, which is
what the component-attribution analysis needs -- search moves *away* from poor
designs and therefore under-samples exactly the region that attribution requires.
Budget it as a data source, not as a checkbox.

Recombination is done on the AST so the output is always syntactically valid:
one parent supplies the control flow, the other supplies instruction strings,
role literals and numeric constants. No meta-agent reasoning is involved.
"""

from __future__ import annotations

import ast
import random
from typing import Dict, List, Optional


def _collect_donor_pool(tree: ast.AST) -> Dict[str, list]:
    pool: Dict[str, list] = {"strings": [], "numbers": [], "roles": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and len(node.value) > 20:
                pool["strings"].append(node.value)
            elif isinstance(node.value, bool):
                continue
            elif isinstance(node.value, (int, float)):
                pool["numbers"].append(node.value)
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "role" and isinstance(keyword.value, ast.Constant):
                    pool["roles"].append(keyword.value.value)
    return pool


class _Splicer(ast.NodeTransformer):
    def __init__(self, pool: Dict[str, list], rng: random.Random, rate: float) -> None:
        self.pool = pool
        self.rng = rng
        self.rate = rate

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        for keyword in node.keywords:
            if (keyword.arg == "role" and self.pool["roles"]
                    and isinstance(keyword.value, ast.Constant)
                    and self.rng.random() < self.rate):
                keyword.value = ast.Constant(value=self.rng.choice(self.pool["roles"]))
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, bool):
            return node
        if isinstance(node.value, str) and len(node.value) > 20 and self.pool["strings"]:
            if self.rng.random() < self.rate:
                return ast.Constant(value=self.rng.choice(self.pool["strings"]))
        if isinstance(node.value, int) and 1 <= node.value <= 12:
            # loop counts and ensemble widths, the design knobs worth perturbing
            if self.rng.random() < self.rate:
                return ast.Constant(value=self.rng.randint(1, 6))
        if isinstance(node.value, float) and 0.0 <= node.value <= 2.0:
            if self.rng.random() < self.rate:
                return ast.Constant(value=round(self.rng.uniform(0.0, 1.0), 2))
        return node


def random_recombination(archive: List[dict], rng: Optional[random.Random] = None,
                         rate: float = 0.35) -> dict:
    """Splice two archive entries into a new candidate. No LLM call is made."""
    rng = rng or random.Random()
    candidates = [entry for entry in archive if entry.get("code")]
    if not candidates:
        raise ValueError("random recombination needs a non-empty archive")

    host = rng.choice(candidates)
    donor = rng.choice(candidates)

    host_tree = ast.parse(host["code"])
    donor_tree = ast.parse(donor["code"])
    pool = _collect_donor_pool(donor_tree)

    spliced = _Splicer(pool, rng, rate).visit(host_tree)
    ast.fix_missing_locations(spliced)
    code = ast.unparse(spliced)

    return {
        "thought": (
            "Random recombination reference condition: no meta reasoning was used. "
            f"Control flow taken from '{host.get('name')}'; instruction strings, role "
            f"literals and numeric constants sampled from '{donor.get('name')}'."
        ),
        "name": f"Recombination of {host.get('name')} and {donor.get('name')}",
        "code": code,
        "recombination_parents": [host.get("name"), donor.get("name")],
    }
