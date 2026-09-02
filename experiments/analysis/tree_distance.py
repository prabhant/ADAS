"""Layer 1b: normalized tree edit distance over the canonicalized AST.

Zhang-Shasha on 30-60 node functions is instant, so this costs nothing and
catches reorganizations the feature vector misses -- while the feature vector
catches semantics that tree edit distance treats as superficial. Neither
subsumes the other, so both are reported.
"""

from __future__ import annotations

import ast
from typing import List, Optional

try:
    import zss
except ImportError:  # analysis-only dependency
    zss = None

from .extract_features import canonicalize


class _Node:
    __slots__ = ("label", "children")

    def __init__(self, label: str, children: Optional[List["_Node"]] = None) -> None:
        self.label = label
        self.children = children or []

    @staticmethod
    def get_children(node: "_Node") -> List["_Node"]:
        return node.children

    @staticmethod
    def get_label(node: "_Node") -> str:
        return node.label


def _label(node: ast.AST) -> str:
    kind = type(node).__name__
    if isinstance(node, ast.Constant) and not isinstance(node.value, str):
        # Numeric literals are retained: N=3 versus N=5 is a design difference.
        return f"Constant:{node.value!r}"
    if isinstance(node, ast.Name):
        return f"Name:{node.id}"
    if isinstance(node, ast.Attribute):
        return f"Attribute:{node.attr}"
    if isinstance(node, ast.keyword):
        return f"keyword:{node.arg}"
    return kind


def to_tree(node: ast.AST) -> _Node:
    return _Node(_label(node), [to_tree(child) for child in ast.iter_child_nodes(node)])


def tree_size(node: _Node) -> int:
    return 1 + sum(tree_size(child) for child in node.children)


def tree_edit_distance(code_a: str, code_b: str) -> float:
    """Edit distance normalized by combined tree size, so it lands in [0, 1]."""
    if zss is None:
        raise ImportError("zss is required for tree edit distance: pip install zss")

    tree_a = to_tree(canonicalize(code_a))
    tree_b = to_tree(canonicalize(code_b))
    raw = zss.simple_distance(tree_a, tree_b,
                              get_children=_Node.get_children,
                              get_label=_Node.get_label)
    denominator = tree_size(tree_a) + tree_size(tree_b)
    return float(raw) / denominator if denominator else 0.0


def pairwise_matrix(codes: List[str]) -> List[List[float]]:
    n = len(codes)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance = tree_edit_distance(codes[i], codes[j])
            matrix[i][j] = matrix[j][i] = distance
    return matrix
