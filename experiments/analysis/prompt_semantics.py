"""Layer 3: prompt semantics.

A large share of each design lives in string literals -- ``cot_instruction``,
``debate_instruction``, the ``role=`` lists. Layer 1 deliberately blanked them,
so they are handled here.

Two prompt sets are matched with the Hungarian algorithm and the matched cosine
distances averaged, with a fixed penalty per unmatched prompt.

The embedding model is held out of the experiment. Embedding with a ladder model
would risk it scoring its own phrasing as more self-similar, which correlates
with the exact axis under study.

Following MaAS, prompts can optionally be replaced by an LLM-written structured
*description* before embedding, which gives more stable clusters than embedding
raw instruction text.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # analysis-only dependency
    linear_sum_assignment = None

HELD_OUT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
UNMATCHED_PENALTY = 1.0
MIN_PROMPT_CHARS = 20

_MODEL_CACHE: Dict[str, object] = {}


def get_embedder(model_name: str = HELD_OUT_EMBEDDING_MODEL):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def extract_prompts(code: str) -> Dict[str, List[str]]:
    """Instruction strings and role literals, kept separate."""
    tree = ast.parse(code)
    instructions: List[str] = []
    roles: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "LLMAgentBase":
            for keyword in node.keywords:
                if keyword.arg != "role":
                    continue
                if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    roles.append(keyword.value.value)
                elif isinstance(keyword.value, (ast.List, ast.Tuple)):
                    roles.extend(e.value for e in keyword.value.elts
                                 if isinstance(e, ast.Constant) and isinstance(e.value, str))

    role_set = set(roles)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if len(value) >= MIN_PROMPT_CHARS and value not in role_set:
                instructions.append(value)

    return {"instructions": instructions, "roles": roles}


def embed(texts: Sequence[str], model_name: str = HELD_OUT_EMBEDDING_MODEL) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_embedder(model_name)
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def matched_distance(texts_a: Sequence[str], texts_b: Sequence[str],
                     model_name: str = HELD_OUT_EMBEDDING_MODEL,
                     penalty: float = UNMATCHED_PENALTY) -> float:
    """Hungarian-matched mean cosine distance, penalising unmatched prompts."""
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for the prompt semantics layer")
    if not texts_a and not texts_b:
        return 0.0
    if not texts_a or not texts_b:
        return penalty

    vectors_a = embed(texts_a, model_name)
    vectors_b = embed(texts_b, model_name)
    cost = 1.0 - vectors_a @ vectors_b.T

    rows, cols = linear_sum_assignment(cost)
    matched = float(cost[rows, cols].sum())
    n_matched = len(rows)
    n_unmatched = (len(texts_a) - n_matched) + (len(texts_b) - n_matched)
    total = matched + penalty * n_unmatched
    return total / (n_matched + n_unmatched)


def prompt_distance(code_a: str, code_b: str,
                    model_name: str = HELD_OUT_EMBEDDING_MODEL,
                    role_weight: float = 0.35) -> float:
    """Combined instruction and role distance for two designs."""
    a = extract_prompts(code_a)
    b = extract_prompts(code_b)
    instruction_distance = matched_distance(a["instructions"], b["instructions"], model_name)
    role_distance = matched_distance(a["roles"], b["roles"], model_name)
    return (1 - role_weight) * instruction_distance + role_weight * role_distance


def describe_operators(code: str, judge_call) -> List[str]:
    """MaAS variant: have a held-out LLM describe each operator, then embed that.

    ``judge_call`` takes a prompt string and returns text. Structured descriptions
    cluster more stably than raw instruction strings.
    """
    prompts = extract_prompts(code)["instructions"]
    descriptions = []
    for prompt in prompts:
        descriptions.append(judge_call(
            "Describe, in one sentence and in neutral terms, what role this instruction "
            "assigns and what operation it asks for. Do not quote it.\n\n" + prompt
        ))
    return descriptions
