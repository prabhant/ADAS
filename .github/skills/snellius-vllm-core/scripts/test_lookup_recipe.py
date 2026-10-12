#!/usr/bin/env python3
"""Self-check for lookup_recipe.py. Run directly: python3 test_lookup_recipe.py

The last two tests hit the real recipes.vllm.ai API and require network access.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lookup_recipe  # noqa: E402


def test_find_candidates_exact_hf_id():
    entries = [
        {"hf_id": "zai-org/GLM-4.7", "title": "GLM-4.7 Base", "url": "/zai-org/GLM-4.7", "json": "/zai-org/GLM-4.7.json"},
        {"hf_id": "other/Model", "title": "Model", "url": "/other/Model", "json": "/other/Model.json"},
    ]
    kind, matches = lookup_recipe.find_candidates("zai-org/GLM-4.7", entries)
    assert kind == "exact"
    assert matches[0]["hf_id"] == "zai-org/GLM-4.7"


def test_find_candidates_ambiguous_substring():
    entries = [
        {"hf_id": "zai-org/GLM-4.7", "title": "GLM-4.7 Base", "url": "/a", "json": "/a.json"},
        {"hf_id": "zai-org/GLM-4.7-FP8", "title": "GLM-4.7 FP8", "url": "/b", "json": "/b.json"},
    ]
    kind, matches = lookup_recipe.find_candidates("glm-4.7", entries)
    assert kind == "ambiguous"
    assert len(matches) == 2


def test_find_candidates_none():
    entries = [
        {"hf_id": "zai-org/GLM-4.7", "title": "GLM-4.7 Base", "url": "/a", "json": "/a.json"},
    ]
    kind, matches = lookup_recipe.find_candidates("totally-unrelated-xyz-987", entries)
    assert kind == "none"
    assert matches == []


def test_find_candidates_suggestions_fuzzy_match():
    entries = [
        {"hf_id": "zai-org/GLM-4.7", "title": "GLM-4.7 Base", "url": "/a", "json": "/a.json"},
        {"hf_id": "other/Unrelated", "title": "Something Else", "url": "/b", "json": "/b.json"},
    ]
    kind, matches = lookup_recipe.find_candidates("GLM-4.6", entries)
    assert kind == "suggestions"
    assert len(matches) == 1
    assert matches[0]["hf_id"] == "zai-org/GLM-4.7"


def test_lookup_live_known_model():
    result = lookup_recipe.lookup("zai-org/GLM-4.7")
    assert result["status"] == "found"
    assert "recipe" in result
    assert result["matched"]["hf_id"].lower() == "zai-org/glm-4.7"


def test_lookup_live_unknown_model():
    result = lookup_recipe.lookup("this-model-definitely-does-not-exist-xyz123")
    assert result["status"] == "not_found"


if __name__ == "__main__":
    tests = [
        test_find_candidates_exact_hf_id,
        test_find_candidates_ambiguous_substring,
        test_find_candidates_none,
        test_find_candidates_suggestions_fuzzy_match,
        test_lookup_live_known_model,
        test_lookup_live_unknown_model,
    ]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print("All tests passed.")
