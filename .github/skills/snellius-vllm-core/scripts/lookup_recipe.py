#!/usr/bin/env python3
"""Look up a vLLM recipe for a model from recipes.vllm.ai.

Usage: lookup_recipe.py <model-query>
Prints JSON to stdout: {"status": "found"|"ambiguous"|"not_found"|"error", ...}
"""
import difflib
import json
import sys
import urllib.error
import urllib.request

INDEX_URL = "https://recipes.vllm.ai/models.json"
BASE_URL = "https://recipes.vllm.ai"


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_candidates(query, entries):
    query_lower = query.strip().lower()

    exact = [
        e for e in entries
        if e["hf_id"].lower() == query_lower or e["title"].lower() == query_lower
    ]
    if exact:
        return "exact", exact[:1]

    substring = [
        e for e in entries
        if query_lower in e["hf_id"].lower() or query_lower in e["title"].lower()
    ]
    if len(substring) == 1:
        return "exact", substring
    if len(substring) > 1:
        return "ambiguous", substring

    names = [e["hf_id"] for e in entries] + [e["title"] for e in entries]
    close = difflib.get_close_matches(query, names, n=5, cutoff=0.6)
    if close:
        matched = [e for e in entries if e["hf_id"] in close or e["title"] in close]
        seen = set()
        deduped = []
        for e in matched:
            if e["url"] not in seen:
                seen.add(e["url"])
                deduped.append(e)
        return "suggestions", deduped[:5]

    return "none", []


def _summarize(entry):
    return {
        "hf_id": entry["hf_id"],
        "title": entry["title"],
        "provider": entry.get("provider"),
        "url": BASE_URL + entry["url"],
    }


def lookup(query):
    entries = fetch_json(INDEX_URL)
    kind, matches = find_candidates(query, entries)

    if kind == "exact":
        entry = matches[0]
        recipe = fetch_json(BASE_URL + entry["json"])
        result = _summarize(entry)
        return {"status": "found", "matched": result, "recipe": recipe}
    if kind == "ambiguous":
        return {"status": "ambiguous", "candidates": [_summarize(e) for e in matches]}
    if kind == "suggestions":
        return {"status": "not_found", "suggestions": [_summarize(e) for e in matches]}
    return {"status": "not_found", "suggestions": []}


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"status": "error", "message": "usage: lookup_recipe.py <model-query>"}))
        sys.exit(2)
    query = sys.argv[1]
    try:
        result = lookup(query)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
