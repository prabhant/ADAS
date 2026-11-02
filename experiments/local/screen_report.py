"""Phase 1 screen table and the pre-registered worker/domain pick (Copilot arm).

Reads the cp1_* cells under the Copilot results root and applies the rule fixed
in experimental_plan_copilot.md before any result was read:

1. keep pairs whose CoT accuracy (validation, 3 passes) is between 30% and 85%;
2. pick the pair with the most of the six non-CoT seed agents at >= CoT + 5 points;
3. break ties toward claude-haiku-5.5.

A tie the rule leaves open is reported, and broken by the larger mean seed gain
over CoT, flagged as outside the pre-registered rule.

    python experiments/local/screen_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, "experiments", "results_copilot")
WORKERS = ("claude-haiku-5.5", "gpt-6-luna", "gpt-5.6-luna")
DOMAINS = ("mgsm", "mmlu", "drop", "gpqa")
PREFERRED_WORKER = "claude-haiku-5.5"
COT_RANGE = (0.30, 0.85)
MARGIN = 0.05


def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _credits(results_root: str, cell: str) -> float:
    path = os.path.join(results_root, cell, "copilot_ledger.jsonl")
    if not os.path.exists(path):
        return 0.0
    with open(path) as handle:
        return sum(json.loads(line).get("credits_charged") or 0.0 for line in handle)


def screen_rows(results_root: str) -> list:
    rows = []
    for domain in DOMAINS:
        for worker in WORKERS:
            tag = f"{domain}_{worker}"
            cot = _load(os.path.join(results_root, f"cp1_cot_{tag}", "cot_baseline.json"))
            bare = _load(os.path.join(results_root, f"cp1_bare_{tag}", "bare_baseline.json"))
            seeds = _load(os.path.join(results_root, f"cp1_seeds_{tag}", f"cp1_seeds_{tag}_run_archive.json"))
            row = {"domain": domain, "worker": worker,
                   "cot": cot["accuracy"] if cot else None,
                   "bare": bare["accuracy"] if bare else None,
                   "credits": sum(_credits(results_root, f"cp1_{k}_{tag}") for k in ("cot", "bare", "seeds"))}
            if seeds and all("raw_median_accuracy" in s for s in seeds):
                others = {s["name"]: s["raw_median_accuracy"] for s in seeds if s["name"] != "Chain-of-Thought"}
                row["seeds"] = others
                row["best_seed"] = max(others.items(), key=lambda kv: kv[1])
                if row["cot"] is not None:
                    row["count"] = sum(acc >= row["cot"] + MARGIN for acc in others.values())
                    row["mean_gain"] = statistics.fmean(acc - row["cot"] for acc in others.values())
            rows.append(row)
    return rows


def pick(rows: list) -> dict:
    complete = [r for r in rows if r["cot"] is not None and "count" in r]
    eligible = [r for r in complete if COT_RANGE[0] <= r["cot"] <= COT_RANGE[1]]
    out = {"complete": len(complete), "total": len(rows), "eligible": [(r["domain"], r["worker"]) for r in eligible]}
    if not eligible:
        out["pick"] = None
        return out
    best = max(r["count"] for r in eligible)
    top = [r for r in eligible if r["count"] == best]
    preferred = [r for r in top if r["worker"] == PREFERRED_WORKER]
    if len(preferred) == 1 or (len(top) == 1):
        choice = (preferred or top)[0]
        out["outside_rule"] = False
    else:
        pool = preferred or top
        choice = max(pool, key=lambda r: r["mean_gain"])
        out["outside_rule"] = True
        out["tied"] = [(r["domain"], r["worker"]) for r in pool]
    out["pick"] = {"domain": choice["domain"], "worker": choice["worker"], "count": choice["count"],
                   "cot": choice["cot"]}
    others = [w for w in WORKERS if w != choice["worker"]]
    out["selection"] = {"worker": choice["worker"], "domain": choice["domain"], "transfer_workers": others}
    return out


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 screen table and pick")
    parser.add_argument("--results_root",
                        default=os.environ.get("ADAS_COPILOT_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    args = parser.parse_args()
    rows = screen_rows(os.path.abspath(args.results_root))
    print(f"{'domain':6} {'worker':18} {'bare':>7} {'CoT':>7} {'best seed':>36} {'>=CoT+5':>8} {'credits':>8}")
    for r in rows:
        best = f"{r['best_seed'][0][:26]} {_pct(r['best_seed'][1])}" if r.get("best_seed") else "-"
        count = f"{r['count']}/6" if "count" in r else "-"
        print(f"{r['domain']:6} {r['worker']:18} {_pct(r['bare']):>7} {_pct(r['cot']):>7} {best:>36} "
              f"{count:>8} {r['credits']:8.1f}")
    print(f"\ntotal screen credits: {sum(r['credits'] for r in rows):.1f}")
    result = pick(rows)
    print(f"complete pairs: {result['complete']}/{result['total']}; eligible (CoT in 30-85%): {result['eligible']}")
    if result["complete"] < result["total"]:
        print("screen incomplete: the pick below is provisional")
    if result.get("pick"):
        note = " (tie broken by mean seed gain, outside the pre-registered rule)" if result["outside_rule"] else ""
        print(f"pick: {result['pick']}{note}")
        print("selection for copilot_cells.yaml:", json.dumps(result["selection"]))
    else:
        print("pick: none -- no pair has CoT accuracy in the 30-85% band")
    return 0


if __name__ == "__main__":
    sys.exit(main())
