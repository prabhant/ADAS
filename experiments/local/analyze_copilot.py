"""Phase 4 analysis for the Copilot arm, on a chosen set of search cells.

The SLURM analysis job (experiments/slurm/analysis.slurm) walks one results root
and reads cells.yaml. The Copilot root also holds screen seeds, superseded cells
and test-only archives, so this driver stages copies of just the named cells and
runs the existing analysis layers on them:

* the panel comparison: the design selected on validation (earliest generation on
  ties) against Self-Consistency and against the best seed, paired bootstrap over
  test questions;
* aggregate (novelty, complexity, Pareto front, per-worker curves), contamination
  scan against published designs, motif counts, designer signatures (no judge),
  failure histograms and figures.

    python experiments/local/analyze_copilot.py cp2_claude-haiku-5.5_drop_claude-haiku-5.5_a \\
        cp2_claude-haiku-5.5_drop_claude-haiku-5.5_b --domain drop
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import traceback
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from experiments.local.run_cell import CELLS_PATH, all_cells, load_yaml, merge  # noqa: E402

DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, "experiments", "results_copilot")
STAGED_FILES = ("{c}_run_archive.json", "{c}_run_archive_evaluate.json")
STAGED_RUN_FILES = ("{c}_abandoned_generations.json", "budget_summary.json", "{c}_failures.jsonl")


def seed_codes(domain: str) -> List[str]:
    path = os.path.join(REPO_ROOT, f"_{domain}", f"{domain}_prompt.py")
    spec = importlib.util.spec_from_file_location(f"_analysis_seeds_{domain}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(os.path.dirname(path))
    return [entry["code"] for entry in module.get_init_archive() if entry.get("code")]


def stage(cells: List[str], results_root: str, stage_root: str) -> None:
    if os.path.isdir(stage_root):
        shutil.rmtree(stage_root)
    for cell in cells:
        src, dst = os.path.join(results_root, cell), os.path.join(stage_root, cell)
        os.makedirs(os.path.join(dst, cell), exist_ok=True)
        for pattern in STAGED_FILES:
            path = os.path.join(src, pattern.format(c=cell))
            if os.path.exists(path):
                shutil.copy2(path, dst)
        for pattern in STAGED_RUN_FILES:
            path = os.path.join(src, cell, pattern.format(c=cell))
            if os.path.exists(path):
                shutil.copy2(path, os.path.join(dst, cell))


def paired_bootstrap(a: List[float], b: List[float], n: int = 10000, seed: int = 0) -> Dict[str, float]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return {"delta": float(diff.mean()), "low": float(np.percentile(means, 2.5)),
            "high": float(np.percentile(means, 97.5))}


def panel_row(cell: str, results_root: str) -> Optional[dict]:
    """The columns of the batch 5/8 panel tables, for one designer cell."""
    archive_path = os.path.join(results_root, cell, f"{cell}_run_archive.json")
    eval_path = os.path.join(results_root, cell, f"{cell}_run_archive_evaluate.json")
    if not os.path.exists(archive_path):
        return None
    archive = json.load(open(archive_path))
    evaluated = json.load(open(eval_path)) if os.path.exists(eval_path) else []
    tested = {i: e for i, e in enumerate(evaluated) if "test_per_question" in e}
    generated = [(i, e) for i, e in enumerate(archive) if isinstance(e.get("generation"), int)]
    seeds = [(i, e) for i, e in enumerate(archive) if e.get("generation") == "initial"]
    row: dict = {"cell": cell, "generations": len(generated)}
    budget = os.path.join(results_root, cell, cell, "budget_summary.json")
    if os.path.exists(budget):
        summary = json.load(open(budget))
        row.update(attempts=summary.get("attempts"), failures=summary.get("failure_histogram"))
    if not generated:
        return row
    sel_i, sel = max(generated, key=lambda ie: (ie[1].get("raw_median_accuracy", 0.0), -ie[1]["generation"]))
    row.update(selected_name=sel["name"], selected_generation=sel["generation"],
               selected_validation=sel.get("raw_median_accuracy"),
               selected_calls_per_question=(sel.get("resource") or {}).get("calls_per_question"),
               selected_credits_per_question=(sel.get("resource") or {}).get("credits_per_question"),
               broken=sum(1 for _, e in generated if e.get("raw_median_accuracy", 0) < 0.3))
    seed_vals = [e.get("raw_median_accuracy", 0.0) for _, e in seeds]
    if seed_vals:
        row["seed_mean_validation"] = float(np.mean(seed_vals))
        row["designs_above_seed_mean_validation"] = sum(
            1 for _, e in generated if e.get("raw_median_accuracy", 0.0) >= np.mean(seed_vals))
    row["mean_calls_per_question"] = float(np.mean([(e.get("resource") or {}).get("calls_per_question", 0.0)
                                                    for _, e in generated]))
    if sel_i in tested:
        sel_vec = tested[sel_i]["test_per_question"]
        row["selected_test"] = float(np.mean(sel_vec))
        named = {e["name"]: tested[i] for i, e in seeds if i in tested}
        for label, name in (("vs_self_consistency", "Self-Consistency with Chain-of-Thought"),
                            ("vs_debate", "LLM Debate"), ("vs_cot", "Chain-of-Thought")):
            if name in named:
                row[label] = paired_bootstrap(sel_vec, named[name]["test_per_question"])
        best_seed_i, best_seed = max(seeds, key=lambda ie: ie[1].get("raw_median_accuracy", 0.0))
        if best_seed_i in tested:
            row["best_seed_by_validation"] = best_seed["name"]
            row["vs_best_seed"] = paired_bootstrap(sel_vec, tested[best_seed_i]["test_per_question"])
        row["seed_test"] = {named_e["name"]: float(np.mean(named_e["test_per_question"]))
                            for named_e in named.values()}
        tested_generated = [(archive[i].get("raw_median_accuracy"), float(np.mean(e["test_per_question"])),
                             archive[i]["generation"]) for i, e in tested.items()
                            if isinstance(archive[i].get("generation"), int)]
        row["tested_designs"] = [{"generation": g, "validation": v, "test": t} for v, t, g in tested_generated]
        best_test = max((t for _, t, _ in tested_generated), default=None)
        row["best_tested_design_test"] = best_test
    return row


def run_layer(name: str, func, errors: Dict[str, str]):
    try:
        return func()
    except Exception as exc:  # noqa: BLE001 - one layer must not sink the report
        errors[name] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 analysis of Copilot search cells")
    parser.add_argument("cells", nargs="+")
    parser.add_argument("--domain", default="drop", help="domain whose seed agents are the novelty reference")
    parser.add_argument("--results_root",
                        default=os.environ.get("ADAS_COPILOT_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    parser.add_argument("--name", default="phase2", help="output subdirectory under <results_root>/_analysis")
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    out_dir = os.path.join(results_root, "_analysis", args.name)
    stage_root = os.path.join(out_dir, "stage")
    stage(args.cells, results_root, stage_root)
    grid = load_yaml(CELLS_PATH)
    cells_config = {"cells": [merge(cell, grid) for cell, problem in all_cells(grid) if not problem]}
    seeds = seed_codes(args.domain)
    errors: Dict[str, str] = {}
    report: dict = {"cells": args.cells, "domain": args.domain}

    report["panel"] = [panel_row(cell, results_root) for cell in args.cells]

    from experiments.analysis import aggregate as agg
    from experiments.analysis.contamination import scan_archive
    from experiments.analysis.lineage import detect_motifs

    aggregate_path = os.path.join(out_dir, "campaign.json")
    payload = run_layer("aggregate", lambda: agg.aggregate(stage_root, seeds, cells_config, aggregate_path), errors)
    if payload:
        report["aggregate_cells"] = payload.get("cells")
        report["pareto_front"] = payload.get("pareto_front")
    campaign = run_layer("load", lambda: agg.load_campaign(stage_root, cells_config), errors) or {}
    report["contamination"] = run_layer("contamination", lambda: {
        cell: scan_archive([e for e in archive if isinstance(e.get("generation"), int)])
        for cell, archive in campaign.items()}, errors)
    report["motifs"] = run_layer("motifs", lambda: {
        cell: dict(sum((Counter({k: 1 for k, v in detect_motifs(e["code"]).items() if v})
                        for e in archive if isinstance(e.get("generation"), int) and e.get("code")), Counter()))
        for cell, archive in campaign.items()}, errors)
    report["designer_signatures"] = run_layer(
        "signatures", lambda: agg.designer_signatures(stage_root, seeds, cells_config), errors)

    def figures():
        from experiments.analysis import figures as fig
        from experiments.analysis.attribution import build_design_table

        rows = build_design_table([e for archive in campaign.values() for e in archive])
        histograms = {}
        for cell in args.cells:
            path = os.path.join(stage_root, cell, cell, "budget_summary.json")
            histograms[cell] = json.load(open(path)).get("failure_histogram", {}) if os.path.exists(path) else {}
        return fig.render_all(aggregate_path, os.path.join(out_dir, "figures"), design_rows=rows,
                              signatures=report.get("designer_signatures") or {},
                              failure_histograms=histograms, crosseval_rows=[])

    report["figures"] = run_layer("figures", figures, errors) if payload else None
    report["layer_errors"] = errors
    with open(os.path.join(out_dir, "report.json"), "w") as handle:
        json.dump(report, handle, indent=2, default=str)

    print(f"\n{'cell':45} {'gens':>5} {'sel gen':>7} {'val':>6} {'test':>6} {'vs SC [95% CI]':>24} "
          f"{'vs best seed [95% CI]':>24} {'calls/q':>8}")
    for row in report["panel"]:
        if not row or "selected_name" not in row:
            continue

        def fmt(c):
            return f"{100 * c['delta']:+.1f} [{100 * c['low']:+.1f}, {100 * c['high']:+.1f}]" if c else "-"

        test = row.get("selected_test")
        print(f"{row['cell'][-45:]:45} {row['generations']:5d} {row['selected_generation']:7d} "
              f"{100 * row['selected_validation']:5.1f}% {(100 * test if test is not None else float('nan')):5.1f}% "
              f"{fmt(row.get('vs_self_consistency')):>24} {fmt(row.get('vs_best_seed')):>24} "
              f"{row.get('selected_calls_per_question') or 0:8.1f}")
    print(f"\nreport: {os.path.join(out_dir, 'report.json')}")
    if errors:
        print("layers that failed:", json.dumps(errors, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
