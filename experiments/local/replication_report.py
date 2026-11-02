"""Report for the GPT-6 Luna replication: the pre-registered comparisons 1 to 9.

Reads the `cpl_*` cells (experimental_plan_copilot.md, "Replication with GPT-6
Luna") and prints Markdown tables; cells still running show as pending, so it can
be run at any point. Selection is by validation only (best validation, earliest
generation on ties); every interval is a paired bootstrap over the 300 test
questions, within a cell where possible.

    python experiments/local/replication_report.py
    python experiments/local/replication_report.py --json experiments/results_copilot/_analysis/luna/report.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import statistics
import sys
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from experiments.local.analyze_copilot import DEFAULT_RESULTS_ROOT, paired_bootstrap  # noqa: E402

WORKER, DOMAIN = "gpt-6-luna", "drop"
DESIGNERS = ("gpt-6-luna", "claude-haiku-5.5")
MODES = ("seq", "batch3")
REPLICATES = ("a", "b")
FORMAT_COT = "Chain-of-Thought, minimal answer format"
SC, COT, DEBATE = "Self-Consistency with Chain-of-Thought", "Chain-of-Thought", "LLM Debate"
# Answer-format wording or answer-normalising code (as in findings_copilot.md, Phase 3.1).
FORMAT_CUE = re.compile(r"digit|verbatim|normali[sz]|shortest|only the (final )?(short )?answer|"
                        r"exact (wording|span|text)|single number|no units|minimal form", re.I)


def search_cell(mode: str, designer: str, replicate: str) -> str:
    return f"cpl_{mode}_{designer}_{DOMAIN}_{WORKER}_{replicate}"


class Cell:
    """One cell's archive, test results and ledger."""

    def __init__(self, root: str, cell_id: str) -> None:
        self.id = cell_id
        directory = os.path.join(root, cell_id)
        self.archive = self._load(os.path.join(directory, f"{cell_id}_run_archive.json")) or []
        self.evaluated = self._load(os.path.join(directory, f"{cell_id}_run_archive_evaluate.json")) or []
        self.budget = self._load(os.path.join(directory, cell_id, "budget_summary.json")) or {}
        self.done = os.path.exists(os.path.join(directory, ".cell_done.json"))
        self.ledger = []
        ledger_path = os.path.join(directory, "copilot_ledger.jsonl")
        if os.path.exists(ledger_path):
            with open(ledger_path) as handle:
                for line in handle:
                    try:
                        self.ledger.append(json.loads(line))
                    except ValueError:
                        continue

    @staticmethod
    def _load(path: str):
        if not os.path.exists(path):
            return None
        try:
            with open(path) as handle:
                return json.load(handle)
        except ValueError:
            return None

    @property
    def generated(self) -> List[dict]:
        return [e for e in self.archive if isinstance(e.get("generation"), int)]

    @property
    def seeds(self) -> List[dict]:
        return [e for e in self.archive if e.get("generation") == "initial"]

    def selected_index(self) -> Optional[int]:
        indexed = [(i, e) for i, e in enumerate(self.archive) if isinstance(e.get("generation"), int)]
        if not indexed:
            return None
        return max(indexed, key=lambda ie: (ie[1].get("raw_median_accuracy", 0.0), -ie[1]["generation"]))[0]

    def test_vector(self, index: Optional[int] = None, name: Optional[str] = None) -> Optional[np.ndarray]:
        if index is not None:
            entry = self.evaluated[index] if index < len(self.evaluated) else {}
            vec = entry.get("test_per_question")
            return np.asarray(vec, float) if vec else None
        for entry in self.evaluated:
            if entry.get("name") == name and entry.get("test_per_question"):
                return np.asarray(entry["test_per_question"], float)
        return None

    def resource(self, index: int) -> dict:
        entry = self.evaluated[index] if index < len(self.evaluated) else {}
        return entry.get("resource") or self.archive[index].get("resource") or {}

    def credits(self) -> float:
        return sum(float(e.get("credits_charged") or 0.0) for e in self.ledger)

    def search_hours(self) -> Optional[float]:
        """First call to the last designer call: the search, without the test phase."""
        meta = [e["t"] for e in self.ledger if e.get("role") == "meta"]
        if not meta or not self.ledger:
            return None
        return (max(meta) - min(e["t"] for e in self.ledger)) / 3600.0


def pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def delta(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[dict]:
    if a is None or b is None or len(a) != len(b):
        return None
    return paired_bootstrap(list(a), list(b))


def fmt_delta(d: Optional[dict]) -> str:
    if not d:
        return "-"
    return f"{100 * d['delta']:+.1f} [{100 * d['low']:+.1f}, {100 * d['high']:+.1f}]"


def mean(vec: Optional[np.ndarray]) -> Optional[float]:
    return None if vec is None else float(np.mean(vec))


def table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def code_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def summarize_search(cell: Cell, format_ref: Optional[np.ndarray]) -> dict:
    out = {"cell": cell.id, "done": cell.done, "generations": len(cell.generated),
           "attempts": cell.budget.get("attempts"), "credits": round(cell.credits(), 1),
           "search_hours": cell.search_hours()}
    vals = [e.get("raw_median_accuracy", 0.0) for e in cell.generated]
    seed_vals = [e.get("raw_median_accuracy", 0.0) for e in cell.seeds]
    if vals:
        out.update(best_validation=max(vals), median_validation=statistics.median(vals),
                   above_best_seed=sum(v > max(seed_vals) for v in vals) if seed_vals else None,
                   format_cue=sum(bool(FORMAT_CUE.search(e.get("code", ""))) for e in cell.generated))
    if seed_vals:
        out["best_seed_validation"] = max(seed_vals)
    index = cell.selected_index()
    if index is None:
        return out
    selected = cell.archive[index]
    out.update(selected_name=selected.get("name"), selected_generation=selected.get("generation"),
               selected_validation=selected.get("raw_median_accuracy"))
    vec = cell.test_vector(index=index)
    if vec is None:
        return out
    res = cell.resource(index)
    out.update(selected_test=mean(vec), calls_per_question=res.get("calls_per_question"),
               credits_per_question=res.get("credits_per_question"),
               vs_sc=delta(vec, cell.test_vector(name=SC)), vs_cot=delta(vec, cell.test_vector(name=COT)),
               vs_format_cot=delta(vec, format_ref), sc_test=mean(cell.test_vector(name=SC)),
               cot_test=mean(cell.test_vector(name=COT)))
    out["_vector"] = vec
    return out


def batch_similarity(cell: Cell) -> dict:
    """Code similarity of designs proposed together (one round) against consecutive ones."""
    by_round: Dict[int, List[dict]] = {}
    for entry in cell.generated:
        if isinstance(entry.get("round"), int):
            by_round.setdefault(entry["round"], []).append(entry)
    within = [code_similarity(a.get("code"), b.get("code"))
              for entries in by_round.values() for a, b in combinations(entries, 2)]
    return {"rounds": len(by_round), "within_round_similarity": statistics.mean(within) if within else None,
            "near_duplicates": sum(s > 0.9 for s in within)}


def consecutive_similarity(cell: Cell) -> Optional[float]:
    gens = cell.generated
    sims = [code_similarity(a.get("code"), b.get("code")) for a, b in zip(gens, gens[1:])]
    return statistics.mean(sims) if sims else None


def report(root: str) -> dict:
    lines: List[str] = []
    result: dict = {}
    fmt = Cell(root, f"cpl_formatcot_{DOMAIN}_{WORKER}")
    fmt_rerun = Cell(root, f"cpl_formatcot_{DOMAIN}_{WORKER}_rerun")
    phase34 = Cell(root, f"cp3_transfer_{DOMAIN}_{WORKER}")
    format_ref = fmt.test_vector(name=FORMAT_COT)

    lines.append("## Reference agents on the 300 test questions (re-run noise floor)\n")
    rows = []
    for label, cell in (("cpl_formatcot", fmt), ("cpl_formatcot_rerun", fmt_rerun), ("cp3_transfer (Phase 3.4)", phase34)):
        rows.append([label, pct(mean(cell.test_vector(name=COT))), pct(mean(cell.test_vector(name=FORMAT_COT)))])
    lines.append(table(["Cell", "CoT", "Format-CoT"], rows))
    fmt_runs = [mean(c.test_vector(name=FORMAT_COT)) for c in (fmt, fmt_rerun, phase34)]
    fmt_runs = [x for x in fmt_runs if x is not None]
    if len(fmt_runs) > 1:
        lines.append(f"\nFormat-CoT re-run spread: {100 * (max(fmt_runs) - min(fmt_runs)):.1f} points.")
    result["format_cot_runs"] = fmt_runs

    # Comparisons 1, 2 and 9: every designer search.
    lines.append("\n## Designer searches (comparisons 1, 2, 9)\n")
    searches: Dict[str, dict] = {}
    rows = []
    for mode in MODES:
        for designer in DESIGNERS:
            for replicate in REPLICATES:
                cell = Cell(root, search_cell(mode, designer, replicate))
                s = summarize_search(cell, format_ref)
                searches[cell.id] = s
                rows.append([mode, designer, replicate, f"{s['generations']} ({s.get('attempts') or '-'})",
                             f"{(s.get('selected_name') or '-')[:44]} ({s.get('selected_generation', '-')})",
                             pct(s.get("selected_validation")), pct(s.get("selected_test")),
                             fmt_delta(s.get("vs_sc")), fmt_delta(s.get("vs_format_cot")),
                             f"{s['calls_per_question']:.1f}" if s.get("calls_per_question") else "-",
                             f"{s['credits_per_question']:.3f}" if s.get("credits_per_question") else "-",
                             f"{s['credits']:.0f}", "done" if s["done"] else "running"])
    lines.append(table(["Mode", "Designer", "Rep", "Gens (attempts)", "Selected (gen)", "Validation", "Test",
                        "vs SC [95% CI]", "vs format-CoT [95% CI]", "Calls/q", "Credits/q", "Credits", "State"],
                       rows))
    complete = [s for s in searches.values() if s.get("vs_sc")]
    if complete:
        n_headline = sum(s["vs_sc"]["delta"] >= 0.05 for s in complete)
        n_format = sum(s["vs_format_cot"] and s["vs_format_cot"]["low"] <= 0 for s in complete)
        tests = [s["selected_test"] for s in complete]
        lines.append(f"\nHeadline (>= +5 over SC): {n_headline} of {len(complete)} searches. "
                     f"Format (interval vs format-CoT includes or lies below 0): {n_format} of {len(complete)}. "
                     f"Selected test across searches: {100 * min(tests):.1f} to {100 * max(tests):.1f}% "
                     f"(SD {100 * statistics.pstdev(tests):.1f}).")
    result["searches"] = {k: {kk: vv for kk, vv in v.items() if kk != "_vector"} for k, v in searches.items()}

    # Comparison 3: designer effect.
    lines.append("\n## Designer: GPT-6 Luna vs Claude Haiku 5.5 designing for GPT-6 Luna (comparison 3)\n")
    rows = []
    for mode in MODES:
        for designer in DESIGNERS:
            group = [searches[search_cell(mode, designer, r)] for r in REPLICATES]
            tests = [s["selected_test"] for s in group if s.get("selected_test") is not None]
            meds = [s["median_validation"] for s in group if s.get("median_validation") is not None]
            cues = [f"{s['format_cue']}/{s['generations']}" for s in group if s.get("format_cue") is not None]
            rows.append([mode, designer, ", ".join(pct(t) for t in tests) or "-",
                         ", ".join(pct(m) for m in meds) or "-", ", ".join(cues) or "-"])
    lines.append(table(["Mode", "Designer", "Selected test (a, b)", "Median validation (a, b)",
                        "Designs with answer-format cues"], rows))
    transferred = phase34.test_vector(name="Token-F1 Minimum Bayes Risk Selection")
    rows = []
    for mode in MODES:
        for replicate in REPLICATES:
            s = searches[search_cell(mode, "claude-haiku-5.5", replicate)]
            rows.append([f"{mode} {replicate}", pct(s.get("selected_test")),
                         fmt_delta(delta(s.get("_vector"), transferred))])
    lines.append(f"\nHaiku designing for GPT-6 Luna vs Haiku's own design transferred to GPT-6 Luna "
                 f"(Phase 3.4, {pct(mean(transferred))}):\n")
    lines.append(table(["Haiku-for-Luna search", "Selected test", "minus transferred design [95% CI]"], rows))

    # Comparison 4: batch vs sequential.
    lines.append("\n## Batch (3 per round) vs sequential, 15 designs each (comparison 4)\n")
    rows = []
    for designer in DESIGNERS:
        for mode in MODES:
            for replicate in REPLICATES:
                cell = Cell(root, search_cell(mode, designer, replicate))
                s = searches[cell.id]
                sim = batch_similarity(cell) if mode == "batch3" else {}
                rows.append([designer, mode, replicate, pct(s.get("best_validation")), pct(s.get("median_validation")),
                             str(s.get("above_best_seed", "-")), pct(s.get("selected_test")),
                             f"{sim['within_round_similarity']:.2f}" if sim.get("within_round_similarity") else "-",
                             str(sim.get("near_duplicates", "-")),
                             f"{consecutive_similarity(cell):.2f}" if consecutive_similarity(cell) else "-",
                             f"{s['search_hours']:.1f}" if s.get("search_hours") else "-"])
    lines.append(table(["Designer", "Mode", "Rep", "Best val", "Median val", "Above best seed", "Selected test",
                        "Within-round code similarity", "Near-duplicates", "Consecutive similarity",
                        "Search hours"], rows))
    lines.append("\nSearch hours are confounded: every cell shares one account pace, and batch cells "
                 "evaluate three designs at once, so they draw more of it.")

    # Comparison 5: best of replicates.
    lines.append("\n## Best of two replicates, chosen on validation (comparison 5)\n")
    rows = []
    for mode in MODES:
        for designer in DESIGNERS:
            group = [searches[search_cell(mode, designer, r)] for r in REPLICATES]
            group = [s for s in group if s.get("selected_validation") is not None and s.get("selected_test") is not None]
            if len(group) < 2:
                rows.append([mode, designer, "-", "-", "-"])
                continue
            best = max(group, key=lambda s: s["selected_validation"])
            rows.append([mode, designer, best["cell"][-1], pct(best["selected_test"]),
                         fmt_delta(delta(best.get("_vector"), format_ref))])
    lines.append(table(["Mode", "Designer", "Chosen replicate", "Test", "vs format-CoT [95% CI]"], rows))

    # Comparison 6: random recombination.
    lines.append("\n## Random recombination, no designer (comparison 6)\n")
    random_cell = Cell(root, f"cpl_random_{DOMAIN}_{WORKER}")
    s = summarize_search(random_cell, format_ref)
    result["random"] = {k: v for k, v in s.items() if k != "_vector"}
    seed_mean = statistics.mean([e.get("raw_median_accuracy", 0) for e in random_cell.seeds]) if random_cell.seeds else None
    above_mean = sum(e.get("raw_median_accuracy", 0) >= seed_mean for e in random_cell.generated) if seed_mean else "-"
    lines.append(table(["Designs", "Above seed mean", "Above best seed", "Best validation", "Best seed validation",
                        "Selected test", "vs SC [95% CI]", "vs format-CoT [95% CI]", "Answer-format cues"],
                       [[str(s["generations"]), str(above_mean), str(s.get("above_best_seed", "-")),
                         pct(s.get("best_validation")), pct(s.get("best_seed_validation")), pct(s.get("selected_test")),
                         fmt_delta(s.get("vs_sc")), fmt_delta(s.get("vs_format_cot")),
                         f"{s.get('format_cue', '-')}/{s['generations']}"]]))

    # Comparison 7: effort.
    lines.append("\n## Worker effort (comparison 7)\n")
    contrast = {e: Cell(root, f"cpl_effortcontrast_{DOMAIN}_{WORKER}_{e}") for e in ("low", "high")}
    design_name = searches[search_cell("seq", "gpt-6-luna", "a")].get("selected_name")
    rows = []
    for label, name in (("Chain-of-Thought", COT), ("Format-CoT", FORMAT_COT), ("Self-design a's selection", design_name)):
        low = contrast["low"].test_vector(name=name) if name else None
        high = contrast["high"].test_vector(name=name) if name else None
        rows.append([label, pct(mean(low)), pct(mean(high)), fmt_delta(delta(high, low))])
    lines.append(table(["Agent", "Low", "High", "High minus low [95% CI]"], rows))
    rows = []
    for effort in ("low", "high"):
        cell = Cell(root, f"cpl_effort_gpt-6-luna_{DOMAIN}_{WORKER}_{effort}")
        s = summarize_search(cell, contrast[effort].test_vector(name=FORMAT_COT))
        result[f"effort_{effort}"] = {k: v for k, v in s.items() if k != "_vector"}
        rows.append([effort, pct(s.get("best_validation")), pct(s.get("median_validation")),
                     f"{(s.get('selected_name') or '-')[:40]} ({s.get('selected_generation', '-')})",
                     pct(s.get("selected_test")), fmt_delta(s.get("vs_sc")), fmt_delta(s.get("vs_format_cot"))])
    lines.append("\nSearches with the worker at low and at high effort (GPT-6 Luna designing):\n")
    lines.append(table(["Effort", "Best val", "Median val", "Selected (gen)", "Test", "vs SC [95% CI]",
                        "vs format-CoT at that effort [95% CI]"], rows))

    # Comparison 8: transfer.
    lines.append("\n## Transfer of self-design a's selection (comparison 8)\n")
    rows = []
    for target in ("claude-haiku-5.5", "gpt-5.6-luna"):
        cell = Cell(root, f"cpl_transfer_{DOMAIN}_{target}_from_{WORKER}")
        design = cell.test_vector(name=design_name) if design_name else None
        rows.append([target, pct(mean(design)), pct(mean(cell.test_vector(name=COT))),
                     pct(mean(cell.test_vector(name=SC))), pct(mean(cell.test_vector(name=FORMAT_COT))),
                     fmt_delta(delta(design, cell.test_vector(name=SC))),
                     fmt_delta(delta(design, cell.test_vector(name=FORMAT_COT)))])
    lines.append(table(["Worker", "Design", "CoT", "SC", "Format-CoT", "Design vs SC [95% CI]",
                        "Design vs format-CoT [95% CI]"], rows))

    total = sum(Cell(root, c).credits() for c in os.listdir(root)
                if c.startswith("cpl_") or (WORKER in c and c.startswith(("cp1_baseline", "cp3_effort_baseline"))))
    lines.append(f"\nCredits spent on the replication so far: {total:.0f}.")
    result["credits"] = total
    result["markdown"] = "\n".join(lines)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results_root", default=os.environ.get("ADAS_COPILOT_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    parser.add_argument("--json", default=None, help="also write the numbers here")
    args = parser.parse_args()
    result = report(os.path.abspath(args.results_root))
    print(result["markdown"])
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
