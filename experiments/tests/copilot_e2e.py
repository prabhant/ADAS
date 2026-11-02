"""Offline end-to-end test of the Copilot path: runner -> gateway -> search.py -> gateway.

Everything is real except the model provider: ``run_cell.py`` starts the gateway
as a subprocess with ``StubBackend`` (below) in place of the Copilot SDK, and the
real ``_mgsm/search.py`` runs against it. ``StubBackend`` answers like the
scripted stub in ``search_e2e.py``, so every accuracy is known in advance, and it
replays each session's history to the stub, so a reflexion or debug turn that
did not continue its session shows up as a wrong chain count.

Checks: exact accuracies through the gateway; designer conversations continued
rather than flattened; worker JSON extracted from fenced replies; credits in
the ledger, in token_usage.json and in each design's resource; the budget halt
stops the search without recording a corrupted score, and a raised cap resumes
it; test-only cells (plain baseline, transfer of the best design) are built and
evaluated.

    python experiments/tests/copilot_e2e.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.serving.copilot_gateway import Backend, TurnResult, UsageEvent  # noqa: E402

VALID, TEST = 8, 8


def _load_stub_module(domain="mgsm"):
    """search_e2e reads its domain from argv at import; give it the one asked for."""
    path = os.path.join(ROOT, "experiments", "tests", "search_e2e.py")
    spec = importlib.util.spec_from_file_location(f"search_e2e_for_copilot_{domain}", path)
    module = importlib.util.module_from_spec(spec)
    saved_argv, saved_path = sys.argv, list(sys.path)
    # Each domain has its own `utils`; drop a cached one so this domain's is imported.
    saved_utils = sys.modules.pop("utils", None)
    sys.argv = [path, domain]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv, sys.path[:] = saved_argv, saved_path
        if saved_utils is not None:
            sys.modules["utils"] = saved_utils
    return module


E2E = _load_stub_module()
_DROP_STUB = None


def drop_stub_module():
    global _DROP_STUB
    if _DROP_STUB is None:
        _DROP_STUB = _load_stub_module("drop")
    return _DROP_STUB


class StubBackend(Backend):
    """Copilot stand-in: one history per session, replies from search_e2e's Stub."""

    def __init__(self):
        self.stub = E2E.Stub()
        self.lock = threading.Lock()

    async def open_session(self, model, system, reasoning_effort, max_tokens):
        return {"model": model, "history": [{"role": "system", "content": system}]}

    async def send(self, handle, prompt, schema, timeout):
        history = handle["history"]
        history.append({"role": "user", "content": prompt})
        if "Reply EXACTLY" in history[0]["content"]:
            body, _ = self.stub.worker(history)
            # Fenced, as a provider without a JSON mode tends to answer.
            content, reasoning = f"```json\n{json.dumps(body)}\n```", None
        else:
            with self.lock:
                body, reasoning = self.stub.meta(history)
            content = json.dumps(body)
        history.append({"role": "assistant", "content": content})
        usage = UsageEvent(model=handle["model"], input_tokens=sum(len(m["content"]) for m in history) // 4,
                           output_tokens=len(content) // 4, nano_aiu=1e5, finish_reason="stop")
        return TurnResult(content=content, reasoning=reasoning, usage=[usage])


class DropStubBackend(StubBackend):
    """DROP answers, and one designer chain per session: concurrent proposals (batch
    rounds) must each continue their own session, or the replies stop matching."""

    def __init__(self):
        self.stub = drop_stub_module().Stub()
        self.lock = threading.Lock()
        self.sessions = 0

    async def open_session(self, model, system, reasoning_effort, max_tokens):
        handle = await super().open_session(model, system, reasoning_effort, max_tokens)
        with self.lock:
            self.sessions += 1
            handle["key"] = f"session{self.sessions}"
        return handle

    async def send(self, handle, prompt, schema, timeout):
        history = handle["history"]
        if "Reply EXACTLY" in history[0]["content"]:
            return await super().send(handle, prompt, schema, timeout)
        history.append({"role": "user", "content": prompt})
        with self.lock:
            body, reasoning = self.stub.meta(history, handle["key"])
        content = json.dumps(body)
        history.append({"role": "assistant", "content": content})
        usage = UsageEvent(model=handle["model"], input_tokens=sum(len(m["content"]) for m in history) // 4,
                           output_tokens=len(content) // 4, nano_aiu=1e5, finish_reason="stop")
        return TurnResult(content=content, reasoning=reasoning, usage=[usage])


def expected_vectors(module=None):
    module = module or E2E
    examples = module.load_examples()
    module.random.seed(0)
    module.random.shuffle(examples)
    valid, test = examples[:VALID], examples[VALID:VALID + TEST]
    return ([int(module.answered_correctly(e["inputs"])) for e in valid],
            [int(module.answered_correctly(e["inputs"])) for e in test])


def grid(max_credits_halt):
    small = {"domain": "mgsm", "valid_size": VALID, "test_size": TEST, "n_repreat": 1,
             "max_attempts": 6, "debug_max": 3, "allow_missing_baseline": True}
    return {
        "global_max_credits": 1000,
        "selection": {"worker": "gpt-6-luna", "source": "e2e_smoke"},
        "common": {"shuffle_seed": 0, "max_workers": 4, "max_concurrency": 8, "informed_meta": True,
                   "request_timeout": 60, "max_credits": 100},
        "domains": {"mgsm": {"valid_size": 128, "test_size": 800, "n_repreat": 1}},
        "stage_defaults": {"e2e": small,
                           "e2e_drop": dict(small, domain="drop", max_concurrency=16,
                                            pace_rates={"openai": 6000, "anthropic": 6000})},
        "cells": [
            {"id": "e2e_smoke", "stage": "e2e", "meta": "claude-haiku-5.5", "worker": "claude-haiku-5.5",
             "n_generation": 2},
            {"id": "e2e_halt", "stage": "e2e", "meta": "claude-haiku-5.5", "worker": "claude-haiku-5.5",
             "n_generation": 1, "max_credits": max_credits_halt},
            {"id": "e2e_plain_{worker}", "stage": "e2e", "condition": "test_archive", "meta": None,
             "worker": "{worker}", "test_agents": {"seeds": ["Chain-of-Thought"], "agents": [
                 {"name": "Inline control", "code": E2E.GOOD_CODE}]}},
            {"id": "e2e_transfer", "stage": "e2e", "condition": "test_archive", "meta": None,
             "worker": "{worker}",
             "test_agents": {"best_of": "{source}", "seeds": ["Chain-of-Thought", "LLM Debate"]}},
            {"id": "e2e_templated_{effort_worker}", "stage": "e2e", "meta": None, "worker": "{effort_worker}"},
            {"id": "e2e_batch", "stage": "e2e_drop", "meta": "claude-haiku-5.5", "worker": "gpt-6-luna",
             "n_generation": 3, "proposals_per_round": 2},
            {"id": "e2e_par_first", "stage": "e2e", "group": "e2e_par", "condition": "test_archive",
             "meta": None, "worker": "gpt-6-luna", "test_agents": {"seeds": ["Chain-of-Thought"]}},
            {"id": "e2e_par_second", "stage": "e2e", "group": "e2e_par", "condition": "test_archive",
             "meta": None, "worker": "gpt-6-luna", "test_agents": {"seeds": ["LLM Debate"]},
             "after": ["e2e_par_first"]},
        ],
    }


def run_cell(cells_path, results_root, cell, *extra, backend="StubBackend"):
    argv = [sys.executable, os.path.join(ROOT, "experiments", "local", "run_cell.py"), cell,
            "--cells", cells_path, "--results_root", results_root, "--python", sys.executable,
            "--gateway_backend", f"experiments.tests.copilot_e2e:{backend}",
            "--no_caffeinate", "--quiet", *extra]
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=1200)
    if proc.returncode not in (0, 2, 3):  # 2 and 3 are asserted by the callers
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:])
    return proc


def load(path):
    with open(path) as handle:
        return json.load(handle)


def ledger(results_root, cell):
    with open(os.path.join(results_root, cell, "copilot_ledger.jsonl")) as handle:
        return [json.loads(line) for line in handle]


def main() -> int:
    check = E2E.Checks()
    valid_vec, test_vec = expected_vectors()
    acc_valid, acc_test = sum(valid_vec) / VALID, sum(test_vec) / TEST
    workdir = tempfile.mkdtemp(prefix="adas_copilot_e2e_")
    results_root = os.path.join(workdir, "results")
    cells_path = os.path.join(workdir, "cells.yaml")
    with open(cells_path, "w") as handle:
        yaml.safe_dump(grid(max_credits_halt=0.001), handle)
    print(f"ground truth: valid {acc_valid:.4f}, test {acc_test:.4f}")

    print("\n=== A: designer and worker behind the gateway, end to end ===")
    proc = run_cell(cells_path, results_root, "e2e_smoke")
    check(proc.returncode == 0, "runner exit 0")
    cell_dir = os.path.join(results_root, "e2e_smoke")
    archive = load(os.path.join(cell_dir, "e2e_smoke_run_archive.json"))
    generated = [e for e in archive if isinstance(e.get("generation"), int)]
    check(len(archive) == 9 and len(generated) == 2, f"7 seeds + 2 designs (got {len(archive)})")
    check(all(e["valid_per_question"] == valid_vec for e in archive), "validation vectors are the ground truth")
    check(generated and generated[0].get("debug_attempts") == 1,
          "the debug turn continued the designer's session (chain recovered after one round)")
    check(all(e.get("meta_reasoning") == E2E.STUB_REASONING for e in generated), "designer reasoning captured")
    check(all((e.get("resource") or {}).get("credits_per_question", 0) > 0 for e in archive),
          "every design carries AI credits per question")
    evaluated = load(os.path.join(cell_dir, "e2e_smoke_run_archive_evaluate.json"))
    check(len(evaluated) == 9 and all(e["test_per_question"] == test_vec for e in evaluated),
          "test vectors are the ground truth")

    lines = ledger(results_root, "e2e_smoke")
    worker = [l for l in lines if l["role"] == "worker"]
    meta = [l for l in lines if l["role"] == "meta"]
    modes = [l["mode"] for l in meta]
    # Three chains (recovered, abandoned, clean) open fresh; every reflexion and debug turn continues.
    check(modes.count("fresh") == 3 and modes.count("flattened") == 0 and modes.count("continued") >= 6,
          f"designer calls: 3 fresh chains, the rest continued (got {modes})")
    check(worker and all(l["mode"] == "fresh" and l["json_extracted"] for l in worker),
          "worker calls are stateless and their fenced JSON is extracted")
    check(all(l["temperature_requested"] is not None for l in lines), "requested temperatures logged")
    usage = load(os.path.join(cell_dir, "e2e_smoke", "e2e_smoke_token_usage.json"))
    tracked = sum(b.get("credits", 0) for k, b in usage.items())
    charged = sum(l["credits_charged"] for l in lines)
    check(tracked > 0 and abs(tracked - charged) / charged < 1e-6,
          f"instrumentation credits equal the ledger ({tracked:.6f} vs {charged:.6f})")

    print("\n=== B: the budget halt stops the search; a raised cap resumes it ===")
    proc = run_cell(cells_path, results_root, "e2e_halt")
    halt_dir = os.path.join(results_root, "e2e_halt")
    check(proc.returncode == 3, f"runner exits 3 on a halt (got {proc.returncode})")
    check(os.path.exists(os.path.join(halt_dir, "GATEWAY_HALTED.json")), "halt file written")
    archive_path = os.path.join(halt_dir, "e2e_halt_run_archive.json")
    partial = load(archive_path) if os.path.exists(archive_path) else []
    check(all(e.get("valid_per_question") == valid_vec for e in partial if "valid_per_question" in e),
          f"no design was scored from held calls ({len(partial)} archived)")
    with open(cells_path, "w") as handle:
        yaml.safe_dump(grid(max_credits_halt=100), handle)
    proc = run_cell(cells_path, results_root, "e2e_halt")
    check(proc.returncode == 0, "rerun with a raised cap exits 0")
    archive = load(archive_path)
    check(len(archive) == 8 and all(e["valid_per_question"] == valid_vec for e in archive),
          "the resumed cell completes with ground-truth scores")
    check(any(name.startswith("GATEWAY_HALTED.json.") for name in os.listdir(halt_dir)),
          "the earlier halt record is kept beside the run")

    print("\n=== C: test-only cells: a plain baseline and a transferred design ===")
    proc = run_cell(cells_path, results_root, "e2e_plain_gpt-6-luna")
    plain = load(os.path.join(results_root, "e2e_plain_gpt-6-luna",
                              "e2e_plain_gpt-6-luna_run_archive_evaluate.json"))
    check(proc.returncode == 0 and [e["name"] for e in plain] == ["Chain-of-Thought", "Inline control"]
          and all(e["test_per_question"] == test_vec for e in plain) and plain[1].get("control") is True,
          "templated plain-CoT cell, with an inline control agent, evaluated on test")
    proc = run_cell(cells_path, results_root, "e2e_transfer")
    transfer = load(os.path.join(results_root, "e2e_transfer", "e2e_transfer_run_archive_evaluate.json"))
    source = load(os.path.join(results_root, "e2e_smoke", "e2e_smoke_run_archive.json"))
    best = max((e for e in source if isinstance(e.get("generation"), int)),
               key=lambda e: (e["raw_median_accuracy"], -e["generation"]))
    check(proc.returncode == 0 and [e["name"] for e in transfer] == [best["name"], "Chain-of-Thought", "LLM Debate"]
          and transfer[0].get("source_cell") == "e2e_smoke"
          and all(e["test_per_question"] == test_vec for e in transfer),
          "transfer cell: best-on-validation design plus references, evaluated under another worker")

    print("\n=== D: configuration guards ===")
    proc = run_cell(cells_path, results_root, "e2e_templated_{effort_worker}")
    check(proc.returncode == 2 and "selection" in proc.stderr, "a cell awaiting `selection` refuses to run")
    listing = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "local", "run_cell.py"),
                              "--list", "--cells", cells_path, "--results_root", results_root],
                             capture_output=True, text=True)
    check(listing.returncode == 0 and "e2e_smoke" in listing.stdout and "selection?" in listing.stdout,
          "--list shows progress, spend and unresolved cells")
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "local", "run_cell.py"),
                           "cp1_cot_gpqa_claude-haiku-5.5", "--dry_run"], capture_output=True, text=True)
    check(proc.returncode == 0 and "--valid_size 32" in proc.stdout and "--n_repreat 3" in proc.stdout
          and "--baseline cot" in proc.stdout, "the real grid resolves a GPQA screen cell")
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "local", "run_cell.py"),
                           "cp1_cot_gpqa_claude-haiku-5.5", "--dry_run", "--max_calls_per_minute", "42"],
                          capture_output=True, text=True)
    check(proc.returncode == 0 and "--max_calls_per_minute 42.0" in proc.stdout,
          "--max_calls_per_minute overrides the cell's pace on the gateway")

    print("\n=== batch rounds through the gateway (DROP) ===")
    drop_valid, _ = expected_vectors(drop_stub_module())
    proc = run_cell(cells_path, results_root, "e2e_batch", backend="DropStubBackend")
    check(proc.returncode == 0, "batch cell exit 0")
    batch_path = os.path.join(results_root, "e2e_batch", "e2e_batch_run_archive.json")
    archive = load(batch_path) if os.path.exists(batch_path) else []
    generated = [e for e in archive if isinstance(e.get("generation"), int)]
    check([e.get("generation") for e in generated] == [1, 2, 3] and [e.get("round") for e in generated] == [1, 2, 2],
          f"two proposals a round, one abandoned in round 1 (got rounds {[e.get('round') for e in generated]})")
    check(generated and all(e["valid_per_question"] == drop_valid for e in generated),
          "batch designs score the DROP ground truth through the gateway")
    meta_modes = [l["mode"] for l in ledger(results_root, "e2e_batch") if l["role"] == "meta"]
    check(meta_modes.count("fresh") == 4 and "flattened" not in meta_modes and meta_modes.count("continued") >= 8,
          f"concurrent proposals each continue their own designer session (modes {sorted(set(meta_modes))}, "
          f"fresh {meta_modes.count('fresh')})")
    check(os.path.exists(os.path.join(results_root, "_pace", "openai.slot"))
          and os.path.exists(os.path.join(results_root, "_pace", "anthropic.slot")),
          "the gateway draws from the shared pace files")
    check(os.path.exists(os.path.join(results_root, "e2e_batch", ".cell_done.json")), "done marker written")

    print("\n=== run_parallel: dependencies, done markers, resume ===")
    driver = [sys.executable, os.path.join(ROOT, "experiments", "local", "run_parallel.py"),
              "--cells", cells_path, "--results_root", results_root, "--group", "e2e_par",
              "--no_caffeinate", "--poll", "0.5", "--python", sys.executable]
    backend_args = ["--run_cell_args", "--gateway_backend", "experiments.tests.copilot_e2e:StubBackend"]
    dry = subprocess.run(driver + ["--dry_run"], cwd=ROOT, capture_output=True, text=True)
    check(dry.returncode == 0 and dry.stdout.index("e2e_par_first") < dry.stdout.index("wave 2")
          < dry.stdout.index("e2e_par_second"), "dry run orders the dependent cell in a later wave")
    proc = subprocess.run(driver + backend_args, cwd=ROOT, capture_output=True, text=True, timeout=1200)
    out = proc.stdout
    check(proc.returncode == 0 and all(os.path.exists(os.path.join(results_root, c, ".cell_done.json"))
                                       for c in ("e2e_par_first", "e2e_par_second")),
          f"both cells finish (exit {proc.returncode})")
    ordered = ("done    e2e_par_first" in out and "started e2e_par_second" in out
               and out.index("done    e2e_par_first") < out.index("started e2e_par_second"))
    check(ordered, "the dependent cell starts only after its dependency finished")
    if not ordered:
        print(out[-3000:])
    second = load(os.path.join(results_root, "e2e_par_second", "e2e_par_second_run_archive_evaluate.json"))
    check([e["name"] for e in second] == ["LLM Debate"] and second[0]["test_per_question"] == test_vec,
          "the dependent cell evaluated its agent")
    again = subprocess.run(driver + backend_args, cwd=ROOT, capture_output=True, text=True, timeout=300)
    check(again.returncode == 0 and "0 cells to run" in again.stdout, "a rerun skips finished cells")

    print(f"\n{'PASSED' if not check.failed else f'{check.failed} CHECK(S) FAILED'}  (artifacts in {workdir})")
    return 1 if check.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
