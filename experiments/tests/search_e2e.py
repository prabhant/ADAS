"""CPU-only end-to-end test of the search loop against a scripted stub server.

Runs the real _mgsm/search.py -- sandbox, instrumentation, budget, archive,
resume, evaluation -- with one change: the model endpoint is a local stub whose
behaviour is fully determined by this file. That buys two things GPU sanity
jobs cannot:

* Ground truth. The stub decides which questions are answered correctly, so
  every design's accuracy is known exactly in advance, and a scoring bug shows
  up as a wrong number rather than as a plausible-looking one.
* Forced branches. The stub meta agent emits broken code on cue, so debug
  recovery, abandonment and the walltime deadline all execute every run
  instead of only when a real model happens to misbehave.

    python experiments/tests/search_e2e.py
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "mgsm"
sys.path.insert(0, os.path.join(ROOT, f"_{DOMAIN}"))
if DOMAIN == "drop":
    from utils import load_drop  # noqa: E402

    def load_examples():
        return load_drop(os.path.join(ROOT, "dataset", "drop_v0_dev.jsonl.gz"))[1:-1]
elif DOMAIN == "arc":
    import pickle

    from utils import format_arc_data, list_to_string  # noqa: E402

    def arc_records(filename):
        with open(os.path.join(ROOT, "dataset", filename), "rb") as handle:
            tasks = pickle.load(handle)
        # "inputs" is exactly the task text an agent receives; the target is test[0]'s output.
        return [{"inputs": format_arc_data(t)[0], "targets": t["test"][0]["output"]} for t in tasks]
else:
    from utils import get_all_examples  # noqa: E402

    def load_examples():
        return get_all_examples()

VALID, TEST, SEED = 12, 12, 0
N_SEEDS = 5 if DOMAIN == "arc" else 7
STUB_REASONING = "stub reasoning trace"

GOOD_CODE = '''def forward(self, taskInfo):
    cot = LLMAgentBase(['thinking', 'answer'], 'Stub CoT Agent')
    thinking, answer = cot([taskInfo], "Think step by step, then answer.")
    return answer
'''
BROKEN_CODE = "def forward(self, taskInfo):\n    return (\n"
CRASHY_CODE = '''def forward(self, taskInfo):
    if len(str(taskInfo.content)) % 2:
        raise ValueError("stub crash")
    cot = LLMAgentBase(['thinking', 'answer'], 'Stub CoT Agent')
    thinking, answer = cot([taskInfo], "Think step by step, then answer.")
    return answer
'''


def answered_correctly(question: str) -> bool:
    return int(hashlib.sha1(question.encode()).hexdigest(), 16) % 10 < 6


def split():
    """Replicates evaluate_forward_fn's selection exactly."""
    if DOMAIN == "arc":
        # ADAS's fixed ARC splits, never shuffled.
        return (arc_records("sampled_arc_val_data.pkl")[:VALID],
                arc_records("sampled_arc_test_data.pkl")[:TEST])
    examples = load_examples()
    random.seed(SEED)
    random.shuffle(examples)
    return examples[:VALID], examples[VALID:VALID + TEST]


def arc_stub_fields(example, field):
    """The gold grid when the stub answers correctly, else the gold with one cell changed.

    `code` fields get a transform that returns that grid through a top-level import
    and helper, returning a numpy array: stock ADAS's exec(code, {}, locals) could not
    see either, and a numpy result crashed its example comparison.
    """
    grid = [list(row) for row in example["targets"]]
    if not answered_correctly(example["inputs"]):
        grid[0][0] = (grid[0][0] + 1) % 10
    if "code" in field:
        return ("import numpy as np\n\n"
                f"def _stub_grid():\n    return {list_to_string(grid)}\n\n"
                "def transform(grid):\n    return np.array(_stub_grid())\n")
    return list_to_string(grid)


VALID_SET, TEST_SET = split()
USED = VALID_SET + TEST_SET


class Stub:
    """Scripted meta agent and deterministic worker behind one endpoint."""

    def __init__(self):
        self.lock = threading.Lock()
        self.chains = 0
        self.meta_calls = 0
        self.meta_prompts = []
        # A slow worker makes an evaluation outlast a deadline or a cap on cue.
        self.worker_delay = 0.0
        self.worker_calls = 0

    def meta(self, messages):
        with self.lock:
            self.meta_calls += 1
            self.meta_prompts.append("\n".join(str(m["content"]) for m in messages))
            if len(messages) == 2:
                self.chains += 1
            chain = self.chains - 1
        debugging = str(messages[-1]["content"]).startswith("Error during evaluation")
        # chain 0: broken, then fixed by debugging -> success with debug_attempts=1
        # chain 1: broken through every debug round -> abandoned
        # later:   correct first time
        if chain == 0:
            code = GOOD_CODE if debugging else BROKEN_CODE
        elif chain == 1:
            code = BROKEN_CODE
        else:
            code = GOOD_CODE
        body = {"thought": f"stub thought {chain}", "name": f"Stub Design {chain}", "code": code}
        if debugging:
            body["debug_thought"] = "stub debug"
        return body, STUB_REASONING

    @staticmethod
    def worker(messages):
        system, user = str(messages[0]["content"]), str(messages[-1]["content"])
        # ARC's code description contains an apostrophe, so its repr is double-quoted.
        fields = re.findall(r"'(\w+)': ['\"]Your", system) or ["answer"]
        example = next((e for e in USED if e["inputs"] in user), None)
        body = {}
        for field in fields:
            if DOMAIN == "arc" and example is not None and ("answer" in field or "code" in field):
                body[field] = arc_stub_fields(example, field)
            elif "answer" in field:
                if example is None:
                    body[field] = 0
                    continue
                if DOMAIN == "drop":
                    # The gold span scores F1 1.0; a token sharing nothing with any answer scores 0.
                    gold = next(t for t in example["targets"] if t.strip())
                    body[field] = gold if answered_correctly(example["inputs"]) else "zzqx"
                    continue
                target = int(example["targets"].replace(",", ""))
                # A JSON number, not a string: the exact shape that once made
                # score_mgsm raise and silently score every answer wrong.
                body[field] = target if answered_correctly(example["inputs"]) else target + 1
            elif field == "correct":
                body[field] = "True"
            else:
                body[field] = f"stub {field}"
        return body, None


def serve(stub: Stub):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, payload):
            data = json.dumps(payload).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the evaluation that asked was killed at its deadline or cap

        def do_GET(self):
            if "models" in self.path:
                self._send({"data": [{"id": "stub-meta"}, {"id": "stub-worker"}]})
            else:
                self._send({})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if request["model"] == "stub-meta":
                body, reasoning = stub.meta(request["messages"])
            else:
                with stub.lock:
                    stub.worker_calls += 1
                time.sleep(stub.worker_delay)
                body, reasoning = stub.worker(request["messages"])
            message = {"role": "assistant", "content": json.dumps(body)}
            if reasoning:
                message["reasoning_content"] = reasoning
            self._send({
                "id": "stub", "object": "chat.completion", "created": int(time.time()),
                "model": request["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            })

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run_search(url, save_dir, run_id, target, extra=()):
    argv = [
        sys.executable, "search.py", "--domain", DOMAIN,
        "--save_dir", save_dir, "--expr_name", run_id, "--run_id", run_id,
        "--model", "stub-meta", "--worker_model", "stub-worker",
        "--meta_base_url", url, "--worker_base_url", url,
        "--meta_max_tokens", "512", "--worker_max_tokens", "256",
        "--n_generation", str(target), "--target_generations", str(target),
        "--max_attempts", "6", "--debug_max", "3",
        "--valid_size", str(VALID), "--test_size", str(TEST), "--n_repreat", "1",
        "--max_workers", "4", "--shuffle_seed", str(SEED),
        "--informed_meta", "--worker_cot_accuracy", "0.6", "--request_timeout", "60",
        *extra,
    ]
    env = dict(os.environ, PYTHONPATH=ROOT, OPENAI_API_KEY="local-vllm")
    proc = subprocess.run(argv, cwd=os.path.join(ROOT, f"_{DOMAIN}"), env=env,
                          capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-3000:])
    return proc.returncode


def load(path):
    with open(path) as handle:
        return json.load(handle)


class Checks:
    def __init__(self):
        self.failed = 0

    def __call__(self, ok, label):
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
        self.failed += 0 if ok else 1


def main() -> int:
    expected_valid = [int(answered_correctly(e["inputs"])) for e in VALID_SET]
    expected_test = [int(answered_correctly(e["inputs"])) for e in TEST_SET]
    acc_valid = sum(expected_valid) / VALID
    acc_test = sum(expected_test) / TEST
    print(f"ground truth: valid {acc_valid:.4f}, test {acc_test:.4f}")
    check = Checks()
    workdir = tempfile.mkdtemp(prefix="adas_e2e_")

    print("\n=== A: full search, forced debug recovery and abandonment, then evaluation ===")
    stub = Stub()
    server = serve(stub)
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    a_dir, a_id = os.path.join(workdir, "A"), "e2eA"
    check(run_search(url, a_dir, a_id, target=3) == 0, "search + evaluate exit 0")

    archive = load(os.path.join(a_dir, f"{a_id}_run_archive.json"))
    generated = [e for e in archive if isinstance(e.get("generation"), int)]
    check(len(archive) == N_SEEDS + 3 and len(generated) == 3,
          f"{N_SEEDS} seeds + 3 generated (got {len(archive)})")
    if not generated:
        print("  no design was generated; the remaining section A checks cannot run")
        print(f"\n{check.failed} CHECK(S) FAILED  (artifacts in {workdir})")
        return 1
    check(all(abs(e["raw_median_accuracy"] - acc_valid) < 1e-9 for e in archive),
          f"every design scores exactly the ground-truth {acc_valid:.4f}")
    check(all(e["valid_per_question"] == expected_valid for e in archive),
          "per-question vectors match ground truth, in order")
    check(generated[0].get("debug_attempts") == 1, "generation 1 recovered after one debug round")
    check(all(e.get("meta_reasoning") == STUB_REASONING for e in generated), "meta reasoning captured")
    ids = [e.get("design_id") for e in archive]
    check(all(ids) and len(set(ids)) == len(ids), "design ids present and unique")
    check([len(e["archive_context"]) for e in generated] == [N_SEEDS, N_SEEDS + 1, N_SEEDS + 2],
          "archive_context records what each generation actually saw")

    evaluated = load(os.path.join(a_dir, f"{a_id}_run_archive_evaluate.json"))
    check(len(evaluated) == N_SEEDS + 3, f"all {N_SEEDS + 3} designs evaluated on test (got {len(evaluated)})")
    check(all(abs(e["test_median_accuracy"] - acc_test) < 1e-9 for e in evaluated),
          f"every test score is exactly {acc_test:.4f}")
    check(all(e["test_per_question"] == expected_test for e in evaluated), "test vectors match ground truth")

    run_dir = os.path.join(a_dir, a_id)
    abandoned = load(os.path.join(run_dir, f"{a_id}_abandoned_generations.json"))
    check(len(abandoned) == 1, f"exactly one abandoned generation (got {len(abandoned)})")
    if abandoned:
        entry = abandoned[0]
        check(entry["terminal_failure"] == "python_syntax_error", "abandoned as python_syntax_error")
        check(entry["debug_attempts"] == 3, "all three debug rounds were spent")
        try:
            compile(entry["code"], "<abandoned>", "exec")
            ran_code = False
        except SyntaxError:
            ran_code = True
        check(ran_code, "recorded code is the code that actually failed")

    budget = load(os.path.join(run_dir, "budget_summary.json"))
    check(budget["successes"] == 3 and budget["attempts"] == 4, "3 successes in 4 attempts")
    check(budget["failure_histogram"] == {"python_syntax_error": 1}, "failure histogram")
    check(budget.get("stopped_on_deadline") is False, "did not stop on deadline")
    check(os.path.getsize(os.path.join(run_dir, f"{a_id}_call_log.jsonl")) > 0, "call log written")
    check(os.path.exists(os.path.join(run_dir, f"{a_id}_token_usage.json")), "token usage written")
    # Analysis fields once leaked into the designer's prompt: reasoning traces,
    # per-question vectors and the worker's identity, which also broke blind mode.
    leaked = sorted({f for p in stub.meta_prompts for f in
                     ("valid_per_question", "meta_reasoning", STUB_REASONING, "archive_context",
                      "\"worker_model\"", "\"meta_model\"", "\"resource\"", "failure_trace")
                     if f in p})
    check(stub.meta_prompts and not leaked, f"no analysis fields reach the designer (leaked: {leaked})")

    print("\n=== D: rerunning a finished cell resumes and does no new work ===")
    before = stub.meta_calls
    check(run_search(url, a_dir, a_id, target=3, extra=["--search_only"]) == 0, "exit 0")
    check(len(load(os.path.join(a_dir, f"{a_id}_run_archive.json"))) == N_SEEDS + 3, "archive unchanged")
    check(stub.meta_calls == before, "no meta calls made")
    server.shutdown()

    print("\n=== B: a passed deadline stops cleanly before any generation ===")
    stub = Stub()
    server = serve(stub)
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    b_dir, b_id = os.path.join(workdir, "B"), "e2eB"
    rc = run_search(url, b_dir, b_id, target=3,
                    extra=["--search_only", "--deadline_epoch", str(time.time() - 1)])
    check(rc == 0, "exit 0 on the deadline path")
    budget = load(os.path.join(b_dir, b_id, "budget_summary.json"))
    check(budget.get("stopped_on_deadline") is True, "stopped_on_deadline recorded")
    check(budget["successes"] == 0 and stub.meta_calls == 0, "no generation started")

    print("\n=== C: the handed-off cell resumes where it stopped ===")
    check(run_search(url, b_dir, b_id, target=1, extra=["--search_only"]) == 0, "exit 0")
    archive = load(os.path.join(b_dir, f"{b_id}_run_archive.json"))
    check(len(archive) == N_SEEDS + 1, f"seeds kept, one generation added (got {len(archive)})")

    print("\n=== G: test evaluation stops at the deadline and resumes aligned ===")
    eval_path = os.path.join(b_dir, f"{b_id}_run_archive_evaluate.json")
    check(run_search(url, b_dir, b_id, target=1,
                     extra=["--evaluate_only", "--deadline_epoch", str(time.time() - 1)]) == 0,
          "exit 0 when the deadline has passed")
    check(not os.path.exists(eval_path) or not load(eval_path), "no design evaluated past the deadline")
    check(run_search(url, b_dir, b_id, target=1, extra=["--evaluate_only"]) == 0, "resumed evaluation exit 0")
    evaluated = load(eval_path)
    check(len(evaluated) == N_SEEDS + 1 and all(e["test_per_question"] == expected_test for e in evaluated),
          f"all {N_SEEDS + 1} designs evaluated, each with the ground-truth test vector")

    print("\n=== H: test crashes count as wrong answers; failed test runs are retried ===")
    archive_path = os.path.join(b_dir, f"{b_id}_run_archive.json")
    archive = load(archive_path)
    archive.append({"thought": "crashes on odd-length questions", "name": "Crashy", "code": CRASHY_CODE,
                    "generation": 2, "raw_median_accuracy": acc_valid})
    with open(archive_path, "w") as handle:
        json.dump(archive, handle)
    evaluated[2] = {k: v for k, v in evaluated[2].items() if not k.startswith("test_")}
    evaluated[2]["test_error"] = "stub failure"
    with open(eval_path, "w") as handle:
        json.dump(evaluated, handle)
    check(run_search(url, b_dir, b_id, target=1, extra=["--evaluate_only"]) == 0, "exit 0")
    evaluated = load(eval_path)
    expected_crashy = [0 if len(e["inputs"]) % 2 else v for e, v in zip(TEST_SET, expected_test)]
    check(len(evaluated) == N_SEEDS + 2, f"crashing design evaluated (got {len(evaluated)} records)")
    check(evaluated[-1].get("test_per_question") == expected_crashy
          and evaluated[-1].get("test_crashed_questions") == sum(len(e["inputs"]) % 2 for e in TEST_SET),
          "crashing questions scored wrong and counted, the rest scored normally")
    check("test_error" not in evaluated[2] and evaluated[2].get("test_per_question") == expected_test,
          "a previously failed test run is retried in place")
    server.shutdown()

    print("\n=== I: a deadline during a test run leaves it unrecorded; the next job runs it ===")
    stub = Stub()
    server = serve(stub)
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    i_dir, i_id = os.path.join(workdir, "I"), "e2eI"
    os.makedirs(i_dir)
    i_archive = os.path.join(i_dir, f"{i_id}_run_archive.json")
    i_eval = os.path.join(i_dir, f"{i_id}_run_archive_evaluate.json")
    i_summary = os.path.join(i_dir, i_id, "evaluation_summary.json")
    designs = [e for e in load(os.path.join(a_dir, f"{a_id}_run_archive.json"))
               if isinstance(e.get("generation"), int)][:2]
    with open(i_archive, "w") as handle:
        json.dump(designs, handle)
    stub.worker_delay = 30  # three rounds of four parallel questions: >= 90 s per design
    started = time.time()
    check(run_search(url, i_dir, i_id, target=1,
                     extra=["--evaluate_only", "--deadline_epoch", str(time.time() + 20)]) == 0,
          "exit 0 when the deadline cuts a test run")
    check(time.time() - started < 80, "stopped at the deadline rather than running to completion")
    check(not os.path.exists(i_eval) or not load(i_eval), "the cut design is not recorded")
    summary = load(i_summary) if os.path.exists(i_summary) else {}
    check(summary.get("stopped_on_deadline") is True and summary.get("untested") == 2,
          f"evaluation summary records the deadline stop (got {summary})")
    stub.worker_delay = 0
    check(run_search(url, i_dir, i_id, target=1, extra=["--evaluate_only"]) == 0, "resumed evaluation exit 0")
    evaluated = load(i_eval)
    check(len(evaluated) == 2 and all(e["test_per_question"] == expected_test for e in evaluated),
          "both designs evaluated, each with the ground-truth test vector")
    check(all(e.get("test_timeout_s") == 3600 for e in evaluated), "fast designs keep the validation cap")
    check(load(i_summary).get("stopped_on_deadline") is False, "evaluation summary records a completed pass")

    print("\n=== J: test caps scale with validation time; a timeout is retried only under a larger cap ===")
    designs[1]["resource"] = dict(designs[1].get("resource") or {}, wallclock_s=5000.0)
    with open(i_archive, "w") as handle:
        json.dump(designs, handle)
    old_timeout = "SandboxTimeout: Sandboxed evaluation exceeded 3600s wall clock"
    stale = [dict({k: v for k, v in e.items() if not k.startswith("test_")}, test_error=old_timeout)
             for e in evaluated]
    with open(i_eval, "w") as handle:
        json.dump(stale, handle)
    check(run_search(url, i_dir, i_id, target=1, extra=["--evaluate_only"]) == 0, "exit 0")
    evaluated = load(i_eval)
    check(evaluated[0].get("test_error") == old_timeout,
          "timed out at 3600 s and still capped at 3600 s: not retried")
    check(not evaluated[1].get("test_error") and evaluated[1].get("test_per_question") == expected_test
          and evaluated[1].get("test_timeout_s") == 2 * 5000.0 * TEST / VALID,
          "the cap scales with the design's validation time, and the old timeout is retried under it")
    designs.append(dict(designs[0], name="Slow"))
    with open(i_archive, "w") as handle:
        json.dump(designs, handle)
    stub.worker_delay = 5
    check(run_search(url, i_dir, i_id, target=1, extra=["--evaluate_only", "--test_timeout", "1"]) == 0,
          "exit 0")
    evaluated = load(i_eval)
    check(len(evaluated) == 3 and str(evaluated[2].get("test_error", "")).startswith("SandboxTimeout")
          and evaluated[2].get("test_timeout_s") == 1, "a run over its cap is recorded with that cap")
    time.sleep(1)  # requests the killed run had in flight
    calls = stub.worker_calls
    check(run_search(url, i_dir, i_id, target=1, extra=["--evaluate_only", "--test_timeout", "1"]) == 0
          and stub.worker_calls == calls, "under the same cap it is not run again")
    stub.worker_delay = 0
    check(run_search(url, i_dir, i_id, target=1, extra=["--evaluate_only", "--test_timeout", "600"]) == 0,
          "exit 0")
    evaluated = load(i_eval)
    check(evaluated[2].get("test_per_question") == expected_test and evaluated[2].get("test_timeout_s") == 600
          and evaluated[0].get("test_error") == old_timeout,
          "a larger cap retries it; a cap still below the earlier one does not")
    server.shutdown()

    print("\n=== F: designers share one evaluated seed archive ===")
    stub = Stub()
    server = serve(stub)
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    shared = os.path.join(workdir, "shared_seeds")
    shared_args = ["--seed_cache_dir", shared, "--require_seed_cache"]
    check(run_search(url, os.path.join(workdir, "F0"), "e2eF0", target=1,
                     extra=["--search_only", *shared_args]) != 0,
          "a cell requiring the shared cache refuses to start without it")
    check(stub.meta_calls == 0, "and makes no meta call")
    check(run_search(url, os.path.join(workdir, "S"), "e2eS", target=1,
                     extra=["--seeds_only", "--seed_cache_dir", shared]) == 0, "seeds-only cell exit 0")
    cache_files = os.listdir(shared) if os.path.isdir(shared) else []
    check(len(cache_files) == 1 and len(load(os.path.join(shared, cache_files[0]))) == N_SEEDS,
          f"seeds-only writes all {N_SEEDS} seeds to the shared cache")
    check(stub.meta_calls == 0, "seeds-only makes no meta call")
    check(not os.path.exists(os.path.join(workdir, "S", "e2eS_run_archive_evaluate.json")),
          "seeds-only skips test evaluation")
    seeds = []
    for name in ("F1", "F2"):
        check(run_search(url, os.path.join(workdir, name), f"e2e{name}", target=1,
                         extra=["--search_only", *shared_args]) == 0, f"cell {name} exit 0")
        archive = load(os.path.join(workdir, name, f"e2e{name}_run_archive.json"))
        seeds.append([{k: e.get(k) for k in ("name", "valid_per_question", "resource")}
                      for e in archive if e.get("generation") == "initial"])
    check(len(seeds[0]) == N_SEEDS and seeds[0] == seeds[1],
          "both cells start from the identical cached seed archive (same wallclock: not re-evaluated)")
    server.shutdown()

    print("\n=== K: a deadline during a generation's evaluation voids the attempt ===")
    stub = Stub()
    server = serve(stub)
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    k_dir, k_id = os.path.join(workdir, "K"), "e2eK"
    stub.worker_delay = 30
    started = time.time()
    check(run_search(url, k_dir, k_id, target=1,
                     extra=["--search_only", *shared_args, "--deadline_epoch", str(time.time() + 25)]) == 0,
          "exit 0 when the deadline cuts a generation")
    check(time.time() - started < 80, "stopped at the deadline rather than finishing the generation")
    budget = load(os.path.join(k_dir, k_id, "budget_summary.json"))
    check(budget.get("stopped_on_deadline") is True and budget["successes"] == 0 and budget["attempts"] == 0,
          f"stopped on the deadline with the cut attempt uncounted (got {budget})")
    abandoned_path = os.path.join(k_dir, k_id, f"{k_id}_abandoned_generations.json")
    check(not os.path.exists(abandoned_path) or not load(abandoned_path), "nothing recorded as abandoned")
    server.shutdown()
    stub = Stub()
    server = serve(stub)
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    check(run_search(url, k_dir, k_id, target=1, extra=["--search_only", *shared_args]) == 0,
          "resumed search exit 0")
    archive = load(os.path.join(k_dir, f"{k_id}_run_archive.json"))
    generated = [e for e in archive if isinstance(e.get("generation"), int)]
    check(len(archive) == N_SEEDS + 1 and len(generated) == 1 and generated[0]["valid_per_question"] == expected_valid,
          "the next job completes the generation")
    server.shutdown()

    print(f"\n{'PASSED' if not check.failed else f'{check.failed} CHECK(S) FAILED'}  (artifacts in {workdir})")
    return 1 if check.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
