"""Offline tests for the Copilot gateway: no SDK, no network beyond 127.0.0.1, no credits.

A scripted fake backend stands in for the Copilot SDK, and every request goes
through the real ``RoleClient``, so the checks cover exactly what a search sees:
JSON extraction, schema closing, meta conversation continuation, error mapping
(context overflow, structured-output fallback, rate limits), the per-port model
allowlist, the ledger, and the budget and authentication holds.

    python experiments/tests/copilot_gateway_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import openai  # noqa: E402

from experiments.adas_ext.instrumentation import Instrumentation, set_instrumentation  # noqa: E402
from experiments.adas_ext.model_router import (  # noqa: E402
    AGENT_DESIGN_SCHEMA, ContextOverflowError, RoleClient, conversation_tag,
)
from experiments.serving.copilot_gateway import (  # noqa: E402
    Backend, Gateway, Ledger, PortConfig, Pricing, SharedPacer, TurnResult, UsageEvent,
    classify_backend_error, close_schema, conversation_key, extract_json_object, parse_rates, serve,
)

WORKER, META = "fake-worker", "fake-meta"
PRICING = {WORKER: Pricing(input=0.10, output=0.50), META: Pricing(input=1.0, output=5.0)}


class Checks:
    def __init__(self):
        self.failed = 0

    def __call__(self, ok, label):
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
        self.failed += 0 if ok else 1


def usage(model, input_tokens=1000, output_tokens=100, nano=None, reasoning=0):
    # initiator "user" is what the real runtime reports for a call we made.
    return UsageEvent(model=model, input_tokens=input_tokens, output_tokens=output_tokens,
                      reasoning_tokens=reasoning, nano_aiu=nano, finish_reason="stop",
                      initiator="user", api_call_id=f"call-{time.time_ns()}")


class FakeBackend(Backend):
    """Sessions are dicts; ``script(handle, prompt, schema)`` returns a TurnResult or raises."""

    def __init__(self, script):
        self.script = script
        self.handles = []

    async def open_session(self, model, system, reasoning_effort, max_tokens):
        handle = {"model": model, "system": system, "effort": reasoning_effort,
                  "max_tokens": max_tokens, "prompts": [], "schemas": [], "closed": False,
                  "partial": []}
        self.handles.append(handle)
        return handle

    async def send(self, handle, prompt, schema, timeout):
        handle["prompts"].append(prompt)
        handle["schemas"].append(schema)
        result = self.script(handle, prompt, schema)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def close_session(self, handle):
        handle["closed"] = True

    def partial_usage(self, handle):
        return list(handle["partial"])


def start(script, workdir, name, **gateway_kwargs):
    ledger_kwargs = {k: gateway_kwargs.pop(k) for k in ("max_credits", "global_max_credits", "ledger_glob")
                     if k in gateway_kwargs}
    backend = FakeBackend(script)
    ledger = Ledger(os.path.join(workdir, name, "ledger.jsonl"), **ledger_kwargs)
    gateway_kwargs.setdefault("retry_backoff", 0.01)
    gateway = Gateway(backend, ledger, PRICING, cell=name,
                      halt_file=os.path.join(workdir, name, "HALTED"), **gateway_kwargs)
    gateway.start_loop()
    ports = [PortConfig("worker", 0, [WORKER], continuation=False),
             PortConfig("meta", 0, [META], continuation=True)]
    servers = serve(gateway, ports)
    urls = {p.role: f"http://127.0.0.1:{p.port}/v1" for p in ports}
    return gateway, backend, servers, urls


def stop(gateway, servers):
    for server in servers:
        server.shutdown()
    gateway.shutdown()


def client(role, url, model, **kwargs):
    kwargs.setdefault("max_retries", 0)
    kwargs.setdefault("timeout", 30.0)
    return RoleClient(role=role, model=model, base_url=url, **kwargs)


def ledger_lines(gateway):
    with open(gateway.ledger.path) as handle:
        return [json.loads(line) for line in handle]


def post(url, body, timeout):
    request = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def test_helpers(check):
    print("\n=== helpers ===")
    check(extract_json_object('{"a": 1}') == '{"a": 1}', "bare object returned unchanged")
    check(json.loads(extract_json_object('```json\n{"answer": "B"}\n```')) == {"answer": "B"},
          "fenced object extracted")
    check(json.loads(extract_json_object('Sure. Here it is: {"x": {"y": [1, 2]}} done')) ==
          {"x": {"y": [1, 2]}}, "object after preamble extracted, nested intact")
    check(extract_json_object("no json here") is None and extract_json_object("[1, 2]") is None,
          "no object -> None")
    # Verbatim shape from Claude Haiku 5.5 on the ADAS worker prompt (Phase 0, 4 of 10 replies).
    check(json.loads(extract_json_object("{'thinking': 'Seattle has 20 sheep, so 4 × 20 = 80.', 'answer': '260'}"))
          == {"thinking": "Seattle has 20 sheep, so 4 × 20 = 80.", "answer": "260"},
          "a Python dict literal, as the ADAS prompt displays the format, is recovered")
    check(extract_json_object("{'a': __import__('os').getcwd()}") is None, "literal parsing never evaluates code")
    # Verbatim Claude Haiku 5.5 replies to the MMLU CoT prompt (Phase 1): unescaped
    # apostrophes inside single-quoted values, fenced or bare.
    haiku = [
        ("```json\n{'thinking': 'Weber defined the key characteristics of bureaucracy. Merton focused on how "
         "bureaucratic structure shaped officials' personalities. Gouldner's \"Patterns of Industrial "
         "Bureaucracy\" fits best.', 'answer': 'C'}\n```", "C"),
        ("{'thinking': 'Protein has the highest TEF, while carbohydrate's is about 5-10%. (C) is true.', "
         "'answer': 'C'}", "C"),
        ("{'thinking': 'Singer's principle qualifies the duty. In this question's wording, that clause is "
         "\"nearly as important.\"', 'answer': 'B'}", "B"),
    ]
    recovered = [json.loads(extract_json_object(text) or "{}") for text, _ in haiku]
    check(all(r.get("answer") == want for r, (_, want) in zip(recovered, haiku)),
          "single-quoted dicts with unescaped apostrophes recover their answer")
    check("officials' personalities" in recovered[0]["thinking"] and "\"nearly as important.\"" in
          recovered[2]["thinking"], "apostrophes and double quotes inside values are kept")
    check(json.loads(extract_json_object('{"thinking": "he said "hi" twice", "answer": "D"}'))["answer"] == "D",
          "double-quoted values with unescaped inner quotes recover too")
    closed = close_schema(AGENT_DESIGN_SCHEMA)
    check(closed["additionalProperties"] is False and closed["required"] == ["thought", "name", "code"]
          and "additionalProperties" not in AGENT_DESIGN_SCHEMA, "schema closed on a copy")
    a = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
         {"role": "assistant", "content": "raw"}, {"role": "user", "content": "u2"}]
    b = [dict(m) for m in a]
    b[2]["content"] = "{'thought': 'x'}"
    check(conversation_key("m", a) == conversation_key("m", b), "key ignores assistant wording")
    check(conversation_key("m", a) != conversation_key("m2", a), "key includes the model")
    check(classify_backend_error(Exception("Session error: prompt is too long: 210000 > 200000"))[0]
          == "overflow", "overflow classified")
    kind, message = classify_backend_error(Exception("context window exceeded"))
    check("maximum context length" in message, "overflow message carries RoleClient's marker")
    check(classify_backend_error(Exception("HTTP 429 Too Many Requests"))[0] == "rate_limit", "429")
    check(classify_backend_error(Exception("Not authenticated. Please authenticate first."))[0] == "fatal",
          "auth failure is fatal")
    # Verbatim from SDK 1.0.24 / runtime 1.0.101 with no credentials.
    check(classify_backend_error(Exception("Session error: Execution failed: InvalidArg, No GitHub OAuth "
                                           "token or Copilot HMAC key provided"))[0] == "fatal",
          "the runtime's missing-token error is fatal")
    check(classify_backend_error(Exception("status: 403"))[0] == "fatal", "status 403 is fatal")
    check(classify_backend_error(Exception("upstream reset, request id 4031-403"))[0] == "transient",
          "a 403 inside a request id is not fatal")
    check(classify_backend_error(Exception("response_format json_schema unsupported"))[0] == "schema",
          "schema rejection")
    check(classify_backend_error(asyncio.TimeoutError())[0] == "timeout", "timeout")
    price = Pricing(input=0.10, output=0.50, long_context_threshold=100_000, long_input=0.50,
                    long_output=2.50)
    check(abs(price.credits(1_000_000, 0) - 50.0) < 1e-9, "1M long-context input = $0.50 = 50 credits")
    check(abs(price.credits(50_000, 10_000) - 1.0) < 1e-9, "default tier: $0.005 + $0.005 = 1 credit")


def test_requests(check, workdir):
    print("\n=== requests through RoleClient ===")
    state = {"meta_calls": 0}

    def script(handle, prompt, schema):
        if handle["model"] == WORKER:
            if "overflow" in prompt:
                raise Exception("Session error: prompt is too long: 300000 tokens")
            if "bare number" in prompt:
                return TurnResult(content="42", usage=[usage(WORKER)])
            # Fenced and with a preamble: providers without a JSON mode do this.
            return TurnResult(content='Answer:\n```json\n{"thinking": "t", "answer": "42"}\n```',
                              usage=[usage(WORKER, nano=2_000_000)])
        state["meta_calls"] += 1
        if schema is not None and handle.get("reject_schema"):
            raise Exception("response_format json_schema is not supported by this model")
        body = {"thought": f"t{state['meta_calls']}", "name": "n", "code": "def forward(self, x): pass"}
        return TurnResult(content=json.dumps(body), reasoning="deliberation",
                          usage=[usage(META, 20_000, 2_000, reasoning=500)])

    gateway, backend, servers, urls = start(script, workdir, "requests")
    instrumentation = set_instrumentation(Instrumentation(run_id="t"))
    try:
        worker = client("worker", urls["worker"], WORKER)
        reply = worker.chat_json([{"role": "system", "content": "Reply in JSON"},
                                  {"role": "user", "content": "6*7?"}], temperature=0.8)
        check(reply == {"thinking": "t", "answer": "42"}, "worker JSON extracted from fenced text")
        line = ledger_lines(gateway)[-1]
        check(line["json_extracted"] is True and line["temperature_requested"] == 0.8
              and line["mode"] == "fresh", "ledger: extraction flagged, inert temperature logged")
        check(abs(line["credits_reported"] - 0.002) < 1e-12 and abs(line["credits_estimated"] - 0.015) < 1e-9
              and line["credits_charged"] == line["credits_estimated"],
              "charged = max(reported, estimated)")
        bucket = instrumentation.usage_for("initial", phase="search").get("worker", {})
        check(abs(bucket.get("credits", 0) - 0.015) < 1e-9 and bucket.get("prompt_tokens") == 1000,
              "instrumentation receives tokens and credits")
        check(all(h["closed"] for h in backend.handles if h["model"] == WORKER), "worker sessions closed")

        try:
            worker.chat_json([{"role": "system", "content": "s"}, {"role": "user", "content": "overflow"}])
            check(False, "overflow raises")
        except ContextOverflowError:
            check(True, "provider overflow surfaces as ContextOverflowError")

        try:
            worker.chat_json([{"role": "system", "content": "s"}, {"role": "user", "content": "bare number"}])
            check(False, "a bare JSON number is refused")
        except ValueError as exc:
            # LLMAgentBase.query catches this and fills the fields with '' (a wrong answer),
            # where a returned int crashed its handler and voided the whole evaluation.
            check("not an object" in str(exc), "a bare JSON number is a format failure, not a crash")

        meta = client("meta", urls["meta"], META)
        messages = [{"role": "system", "content": "design"}, {"role": "user", "content": "archive v1"}]
        first = meta.chat_json(list(messages), schema=AGENT_DESIGN_SCHEMA)
        handle = backend.handles[-1]
        check(handle["schemas"][0]["additionalProperties"] is False, "meta schema forwarded closed")
        check(meta.last_reasoning == "deliberation", "reasoning trace reaches RoleClient.last_reasoning")
        messages += [{"role": "assistant", "content": str(first)}, {"role": "user", "content": "reflect 1"}]
        meta.chat_json(list(messages), schema=AGENT_DESIGN_SCHEMA)
        messages += [{"role": "assistant", "content": "{'whatever': 1}"}, {"role": "user", "content": "reflect 2"}]
        meta.chat_json(list(messages), schema=AGENT_DESIGN_SCHEMA)
        check(handle["prompts"] == ["archive v1", "reflect 1", "reflect 2"] and not handle["closed"],
              "reflexion turns continue one session, sending only the new user turn")
        check([l["mode"] for l in ledger_lines(gateway)[-3:]] == ["fresh", "continued", "continued"],
              "ledger records fresh, continued, continued")

        orphan = [{"role": "system", "content": "design"}, {"role": "user", "content": "other"},
                  {"role": "assistant", "content": "x"}, {"role": "user", "content": "follow-up"}]
        meta.chat_json(orphan, schema=AGENT_DESIGN_SCHEMA)
        flat = backend.handles[-1]["prompts"][0]
        check(ledger_lines(gateway)[-1]["mode"] == "flattened" and "<assistant>" in flat
              and flat.rstrip().endswith("follow-up"), "unknown history is flattened into one prompt")

        def reject(handle, prompt, schema):
            handle["reject_schema"] = True
            return script(handle, prompt, schema)

        backend.script = reject
        # The fallback re-request spends one of RoleClient's retries (it has 4 in production).
        fallback = client("meta", urls["meta"], META, max_retries=1)
        reply = fallback.chat_json([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                                   schema=AGENT_DESIGN_SCHEMA)
        check(set(reply) == {"thought", "name", "code"} and fallback.use_guided_json is False,
              "schema rejection falls back to JSON mode through RoleClient's existing path")
        backend.script = script

        try:
            client("worker", urls["worker"], META).chat_json([{"role": "user", "content": "x"}])
            check(False, "foreign model refused")
        except openai.PermissionDeniedError:
            check(True, "the worker port refuses the designer model (403)")
        with urllib.request.urlopen(f"{urls['worker']}/models") as response:
            listed = [m["id"] for m in json.loads(response.read())["data"]]
        check(listed == [WORKER], "/v1/models lists only the port's allowlist")
    finally:
        stop(gateway, servers)


def test_retries(check, workdir):
    print("\n=== retries and partial usage ===")
    attempts = {"n": 0}

    def flaky(handle, prompt, schema):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            handle["partial"] = [usage(WORKER, 500, 0)]
            raise Exception("upstream returned status 429: too many requests")
        return TurnResult(content='{"answer": "1"}', usage=[usage(WORKER)])

    gateway, backend, servers, urls = start(flaky, workdir, "retries")
    try:
        reply = client("worker", urls["worker"], WORKER).chat_json([{"role": "user", "content": "q"}])
        lines = ledger_lines(gateway)
        check(reply == {"answer": "1"} and attempts["n"] == 3, "rate-limited twice, then served")
        check([bool(l["error"]) for l in lines] == [True, True, False], "each failed attempt is in the ledger")
        check(lines[0]["input_tokens"] == 500 and lines[0]["credits_charged"] > 0,
              "tokens billed before a failure are charged")
    finally:
        stop(gateway, servers)

    print("\n=== retry budget (an outage outlasting max_retries) ===")
    outage = {"n": 0}

    def network_down(handle, prompt, schema):
        outage["n"] += 1
        if outage["n"] <= 6:
            raise Exception("Session error: Execution failed: Failed to get response from the AI model; "
                            "dns error: failed to lookup address information")
        return TurnResult(content='{"answer": "1"}', usage=[usage(WORKER)])

    gateway, backend, servers, urls = start(network_down, workdir, "outage", max_retries=1, max_retry_wait=30)
    try:
        reply = client("worker", urls["worker"], WORKER).chat_json([{"role": "user", "content": "q"}])
        check(reply == {"answer": "1"} and outage["n"] == 7,
              "with max_retries=1, the retry budget carries the call through six failures")
    finally:
        stop(gateway, servers)


def test_holds(check, workdir):
    print("\n=== budget and authentication holds ===")

    def pricey(handle, prompt, schema):
        return TurnResult(content='{"answer": "1"}', usage=[usage(WORKER, 1_000_000, 0)])  # 10 credits

    gateway, backend, servers, urls = start(pricey, workdir, "budget", max_credits=15.0)
    body = {"model": WORKER, "messages": [{"role": "user", "content": "q"}]}
    try:
        post(urls["worker"], body, timeout=10)
        check(gateway.halt_reason is None, "under the cap after one 10-credit call")
        post(urls["worker"], body, timeout=10)
        check(gateway.halt_reason is not None and os.path.exists(gateway.halt_file),
              "the call that crosses the cap halts the gateway and writes the halt file")
        outcome = {}

        def held():
            try:
                post(urls["worker"], body, timeout=30)
                outcome["status"] = 200
            except urllib.error.HTTPError as exc:
                outcome["status"] = exc.code
            except Exception as exc:  # noqa: BLE001
                outcome["status"] = type(exc).__name__

        thread = threading.Thread(target=held)
        thread.start()
        thread.join(1.5)
        check(thread.is_alive() and gateway.counters["held"] == 1,
              "a request after the cap is held, not failed (it would score as a wrong answer)")
        with urllib.request.urlopen(f"{urls['worker']}/health") as response:
            health = json.loads(response.read())
        check(health["status"] == "halted" and health["cell_spent"] == 20.0, "/health reports the halt")
    finally:
        stop(gateway, servers)
        thread.join(10)
    check(outcome.get("status") not in (None, 200), f"the held request is released with an error at shutdown "
                                                    f"(got {outcome.get('status')})")
    calls = len(ledger_lines(gateway))
    check(calls == 2, "the held request never reached the backend")

    resumed = Ledger(os.path.join(workdir, "budget", "ledger.jsonl"), max_credits=15.0)
    check(resumed.cell_spent == 20.0 and resumed.exhausted_reason(), "a restarted cell keeps its spend")
    other = Ledger(os.path.join(workdir, "other", "ledger.jsonl"), global_max_credits=25.0,
                   ledger_glob=os.path.join(workdir, "bud*", "ledger.jsonl"))
    check(other.global_spent == 20.0 and other.exhausted_reason() is None, "global spend sums other cells")

    def unauthenticated(handle, prompt, schema):
        raise Exception("Request session.send failed with message: Not authenticated. Please authenticate first.")

    def quiet_post():
        try:
            post(urls["worker"], body, timeout=30)
        except Exception:  # noqa: BLE001 - released with an error at shutdown
            pass

    gateway, backend, servers, urls = start(unauthenticated, workdir, "auth")
    try:
        thread = threading.Thread(target=quiet_post, daemon=True)
        thread.start()
        thread.join(1.5)
        check(thread.is_alive() and "Not authenticated" in (gateway.halt_reason or ""),
              "an authentication failure halts and holds instead of scoring wrong answers")
    finally:
        stop(gateway, servers)


def _probe_script(hidden_tokens):
    def script(handle, prompt, schema):
        billed = (len(handle["system"]) + len(prompt)) // 4 + hidden_tokens
        output = 400 if handle["effort"] == "high" else 100
        content = (json.dumps({"thought": "t", "name": "n", "code": "c"}) if schema
                   else '{"thinking": "t", "answer": "1"}')
        return TurnResult(content=content, reasoning="r",
                          usage=[usage(handle["model"], billed, output, reasoning=output // 2)])
    return script


def probe_backend():
    """A provider that bills the prompt plus a chat-template-sized constant."""
    return FakeBackend(_probe_script(hidden_tokens=12))


def probe_backend_overhead():
    """A provider that silently adds a large hidden system prompt."""
    return FakeBackend(_probe_script(hidden_tokens=3000))


def test_probe(check, workdir):
    print("\n=== Phase 0 probe (fake provider) ===")
    from experiments.serving import copilot_probe

    common = ["--results_root", os.path.join(workdir, "probe_results"), "--max_credits", "50"]
    me = "experiments.tests.copilot_gateway_test"
    check(copilot_probe.main([*common, "--backend", f"{me}:probe_backend",
                              "audit", "--model", WORKER, "--n", "8"]) == 0,
          "audit passes when billed tokens are the prompt plus a small constant")
    check(copilot_probe.main([*common, "--backend", f"{me}:probe_backend_overhead",
                              "audit", "--model", WORKER, "--n", "8"]) == 1,
          "audit fails when the provider adds a hidden prompt")
    check(copilot_probe.main([*common, "--backend", f"{me}:probe_backend", "capabilities",
                              "--models", WORKER, "--n", "3", "--effort_repeats", "1"]) == 0,
          "capability matrix runs")
    reports = sorted(os.listdir(os.path.join(workdir, "probe_results", "_probes")))
    row = load_json(os.path.join(workdir, "probe_results", "_probes",
                                 next(r for r in reports if r.startswith("capabilities"))))["rows"][0]
    check(row["json_schema"] == "ok" and row["effort_effective"] and row["json_valid"] == 1.0,
          "capability row: schema ok, effort effective, JSON valid")
    check(copilot_probe.main([*common, "--backend", f"{me}:probe_backend", "throughput",
                              "--model", WORKER, "--concurrency", "2,4", "--rounds", "2"]) == 0,
          "throughput sweep recommends a concurrency")


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def test_pacing(check, workdir):
    print("\n=== pacing ===")

    def instant(handle, prompt, schema):
        return TurnResult(content='{"answer": "1"}', usage=[usage(WORKER)])

    gateway, backend, servers, urls = start(instant, workdir, "pacing", max_calls_per_minute=600)
    body = {"model": WORKER, "messages": [{"role": "user", "content": "q"}]}
    try:
        started = time.time()
        threads = [threading.Thread(target=post, args=(urls["worker"], body, 30)) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.time() - started
        # Recorded latency includes the pacing wait, so spacing shows in completion times
        # (the fake backend answers instantly).
        finished = sorted(l["t"] for l in ledger_lines(gateway))
    finally:
        stop(gateway, servers)
    check(elapsed >= 0.85, f"10 concurrent calls at 600/min are spread over >= 0.9 s (took {elapsed:.2f} s)")
    gaps = [b - a for a, b in zip(finished, finished[1:])]
    check(len(gaps) == 9 and min(gaps) >= 0.08, f"calls are spaced ~0.1 s apart (min gap {min(gaps):.3f} s)")


def test_shared_pacing(check, workdir):
    print("\n=== shared pacing across gateways ===")

    def instant(handle, prompt, schema):
        return TurnResult(content='{"answer": "1"}', usage=[usage(handle["model"])])

    pace_dir = os.path.join(workdir, "pace")
    families = {WORKER: "fam_a", META: "fam_b"}
    pacer = SharedPacer(pace_dir, {"fam_a": 600, "fam_b": 600})
    check(pacer.group_for(WORKER, families) == "fam_a" and pacer.group_for("other", families) is None,
          "models pace by family; unlisted models are not paced")
    check(SharedPacer(pace_dir, {WORKER: 30, "fam_a": 600}).group_for(WORKER, families) == WORKER,
          "a model's own rate overrides its family's")
    check(SharedPacer(pace_dir, {"all": 500}).group_for("other", families) == "all"
          and SharedPacer(pace_dir, {"all": 500, "fam_b": 50}).group_for(META, families) == "fam_b",
          "unlisted models fall back to the account-wide `all` group")
    check(parse_rates("openai=600, anthropic=350,bad=0") == {"openai": 600.0, "anthropic": 350.0},
          "pace rates parse; zero rates are dropped")
    one = start(instant, workdir, "shared1", pacer=SharedPacer(pace_dir, {"fam_a": 600, "fam_b": 600}),
                families=families)
    two = start(instant, workdir, "shared2", pacer=SharedPacer(pace_dir, {"fam_a": 600, "fam_b": 600}),
                families=families)

    def burst(targets):
        threads = [threading.Thread(target=post, args=(url, {"model": model, "messages": [
            {"role": "user", "content": f"q{i}"}]}, 30)) for i, (url, model) in enumerate(targets)]
        started = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return time.time() - started

    try:
        elapsed = burst([(one[3]["worker"], WORKER)] * 10 + [(two[3]["worker"], WORKER)] * 10)
        finished = sorted(l["t"] for g in (one[0], two[0]) for l in ledger_lines(g))
        gaps = [b - a for a, b in zip(finished, finished[1:])]
        check(elapsed >= 1.8, f"20 calls from two gateways share one 600/min slot file (took {elapsed:.2f} s)")
        check(len(gaps) == 19 and min(gaps) >= 0.08,
              f"starts are spaced across both gateways (min gap {min(gaps):.3f} s)")
        elapsed = burst([(one[3]["worker"], WORKER)] * 10 + [(one[3]["meta"], META)] * 10)
        check(0.85 <= elapsed < 1.6, f"two groups pace independently (20 calls took {elapsed:.2f} s)")
    finally:
        stop(one[0], one[2])
        stop(two[0], two[2])


def test_tagged_sessions(check, workdir):
    print("\n=== tagged designer sessions (batch rounds) ===")

    def script(handle, prompt, schema):
        body = {"thought": prompt, "name": "n", "code": "def forward(self, x): pass"}
        return TurnResult(content=json.dumps(body), reasoning=f"trace for {prompt}",
                          usage=[usage(META, 2_000, 200)])

    gateway, backend, servers, urls = start(script, workdir, "tagged")
    set_instrumentation(Instrumentation(run_id="t"))
    try:
        meta = client("meta", urls["meta"], META)
        opening = [{"role": "system", "content": "design"}, {"role": "user", "content": "archive"}]
        replies = {}
        for tag in ("p0", "p1"):
            with conversation_tag(tag):
                replies[tag] = meta.chat_json(list(opening), schema=AGENT_DESIGN_SCHEMA)
        for tag in ("p0", "p1"):
            with conversation_tag(tag):
                meta.chat_json(opening + [{"role": "assistant", "content": str(replies[tag])},
                                          {"role": "user", "content": f"reflect {tag}"}],
                               schema=AGENT_DESIGN_SCHEMA)
        sessions = [h for h in backend.handles if h["model"] == META]
        check(len(sessions) == 2 and [h["prompts"] for h in sessions]
              == [["archive", "reflect p0"], ["archive", "reflect p1"]],
              "identical opening prompts with different tags keep separate sessions")
        check([l["mode"] for l in ledger_lines(gateway)] == ["fresh", "fresh", "continued", "continued"],
              "both conversations continue")

        meta.last_reasoning = None
        seen = {}

        def other_thread():
            meta.chat_json(list(opening), schema=AGENT_DESIGN_SCHEMA)
            seen["trace"] = meta.last_reasoning

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        check(seen.get("trace") == "trace for archive" and meta.last_reasoning is None,
              "last_reasoning is per thread")
    finally:
        stop(gateway, servers)


def main() -> int:
    check = Checks()
    workdir = tempfile.mkdtemp(prefix="copilot_gateway_test_")
    test_helpers(check)
    test_requests(check, workdir)
    test_retries(check, workdir)
    test_holds(check, workdir)
    test_pacing(check, workdir)
    test_shared_pacing(check, workdir)
    test_tagged_sessions(check, workdir)
    test_probe(check, workdir)
    print(f"\n{'PASSED' if not check.failed else f'{check.failed} CHECK(S) FAILED'}  (artifacts in {workdir})")
    return 1 if check.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
