"""Phase 0 probes for the Copilot arm: what the account offers and what a call costs.

Every probe except ``inventory`` goes through an in-process copy of the gateway,
so it measures exactly the path the searches use, and spends against its own
capped ledger (counted toward the campaign's global cap).

    python experiments/serving/copilot_probe.py inventory
    python experiments/serving/copilot_probe.py audit --model claude-haiku-5.5
    python experiments/serving/copilot_probe.py capabilities --models claude-haiku-5.5,gpt-6-luna
    python experiments/serving/copilot_probe.py throughput --model claude-haiku-5.5

``audit`` is the gate the plan puts before any search: a call must bill our
prompt and nothing else (one model call, no sub-agents, no hidden system prompt
or tool definitions, no model substitution). It exits 1 when that fails.
Reports go to ``<results_root>/_probes/``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import yaml  # noqa: E402

from experiments.adas_ext.model_router import AGENT_DESIGN_SCHEMA  # noqa: E402
from experiments.serving.copilot_gateway import (  # noqa: E402
    DEFAULT_COPILOT_HOME, CopilotBackend, Gateway, Ledger, PortConfig, load_pricing, serve,
)

EXIT_SUCCESS, EXIT_FAILURE, EXIT_ERROR = 0, 1, 2
MODELS_PATH = os.path.join(REPO_ROOT, "experiments", "config", "copilot_models.yaml")
CELLS_PATH = os.path.join(REPO_ROOT, "experiments", "config", "copilot_cells.yaml")
DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, "experiments", "results_copilot")

# The ADAS worker prompt shape (_mgsm/search.py LLMAgentBase.generate_prompt).
WORKER_SYSTEM = ("You are a helpful assistant.\n\nReply EXACTLY with the following JSON format.\n"
                 "{'thinking': 'Your thinking.', 'answer': 'Your answer. Return ONLY an integer. "
                 "DO NOT return anything other than the integer answer.'}\n"
                 "DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n")
WORKER_INSTRUCTION = "Please think step by step and then solve the task."
PADDING = "This sentence is padding for a token-accounting probe and carries no information. "


def mgsm_questions(n: int) -> List[str]:
    path = os.path.join(REPO_ROOT, "dataset", "mgsm", "mgsm_en.tsv")
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    return [row[0] for row in rows[:n]]


def worker_messages(question: str, padding_sentences: int = 0) -> List[dict]:
    task = question + ("\n\n" + PADDING * padding_sentences if padding_sentences else "")
    return [{"role": "system", "content": WORKER_SYSTEM},
            {"role": "user", "content": f"# Your Task:\n{task}\n\n{WORKER_INSTRUCTION}"}]


class ProbeGateway:
    """The gateway on loopback ports, with a dedicated capped ledger."""

    def __init__(self, args: argparse.Namespace, name: str, models: List[str],
                 max_retries: int = 5, honor_max_tokens: bool = False) -> None:
        results_root = os.path.abspath(args.results_root)
        cells = yaml.safe_load(open(CELLS_PATH)) or {}
        self.ledger = Ledger(os.path.join(results_root, f"_probe_{name}", "copilot_ledger.jsonl"),
                             max_credits=args.max_credits,
                             global_max_credits=cells.get("global_max_credits"),
                             ledger_glob=os.path.join(results_root, "*", "copilot_ledger.jsonl"))
        self.start_line = self._lines()
        backend = _backend(args, honor_max_tokens)
        self.gateway = Gateway(backend, self.ledger, load_pricing(MODELS_PATH), cell=f"probe:{name}",
                               max_concurrency=args.max_concurrency, max_retries=max_retries,
                               request_timeout=args.request_timeout,
                               halt_file=os.path.join(results_root, f"_probe_{name}", "GATEWAY_HALTED.json"))
        self.gateway.start_loop()
        self.port = PortConfig("probe", 0, models, continuation=False)
        self.servers = serve(self.gateway, [self.port])
        self.url = f"http://127.0.0.1:{self.port.port}/v1"

    def _lines(self) -> int:
        if not os.path.exists(self.ledger.path):
            return 0
        with open(self.ledger.path) as handle:
            return sum(1 for _ in handle)

    def entries(self) -> List[dict]:
        """Ledger lines written by this probe run."""
        if not os.path.exists(self.ledger.path):
            return []
        with open(self.ledger.path) as handle:
            return [json.loads(line) for line in list(handle)[self.start_line:]]

    def post(self, body: dict, timeout: float = 600.0) -> dict:
        if self.gateway.halt_reason:
            return {"status": "halted", "error": self.gateway.halt_reason}
        request = urllib.request.Request(f"{self.url}/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            return {"status": 200, "latency": time.time() - started, "body": payload}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            return {"status": exc.code, "latency": time.time() - started, "error": detail}
        except Exception as exc:  # noqa: BLE001 - a held request times out here
            return {"status": type(exc).__name__, "latency": time.time() - started, "error": str(exc)}

    def close(self) -> None:
        for server in self.servers:
            server.shutdown()
        self.gateway.shutdown()


def _backend(args: argparse.Namespace, honor_max_tokens: bool):
    if args.backend:
        import importlib

        module, _, factory = args.backend.partition(":")
        return getattr(importlib.import_module(module), factory)()
    return _copilot_backend(args, honor_max_tokens)


def _copilot_backend(args: argparse.Namespace, honor_max_tokens: bool = False) -> CopilotBackend:
    return CopilotBackend(honor_max_tokens=honor_max_tokens, client_mode=args.client_mode,
                          copilot_home=args.copilot_home,
                          github_token=os.environ.get("ADAS_COPILOT_TOKEN"))


def write_report(args: argparse.Namespace, name: str, report: dict) -> str:
    out_dir = os.path.join(os.path.abspath(args.results_root), "_probes")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nreport: {path}")
    return path


def _content(result: dict) -> Optional[str]:
    try:
        return result["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _is_json_object(text: Optional[str]) -> bool:
    try:
        return isinstance(json.loads(text or ""), dict)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {k: _plain(v) for k, v in vars(value).items()}
    return value


def cmd_inventory(args: argparse.Namespace) -> int:
    async def fetch():
        backend = _copilot_backend(args)
        await backend.start()
        try:
            models = [_plain(m) for m in await backend.list_models()]
            quota = None
            try:
                from copilot.generated.rpc import AccountGetQuotaRequest

                quota = _plain(await backend.client.rpc.account.get_quota(AccountGetQuotaRequest()))
            except Exception as exc:  # noqa: BLE001 - quota is informative only
                quota = {"error": str(exc)}
            return models, quota
        finally:
            await backend.stop()

    models, quota = asyncio.run(fetch())
    registry = (yaml.safe_load(open(MODELS_PATH)) or {}).get("models") or {}
    by_id = {m.get("id"): m for m in models}
    print(f"{'id':28} {'policy':10} {'context':>9} {'prompt':>9}  efforts")
    for m in sorted(models, key=lambda m: str(m.get("id"))):
        caps = m.get("capabilities") or {}
        limits = caps.get("limits") or {}
        efforts = m.get("supportedReasoningEfforts") or m.get("supported_reasoning_efforts") or []
        policy = (m.get("policy") or {}).get("state", "-")
        print(f"{str(m.get('id')):28} {str(policy):10} {str(limits.get('max_context_window_tokens', '-')):>9} "
              f"{str(limits.get('max_prompt_tokens', '-')):>9}  {','.join(efforts) or '-'}")
    problems = []
    for key in registry:
        if key not in by_id:
            problems.append(f"{key}: not offered to this account (fix the id in copilot_models.yaml)")
        elif str((by_id[key].get("policy") or {}).get("state", "enabled")) not in ("enabled", "None"):
            problems.append(f"{key}: policy state {(by_id[key].get('policy') or {}).get('state')}")
    print("\nregistry check:", "all registered ids are offered" if not problems else "")
    for problem in problems:
        print("  -", problem)
    write_report(args, "inventory", {"models": models, "quota": quota, "registry_problems": problems})
    return EXIT_SUCCESS if not problems else EXIT_FAILURE


# ---------------------------------------------------------------------------
# audit (gate)
# ---------------------------------------------------------------------------

def _fit(xs: List[float], ys: List[float]):
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return mean_y - slope * mean_x, slope


def cmd_audit(args: argparse.Namespace) -> int:
    probe = ProbeGateway(args, "audit", [args.model])
    try:
        minimal = probe.post({"model": args.model, "messages": [
            {"role": "system", "content": "Reply with one word."},
            {"role": "user", "content": "Say OK."}]})
        pairs = []
        for i, question in enumerate(mgsm_questions(args.n)):
            body = {"model": args.model, "messages": worker_messages(question, (0, 25, 50, 100)[i % 4]),
                    "response_format": {"type": "json_object"}, "temperature": 0.5}
            result = probe.post(body)
            if result["status"] == 200:
                pairs.append((sum(len(m["content"]) for m in body["messages"]),
                              result["body"]["usage"]["prompt_tokens"]))
        entries = probe.entries()
    finally:
        probe.close()

    ok = [e for e in entries if not e.get("error")]
    intercept, slope = _fit(*map(list, zip(*pairs))) if len(pairs) >= 4 else (float("nan"), float("nan"))
    minimal_tokens = minimal["body"]["usage"]["prompt_tokens"] if minimal["status"] == 200 else None
    served = sorted({m for e in ok for m in e.get("served_models") or []})
    # The runtime labels our own calls initiator "user"; anything else it started itself.
    foreign = sorted({i for e in ok for i in e["initiators"] if i != "user"})
    checks = {
        "one model call per request": bool(ok) and all(e["n_usage_events"] == 1 for e in ok),
        "no sub-agent or runtime-initiated calls": not foreign and all(not e["subagent_calls"] for e in ok),
        f"minimal prompt bills <= {args.max_overhead_tokens} input tokens":
            minimal_tokens is not None and minimal_tokens <= args.max_overhead_tokens,
        # An extrapolation: padding and task text tokenize differently, so this only
        # catches a large hidden prompt; the minimal prompt bounds the real overhead.
        f"no large hidden prompt (regression intercept <= {args.max_hidden_tokens} tokens)":
            intercept == intercept and intercept <= args.max_hidden_tokens,
        "no model substitution": all(m.startswith(args.model) or args.model.startswith(m) for m in served),
        "every request answered": len(pairs) == args.n and minimal_tokens is not None,
        "reported cost matches the price table (within 2%)": all(
            e["credits_reported"] is None or e["credits_estimated"] is None
            or abs(e["credits_reported"] - e["credits_estimated"]) <= 0.02 * max(e["credits_estimated"], 1e-9)
            for e in ok),
    }
    print(f"\naudit of {args.model}: {len(ok)} calls, {sum(e['credits_charged'] for e in entries):.3f} credits")
    print(f"  minimal prompt billed {minimal_tokens} input tokens; regression: "
          f"{intercept:.1f} tokens + {slope:.3f} tokens/char; served models {served}")
    for label, passed in checks.items():
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")
    passed = all(checks.values())
    print("GATE PASSED" if passed else "GATE FAILED: do not start searches until this is understood")
    write_report(args, f"audit_{args.model}", {
        "model": args.model, "checks": checks, "passed": passed, "minimal_input_tokens": minimal_tokens,
        "intercept_tokens": intercept, "tokens_per_char": slope, "served_models": served, "entries": entries})
    return EXIT_SUCCESS if passed else EXIT_FAILURE


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------

def _usage_stats(results: List[dict]) -> dict:
    usage = [r["body"]["usage"] for r in results if r["status"] == 200]
    return {
        "reasoning": statistics.fmean([u["completion_tokens_details"]["reasoning_tokens"] for u in usage])
        if usage else None,
        "output": statistics.fmean([u["completion_tokens"] for u in usage]) if usage else None,
        "latency_s": statistics.fmean([r["latency"] for r in results if r["status"] == 200]) if usage else None,
        "errors": [r.get("error") for r in results if r["status"] != 200][:2],
    }


def _capabilities(args: argparse.Namespace, model: str) -> dict:
    probe = ProbeGateway(args, f"capabilities_{model}", [model], max_retries=2,
                         honor_max_tokens=args.honor_max_tokens)
    row: Dict[str, Any] = {"model": model}
    try:
        results = [probe.post({"model": model, "messages": worker_messages(q),
                               "response_format": {"type": "json_object"}})
                   for q in mgsm_questions(args.n)]
        replies = [_content(r) for r in results]
        row["json_valid"] = sum(_is_json_object(c) for c in replies) / max(len(replies), 1)
        row["json_needed_extraction"] = sum(1 for e in probe.entries()
                                            if e.get("json_extracted") and not e.get("schema"))

        design = probe.post({"model": model, "response_format": {
            "type": "json_schema", "json_schema": {"name": "agent_design", "schema": AGENT_DESIGN_SCHEMA,
                                                   "strict": True}},
            "messages": [{"role": "system", "content": "You design agents. Reply in JSON."},
                         {"role": "user", "content": "Propose a one-line Python agent: thought, name, code."}]})
        body = _content(design)
        row["json_schema"] = ("ok" if design["status"] == 200 and _is_json_object(body)
                              and set(json.loads(body)) >= {"thought", "name", "code"}
                              else f"failed: {design.get('error', body)}"[:300])
        row["reasoning_trace"] = bool(design.get("body", {}).get("choices", [{}])[0]
                                      .get("message", {}).get("reasoning_content"))

        # Effort is measured on the worker workload itself (ADAS worker prompt, MGSM),
        # beside the default-effort calls above on the same questions.
        subset = mgsm_questions(args.effort_repeats)
        effort_tokens = {"default": _usage_stats(results[:args.effort_repeats])}
        for effort in ("low", "high"):
            effort_tokens[effort] = _usage_stats([
                probe.post({"model": model, "reasoning_effort": effort, "messages": worker_messages(q),
                            "response_format": {"type": "json_object"}}) for q in subset])
        low, high = effort_tokens["low"]["output"], effort_tokens["high"]["output"]
        row["effort_tokens"] = effort_tokens
        row["effort_effective"] = bool(low and high and high >= 1.2 * low)

        if args.honor_max_tokens:
            capped = probe.post({"model": model, "max_tokens": 64, "messages": [
                {"role": "user", "content": "Write 400 words on the history of the abacus."}]})
            usage = (capped.get("body") or {}).get("usage") or {}
            finish = ((capped.get("body") or {}).get("choices") or [{}])[0].get("finish_reason")
            row["max_tokens_honoured"] = bool(usage) and usage.get("completion_tokens", 1e9) <= 96 \
                and finish == "length"
        entries = probe.entries()
        row["credits"] = sum(e["credits_charged"] for e in entries)
        row["reported_vs_estimated"] = [
            (e["credits_reported"], e["credits_estimated"]) for e in entries[:5]]
    finally:
        probe.close()
    return row


def cmd_capabilities(args: argparse.Namespace) -> int:
    rows = [_capabilities(args, model) for model in [m.strip() for m in args.models.split(",") if m.strip()]]
    print(f"\n{'model':22} {'json ok':>8} {'extracted':>9} {'schema':>8} {'trace':>6} {'effort':>7} "
          f"{'max_tok':>8} {'credits':>8}")
    for r in rows:
        print(f"{r['model']:22} {r['json_valid']:8.0%} {r['json_needed_extraction']:9d} "
              f"{'ok' if r['json_schema'] == 'ok' else 'NO':>8} {str(r['reasoning_trace']):>6} "
              f"{str(r['effort_effective']):>7} {str(r.get('max_tokens_honoured', '-')):>8} {r['credits']:8.2f}")
    print("\nworker-call output tokens (reasoning) and latency by effort:")
    for r in rows:
        cells = [f"{level} {s['output'] or 0:.0f} ({s['reasoning'] or 0:.0f}) {s['latency_s'] or 0:.1f}s"
                 for level, s in r["effort_tokens"].items()]
        print(f"  {r['model']:22} " + " | ".join(cells))
    print("\nCompare credits_reported with credits_estimated in the report: a constant ratio far from 1 "
          "means the nano-AIU conversion or a price in copilot_models.yaml is off.")
    write_report(args, "capabilities", {"rows": rows})
    return EXIT_SUCCESS


# ---------------------------------------------------------------------------
# throughput
# ---------------------------------------------------------------------------

def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def cmd_throughput(args: argparse.Namespace) -> int:
    levels = [int(x) for x in args.concurrency.split(",")]
    home = args.copilot_home or (DEFAULT_COPILOT_HOME if args.client_mode == "empty"
                                 else os.path.expanduser("~/.copilot"))
    state_dir = os.path.join(home, "session-state")
    probe = ProbeGateway(args, f"throughput_{args.model}", [args.model], max_retries=0)
    questions = mgsm_questions(250)
    rows = []
    try:
        for level in levels:
            before_disk = _dir_size(state_dir)
            n = level * args.rounds
            bodies = [{"model": args.model, "messages": worker_messages(questions[i % len(questions)]),
                       "response_format": {"type": "json_object"}} for i in range(n)]
            started = time.time()
            with ThreadPoolExecutor(max_workers=level) as pool:
                results = list(pool.map(lambda b: probe.post(b, timeout=args.request_timeout), bodies))
            wall = time.time() - started
            latencies = sorted(r["latency"] for r in results if r["status"] == 200)
            rows.append({
                "concurrency": level, "requests": n, "wall_s": round(wall, 1),
                "calls_per_min": round(60 * len(latencies) / wall, 1) if wall else 0,
                "p50_s": round(latencies[len(latencies) // 2], 2) if latencies else None,
                "p95_s": round(latencies[int(len(latencies) * 0.95) - 1], 2) if latencies else None,
                "rate_limited": sum(1 for r in results if r["status"] == 429),
                "other_errors": sum(1 for r in results if r["status"] not in (200, 429)),
                "session_state_growth_kb": round((_dir_size(state_dir) - before_disk) / 1024, 1),
            })
            print(rows[-1], flush=True)
            if probe.gateway.halt_reason:
                print(f"halted: {probe.gateway.halt_reason}")
                break
        entries = probe.entries()
    finally:
        probe.close()
    good = [r for r in rows if r["rate_limited"] <= 0.01 * r["requests"] and not r["other_errors"]]
    best = max(good, key=lambda r: r["calls_per_min"]) if good else None
    print("\nrecommended max_workers:", best["concurrency"] if best else "none passed (<1% 429s, no errors)")
    print(f"credits: {sum(e['credits_charged'] for e in entries):.2f}")
    write_report(args, f"throughput_{args.model}", {"rows": rows, "recommended": best})
    return EXIT_SUCCESS if best else EXIT_FAILURE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0 probes for the Copilot arm")
    parser.add_argument("--results_root",
                        default=os.environ.get("ADAS_COPILOT_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    parser.add_argument("--max_credits", type=float, default=100.0, help="cap for this probe run")
    parser.add_argument("--max_concurrency", type=int, default=128)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--client_mode", choices=["copilot-cli", "empty"], default="copilot-cli")
    parser.add_argument("--copilot_home", default=None)
    parser.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory", help="list models, limits, efforts and quota; check the registry")
    audit = sub.add_parser("audit", help="gate: billed tokens are only our prompt")
    audit.add_argument("--model", required=True)
    audit.add_argument("--n", type=int, default=20)
    audit.add_argument("--max_overhead_tokens", type=int, default=100)
    audit.add_argument("--max_hidden_tokens", type=int, default=500)
    caps = sub.add_parser("capabilities", help="JSON, schema, effort, reasoning, max_tokens per model")
    caps.add_argument("--models", required=True)
    caps.add_argument("--n", type=int, default=10)
    caps.add_argument("--effort_repeats", type=int, default=3)
    caps.add_argument("--honor_max_tokens", action="store_true")
    throughput = sub.add_parser("throughput", help="concurrency sweep; recommends max_workers")
    throughput.add_argument("--model", required=True)
    throughput.add_argument("--concurrency", default="8,16,32,64")
    throughput.add_argument("--rounds", type=int, default=4, help="requests per level = rounds x concurrency")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    commands = {"inventory": cmd_inventory, "audit": cmd_audit,
                "capabilities": cmd_capabilities, "throughput": cmd_throughput}
    try:
        return commands[args.command](args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
