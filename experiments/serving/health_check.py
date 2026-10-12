#!/usr/bin/env python
"""Block until both vLLM servers are ready, then validate the two features the
whole campaign depends on.

Stage 0 does not exit until:

* ``/health`` and ``/v1/models`` respond on every configured endpoint
* a request with ``response_format={"type": "json_object"}`` returns parseable
  JSON -- the gpt-oss channel-separated response format with reasoning traces can
  interact badly with JSON mode, and both ends of the meta ladder use it
* the reasoning-effort control is accepted and actually changes behaviour, since
  the entire Stage 2r factor depends on it
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional
import urllib.error
import urllib.request


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read()


def _post(url: str, payload: dict, timeout: float = 240.0):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-vllm"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _process_alive(pid: int) -> bool:
    # A dead server child stays a zombie until the job shell reaps it, so a bare
    # os.kill(pid, 0) would report it alive; read its state instead.
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, IndexError):
        return False


# Specific on purpose: vLLM also logs harmless tracebacks as warnings.
_FATAL_MARKERS = (
    "Engine core initialization failed",
    "Timed out waiting for engine core",
    "CUDA out of memory",
    "OutOfMemoryError",
)


def _fatal_in_log(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        handle.seek(max(0, os.path.getsize(path) - 262144))
        tail = handle.read().decode(errors="replace")
    return next((m for m in _FATAL_MARKERS if m in tail), None)


def wait_for_health(base_url: str, timeout: float, poll: float = 5.0,
                    expect_model: Optional[str] = None, pid: Optional[int] = None,
                    log: Optional[str] = None) -> str:
    """Block until the endpoint serves ``expect_model``.

    A 200 from /health only proves something is listening on the port, not that
    it is ours. Compute nodes are shared by up to four jobs and SLURM does not
    always reap a previous job's server, so cross-checking /v1/models is what
    stops a whole cell running against another cell's models and being recorded
    under the wrong pair.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    deadline = time.time() + timeout
    last_error = "no attempt made"
    while time.time() < deadline:
        if pid is not None and not _process_alive(pid):
            raise RuntimeError(f"{base_url}: server process {pid} exited before becoming ready "
                               f"(last: {last_error}); see its log")
        # A server can fail and still leave its process alive, e.g. hung in
        # shutdown waiting on engine workers; its log says so long before timeout.
        fatal = _fatal_in_log(log)
        if fatal:
            raise RuntimeError(f"{base_url}: server log reports '{fatal}' ({log})")
        try:
            status, _ = _get(f"{root}/health")
            if status == 200:
                status, body = _get(f"{root}/v1/models")
                models = json.loads(body).get("data", [])
                served = [m["id"] for m in models]
                if not served:
                    last_error = "/v1/models returned an empty list"
                elif expect_model is None or expect_model in served:
                    return served[0]
                else:
                    last_error = (f"port is serving {served}, not {expect_model!r} "
                                  f"-- likely another job's server; still waiting for ours")
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(poll)
    raise TimeoutError(f"{base_url} not ready after {timeout:.0f}s (last: {last_error})")


def check_json_mode(base_url: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You reply only with well-formed JSON."},
            {"role": "user", "content": 'Reply with exactly {"thought": "...", "name": "...", "code": "..."}. '
                                        'Set name to "probe" and keep the other fields to one short sentence.'},
        ],
        "temperature": 0.0,
        # Thinking models spend tokens reasoning before the JSON starts; 512 left
        # them with no content at all and read as a JSON failure.
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    response = _post(f"{base_url.rstrip('/')}/chat/completions", payload)
    choice = response["choices"][0]
    content = choice["message"]["content"]
    if content is None:
        raise ValueError(f"no content returned (finish_reason={choice.get('finish_reason')})")
    parsed = json.loads(content)  # raises if JSON mode is not actually honoured
    usage = response.get("usage", {})
    details = usage.get("completion_tokens_details") or {}
    return {
        "parsed_keys": sorted(parsed.keys()),
        "finish_reason": response["choices"][0].get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "reasoning_split_exposed": "reasoning_tokens" in details,
    }


def check_guided_json(base_url: str, model: str) -> bool:
    """Schema-constrained decoding is preferred over plain JSON mode.

    Uses the same response_format json_schema request the search sends, so this
    reports whether production schema enforcement works on this server.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Return a design named probe."}],
        "temperature": 0.0,
        "max_tokens": 4096,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"thought": {"type": "string"}, "name": {"type": "string"},
                                   "code": {"type": "string"}},
                    "required": ["thought", "name", "code"],
                },
            },
        },
    }
    try:
        response = _post(f"{base_url.rstrip('/')}/chat/completions", payload)
        parsed = json.loads(response["choices"][0]["message"]["content"])
        return set(parsed) >= {"thought", "name", "code"}
    except Exception:
        return False


def check_reasoning_effort(base_url: str, model: str) -> dict:
    """Confirm the control is both accepted and load-bearing."""
    prompt = "A train leaves at 14:05 and arrives at 17:40. How long is the journey? Reply as JSON."
    observed = {}
    for effort in ("low", "high"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "reasoning_effort": effort,
        }
        try:
            response = _post(f"{base_url.rstrip('/')}/chat/completions", payload)
        except urllib.error.HTTPError as exc:
            return {"accepted": False, "error": f"HTTP {exc.code}: {exc.read()[:400]!r}"}
        usage = response.get("usage", {})
        details = usage.get("completion_tokens_details") or {}
        observed[effort] = {
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": details.get("reasoning_tokens", 0),
        }
    low = observed["low"]["reasoning_tokens"] or observed["low"]["completion_tokens"]
    high = observed["high"]["reasoning_tokens"] or observed["high"]["completion_tokens"]
    return {
        "accepted": True,
        "observed": observed,
        # If this is False the control is being silently ignored and Stage 2r
        # would measure nothing.
        "load_bearing": high > low,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-url", default=None)
    parser.add_argument("--meta-url", default=None)
    parser.add_argument("--expect-worker-model", default=None,
                        help="Reject a port that is serving something else, e.g. another job's server.")
    parser.add_argument("--expect-meta-model", default=None)
    parser.add_argument("--worker-pid", type=int, default=None,
                        help="Fail as soon as this server process dies, instead of polling to timeout.")
    parser.add_argument("--meta-pid", type=int, default=None)
    parser.add_argument("--worker-log", default=None, help="Server log to watch for fatal errors.")
    parser.add_argument("--meta-log", default=None)
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--check-json", action="store_true")
    parser.add_argument("--check-reasoning-effort", action="store_true")
    parser.add_argument("--report", default=None, help="Write the report as JSON here.")
    args = parser.parse_args()
    if not (args.worker_url or args.meta_url):
        parser.error("give at least one of --worker-url / --meta-url")

    endpoints = {}
    if args.worker_url:
        endpoints["worker"] = (args.worker_url, args.expect_worker_model, args.worker_pid, args.worker_log)
    if args.meta_url:
        endpoints["meta"] = (args.meta_url, args.expect_meta_model, args.meta_pid, args.meta_log)

    report = {}
    ok = True
    for role, (url, expected, pid, log) in endpoints.items():
        model = wait_for_health(url, args.timeout, expect_model=expected, pid=pid, log=log)
        entry = {"url": url, "model": model, "expected": expected, "healthy": True}
        print(f"[{role}] ready: {model} @ {url}")

        if args.check_json:
            try:
                entry["json_mode"] = check_json_mode(url, model)
                entry["guided_json"] = check_guided_json(url, model)
                print(f"[{role}] json_mode ok, guided_json={entry['guided_json']}, "
                      f"reasoning split exposed={entry['json_mode']['reasoning_split_exposed']}")
            except Exception as exc:
                entry["json_mode_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{role}] JSON MODE FAILED: {entry['json_mode_error']}", file=sys.stderr)
                ok = False

        if args.check_reasoning_effort:
            entry["reasoning_effort"] = check_reasoning_effort(url, model)
            print(f"[{role}] reasoning_effort: {json.dumps(entry['reasoning_effort'])}")
            if not entry["reasoning_effort"].get("accepted"):
                ok = False

        report[role] = entry

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
