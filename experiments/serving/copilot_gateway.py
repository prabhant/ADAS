"""OpenAI-compatible gateway in front of the GitHub Copilot SDK.

The ADAS pipeline talks to models through ``RoleClient`` (OpenAI chat
completions). This gateway serves that same endpoint on 127.0.0.1 and answers
each request through a Copilot SDK session, so search, sandbox, instrumentation
and every ``_<domain>/search.py`` run unchanged against models billed in GitHub
AI Credits. Only ``--meta_base_url`` / ``--worker_base_url`` change.

Why a separate process rather than the SDK inside ``search.py``:

* the sandboxed child that runs generated agent code may only reach allowlisted
  host:port pairs and cannot spawn processes, so it cannot host an SDK runtime;
* generated code must never hold a handle to a Copilot runtime with shell and
  file tools. The gateway exposes chat completions only, every session has tools
  off and permissions denied, and each port serves an explicit model allowlist,
  so worker code cannot reach the designer model;
* credentials stay in this process (stored ``copilot`` login, GitHub CLI login or
  ``COPILOT_GITHUB_TOKEN``), never in the search process or in results.

Protocol mapping (deviations are recorded per call in the ledger):

* ``temperature`` has no SDK equivalent. It is accepted, logged and ignored.
* ``response_format: json_object`` has no SDK equivalent either; the reply is
  requested as plain text (the ADAS prompts already demand JSON) and the JSON
  object is extracted from it. ``json_schema`` is forwarded as a closed strict
  schema.
* A meta-agent conversation (reflexion and debug turns) continues one SDK
  session: a request whose history, assistant turns aside, extends a live
  conversation by one assistant/user pair sends only the new user turn. The
  model therefore sees its own raw reply rather than ADAS's ``str(dict)`` echo
  of it. Unmatched multi-turn requests are flattened into one transcript.
* ``max_tokens`` is only applied when ``--honor_max_tokens`` is set, because the
  runtime treats it as a model-capability override; Phase 0 checks that it holds.

Budget: every call is appended to a JSONL ledger. When the cell or global AI
Credit cap is reached, or authentication fails, the gateway *holds* new
requests instead of failing them (a failed worker call would be scored as a
wrong answer and silently corrupt fitness), writes ``--halt_file`` and waits for
the runner to stop the search.

    python -m experiments.serving.copilot_gateway \\
        --worker_port 8100 --worker_models claude-haiku-5.5 \\
        --meta_port 8101 --meta_models claude-haiku-5.5 \\
        --ledger results/cell/copilot_ledger.jsonl --max_credits 2000
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import fcntl
import glob
import hashlib
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2

NANO_AIU_PER_CREDIT = 1e9
# Pace group every model falls back to: the Copilot rate limit is per account.
ACCOUNT_GROUP = "all"

logger = logging.getLogger("copilot_gateway")

# RoleClient raises ContextOverflowError when the 400 body contains one of its
# markers; this phrase contains "maximum context length".
OVERFLOW_MESSAGE = "This model's maximum context length was exceeded"
STRUCTURED_OUTPUT_MESSAGE = "structured output is not supported for this request"

_OVERFLOW_HINTS = (
    "context length", "context window", "maximum context", "prompt is too long",
    "input is too long", "too many tokens", "token limit", "exceeds the limit",
    "max_prompt_tokens", "request too large",
)
_RATE_HINTS = ("rate limit", "rate_limit", "too many requests", "throttl")
_SCHEMA_HINTS = ("response_format", "json_schema", "structured output", "responseformat",
                 "invalid schema")
# Failures that no retry fixes and that would otherwise be scored as wrong answers.
_FATAL_HINTS = (
    "not authenticated", "unauthorized", "authentication", "forbidden", "oauth token",
    "hmac key", "not logged in", "please log in",
    "model not found", "model is not available", "not enabled for", "is not supported",
    "unknown model", "insufficient", "budget exceeded", "limit exhausted", "quota exceeded",
    "exceeded your", "payment required",
)
# Bare status codes only next to a status word, so a request id containing "403" is not one.
_STATUS_CODE = re.compile(r"\b(?:status|http|code|error)\b\D{0,12}\b(40[123]|429)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pure helpers (no SDK, unit-tested)
# ---------------------------------------------------------------------------

def message_text(content: Any) -> str:
    """OpenAI message content as plain text (strings or a list of text parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def conversation_key(model: str, messages: List[dict], tag: Optional[str] = None) -> str:
    """Identity of a conversation state, blind to assistant wording.

    ADAS echoes the designer's reply back as ``str(dict)``, which never matches
    the raw text the session holds, so assistant turns are keyed by position only.
    ``tag`` (the request's ``user`` field) separates conversations whose messages
    are identical, such as the concurrent proposals of one batch round.
    """
    canonical = [[m.get("role"), "" if m.get("role") == "assistant" else message_text(m.get("content"))]
                 for m in messages]
    identity = [model, canonical] if tag is None else [model, canonical, tag]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def split_system(messages: List[dict]) -> Tuple[str, List[dict]]:
    if messages and messages[0].get("role") in ("system", "developer"):
        return message_text(messages[0].get("content")), list(messages[1:])
    return "", list(messages)


def flatten_conversation(turns: List[dict]) -> str:
    """One prompt holding every earlier turn, for a conversation with no live session."""
    earlier, last = turns[:-1], turns[-1]
    parts = ["The conversation so far is reproduced below.", ""]
    for turn in earlier:
        role = turn.get("role", "user")
        parts.append(f"<{role}>\n{message_text(turn.get('content'))}\n</{role}>\n")
    parts.append("Reply to the latest user message:")
    parts.append("")
    parts.append(message_text(last.get("content")))
    return "\n".join(parts)


_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def is_json_object(text: Optional[str]) -> bool:
    try:
        return isinstance(json.loads((text or "").strip()), dict)
    except (ValueError, TypeError):
        return False


def extract_json_object(text: Optional[str]) -> Optional[str]:
    """The first JSON object in a reply, serialised; None when there is none.

    Providers without a JSON mode answer in text that usually is, or contains,
    the requested object: bare, fenced, or after a sentence of preamble.
    """
    if text is None:
        return None
    stripped = text.strip()
    if is_json_object(stripped):
        return stripped
    for candidate in [m.strip() for m in _FENCE.findall(stripped)]:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            value, _ = decoder.raw_decode(stripped, match.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
    # The ADAS worker prompt shows the requested fields as a Python dict repr, and
    # without a JSON mode some models answer in that form ({'answer': '42'}).
    start, end = stripped.find("{"), stripped.rfind("}")
    for candidate in [m.strip() for m in _FENCE.findall(stripped)] + (
            [stripped[start:end + 1]] if 0 <= start < end else []):
        try:
            value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, default=str)
    if 0 <= start < end:
        loose = _loose_quoted_dict(stripped[start:end + 1])
        if loose:
            return json.dumps(loose, ensure_ascii=False)
    return None


# A key opens a dict or follows the closing quote of the previous value.
_LOOSE_KEY = re.compile(r"""(?:^\{|(?<=['"])\s*,)\s*(['"])(\w+)\1\s*:\s*(['"])""")


def _loose_quoted_dict(text: str) -> Optional[Dict[str, str]]:
    """Recover {'thinking': 'officials' personalities ...', 'answer': 'C'}.

    Claude Haiku 5.5 imitates the prompt's dict repr but leaves apostrophes inside
    values unescaped, which neither JSON nor Python can parse (Phase 1: 2-9% of its
    answers). Each value runs from its opening quote to the next key boundary, so
    an apostrophe inside it is kept. Only string values are recovered.
    """
    keys = list(_LOOSE_KEY.finditer(text))
    if not keys:
        return None
    body_end = text.rstrip().rstrip("}").rstrip()
    out: Dict[str, str] = {}
    for i, key in enumerate(keys):
        quote = key.group(3)
        if i + 1 < len(keys):
            # The next match starts right after this value's closing quote.
            raw = text[key.end():keys[i + 1].start() - 1]
        else:
            if not body_end.endswith(quote):
                return None
            raw = body_end[key.end():len(body_end) - 1]
        out[key.group(2)] = raw.replace("\\" + quote, quote).replace("\\n", "\n")
    return out


def close_schema(schema: Any) -> Any:
    """Strict structured output needs closed objects with every property required."""
    schema = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                properties = node.get("properties") or {}
                node.setdefault("additionalProperties", False)
                node["required"] = list(properties.keys())
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


def classify_backend_error(exc: BaseException) -> Tuple[str, str]:
    """Map a backend failure to (kind, message); kind drives retry and HTTP status."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout", f"upstream timeout: {exc}"
    message = str(exc) or type(exc).__name__
    lowered = message.lower()
    status = _STATUS_CODE.search(message)
    code = status.group(1) if status else None
    if any(hint in lowered for hint in _OVERFLOW_HINTS):
        return "overflow", f"{OVERFLOW_MESSAGE}: {message}"
    if code == "429" or any(hint in lowered for hint in _RATE_HINTS):
        return "rate_limit", message
    if any(hint in lowered for hint in _SCHEMA_HINTS):
        return "schema", f"{STRUCTURED_OUTPUT_MESSAGE}: {message}"
    if code in ("401", "402", "403") or any(hint in lowered for hint in _FATAL_HINTS):
        return "fatal", message
    return "transient", message


HTTP_STATUS = {"overflow": 400, "schema": 400, "rate_limit": 429, "timeout": 504,
               "transient": 502, "bad_request": 400, "forbidden": 403}


@dataclass
class Pricing:
    """USD per million tokens, with an optional long-context tier."""

    input: float
    output: float
    cached_input: float = 0.0
    cache_write: float = 0.0
    long_context_threshold: Optional[int] = None
    long_input: Optional[float] = None
    long_output: Optional[float] = None
    long_cached_input: Optional[float] = None
    long_cache_write: Optional[float] = None

    @classmethod
    def from_dict(cls, payload: dict) -> "Pricing":
        known = {k: payload[k] for k in cls.__dataclass_fields__ if k in payload}
        return cls(**known)

    def credits(self, input_tokens: int, output_tokens: int, cache_read: int = 0,
                cache_write: int = 0) -> float:
        long = self.long_context_threshold is not None and input_tokens > self.long_context_threshold
        price_in = self.long_input if long and self.long_input is not None else self.input
        price_out = self.long_output if long and self.long_output is not None else self.output
        price_cached = (self.long_cached_input if long and self.long_cached_input is not None
                        else self.cached_input)
        price_write = (self.long_cache_write if long and self.long_cache_write is not None
                       else self.cache_write)
        uncached = max(input_tokens - cache_read - cache_write, 0)
        usd = (uncached * price_in + cache_read * price_cached + cache_write * price_write
               + output_tokens * price_out) / 1e6
        return usd * 100.0  # 1 AI credit = $0.01


def load_pricing(path: Optional[str]) -> Dict[str, Pricing]:
    if not path or not os.path.exists(path):
        return {}
    import yaml

    with open(path) as handle:
        registry = yaml.safe_load(handle) or {}
    table = {}
    for key, spec in (registry.get("models") or {}).items():
        pricing = (spec or {}).get("pricing")
        if pricing:
            for name in {key, spec.get("copilot_id", key)}:
                table[name] = Pricing.from_dict(pricing)
    return table


def load_families(path: Optional[str]) -> Dict[str, str]:
    """Model id -> provider family (openai, anthropic), the default pace group."""
    if not path or not os.path.exists(path):
        return {}
    import yaml

    with open(path) as handle:
        registry = yaml.safe_load(handle) or {}
    families = {}
    for key, spec in (registry.get("models") or {}).items():
        family = (spec or {}).get("family")
        if family:
            for name in {key, (spec or {}).get("copilot_id", key)}:
                families[name] = family
    return families


def parse_rates(text: Optional[str]) -> Dict[str, float]:
    """``"all=550"`` or ``"openai=600,anthropic=350"`` -> {group: calls per minute}."""
    rates: Dict[str, float] = {}
    for item in (text or "").split(","):
        if "=" in item:
            group, _, value = item.partition("=")
            if group.strip() and float(value) > 0:
                rates[group.strip()] = float(value)
    return rates


class SharedPacer:
    """Spaces call starts across processes.

    Every gateway on the account reserves start slots from one file per pace group
    (a single model, a provider family, or ``all``), under an exclusive lock, so
    cells running side by side together stay under the account rate limit, and each
    speeds up as the others finish. Slots are wall-clock times, so a machine waking
    from sleep finds them in the past and resumes at once. The Copilot limit is
    account-wide (GPT-6 Luna and Claude Haiku 5.5 throttle each other, 2026-10-30),
    so ``all`` is the group that matters; per-model rates can sit below it.
    """

    def __init__(self, directory: str, rates: Dict[str, float]) -> None:
        self.directory = directory
        self.rates = {g: r for g, r in rates.items() if r and r > 0}
        os.makedirs(directory, exist_ok=True)

    def group_for(self, model: str, families: Dict[str, str]) -> Optional[str]:
        """The model's own rate, else its family's, else the account-wide ``all``."""
        if model in self.rates:
            return model
        family = families.get(model)
        if family in self.rates:
            return family
        return ACCOUNT_GROUP if ACCOUNT_GROUP in self.rates else None

    def reserve(self, group: Optional[str]) -> float:
        """Claim the next start slot of ``group``; returns the seconds to wait."""
        rate = self.rates.get(group or "")
        if not rate:
            return 0.0
        path = os.path.join(self.directory, f"{group}.slot")
        with open(path, "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0)
                try:
                    next_slot = float(handle.read().strip() or 0.0)
                except ValueError:
                    next_slot = 0.0
                now = time.time()
                start = max(now, next_slot)
                handle.seek(0)
                handle.truncate()
                handle.write(repr(start + 60.0 / rate))
                handle.flush()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        return start - now


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

@dataclass
class UsageEvent:
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    nano_aiu: Optional[float] = None
    cost_multiplier: Optional[float] = None
    finish_reason: Optional[str] = None
    initiator: Optional[str] = None
    agent_id: Optional[str] = None
    api_call_id: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class TurnResult:
    content: Optional[str]
    reasoning: Optional[str] = None
    usage: List[UsageEvent] = field(default_factory=list)
    model: Optional[str] = None


class Backend:
    """What the gateway needs from a model provider. Implementations: CopilotBackend, test fakes."""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def open_session(self, model: str, system: str, reasoning_effort: Optional[str],
                           max_tokens: Optional[int]) -> Any:
        raise NotImplementedError

    async def send(self, handle: Any, prompt: str, schema: Optional[dict],
                   timeout: float) -> TurnResult:
        raise NotImplementedError

    async def close_session(self, handle: Any) -> None:
        pass

    def partial_usage(self, handle: Any) -> List[UsageEvent]:
        """Usage already reported for a turn that then failed; failed calls can still bill."""
        return []


def _seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        return value.total_seconds() * 1000.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _Collector:
    """Session event sink: usage per model call, reasoning, errors."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.usage: List[UsageEvent] = []
        self.reasoning: List[str] = []

    def __call__(self, event: Any) -> None:
        from copilot.session_events import AssistantReasoningData, AssistantUsageData

        data = getattr(event, "data", None)
        if isinstance(data, AssistantUsageData):
            copilot_usage = getattr(data, "copilot_usage", None)
            self.usage.append(UsageEvent(
                model=data.model,
                input_tokens=data.input_tokens or 0,
                output_tokens=data.output_tokens or 0,
                reasoning_tokens=data.reasoning_tokens or 0,
                cache_read_tokens=data.cache_read_tokens or 0,
                cache_write_tokens=data.cache_write_tokens or 0,
                nano_aiu=getattr(copilot_usage, "total_nano_aiu", None) if copilot_usage else None,
                cost_multiplier=data.cost,
                finish_reason=data.finish_reason,
                initiator=data.initiator,
                agent_id=getattr(event, "agent_id", None) or None,
                api_call_id=data.api_call_id,
                duration_ms=_seconds(data.duration),
            ))
        elif isinstance(data, AssistantReasoningData) and not getattr(event, "agent_id", None):
            if data.content:
                self.reasoning.append(data.content)


@dataclass
class _CopilotHandle:
    session: Any
    collector: _Collector
    unsubscribe: Any


def _deny_all(request: Any, invocation: Any) -> Any:
    from copilot.generated.rpc import PermissionDecisionReject

    return PermissionDecisionReject(feedback="tools are disabled in the ADAS gateway")


DEFAULT_COPILOT_HOME = os.path.expanduser(os.path.join("~", ".cache", "adas-copilot", "home"))


class CopilotBackend(Backend):
    """Each session: replaced system prompt, no tools, permissions denied, nothing persisted.

    ``copilot-cli`` client mode is the default because the stored Copilot login
    (``copilot`` then ``/login``) is only found under ~/.copilot; every session is
    deleted after use, and the overhead audit checks that nothing from the user's
    configuration reaches the prompt. ``empty`` mode isolates the runtime in its
    own home (no instruction discovery, skills or session store) but then needs
    a token in ``COPILOT_GITHUB_TOKEN``.
    """

    def __init__(self, copilot_home: Optional[str] = None, github_token: Optional[str] = None,
                 honor_max_tokens: bool = False, log_level: str = "error",
                 client_mode: str = "copilot-cli") -> None:
        self.client_mode = client_mode
        self.copilot_home = copilot_home or (DEFAULT_COPILOT_HOME if client_mode == "empty" else None)
        self.github_token = github_token
        self.honor_max_tokens = honor_max_tokens
        self.log_level = log_level
        self.client = None

    async def start(self) -> None:
        from copilot import CopilotClient

        kwargs: Dict[str, Any] = {"mode": self.client_mode, "log_level": self.log_level}
        if self.copilot_home:
            os.makedirs(self.copilot_home, exist_ok=True)
            kwargs["base_directory"] = self.copilot_home
        if self.github_token:
            kwargs["github_token"] = self.github_token
        self.client = CopilotClient(**kwargs)
        await self.client.start()

    async def stop(self) -> None:
        if self.client is not None:
            await self.client.stop()

    async def list_models(self) -> list:
        return await self.client.list_models()

    async def open_session(self, model: str, system: str, reasoning_effort: Optional[str],
                           max_tokens: Optional[int]) -> Any:
        kwargs: Dict[str, Any] = {
            "model": model,
            "system_message": {"mode": "replace", "content": system},
            "available_tools": [],
            "on_permission_request": _deny_all,
            "infinite_sessions": {"enabled": False},
            "enable_session_store": False,
            "enable_session_telemetry": False,
            "enable_skills": False,
            "streaming": False,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if self.honor_max_tokens and max_tokens:
            from copilot import ModelCapabilitiesOverride, ModelLimitsOverride

            kwargs["model_capabilities"] = ModelCapabilitiesOverride(
                limits=ModelLimitsOverride(max_output_tokens=int(max_tokens)))
        session = await self.client.create_session(**kwargs)
        collector = _Collector()
        unsubscribe = session.on(collector)
        return _CopilotHandle(session=session, collector=collector, unsubscribe=unsubscribe)

    async def send(self, handle: Any, prompt: str, schema: Optional[dict],
                   timeout: float) -> TurnResult:
        from copilot.session_events import SessionErrorData

        handle.collector.reset()
        loop = asyncio.get_running_loop()
        session_error: asyncio.Future = loop.create_future()

        def watch(event: Any) -> None:
            # SDK 1.0.24's structured wait ignores an error raised before the
            # message is admitted (e.g. no credentials) and sits out the timeout.
            data = getattr(event, "data", None)
            if isinstance(data, SessionErrorData) and not getattr(event, "agent_id", None):
                loop.call_soon_threadsafe(
                    lambda: session_error.done() or session_error.set_result(data.message or str(data)))

        unsubscribe = handle.session.on(watch)
        turn = asyncio.ensure_future(
            handle.session.send_and_wait(prompt, response_schema=schema, timeout=timeout))
        try:
            await asyncio.wait({turn, session_error}, return_when=asyncio.FIRST_COMPLETED)
            if not turn.done():
                turn.cancel()
                await asyncio.gather(turn, return_exceptions=True)
                await self._abort(handle)
                raise RuntimeError(f"Session error: {session_error.result()}")
            event = turn.result()
        except (asyncio.TimeoutError, TimeoutError):
            await self._abort(handle)  # stop generation that would still be billed
            raise
        finally:
            if callable(unsubscribe):
                unsubscribe()
            if not session_error.done():
                session_error.cancel()
        data = getattr(event, "data", None) if event is not None else None
        content = getattr(data, "content", None)
        reasoning = "\n".join(handle.collector.reasoning) or getattr(data, "reasoning_text", None)
        return TurnResult(content=content, reasoning=reasoning, usage=list(handle.collector.usage),
                          model=getattr(data, "model", None))

    @staticmethod
    async def _abort(handle: Any) -> None:
        try:
            await handle.session.abort()
        except Exception:  # noqa: BLE001
            pass

    def partial_usage(self, handle: Any) -> List[UsageEvent]:
        return list(handle.collector.usage)

    async def close_session(self, handle: Any) -> None:
        try:
            if callable(handle.unsubscribe):
                handle.unsubscribe()
            session_id = getattr(handle.session, "session_id", None)
            await handle.session.disconnect()
            if session_id:
                await self.client.delete_session(session_id)
        except Exception as exc:  # noqa: BLE001 - cleanup must never fail a call
            logger.debug("session cleanup failed: %s", exc)


# ---------------------------------------------------------------------------
# Ledger and budget
# ---------------------------------------------------------------------------

def _ledger_total(path: str) -> float:
    total = 0.0
    if not os.path.exists(path):
        return total
    with open(path) as handle:
        for line in handle:
            try:
                total += float(json.loads(line).get("credits_charged") or 0.0)
            except (ValueError, TypeError):
                continue
    return total


class Ledger:
    """Append-only call record plus the two spend caps (this cell, all cells)."""

    def __init__(self, path: str, max_credits: Optional[float] = None,
                 global_max_credits: Optional[float] = None, ledger_glob: Optional[str] = None) -> None:
        self.path = path
        self.max_credits = max_credits
        self.global_max_credits = global_max_credits
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        # Resuming a cell keeps its earlier spend: the cap is per cell, not per process.
        self.cell_spent = _ledger_total(path)
        others = 0.0
        if ledger_glob:
            mine = os.path.realpath(path)
            for other in glob.glob(ledger_glob, recursive=True):
                if os.path.realpath(other) != mine:
                    others += _ledger_total(other)
        self.global_spent = others + self.cell_spent
        self.calls = 0

    def exhausted_reason(self) -> Optional[str]:
        with self._lock:
            if self.max_credits is not None and self.cell_spent >= self.max_credits:
                return f"cell AI credit cap reached ({self.cell_spent:.1f} >= {self.max_credits:.1f})"
            if self.global_max_credits is not None and self.global_spent >= self.global_max_credits:
                return (f"global AI credit cap reached "
                        f"({self.global_spent:.1f} >= {self.global_max_credits:.1f})")
        return None

    def record(self, entry: dict) -> None:
        charged = float(entry.get("credits_charged") or 0.0)
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            self.cell_spent += charged
            self.global_spent += charged
            self.calls += 1
            with open(self.path, "a") as handle:
                handle.write(line + "\n")

    def snapshot(self) -> dict:
        with self._lock:
            return {"calls": self.calls, "cell_spent": round(self.cell_spent, 4),
                    "global_spent": round(self.global_spent, 4), "max_credits": self.max_credits,
                    "global_max_credits": self.global_max_credits, "ledger": self.path}


# ---------------------------------------------------------------------------
# Gateway core
# ---------------------------------------------------------------------------

@dataclass
class PortConfig:
    role: str
    port: int
    models: List[str]
    continuation: bool
    reasoning_effort: Optional[str] = None
    host: str = "127.0.0.1"
    timeout: Optional[float] = None  # per-call limit; defaults to the gateway's request_timeout


class GatewayError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class _Conversation:
    handle: Any
    model: str
    last_used: float


class Gateway:
    """Request handling on one asyncio loop shared by every port."""

    def __init__(self, backend: Backend, ledger: Ledger, pricing: Dict[str, Pricing],
                 cell: str = "", request_timeout: float = 900.0, max_concurrency: int = 32,
                 max_retries: int = 5, retry_backoff: float = 2.0, max_backoff: float = 60.0,
                 halt_file: Optional[str] = None, max_conversations: int = 32,
                 conversation_ttl: float = 7200.0, max_calls_per_minute: float = 0.0,
                 max_retry_wait: float = 0.0, rate_limit_backoff: float = 0.0,
                 pacer: Optional[SharedPacer] = None,
                 families: Optional[Dict[str, str]] = None) -> None:
        self.backend = backend
        self.ledger = ledger
        self.pricing = pricing
        self.cell = cell
        self.request_timeout = request_timeout
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_backoff = max_backoff
        self.halt_file = halt_file
        self.max_conversations = max_conversations
        self.conversation_ttl = conversation_ttl
        self.max_calls_per_minute = max_calls_per_minute
        # Retry transient failures for this long in total, beyond max_retries: a laptop
        # asleep with its lid shut drops the network for minutes to hours, and a worker
        # call that gives up is scored as a wrong answer.
        self.max_retry_wait = max_retry_wait
        # Rate-limited calls are billed (the runtime retries internally and Copilot
        # charges each attempt's input), so wait out the limit before trying again.
        self.rate_limit_backoff = rate_limit_backoff
        # Account-wide pacing shared with every other gateway (see SharedPacer); the
        # per-gateway max_calls_per_minute still applies on top as a per-cell cap.
        self.pacer = pacer
        self.families = families or {}
        self._next_slot = 0.0
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._conversations: Dict[str, _Conversation] = {}
        self._halt_reason: Optional[str] = None
        self._halt_event: Optional[asyncio.Event] = None
        self._released = False
        self._warned_unpriced: set = set()
        self.counters = {"fresh": 0, "continued": 0, "flattened": 0, "errors": 0, "held": 0}

    # -- loop management -----------------------------------------------------

    def start_loop(self) -> None:
        ready = threading.Event()

        def run() -> None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self._halt_event = asyncio.Event()
            ready.set()
            self.loop.run_forever()

        threading.Thread(target=run, name="copilot-gateway-loop", daemon=True).start()
        ready.wait()
        self.run(self.backend.start())

    def run(self, coroutine, timeout: Optional[float] = None):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    def shutdown(self) -> None:
        if self.loop is None:
            return
        self._released = True
        self.loop.call_soon_threadsafe(self._halt_event.set)

        async def close_all() -> None:
            for conversation in list(self._conversations.values()):
                await self.backend.close_session(conversation.handle)
            self._conversations.clear()
            await self.backend.stop()

        try:
            self.run(close_all(), timeout=60)
        except Exception as exc:  # noqa: BLE001
            logger.warning("shutdown cleanup failed: %s", exc)
        self.loop.call_soon_threadsafe(self.loop.stop)

    # -- halting ---------------------------------------------------------------

    @property
    def halt_reason(self) -> Optional[str]:
        return self._halt_reason

    def halt(self, reason: str) -> None:
        if self._halt_reason is None:
            self._halt_reason = reason
            logger.error("HALTED: %s -- holding all further requests", reason)
            if self.halt_file:
                os.makedirs(os.path.dirname(os.path.abspath(self.halt_file)), exist_ok=True)
                with open(self.halt_file, "w") as handle:
                    json.dump({"reason": reason, "t": time.time(), **self.ledger.snapshot()}, handle)

    async def _hold(self) -> None:
        # A held request is answered only when the gateway shuts down; the runner
        # stops the search first, so nothing it would have scored is recorded.
        self.counters["held"] += 1
        await self._halt_event.wait()
        raise GatewayError("transient", f"gateway halted: {self._halt_reason}")

    # -- request handling ------------------------------------------------------

    async def _pace(self, model: Optional[str] = None) -> None:
        """Space call starts evenly. Bursts above the account's sustained rate draw
        a throttle in which every in-flight call waits ~15 minutes (Phase 1)."""
        if self.max_calls_per_minute:
            now = self.loop.time()
            start = max(now, self._next_slot)
            self._next_slot = start + 60.0 / self.max_calls_per_minute
            if start > now:
                await asyncio.sleep(start - now)
        if self.pacer is not None and model:
            wait = self.pacer.reserve(self.pacer.group_for(model, self.families))
            if wait > 0:
                await asyncio.sleep(wait)

    def _semaphore(self, role: str) -> asyncio.Semaphore:
        if role not in self._semaphores:
            self._semaphores[role] = asyncio.Semaphore(self.max_concurrency)
        return self._semaphores[role]

    def _credits(self, model: str, usage: List[UsageEvent]) -> Tuple[Optional[float], Optional[float]]:
        reported = None
        if any(u.nano_aiu is not None for u in usage):
            reported = sum((u.nano_aiu or 0.0) for u in usage) / NANO_AIU_PER_CREDIT
        estimated = None
        for u in usage:
            pricing = self.pricing.get(u.model or model) or self.pricing.get(model)
            if pricing is None:
                if (u.model or model) not in self._warned_unpriced:
                    self._warned_unpriced.add(u.model or model)
                    logger.warning("no pricing for %s; spend is only what Copilot reports",
                                   u.model or model)
                continue
            estimated = (estimated or 0.0) + pricing.credits(
                u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens)
        return reported, estimated

    async def _evict(self) -> None:
        now = time.time()
        stale = [k for k, c in self._conversations.items() if now - c.last_used > self.conversation_ttl]
        while len(self._conversations) - len(stale) > self.max_conversations:
            oldest = min((k for k in self._conversations if k not in stale),
                         key=lambda k: self._conversations[k].last_used)
            stale.append(oldest)
        for key in stale:
            conversation = self._conversations.pop(key, None)
            if conversation is not None:
                await self.backend.close_session(conversation.handle)

    async def handle(self, port: PortConfig, request: dict) -> dict:
        model = request.get("model")
        if model not in port.models:
            raise GatewayError("forbidden", f"model {model!r} is not served on the {port.role} port "
                                            f"(allowed: {port.models})")
        messages = request.get("messages") or []
        if not messages or messages[-1].get("role") != "user":
            raise GatewayError("bad_request", "the last message must be a user message")
        if self._halt_reason is not None:
            await self._hold()

        system, turns = split_system(messages)
        response_format = request.get("response_format") or {}
        schema = None
        if response_format.get("type") == "json_schema":
            schema = close_schema((response_format.get("json_schema") or {}).get("schema") or {})
        wants_json = response_format.get("type") in ("json_object", "json_schema")
        effort = request.get("reasoning_effort") or port.reasoning_effort
        max_tokens = request.get("max_tokens") or request.get("max_completion_tokens")

        started = time.time()
        attempt = 0
        while True:
            try:
                await self._pace(model)
                async with self._semaphore(port.role):
                    result, mode = await self._turn(port, model, system, turns, messages, schema,
                                                    effort, max_tokens, tag=request.get("user"))
                break
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                kind, message = classify_backend_error(exc)
                self.counters["errors"] += 1
                partial = getattr(exc, "partial_result", None)
                self._record(port, request, model, partial, "error", started, effort, schema,
                             error=f"{kind}: {message}"[:2000])
                reason = self.ledger.exhausted_reason()
                if reason:
                    self.halt(reason)
                    await self._hold()
                if kind == "fatal":
                    self.halt(f"{type(exc).__name__}: {message}"[:500])
                    await self._hold()
                within_budget = attempt < self.max_retries or (
                    self.max_retry_wait and time.time() - started < self.max_retry_wait)
                if kind in ("rate_limit", "transient", "timeout") and within_budget:
                    delay = min(self.retry_backoff ** attempt, self.max_backoff)
                    if kind == "rate_limit":
                        delay = max(delay, self.rate_limit_backoff)
                    await asyncio.sleep(delay * (0.5 + random.random()))
                    attempt += 1
                    continue
                raise GatewayError(kind, message) from exc

        content = result.content
        extracted = None
        if wants_json:
            json_text = extract_json_object(content)
            extracted = json_text is not None and not is_json_object(content)
            if json_text is not None:
                content = json_text
        entry = self._record(port, request, model, result, mode, started, effort, schema,
                             json_extracted=extracted)
        reason = self.ledger.exhausted_reason()
        if reason:
            self.halt(reason)
        return self._openai_response(model, content, result, entry)

    async def _turn(self, port: PortConfig, model: str, system: str, turns: List[dict],
                    messages: List[dict], schema: Optional[dict], effort: Optional[str],
                    max_tokens: Optional[int], tag: Optional[str] = None) -> Tuple[TurnResult, str]:
        handle = None
        prompt = message_text(turns[-1].get("content"))
        mode = "fresh"
        if port.continuation and len(turns) >= 3 and turns[-2].get("role") == "assistant":
            conversation = self._conversations.pop(conversation_key(model, messages[:-2], tag), None)
            if conversation is not None:
                handle, mode = conversation.handle, "continued"
        if handle is None:
            if len(turns) > 1:
                prompt, mode = flatten_conversation(turns), "flattened"
            handle = await self.backend.open_session(model, system, effort, max_tokens)
        keep = False
        try:
            try:
                result = await self.backend.send(handle, prompt, schema, port.timeout or self.request_timeout)
            except Exception as exc:
                partial = self.backend.partial_usage(handle)
                if partial:
                    exc.partial_result = TurnResult(content=None, usage=partial)
                raise
            if port.continuation:
                key = conversation_key(model, messages, tag)
                # An identical prompt (a failed generation retried on an unchanged
                # archive) replaces the older state; close it rather than leak it.
                replaced = self._conversations.pop(key, None)
                if replaced is not None:
                    await self.backend.close_session(replaced.handle)
                self._conversations[key] = _Conversation(handle=handle, model=model,
                                                         last_used=time.time())
                keep = True
                await self._evict()
        finally:
            if not keep:
                await self.backend.close_session(handle)
        self.counters[mode] += 1
        return result, mode

    def _record(self, port: PortConfig, request: dict, model: str, result: Optional[TurnResult],
                mode: str, started: float, effort: Optional[str], schema: Optional[dict],
                error: Optional[str] = None, json_extracted: Optional[bool] = None) -> dict:
        usage = result.usage if result else []
        reported, estimated = self._credits(model, usage) if usage else (None, None)
        # Conservative: a scale error in either source cannot under-count spend.
        charged = max(v for v in (reported, estimated, 0.0) if v is not None)
        entry = {
            "t": time.time(),
            "cell": self.cell,
            "role": port.role,
            "model": model,
            "served_models": sorted({u.model for u in usage if u.model}),
            "mode": mode,
            "n_messages": len(request.get("messages") or []),
            "temperature_requested": request.get("temperature"),
            "max_tokens_requested": request.get("max_tokens"),
            "reasoning_effort": effort,
            "schema": schema is not None,
            "json_extracted": json_extracted,
            "n_usage_events": len(usage),
            "input_tokens": sum(u.input_tokens for u in usage),
            "output_tokens": sum(u.output_tokens for u in usage),
            "reasoning_tokens": sum(u.reasoning_tokens for u in usage),
            "cache_read_tokens": sum(u.cache_read_tokens for u in usage),
            "cache_write_tokens": sum(u.cache_write_tokens for u in usage),
            "initiators": [u.initiator for u in usage if u.initiator],
            "subagent_calls": sum(1 for u in usage if u.agent_id),
            "finish_reasons": [u.finish_reason for u in usage],
            "api_call_ids": [u.api_call_id for u in usage if u.api_call_id],
            "credits_reported": reported,
            "credits_estimated": estimated,
            "credits_charged": charged,
            "latency_s": round(time.time() - started, 3),
            "error": error,
        }
        self.ledger.record(entry)
        return entry

    @staticmethod
    def _openai_response(model: str, content: Optional[str], result: TurnResult, entry: dict) -> dict:
        finish = next((r for r in reversed(entry["finish_reasons"]) if r), "stop")
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if result.reasoning:
            message["reasoning_content"] = result.reasoning
        return {
            "id": (entry["api_call_ids"] or [f"copilot-{int(entry['t'] * 1000)}"])[-1],
            "object": "chat.completion",
            "created": int(entry["t"]),
            "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "length" if finish == "length" else "stop"}],
            "usage": {
                "prompt_tokens": entry["input_tokens"],
                "completion_tokens": entry["output_tokens"],
                "total_tokens": entry["input_tokens"] + entry["output_tokens"],
                "completion_tokens_details": {"reasoning_tokens": entry["reasoning_tokens"]},
            },
            "x_copilot": {
                "credits": entry["credits_charged"],
                "credits_reported": entry["credits_reported"],
                "credits_estimated": entry["credits_estimated"],
                "cache_read_tokens": entry["cache_read_tokens"],
                "served_models": entry["served_models"],
                "mode": entry["mode"],
                "json_extracted": entry["json_extracted"],
            },
        }

    def status(self) -> dict:
        return {"status": "halted" if self._halt_reason else "ok", "reason": self._halt_reason,
                "cell": self.cell, "conversations": len(self._conversations),
                "counters": dict(self.counters), **self.ledger.snapshot()}


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

def make_handler(gateway: Gateway, port: PortConfig):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the caller was killed (deadline, sandbox cap, runner)

        def _error(self, kind: str, message: str) -> None:
            status = HTTP_STATUS.get(kind, 502)
            self._send(status, {"error": {"message": message, "type": kind, "code": kind}})

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"object": "list",
                                 "data": [{"id": m, "object": "model"} for m in port.models]})
            elif self.path.rstrip("/").endswith("/health"):
                self._send(200, gateway.status())
            else:
                self._error("bad_request", f"unknown path {self.path}")

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._error("bad_request", f"unknown path {self.path}")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, TypeError) as exc:
                self._error("bad_request", f"invalid JSON body: {exc}")
                return
            if request.get("stream"):
                self._error("bad_request", "streaming is not supported by the ADAS gateway")
                return
            try:
                payload = gateway.run(gateway.handle(port, request))
            except GatewayError as exc:
                self._error(exc.kind, str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                self._error("transient", f"{type(exc).__name__}: {exc}")
                return
            self._send(200, payload)

        def log_message(self, *args) -> None:
            pass

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512


def serve(gateway: Gateway, ports: List[PortConfig]) -> List[ThreadingHTTPServer]:
    servers = []
    for port in ports:
        server = _Server((port.host, port.port), make_handler(gateway, port))
        port.port = server.server_address[1]
        threading.Thread(target=server.serve_forever, name=f"gateway-{port.role}", daemon=True).start()
        servers.append(server)
    return servers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-compatible gateway over the GitHub Copilot SDK")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--worker_port", type=int, default=None, help="0 picks a free port")
    parser.add_argument("--worker_models", default="", help="comma-separated allowlist")
    parser.add_argument("--worker_reasoning_effort", default=None)
    parser.add_argument("--meta_port", type=int, default=None, help="0 picks a free port")
    parser.add_argument("--meta_models", default="", help="comma-separated allowlist")
    parser.add_argument("--meta_reasoning_effort", default=None)
    parser.add_argument("--cell", default="", help="label written into every ledger line")
    parser.add_argument("--ledger", required=True, help="JSONL call ledger for this cell")
    parser.add_argument("--max_credits", type=float, default=None, help="cap for this cell (AI credits)")
    parser.add_argument("--global_max_credits", type=float, default=None,
                        help="cap across every ledger matched by --ledger_glob")
    parser.add_argument("--ledger_glob", default=None)
    parser.add_argument("--pricing", default=None,
                        help="copilot_models.yaml; estimates spend when Copilot reports none")
    parser.add_argument("--request_timeout", type=float, default=900.0)
    parser.add_argument("--worker_request_timeout", type=float, default=None,
                        help="per worker call; a stuck call is cut and retried (default --request_timeout)")
    parser.add_argument("--max_concurrency", type=int, default=32, help="in-flight calls per port")
    parser.add_argument("--max_calls_per_minute", type=float, default=0.0,
                        help="pace call starts across all ports; 0 disables pacing")
    parser.add_argument("--pace_dir", default=None,
                        help="shared slot files, one per pace group, for every gateway on the account")
    parser.add_argument("--pace_rates", default="",
                        help="calls/min per pace group, e.g. 'openai=600,anthropic=350'; a group is a "
                             "model id or its family in --pricing")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--rate_limit_backoff", type=float, default=30.0,
                        help="minimum wait after a rate-limit error (rate-limited attempts are billed)")
    parser.add_argument("--max_retry_wait", type=float, default=0.0,
                        help="keep retrying transient failures for this many seconds in total")
    parser.add_argument("--honor_max_tokens", action="store_true")
    parser.add_argument("--client_mode", choices=["copilot-cli", "empty"], default="copilot-cli",
                        help="SDK client mode; empty isolates the runtime but needs COPILOT_GITHUB_TOKEN")
    parser.add_argument("--copilot_home", default=None,
                        help=f"runtime home (COPILOT_HOME); empty mode defaults to {DEFAULT_COPILOT_HOME}")
    parser.add_argument("--ready_file", default=None, help="written with the bound ports once serving")
    parser.add_argument("--halt_file", default=None, help="written when the gateway halts")
    parser.add_argument("--backend", default=None, help=argparse.SUPPRESS)  # module:factory, for tests
    parser.add_argument("--log_level", default="INFO")
    return parser


def _models(text: str) -> List[str]:
    return [m.strip() for m in text.split(",") if m.strip()]


def run(args: argparse.Namespace, backend: Optional[Backend] = None) -> int:
    ports = []
    if args.worker_port is not None:
        ports.append(PortConfig("worker", args.worker_port, _models(args.worker_models),
                                continuation=False, reasoning_effort=args.worker_reasoning_effort,
                                host=args.host, timeout=args.worker_request_timeout))
    if args.meta_port is not None:
        ports.append(PortConfig("meta", args.meta_port, _models(args.meta_models),
                                continuation=True, reasoning_effort=args.meta_reasoning_effort,
                                host=args.host))
    if not ports or any(not p.models for p in ports):
        logger.error("configure at least one port, each with a non-empty model allowlist")
        return EXIT_ERROR

    ledger = Ledger(args.ledger, args.max_credits, args.global_max_credits, args.ledger_glob)
    if backend is None and args.backend:
        import importlib

        module, _, factory = args.backend.partition(":")
        backend = getattr(importlib.import_module(module), factory)()
    if backend is None:
        backend = CopilotBackend(copilot_home=args.copilot_home,
                                 github_token=os.environ.get("ADAS_COPILOT_TOKEN"),
                                 honor_max_tokens=args.honor_max_tokens,
                                 client_mode=args.client_mode)
    pacer = None
    rates = parse_rates(args.pace_rates)
    if args.pace_dir and rates:
        pacer = SharedPacer(args.pace_dir, rates)
    gateway = Gateway(backend, ledger, load_pricing(args.pricing), cell=args.cell,
                      request_timeout=args.request_timeout, max_concurrency=args.max_concurrency,
                      max_retries=args.max_retries, halt_file=args.halt_file,
                      max_calls_per_minute=args.max_calls_per_minute,
                      max_retry_wait=args.max_retry_wait,
                      rate_limit_backoff=args.rate_limit_backoff,
                      pacer=pacer, families=load_families(args.pricing))
    reason = ledger.exhausted_reason()
    gateway.start_loop()
    if reason:
        gateway.halt(reason)
    servers = serve(gateway, ports)
    bound = {p.role: f"http://{p.host}:{p.port}/v1" for p in ports}
    logger.info("serving %s; spend so far %s", bound, ledger.snapshot())
    if args.ready_file:
        tmp = f"{args.ready_file}.tmp"
        with open(tmp, "w") as handle:
            json.dump({"pid": os.getpid(), "urls": bound,
                       "models": {p.role: p.models for p in ports}}, handle)
        os.replace(tmp, args.ready_file)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.is_set():
        stop.wait(1.0)
    for server in servers:
        server.shutdown()
    gateway.shutdown()
    logger.info("stopped; %s", json.dumps(gateway.status()))
    return EXIT_SUCCESS


def main() -> int:
    args = create_parser().parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
