"""Per-role OpenAI-compatible client factory.

Stock ADAS uses a single module-level ``openai.OpenAI()``. The study needs two
independent endpoints -- the meta agent that writes code and the worker that
executes it -- so this module hands out one :class:`RoleClient` per role and
routes every call through it.

The role client also owns the things that must not be hardcoded per model:
the usable completion budget (see the fairness control note below), the
reasoning-effort setting, and the structured-output backend.

Fairness control: ``max_tokens`` is *not* held constant across models. gpt-oss
emits reasoning traces that draw on the same budget as the code, so holding the
limit fixed hands it materially less usable output than a non-reasoning model of
similar size. Limits are calibrated per model in ``config/models.yaml`` and both
token counts are logged either way.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import openai

from .instrumentation import get_instrumentation

DUMMY_API_KEY = "local-vllm"

# Response schema for every meta-agent turn. vLLM honours this through guided
# decoding when the backend supports it; the json_object response_format is kept
# as the fallback for builds that do not.
AGENT_DESIGN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "name": {"type": "string"},
        "code": {"type": "string"},
    },
    "required": ["thought", "name", "code"],
}


class ContextOverflowError(RuntimeError):
    """Raised when the server rejects a request because the prompt is too long.

    vLLM returns HTTP 400 for this, not a rate-limit error, so it has to be
    classified separately from ordinary code errors.
    """


_OVERFLOW_MARKERS = (
    "maximum context length",
    "longer than the maximum model length",
    "max_model_len",
    "context length exceeded",
    "reduce the length of the messages",
    "please reduce the length",
)


def _looks_like_overflow(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS)


@dataclass
class RoleClient:
    """One vLLM endpoint bound to one experimental role."""

    role: str
    model: str
    base_url: str
    api_key: str = DUMMY_API_KEY
    temperature: float = 0.5
    max_tokens: int = 4096
    reasoning_effort: Optional[str] = None
    use_guided_json: bool = True
    timeout: float = 900.0
    max_retries: int = 4
    retry_backoff: float = 2.0
    client: openai.OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,  # retries are handled here so failures stay classifiable
        )

    # -- request construction -------------------------------------------------

    def _request_kwargs(self, messages: List[dict], temperature: float, schema) -> dict:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stop": None,
            "response_format": {"type": "json_object"},
        }
        extra_body: Dict[str, Any] = {}
        if self.use_guided_json and schema is not None:
            extra_body["guided_json"] = schema
        if self.reasoning_effort is not None:
            # gpt-oss exposes this both as a top-level field and via chat kwargs
            # depending on the vLLM version; sending it in extra_body works for
            # both and is ignored by models that do not support it.
            extra_body["reasoning_effort"] = self.reasoning_effort
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    # -- instrumentation ------------------------------------------------------

    def _record_usage(self, response, latency: float) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = 0
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        finish_reason = None
        if response.choices:
            finish_reason = response.choices[0].finish_reason
        get_instrumentation().add_usage(
            role=self.role,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            reasoning_tokens=reasoning_tokens,
            latency_s=latency,
            truncated=finish_reason == "length",
        )

    # -- public API -----------------------------------------------------------

    def chat_json(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        schema: Optional[dict] = None,
    ) -> dict:
        """Send a chat request and return the parsed JSON body.

        Raises :class:`ContextOverflowError` on prompt overflow, and lets
        ``json.JSONDecodeError`` propagate so the caller can classify a parse
        failure separately from a transport failure.
        """
        temperature = self.temperature if temperature is None else temperature
        kwargs = self._request_kwargs(messages, temperature, schema)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                response = self.client.chat.completions.create(**kwargs)
            except openai.BadRequestError as exc:
                message = str(exc)
                if _looks_like_overflow(message):
                    raise ContextOverflowError(message) from exc
                if "guided_json" in message or "guided decoding" in message.lower():
                    # Backend does not support schema-constrained decoding; fall
                    # back to plain JSON mode once and keep going.
                    self.use_guided_json = False
                    kwargs = self._request_kwargs(messages, temperature, schema)
                    last_error = exc
                    continue
                raise
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff ** attempt)
                continue
            except openai.RateLimitError as exc:  # not expected against vLLM, kept for hosted APIs
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff ** attempt)
                continue

            self._record_usage(response, time.time() - started)
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Model returned an empty message body")
            parsed = json.loads(content)
            if parsed is None:
                raise ValueError("Model returned JSON null")
            return parsed

        raise last_error if last_error is not None else RuntimeError("chat_json exhausted retries")


@dataclass
class RoleClients:
    meta: Optional[RoleClient]
    worker: RoleClient


def _resolve(value, env_name: str, default):
    if value:
        return value
    return os.environ.get(env_name, default)


def build_role_clients(args) -> RoleClients:
    """Build the meta and worker clients from parsed CLI arguments.

    ``args.model`` names the meta agent and ``args.worker_model`` the worker.
    Reference conditions that run no search (bare model, solo CoT) pass
    ``--model none`` and get ``meta=None``.
    """
    worker_base_url = _resolve(
        getattr(args, "worker_base_url", None),
        "ADAS_WORKER_BASE_URL",
        "http://127.0.0.1:8000/v1",
    )
    meta_base_url = _resolve(
        getattr(args, "meta_base_url", None),
        "ADAS_META_BASE_URL",
        "http://127.0.0.1:8001/v1",
    )
    api_key = os.environ.get("OPENAI_API_KEY", DUMMY_API_KEY)

    worker = RoleClient(
        role="worker",
        model=getattr(args, "worker_model", None) or os.environ.get("ADAS_WORKER_MODEL", "gpt-3.5-turbo-0125"),
        base_url=worker_base_url,
        api_key=api_key,
        temperature=getattr(args, "worker_temperature", 0.5),
        max_tokens=getattr(args, "worker_max_tokens", 4096),
        reasoning_effort=getattr(args, "worker_reasoning_effort", None),
        timeout=getattr(args, "request_timeout", 900.0),
    )

    meta_model = getattr(args, "model", None)
    if not meta_model or str(meta_model).lower() == "none":
        return RoleClients(meta=None, worker=worker)

    meta = RoleClient(
        role="meta",
        model=meta_model,
        base_url=meta_base_url,
        api_key=api_key,
        temperature=getattr(args, "meta_temperature", 0.8),
        max_tokens=getattr(args, "meta_max_tokens", 8192),
        reasoning_effort=getattr(args, "meta_reasoning_effort", None),
        timeout=getattr(args, "request_timeout", 900.0),
    )
    return RoleClients(meta=meta, worker=worker)


def worker_client_config(clients: RoleClients) -> dict:
    """Picklable description of the worker client, for the sandbox subprocess."""
    w = clients.worker
    return {
        "role": w.role,
        "model": w.model,
        "base_url": w.base_url,
        "api_key": w.api_key,
        "temperature": w.temperature,
        "max_tokens": w.max_tokens,
        "reasoning_effort": w.reasoning_effort,
        "use_guided_json": w.use_guided_json,
        "timeout": w.timeout,
    }


def role_client_from_config(config: dict) -> RoleClient:
    return RoleClient(**config)
