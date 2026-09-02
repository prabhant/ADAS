"""Thread-safe token, call and failure logging for the ADAS scaling study.

Everything the analysis layers need has to exist in the archive before any data
is generated, because none of it can be reconstructed afterwards:

* token and call counts per generation and role (Pareto front, resource table)
* the per-question correctness vector (Layer 4 behavioural distance)
* one record per ``LLMAgentBase.query`` including the ``author`` of every input
  ``Info`` (Layer 2 runtime dataflow graph)
* debug attempt counts and a terminal failure category (Q6 failure taxonomy)
* abandoned generations, which stock ADAS discards -- keeping them is what stops
  every downstream metric from being conditioned on success
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

# The seven categories from the operational definitions. Q6 is answered from the
# distribution across these, so the set is fixed and must not drift.
FAILURE_CATEGORIES = (
    "json_parse_error",
    "python_syntax_error",
    "runtime_exception",
    "context_overflow",
    "wallclock_timeout",
    "all_zero_accuracy",
    "semantically_wrong_but_running",
)


def classify_failure(exc: BaseException) -> str:
    """Map an exception onto the failure taxonomy.

    A model whose failures are dominated by ``json_parse_error`` is telling you
    about format compliance, not design ability, which is why parse errors are
    never folded into ``runtime_exception``.
    """
    # A failure raised inside the sandbox was already classified in the child.
    preset = getattr(exc, "failure_category", None)
    if preset in FAILURE_CATEGORIES:
        return preset

    name = getattr(exc, "original_type", None) or type(exc).__name__
    message = str(exc)
    lowered = message.lower()

    if name in ("JSONDecodeError", "ContextOverflowError", "SandboxTimeout"):
        if name == "JSONDecodeError":
            return "json_parse_error"
        if name == "ContextOverflowError":
            return "context_overflow"
        return "wallclock_timeout"
    if isinstance(exc, SyntaxError):
        return "python_syntax_error"
    if "all 0 accuracy" in lowered:
        return "all_zero_accuracy"
    if "timed out" in lowered or "timeout" in lowered:
        return "wallclock_timeout"
    if "maximum context length" in lowered or "context is too long" in lowered:
        return "context_overflow"
    if "expecting value" in lowered or "json" in lowered and "decode" in lowered:
        return "json_parse_error"
    return "runtime_exception"


class Instrumentation:
    """Accumulates everything a run needs to persist.

    Safe to call from the evaluation thread pool. The evaluation subprocess owns
    its own instance and hands a snapshot back to the parent, which merges it.
    """

    def __init__(self, run_id: str = "unset", output_dir: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self.run_id = run_id
        self.output_dir = output_dir
        self.generation: Any = "initial"
        self.attempt: int = 0
        self.phase: str = "search"

        self.usage: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "n_calls": 0,
                "n_truncated": 0,
                "latency_s": 0.0,
            }
        )
        self.call_log: List[dict] = []
        self.failures: List[dict] = []
        self.abandoned: List[dict] = []

    # -- context --------------------------------------------------------------

    def set_context(self, generation: Any = None, attempt: Optional[int] = None,
                    phase: Optional[str] = None) -> None:
        with self._lock:
            if generation is not None:
                self.generation = generation
            if attempt is not None:
                self.attempt = attempt
            if phase is not None:
                self.phase = phase

    def _key(self, role: str) -> str:
        return f"{self.phase}|{self.generation}|{role}"

    # -- accumulation ---------------------------------------------------------

    def add_usage(self, role: str, model: str, prompt_tokens: int, completion_tokens: int,
                  reasoning_tokens: int = 0, latency_s: float = 0.0,
                  truncated: bool = False) -> None:
        with self._lock:
            bucket = self.usage[self._key(role)]
            bucket["prompt_tokens"] += prompt_tokens
            bucket["completion_tokens"] += completion_tokens
            bucket["reasoning_tokens"] += reasoning_tokens
            bucket["n_calls"] += 1
            bucket["n_truncated"] += int(truncated)
            bucket["latency_s"] += latency_s
            bucket["model"] = model

    def log_call(self, agent_name: str, agent_id: str, role: str, output_fields: List[str],
                 iteration_idx: int, input_authors: List[str], input_names: List[str],
                 question_idx: Optional[int] = None, temperature: Optional[float] = None,
                 error: Optional[str] = None) -> str:
        """Record one worker invocation; returns the call id.

        ``input_authors`` is what makes the runtime dataflow graph reconstructible:
        the ``Info`` namedtuple already carries provenance, this just persists it.
        """
        call_id = uuid.uuid4().hex[:12]
        record = {
            "call_id": call_id,
            "generation": self.generation,
            "attempt": self.attempt,
            "phase": self.phase,
            "question_idx": question_idx,
            "agent_name": agent_name,
            "agent_id": agent_id,
            "role": role,
            "output_fields": list(output_fields),
            "iteration_idx": iteration_idx,
            "input_authors": list(input_authors),
            "input_names": list(input_names),
            "temperature": temperature,
            "error": error,
            "t": time.time(),
        }
        with self._lock:
            self.call_log.append(record)
        return call_id

    def record_failure(self, category: str, message: str, stage: str,
                       generation: Any = None, attempt: Optional[int] = None) -> None:
        with self._lock:
            self.failures.append({
                "generation": self.generation if generation is None else generation,
                "attempt": self.attempt if attempt is None else attempt,
                "stage": stage,
                "category": category,
                "message": message[:4000],
                "t": time.time(),
            })

    def record_abandoned(self, generation: Any, attempt: int, terminal_failure: str,
                         debug_attempts: int, solution: Optional[dict],
                         failure_trace: List[dict]) -> None:
        """Persist a generation that never produced an archive entry."""
        with self._lock:
            self.abandoned.append({
                "generation": generation,
                "attempt": attempt,
                "terminal_failure": terminal_failure,
                "debug_attempts": debug_attempts,
                "thought": (solution or {}).get("thought"),
                "name": (solution or {}).get("name"),
                "code": (solution or {}).get("code"),
                "failure_trace": failure_trace,
                "t": time.time(),
            })

    # -- transfer between processes ------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "usage": {k: dict(v) for k, v in self.usage.items()},
                "call_log": list(self.call_log),
                "failures": list(self.failures),
                "abandoned": list(self.abandoned),
            }

    def merge(self, snapshot: dict) -> None:
        """Fold a snapshot produced by the sandbox subprocess back in."""
        with self._lock:
            for key, bucket in snapshot.get("usage", {}).items():
                target = self.usage[key]
                for field, value in bucket.items():
                    if field == "model":
                        target["model"] = value
                    else:
                        target[field] = target.get(field, 0) + value
            self.call_log.extend(snapshot.get("call_log", []))
            self.failures.extend(snapshot.get("failures", []))
            self.abandoned.extend(snapshot.get("abandoned", []))

    # -- persistence ----------------------------------------------------------

    def usage_for(self, generation: Any, phase: Optional[str] = None) -> dict:
        """Token and call totals for one generation, split by role."""
        phase = self.phase if phase is None else phase
        out: Dict[str, dict] = {}
        with self._lock:
            for key, bucket in self.usage.items():
                bucket_phase, bucket_gen, role = key.split("|", 2)
                if bucket_phase == phase and bucket_gen == str(generation):
                    out[role] = dict(bucket)
        return out

    def write(self, output_dir: Optional[str] = None) -> None:
        directory = output_dir or self.output_dir
        if not directory:
            return
        os.makedirs(directory, exist_ok=True)
        payloads = {
            "token_usage.json": {k: dict(v) for k, v in self.usage.items()},
            "call_log.jsonl": self.call_log,
            "failures.jsonl": self.failures,
            "abandoned_generations.json": self.abandoned,
        }
        with self._lock:
            for filename, payload in payloads.items():
                path = os.path.join(directory, f"{self.run_id}_{filename}")
                if filename.endswith(".jsonl"):
                    with open(path, "w") as handle:
                        for record in payload:
                            handle.write(json.dumps(record) + "\n")
                else:
                    with open(path, "w") as handle:
                        json.dump(payload, handle, indent=2)


_INSTRUMENTATION = Instrumentation()


def get_instrumentation() -> Instrumentation:
    return _INSTRUMENTATION


def set_instrumentation(instrumentation: Instrumentation) -> Instrumentation:
    global _INSTRUMENTATION
    _INSTRUMENTATION = instrumentation
    return _INSTRUMENTATION
