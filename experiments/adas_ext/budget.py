"""Equal-successes budget control.

Stock ADAS runs ``for n in range(start, args.n_generation)`` and, when a
generation fails, executes ``n -= 1; continue``. Rebinding the loop variable of a
``for`` over ``range`` has no effect, so the generation is silently skipped
rather than retried. A weaker meta agent therefore ends up with fewer archive
entries from the same nominal budget, which makes search budget correlate with
model capability -- the single most dangerous confound in the study.

This controller replaces that with a ``while`` loop that runs until
``target_generations`` successful entries have landed, capped at ``max_attempts``
total tries. The attempt count is itself an outcome variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationOutcome:
    generation: int
    attempt: int
    success: bool
    terminal_failure: Optional[str] = None
    debug_attempts: int = 0
    failure_trace: List[dict] = field(default_factory=list)


class BudgetController:
    def __init__(self, target_generations: int, max_attempts: Optional[int] = None,
                 completed: int = 0) -> None:
        if target_generations < 0:
            raise ValueError("target_generations must be non-negative")
        self.target_generations = target_generations
        # Default headroom: three tries per wanted generation, which is enough
        # for a weak meta agent to be measured rather than truncated.
        self.max_attempts = max_attempts if max_attempts and max_attempts > 0 else target_generations * 3
        self.successes = completed
        self.attempts = 0
        self.outcomes: List[GenerationOutcome] = []
        self._current_attempt = 0

    # -- loop control ---------------------------------------------------------

    def should_continue(self) -> bool:
        return self.successes < self.target_generations and self.attempts < self.max_attempts

    def begin_attempt(self) -> tuple:
        """Return ``(generation_index, attempt_index)`` for the attempt starting now."""
        self.attempts += 1
        self._current_attempt = self.attempts
        return self.successes, self.attempts

    def void_attempt(self) -> None:
        """Uncount the current attempt: the job's walltime ended it, not the designer."""
        self.attempts -= 1
        self._current_attempt = self.attempts

    # -- outcomes -------------------------------------------------------------

    def record_success(self, debug_attempts: int = 0,
                       failure_trace: Optional[List[dict]] = None) -> GenerationOutcome:
        outcome = GenerationOutcome(
            generation=self.successes + 1,
            attempt=self._current_attempt,
            success=True,
            debug_attempts=debug_attempts,
            failure_trace=list(failure_trace or []),
        )
        self.successes += 1
        self.outcomes.append(outcome)
        return outcome

    def record_failure(self, terminal_failure: str, debug_attempts: int = 0,
                       failure_trace: Optional[List[dict]] = None) -> GenerationOutcome:
        outcome = GenerationOutcome(
            generation=self.successes + 1,
            attempt=self._current_attempt,
            success=False,
            terminal_failure=terminal_failure,
            debug_attempts=debug_attempts,
            failure_trace=list(failure_trace or []),
        )
        self.outcomes.append(outcome)
        return outcome

    # -- reporting ------------------------------------------------------------

    @property
    def exhausted(self) -> bool:
        return self.successes < self.target_generations and self.attempts >= self.max_attempts

    def failure_histogram(self) -> Dict[str, int]:
        histogram: Dict[str, int] = {}
        for outcome in self.outcomes:
            if outcome.success:
                continue
            key = outcome.terminal_failure or "unknown"
            histogram[key] = histogram.get(key, 0) + 1
        return histogram

    def is_degenerate(self, threshold: float = 0.5) -> bool:
        """More than half the attempts failed terminally.

        Degenerate cells are kept and reported -- they are the raw material for
        the failure taxonomy, not something to discard.
        """
        if not self.attempts:
            return False
        failed = sum(1 for outcome in self.outcomes if not outcome.success)
        return failed / self.attempts > threshold

    def summary(self) -> Dict[str, Any]:
        return {
            "target_generations": self.target_generations,
            "max_attempts": self.max_attempts,
            "successes": self.successes,
            "attempts": self.attempts,
            "exhausted": self.exhausted,
            "degenerate": self.is_degenerate(),
            "failure_histogram": self.failure_histogram(),
            "attempts_per_success": (self.attempts / self.successes) if self.successes else None,
        }
