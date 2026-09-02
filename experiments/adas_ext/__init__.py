"""Runtime extensions bolted onto the stock ADAS search loop.

Nothing in here changes the search algorithm. The modules add the plumbing the
scaling study needs: per-role model routing, instrumentation, an equal-successes
budget controller, and a sandbox for the generated agent code.
"""

from .budget import BudgetController, GenerationOutcome
from .instrumentation import (
    FAILURE_CATEGORIES,
    Instrumentation,
    classify_failure,
    get_instrumentation,
    set_instrumentation,
)
from .model_router import RoleClient, RoleClients, build_role_clients
from .recombination import random_recombination
from .sandbox import SandboxError, SandboxPolicy, SandboxTimeout, run_sandboxed

__all__ = [
    "BudgetController",
    "GenerationOutcome",
    "FAILURE_CATEGORIES",
    "Instrumentation",
    "classify_failure",
    "get_instrumentation",
    "set_instrumentation",
    "RoleClient",
    "RoleClients",
    "build_role_clients",
    "random_recombination",
    "SandboxError",
    "SandboxPolicy",
    "SandboxTimeout",
    "run_sandboxed",
]
