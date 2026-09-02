"""Restricted execution of model-generated agent code.

``evaluate_forward_fn`` calls ``exec()`` on code written by an LLM with no
isolation at all. This study runs far more untrusted code, from models less
aligned than GPT-4o, so execution is routed through a subprocess that is:

* network-restricted -- only the configured vLLM endpoints are reachable, so a
  generated agent cannot call out to the internet
* filesystem read-only -- writes are confined to an explicit allowlist
* process-restricted -- no ``subprocess``, ``fork`` or ``exec``
* wall-clock bounded -- the parent kills the child on timeout and reports it as
  its own failure category

The child returns a plain data payload, so nothing the untrusted code produced is
unpickled as a live object in the parent.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import queue as queue_mod
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class SandboxError(RuntimeError):
    """A failure that happened inside the sandboxed child."""

    def __init__(self, message: str, failure_category: str = "runtime_exception",
                 original_type: str = "Exception", child_traceback: str = "") -> None:
        super().__init__(message)
        self.failure_category = failure_category
        self.original_type = original_type
        self.child_traceback = child_traceback


class SandboxTimeout(SandboxError):
    def __init__(self, message: str) -> None:
        super().__init__(message, failure_category="wallclock_timeout",
                         original_type="SandboxTimeout")


class SandboxPolicy:
    """What the sandboxed child is allowed to do."""

    def __init__(self, allowed_endpoints: Sequence[Tuple[str, int]] = (),
                 writable_dirs: Sequence[str] = (),
                 timeout_s: float = 3600.0,
                 memory_limit_bytes: Optional[int] = None,
                 enabled: bool = True) -> None:
        self.allowed_endpoints = [(str(h), int(p)) for h, p in allowed_endpoints]
        self.writable_dirs = [os.path.realpath(d) for d in writable_dirs]
        self.timeout_s = float(timeout_s)
        self.memory_limit_bytes = memory_limit_bytes
        self.enabled = enabled

    @classmethod
    def from_args(cls, args, base_urls: Sequence[str]) -> "SandboxPolicy":
        from urllib.parse import urlparse

        endpoints: List[Tuple[str, int]] = []
        for url in base_urls:
            if not url:
                continue
            parsed = urlparse(url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            endpoints.append((host, port))
            # loopback aliases resolve to different literals
            if host in ("localhost", "127.0.0.1", "::1"):
                endpoints.extend([("127.0.0.1", port), ("localhost", port), ("::1", port)])

        writable = [
            os.path.realpath(getattr(args, "save_dir", "results/")),
            os.path.realpath(os.environ.get("TMPDIR", "/tmp")),
            "/dev/shm",
        ]
        return cls(
            allowed_endpoints=endpoints,
            writable_dirs=writable,
            timeout_s=float(getattr(args, "sandbox_timeout", 3600.0)),
            memory_limit_bytes=getattr(args, "sandbox_memory_bytes", None),
            enabled=not getattr(args, "no_sandbox", False),
        )

    def as_dict(self) -> dict:
        return {
            "allowed_endpoints": self.allowed_endpoints,
            "writable_dirs": self.writable_dirs,
            "timeout_s": self.timeout_s,
            "memory_limit_bytes": self.memory_limit_bytes,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SandboxPolicy":
        return cls(**payload)


# ---------------------------------------------------------------------------
# Guards installed inside the child
# ---------------------------------------------------------------------------

def _install_guards(policy: SandboxPolicy) -> None:
    import builtins
    import socket

    allowed = set(policy.allowed_endpoints)
    allowed_hosts = {host for host, _ in policy.allowed_endpoints}
    writable = tuple(policy.writable_dirs)

    def _endpoint_allowed(address) -> bool:
        if not isinstance(address, tuple) or len(address) < 2:
            return False
        host, port = address[0], address[1]
        return (str(host), int(port)) in allowed

    # -- network ------------------------------------------------------------
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address):
        if not _endpoint_allowed(address):
            raise PermissionError(f"sandbox: outbound connection to {address!r} is blocked")
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if not _endpoint_allowed(address):
            raise PermissionError(f"sandbox: outbound connection to {address!r} is blocked")
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        if str(host) not in allowed_hosts:
            raise PermissionError(f"sandbox: DNS resolution of {host!r} is blocked")
        return real_getaddrinfo(host, port, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_getaddrinfo

    # -- filesystem ---------------------------------------------------------
    def _write_allowed(path) -> bool:
        try:
            resolved = os.path.realpath(os.fspath(path))
        except TypeError:
            return False  # file descriptors and other exotica
        return any(resolved == root or resolved.startswith(root + os.sep) for root in writable)

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")) and not _write_allowed(file):
            raise PermissionError(f"sandbox: write access to {file!r} is blocked")
        return real_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open

    def _guard_path_call(func, arity: int = 1):
        def guarded(*paths, **kwargs):
            for path in paths[:arity]:
                if not _write_allowed(path):
                    raise PermissionError(f"sandbox: mutating {path!r} is blocked")
            return func(*paths, **kwargs)
        return guarded

    os.remove = _guard_path_call(os.remove)
    os.unlink = _guard_path_call(os.unlink)
    os.rmdir = _guard_path_call(os.rmdir)
    os.mkdir = _guard_path_call(os.mkdir)
    os.makedirs = _guard_path_call(os.makedirs)
    os.rename = _guard_path_call(os.rename, arity=2)
    os.replace = _guard_path_call(os.replace, arity=2)

    # -- process creation ---------------------------------------------------
    def _blocked(*_args, **_kwargs):
        raise PermissionError("sandbox: spawning processes is blocked")

    os.system = _blocked
    os.fork = _blocked
    os.forkpty = _blocked
    for name in ("execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "spawnv", "spawnve"):
        if hasattr(os, name):
            setattr(os, name, _blocked)
    try:
        import subprocess

        subprocess.Popen = _blocked
        subprocess.run = _blocked
        subprocess.call = _blocked
        subprocess.check_call = _blocked
        subprocess.check_output = _blocked
    except ImportError:
        pass

    # -- resource limits ----------------------------------------------------
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if policy.memory_limit_bytes:
            resource.setrlimit(resource.RLIMIT_AS,
                               (policy.memory_limit_bytes, policy.memory_limit_bytes))
    except (ImportError, ValueError, OSError):
        pass


def _child_main(result_queue, entry_module: str, entry_func: str, kwargs: dict,
                policy_payload: dict, sys_path: List[str], env: Dict[str, str]) -> None:
    try:
        for path in reversed(sys_path):
            if path not in sys.path:
                sys.path.insert(0, path)
        os.environ.update(env)

        policy = SandboxPolicy.from_dict(policy_payload)
        module = importlib.import_module(entry_module)
        func = getattr(module, entry_func)

        # Guards go on last so module import is not itself restricted.
        if policy.enabled:
            _install_guards(policy)

        result = func(**kwargs)
        result_queue.put(("ok", result))
    except BaseException as exc:  # noqa: BLE001 - everything must come back as data
        from .instrumentation import classify_failure

        result_queue.put((
            "error",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "category": classify_failure(exc),
                "traceback": traceback.format_exc(limit=20),
            },
        ))


def run_sandboxed(entry_module: str, entry_func: str, kwargs: dict,
                  policy: SandboxPolicy, env: Optional[Dict[str, str]] = None) -> Any:
    """Run ``entry_module.entry_func(**kwargs)`` in a restricted child process.

    The child is killed on timeout. Failures come back as :class:`SandboxError`
    carrying an already-classified failure category.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_child_main,
        args=(result_queue, entry_module, entry_func, kwargs, policy.as_dict(),
              list(sys.path), dict(env or {})),
        daemon=False,
    )
    process.start()
    try:
        # Read before join: a child that put a large payload blocks on exit
        # until the pipe is drained.
        try:
            status, payload = result_queue.get(timeout=policy.timeout_s)
        except queue_mod.Empty:
            raise SandboxTimeout(
                f"Sandboxed evaluation exceeded {policy.timeout_s:.0f}s wall clock"
            ) from None
        process.join(timeout=60)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        result_queue.close()

    if status == "ok":
        return payload
    raise SandboxError(
        payload["message"],
        failure_category=payload["category"],
        original_type=payload["type"],
        child_traceback=payload["traceback"],
    )


def run_unsandboxed(entry_module: str, entry_func: str, kwargs: dict) -> Any:
    """In-process escape hatch for debugging. Never use for a real campaign."""
    module = importlib.import_module(entry_module)
    return getattr(module, entry_func)(**kwargs)


def self_test(policy: Optional[SandboxPolicy] = None) -> Dict[str, bool]:
    """Stage 0 exit condition: confirm the sandbox blocks a deliberate escape.

    Must be invoked *through* :func:`run_sandboxed`, which installs the guards
    before calling it. Calling it in-process installs nothing and reports that
    the guards do not hold, which is the honest answer for that situation.
    """
    checks: Dict[str, bool] = {}

    def _blocked_by(func: Callable) -> bool:
        try:
            func()
        except PermissionError:
            return True
        except Exception:
            return True
        return False

    import socket as _socket

    checks["network_blocked"] = _blocked_by(
        lambda: _socket.create_connection(("example.com", 80), timeout=3)
    )
    # Probe a path the user genuinely owns. Writing to /etc/ fails on ordinary
    # filesystem permissions whether or not the guard is installed, so it would
    # report success against a completely inert sandbox.
    probe = os.path.join(os.path.expanduser("~"), ".adas_sandbox_write_probe")
    checks["write_blocked"] = _blocked_by(lambda: open(probe, "w").close())
    if not checks["write_blocked"]:
        try:
            os.unlink(probe)
        except OSError:
            pass
    checks["subprocess_blocked"] = _blocked_by(lambda: __import__("subprocess").run(["true"]))
    return checks


if __name__ == "__main__":
    # Run as `python -m experiments.adas_ext.sandbox` from the repository root.
    #
    # This has to be a real module rather than a heredoc: the multiprocessing
    # "spawn" start method re-imports __main__ in the child, and a parent read
    # from stdin has no importable __main__, so the child dies before the guards
    # are ever installed. Running as a script or with -m both work; `python - <<EOF`
    # does not.
    import tempfile as _tempfile

    _policy = SandboxPolicy(
        timeout_s=120,
        writable_dirs=(_tempfile.gettempdir(),),
        allowed_endpoints=(),
    )
    _report = run_sandboxed(__spec__.name if __spec__ else "experiments.adas_ext.sandbox",
                            "self_test", {}, _policy)
    for _name, _blocked in sorted(_report.items()):
        print(f"  [{'ok' if _blocked else 'LEAK'}] {_name}")
    if not all(_report.values()):
        print(f"sandbox guards did not hold: {_report}", file=sys.stderr)
        sys.exit(1)
    print("sandbox guards hold")
