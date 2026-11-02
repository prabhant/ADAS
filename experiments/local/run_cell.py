"""Run one Copilot-backed ADAS cell on a local machine.

Resolves a cell from ``experiments/config/copilot_cells.yaml``, starts the
Copilot gateway (``experiments/serving/copilot_gateway.py``) on two loopback
ports, runs ``_<domain>/search.py`` against it, and stops the search if the
gateway halts on its AI-credit cap or on an authentication failure.

Rerunning a cell resumes it: search.py continues from the archive and
re-evaluates only untested designs, and the gateway reads the cell's ledger back
so the credit cap holds across restarts.

    python experiments/local/run_cell.py --list
    python experiments/local/run_cell.py cp0_smoke --dry_run
    python experiments/local/run_cell.py cp0_smoke

Exit codes: 0 done, 1 the search failed, 2 configuration error, 3 halted by the
gateway (raise the cap or fix authentication, then rerun to resume).
"""

from __future__ import annotations

import argparse
import copy
import glob
import importlib.util
import itertools
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(REPO_ROOT, "experiments", "config")
MODELS_PATH = os.path.join(CONFIG_DIR, "copilot_models.yaml")
CELLS_PATH = os.path.join(CONFIG_DIR, "copilot_cells.yaml")
DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, "experiments", "results_copilot")
LEDGER_NAME = "copilot_ledger.jsonl"
HALT_NAME = "GATEWAY_HALTED.json"
DONE_NAME = ".cell_done.json"
PACE_DIR_NAME = "_pace"

EXIT_SUCCESS, EXIT_FAILURE, EXIT_ERROR, EXIT_HALTED = 0, 1, 2, 3

CONDITION_FLAGS = {
    "bare_model": ["--baseline", "bare"],
    "solo_cot": ["--baseline", "cot"],
    "seed_eval": ["--seeds_only"],
    "random_recombination": ["--search_mode", "random"],
    "no_reflection": ["--no_reflection"],
    "empty_archive": ["--init_archive", "empty"],
    "test_archive": ["--evaluate_only"],
}


class ConfigError(Exception):
    """The cell cannot be run as configured."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: str) -> Any:
    with open(path) as handle:
        return json.load(handle)


def expand_matrices(grid: dict) -> List[dict]:
    cells = [dict(c) for c in grid.get("cells") or []]
    for matrix in grid.get("matrices") or []:
        axes = {k: v for k, v in matrix.items() if isinstance(v, list)}
        fixed = {k: v for k, v in matrix.items() if not isinstance(v, list)}
        for combo in itertools.product(*axes.values()):
            cell = dict(fixed, **dict(zip(axes.keys(), combo)))
            cell["id"] = cell["id"].format(**cell)
            cells.append(cell)
    return cells


class _Unset:
    """Stands in for a selection value that has not been filled in yet."""

    def __format__(self, spec: str) -> str:
        return "None"

    def __getitem__(self, index: Any) -> "_Unset":
        return self


def render(value: Any, selection: dict) -> Any:
    """Fill {worker}, {domain}, {transfer_workers[0]} ... from the selection block."""
    if isinstance(value, str):
        if "{" not in value:
            return value
        values = {k: (_Unset() if v in (None, []) else v) for k, v in selection.items()}
        try:
            rendered = value.format(**values)
        except (KeyError, IndexError) as exc:
            raise ConfigError(f"{value!r} needs selection entry {exc}") from None
        if "None" in rendered:
            raise ConfigError(f"{value!r} needs a selection that is still unset")
        return rendered
    if isinstance(value, dict):
        return {k: render(v, selection) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, selection) for v in value]
    return value


def all_cells(grid: dict) -> List[Tuple[dict, Optional[str]]]:
    """Every cell, rendered where its selection is set; the second item says why not."""
    selection = grid.get("selection") or {}
    out = []
    for raw in expand_matrices(grid):
        try:
            out.append((render(raw, selection), None))
        except ConfigError as exc:
            out.append((raw, str(exc)))
    return out


def resolve_cell(cell_id: str, grid: dict) -> dict:
    cells = all_cells(grid)
    for cell, problem in cells:
        if cell["id"] == cell_id:
            if problem:
                raise ConfigError(f"{cell_id}: {problem}; fill `selection` in the cell grid")
            return cell
    templated = [c["id"] for c, problem in cells if problem]
    raise ConfigError(f"no cell {cell_id!r} in the cell grid"
                      + (f" (cells awaiting `selection`: {templated})" if templated else ""))


def merge(cell: dict, grid: dict) -> dict:
    stage = (grid.get("stage_defaults") or {}).get(cell.get("stage"), {})
    common = grid.get("common") or {}
    domain = cell.get("domain") or stage.get("domain") or common.get("domain") or "mgsm"
    merged = dict(common)
    merged.update((grid.get("domains") or {}).get(domain, {}))
    merged.update(stage)
    merged.update({k: v for k, v in cell.items() if v is not None})
    merged["domain"] = domain
    merged.setdefault("meta", None)
    return merged


def check_models(cfg: dict, registry: dict) -> None:
    models = registry.get("models") or {}
    for role in ("meta", "worker"):
        key = cfg.get(role)
        if key is None:
            continue
        if key not in models:
            raise ConfigError(f"{role} model {key!r} is not in {MODELS_PATH}")
        if role not in (models[key].get("roles") or []):
            raise ConfigError(f"{key!r} is not registered for the {role} role")


# ---------------------------------------------------------------------------
# Search arguments
# ---------------------------------------------------------------------------

def find_cot_baseline(cfg: dict, results_root: str) -> Optional[float]:
    """The validation CoT accuracy the informed designer is told, from this campaign only."""
    worker, domain = cfg["worker"], cfg["domain"]
    effort = cfg.get("worker_reasoning_effort")
    cell_id = (f"cp3_effort_baseline_{domain}_{worker}_{effort}" if effort
               else f"cp1_baseline_cot_{domain}_{worker}")
    path = os.path.join(results_root, cell_id, "cot_baseline.json")
    if not os.path.exists(path):
        return None
    payload = load_json(path)
    if payload.get("domain") != domain or payload.get("worker_model") != worker:
        raise ConfigError(f"{path} is for {payload.get('worker_model')} on "
                          f"{payload.get('domain')}; refusing it for {worker} on {domain}")
    return float(payload["accuracy"])


def search_args(cfg: dict, registry: dict, urls: Dict[str, str], results_root: str,
                evaluate_only: bool = False) -> List[str]:
    models = registry.get("models") or {}
    worker_spec = models[cfg["worker"]]
    save_dir = os.path.join(results_root, cfg["id"])
    n_generation = int(cfg.get("n_generation", 0) or 0)
    args = [
        "--domain", cfg["domain"],
        "--save_dir", save_dir,
        "--expr_name", cfg["id"],
        "--run_id", cfg["id"],
        "--worker_model", cfg["worker"],
        "--worker_base_url", urls["worker"],
        "--meta_base_url", urls.get("meta", urls["worker"]),
        "--worker_max_tokens", str(worker_spec.get("max_output_tokens", 4096)),
        "--meta_temperature", str(cfg.get("meta_temperature", 0.8)),
        "--worker_temperature", str(cfg.get("worker_temperature", 0.5)),
        "--n_generation", str(n_generation),
        "--target_generations", str(n_generation),
        "--max_attempts", str(cfg.get("max_attempts", 3 * max(n_generation, 1))),
        "--debug_max", str(cfg.get("debug_max", 3)),
        "--valid_size", str(cfg["valid_size"]),
        "--test_size", str(cfg["test_size"]),
        "--n_repreat", str(cfg.get("n_repreat", 1)),
        "--max_workers", str(cfg.get("max_workers", 32)),
        "--shuffle_seed", str(cfg.get("shuffle_seed", 0)),
        "--cost_lambda", str(cfg.get("cost_lambda", 0.0)),
        # Client side: long enough that search.py never resends while the gateway is
        # still retrying through an outage (which would bill the call twice).
        "--request_timeout", str(cfg.get("client_request_timeout", cfg.get("request_timeout", 900))),
        "--test_top_k", str(cfg.get("test_top_k", 0)),
        "--seed_cache_dir", os.path.join(results_root, "_seed_cache"),
    ]
    if cfg.get("meta"):
        meta_spec = models[cfg["meta"]]
        args += ["--model", cfg["meta"], "--meta_max_tokens",
                 str(meta_spec.get("meta_max_output_tokens", meta_spec.get("max_output_tokens", 8192)))]
    args.append("--informed_meta" if cfg.get("informed_meta", True) else "--blind_meta")
    if cfg.get("worker_reasoning_effort"):
        args += ["--worker_reasoning_effort", cfg["worker_reasoning_effort"]]
    if cfg.get("meta_reasoning_effort"):
        args += ["--meta_reasoning_effort", cfg["meta_reasoning_effort"]]
    if cfg.get("meta") and cfg.get("informed_meta", True):
        baseline = find_cot_baseline(cfg, results_root)
        if baseline is not None:
            args += ["--worker_cot_accuracy", str(baseline)]
        elif not cfg.get("allow_missing_baseline"):
            effort = cfg.get("worker_reasoning_effort")
            raise ConfigError(
                f"no CoT baseline for {cfg['worker']} on {cfg['domain']}"
                f"{' at effort ' + effort if effort else ''}; run its baseline cell first, "
                f"or pass --allow_missing_baseline")
    if cfg.get("require_seed_cache"):
        args.append("--require_seed_cache")
    if cfg.get("sandbox_timeout"):
        # Wall-clock cap per evaluation. Cells sharing the account pace each get a slice of
        # it, so an evaluation runs several times longer than it would alone.
        args += ["--sandbox_timeout", str(cfg["sandbox_timeout"])]
    proposals = int(cfg.get("proposals_per_round", 1) or 1)
    if proposals > 1:
        if cfg["domain"] != "drop":
            raise ConfigError(f"{cfg['id']}: proposals_per_round is implemented for DROP only")
        args += ["--proposals_per_round", str(proposals)]
    condition = cfg.get("condition") or ""
    if condition and condition not in CONDITION_FLAGS:
        raise ConfigError(f"unknown condition {condition!r}")
    args += CONDITION_FLAGS.get(condition, [])
    if evaluate_only and "--evaluate_only" not in args:
        args.append("--evaluate_only")
    return args


# ---------------------------------------------------------------------------
# Test-only archives (plain frontier baselines, transfer)
# ---------------------------------------------------------------------------

def seed_archive(domain: str) -> List[dict]:
    path = os.path.join(REPO_ROOT, f"_{domain}", f"{domain}_prompt.py")
    spec = importlib.util.spec_from_file_location(f"_copilot_seeds_{domain}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(os.path.dirname(path))
    return module.get_init_archive()


def best_on_validation(archive: List[dict]) -> dict:
    generated = [e for e in archive if isinstance(e.get("generation"), int)]
    if not generated:
        raise ConfigError("the source cell has no generated design yet")
    # Validation only, earliest generation on ties: selection never sees test scores.
    return max(generated, key=lambda e: (e.get("raw_median_accuracy", 0.0), -e["generation"]))


def prepare_test_archive(cfg: dict, results_root: str) -> Optional[str]:
    """Write the archive a test_archive cell evaluates; an existing one is kept."""
    if cfg.get("condition") != "test_archive":
        return None
    save_dir = os.path.join(results_root, cfg["id"])
    path = os.path.join(save_dir, f"{cfg['id']}_run_archive.json")
    if os.path.exists(path):
        return path
    spec = cfg.get("test_agents") or {}
    entries = []
    source = spec.get("best_of")
    if source:
        source_path = os.path.join(results_root, source, f"{source}_run_archive.json")
        if not os.path.exists(source_path):
            raise ConfigError(f"{cfg['id']} needs {source_path}; run {source} first")
        best = copy.deepcopy(best_on_validation(load_json(source_path)))
        entries.append({"thought": best.get("thought"), "name": best.get("name"), "code": best["code"],
                        "generation": best["generation"], "source_cell": source,
                        "source_design_id": best.get("design_id"),
                        "source_raw_median_accuracy": best.get("raw_median_accuracy")})
    wanted = spec.get("seeds") or []
    seeds = {s["name"]: s for s in seed_archive(cfg["domain"])}
    missing = [name for name in wanted if name not in seeds]
    if missing:
        raise ConfigError(f"unknown seed agents {missing}; known: {sorted(seeds)}")
    for name in wanted:
        entries.append({"thought": seeds[name]["thought"], "name": name, "code": seeds[name]["code"],
                        "generation": "initial"})
    # Controls written for the campaign (for example CoT with one instruction changed).
    for agent in spec.get("agents") or []:
        entries.append({"thought": agent.get("thought", ""), "name": agent["name"], "code": agent["code"],
                        "generation": "initial", "control": True})
    if not entries:
        raise ConfigError(f"{cfg['id']}: test_agents names no agent")
    os.makedirs(save_dir, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(entries, handle, indent=2)
    return path


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

def gateway_command(cfg: dict, results_root: str, global_max_credits: Optional[float],
                    ready_file: str, python: str, honor_max_tokens: bool = False,
                    backend: Optional[str] = None, client_mode: str = "copilot-cli",
                    copilot_home: Optional[str] = None) -> List[str]:
    run_dir = os.path.join(results_root, cfg["id"])
    command = [
        python, "-m", "experiments.serving.copilot_gateway",
        "--worker_port", "0", "--worker_models", cfg["worker"],
        "--cell", cfg["id"],
        "--ledger", os.path.join(run_dir, LEDGER_NAME),
        "--ledger_glob", os.path.join(results_root, "*", LEDGER_NAME),
        "--pricing", MODELS_PATH,
        "--request_timeout", str(cfg.get("request_timeout", 900)),
        "--worker_request_timeout", str(cfg.get("worker_request_timeout", cfg.get("request_timeout", 900))),
        "--max_concurrency", str(cfg.get("max_concurrency", cfg.get("max_workers", 32))),
        "--max_calls_per_minute", str(cfg.get("max_calls_per_minute", 0)),
        "--max_retry_wait", str(cfg.get("max_retry_wait", 0)),
        "--ready_file", ready_file,
        "--halt_file", os.path.join(run_dir, HALT_NAME),
    ]
    if cfg.get("meta"):
        command += ["--meta_port", "0", "--meta_models", cfg["meta"]]
    if cfg.get("max_credits") is not None:
        command += ["--max_credits", str(cfg["max_credits"])]
    if global_max_credits is not None:
        command += ["--global_max_credits", str(global_max_credits)]
    rates = cfg.get("pace_rates") or {}
    if rates:
        # One slot file per pace group under the results root, shared by every cell.
        command += ["--pace_dir", os.path.join(results_root, PACE_DIR_NAME),
                    "--pace_rates", ",".join(f"{group}={rate}" for group, rate in rates.items())]
    if honor_max_tokens:
        command.append("--honor_max_tokens")
    command += ["--client_mode", client_mode]
    if copilot_home:
        command += ["--copilot_home", copilot_home]
    if backend:
        command += ["--backend", backend]
    return command


def start_gateway(command: List[str], ready_file: str, log_path: str,
                  timeout: float = 180.0) -> Tuple[subprocess.Popen, Dict[str, str]]:
    if os.path.exists(ready_file):
        os.remove(ready_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ, PYTHONPATH=REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with open(log_path, "a") as log:
        proc = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(ready_file):
            return proc, load_json(ready_file)["urls"]
        if proc.poll() is not None:
            raise RuntimeError(f"gateway exited with {proc.returncode}; see {log_path}")
        time.sleep(0.2)
    stop_process(proc)
    raise RuntimeError(f"gateway not ready after {timeout:.0f}s; see {log_path}")


def stop_process(proc: Optional[subprocess.Popen], grace: float = 30.0) -> None:
    """Stop a process and everything it spawned (search.py's sandbox children)."""
    if proc is None or proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _tee(stream, log_path: str, echo: bool) -> threading.Thread:
    def pump() -> None:
        with open(log_path, "a") as log:
            for line in iter(stream.readline, ""):
                log.write(line)
                log.flush()
                if echo:
                    sys.stdout.write(line)
                    sys.stdout.flush()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    return thread


def preserve_run_logs(results_root: str, cell_id: str) -> None:
    """A resumed run rewrites the per-run logs; keep the earlier part beside them."""
    run_dir = os.path.join(results_root, cell_id, cell_id)
    if not os.path.isdir(run_dir):
        return
    n = 1
    while glob.glob(os.path.join(run_dir, f"part{n}_*")):
        n += 1
    files = glob.glob(os.path.join(run_dir, f"{cell_id}_*")) + [
        os.path.join(run_dir, "evaluation_summary.json"), os.path.join(run_dir, "budget_summary.json")]
    kept = 0
    for path in files:
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(run_dir, f"part{n}_{os.path.basename(path)}"))
            kept += 1
    if kept:
        print(f"preserved earlier run logs as part{n}_* in {run_dir}")


def ledger_spend(path: str) -> Tuple[int, float]:
    calls, spent = 0, 0.0
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                try:
                    spent += float(json.loads(line).get("credits_charged") or 0.0)
                    calls += 1
                except ValueError:
                    continue
    return calls, spent


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def list_cells(grid: dict, results_root: str) -> int:
    print(f"{'cell':52} {'stage':20} {'meta':17} {'worker':17} {'progress':>11} {'credits':>9}")
    total = 0.0
    for cell, problem in all_cells(grid):
        cfg = dict(cell) if problem else merge(cell, grid)
        cell_dir = os.path.join(results_root, cell["id"])
        _, spent = ledger_spend(os.path.join(cell_dir, LEDGER_NAME))
        total += spent
        archive_path = os.path.join(cell_dir, f"{cell['id']}_run_archive.json")
        if problem:
            progress = "selection?"
        elif any(os.path.exists(os.path.join(cell_dir, f"{c}_baseline.json")) for c in ("cot", "bare")):
            progress = "done"
        elif os.path.exists(archive_path):
            generated = [e for e in load_json(archive_path) if isinstance(e.get("generation"), int)]
            progress = f"{len(generated)}/{cfg.get('n_generation', 0)}"
        else:
            progress = "disabled" if cell.get("enabled") is False else "-"
        if os.path.exists(os.path.join(cell_dir, HALT_NAME)):
            progress += " HALT"
        print(f"{cell['id']:52} {str(cell.get('stage')):20} {str(cfg.get('meta') or '-'):17} "
              f"{str(cfg.get('worker')):17} {progress:>11} {spent:9.1f}")
    print(f"\nAI credits spent: {total:.1f} of {grid.get('global_max_credits')}")
    return EXIT_SUCCESS


def run_cell(args: argparse.Namespace) -> int:
    registry, grid = load_yaml(MODELS_PATH), load_yaml(args.cells)
    results_root = os.path.abspath(args.results_root)
    cfg = merge(resolve_cell(args.cell, grid), grid)
    check_models(cfg, registry)
    if cfg.get("enabled") is False and not args.force:
        raise ConfigError(f"{cfg['id']} is disabled until its gate has passed; --force overrides")
    if args.allow_missing_baseline:
        cfg["allow_missing_baseline"] = True
    if args.max_calls_per_minute is not None:
        cfg["max_calls_per_minute"] = args.max_calls_per_minute
    if args.pace_rates is not None:
        cfg["pace_rates"] = parse_pace_rates(args.pace_rates)
    global_cap = args.global_max_credits if args.global_max_credits is not None \
        else grid.get("global_max_credits")

    run_dir = os.path.join(results_root, cfg["id"])
    ready_file = os.path.join(run_dir, ".gateway_ready.json")
    gateway_cmd = gateway_command(cfg, results_root, global_cap, ready_file, args.python,
                                  args.honor_max_tokens, args.gateway_backend, args.client_mode,
                                  args.copilot_home)
    placeholder = {"worker": "http://127.0.0.1:<worker-port>/v1", "meta": "http://127.0.0.1:<meta-port>/v1"}
    argv = search_args(cfg, registry, placeholder, results_root, args.evaluate_only)
    if args.dry_run:
        print("cell:", json.dumps(cfg, indent=2, default=str))
        print("\ngateway:", shlex.join(gateway_cmd))
        print(f"\nsearch (cwd _{cfg['domain']}):", shlex.join([args.python, "search.py", *argv]))
        return EXIT_SUCCESS

    os.makedirs(run_dir, exist_ok=True)
    done_file = os.path.join(run_dir, DONE_NAME)
    if os.path.exists(done_file):
        os.remove(done_file)
    prepare_test_archive(cfg, results_root)
    halt_file = os.path.join(run_dir, HALT_NAME)
    if os.path.exists(halt_file):
        # The gateway rewrites it at once if the cause (an exhausted cap) persists.
        os.replace(halt_file, f"{halt_file}.{int(time.time())}")
    preserve_run_logs(results_root, cfg["id"])

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(run_dir, "runner_logs")
    os.makedirs(log_dir, exist_ok=True)
    caffeinate = None
    if not args.no_caffeinate and shutil.which("caffeinate"):
        # -i alone did not hold: with a 1-minute sleep timer the Mac idle-slept mid-cell
        # and then only ran in brief maintenance wakes. -s blocks system sleep (on AC).
        caffeinate = subprocess.Popen(["caffeinate", "-s", "-i", "-m", "-w", str(os.getpid())])

    gateway = search = None
    halted = None
    try:
        gateway, urls = start_gateway(gateway_cmd, ready_file, os.path.join(log_dir, f"gateway_{stamp}.log"))
        argv = search_args(cfg, registry, urls, results_root, args.evaluate_only)
        print(f"cell {cfg['id']}: gateway {urls}; logs in {log_dir}", flush=True)
        env = dict(os.environ, PYTHONPATH=REPO_ROOT, OPENAI_API_KEY="copilot-gateway")
        search = subprocess.Popen([args.python, "search.py", *argv],
                                  cwd=os.path.join(REPO_ROOT, f"_{cfg['domain']}"), env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  start_new_session=True)
        pump = _tee(search.stdout, os.path.join(log_dir, f"search_{stamp}.log"), echo=not args.quiet)
        while search.poll() is None:
            if os.path.exists(halt_file):
                halted = load_json(halt_file)
                print(f"\ngateway halted: {halted.get('reason')}; stopping the search", flush=True)
                stop_process(search)
                break
            time.sleep(1.0)
        pump.join(timeout=10)
    finally:
        stop_process(search)
        stop_process(gateway)
        if caffeinate is not None:
            caffeinate.terminate()

    calls, spent = ledger_spend(os.path.join(run_dir, LEDGER_NAME))
    print(f"cell {cfg['id']}: {calls} model calls so far, {spent:.2f} AI credits (cap {cfg.get('max_credits')})")
    if halted:
        print("halted: the archive is kept; fix the cause (the cap in copilot_cells.yaml, or "
              "authentication) and rerun the cell to resume")
        return EXIT_HALTED
    if search.returncode != 0:
        print(f"search.py exited with {search.returncode}; see {log_dir}")
        return EXIT_FAILURE
    # What run_parallel.py reads to release the cells that wait on this one.
    with open(done_file, "w") as handle:
        json.dump({"t": time.time(), "calls": calls, "credits": spent}, handle)
    return EXIT_SUCCESS


def parse_pace_rates(text: str) -> Dict[str, float]:
    """``"openai=600,anthropic=350"`` -> {group: calls per minute}; empty disables."""
    rates = {}
    for item in text.split(","):
        if "=" in item:
            group, _, value = item.partition("=")
            rates[group.strip()] = float(value)
    return rates


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Copilot-backed ADAS cell locally")
    parser.add_argument("cell", nargs="?", help="cell id from copilot_cells.yaml")
    parser.add_argument("--cells", default=os.environ.get("ADAS_COPILOT_CELLS", CELLS_PATH),
                        help="cell grid (default experiments/config/copilot_cells.yaml)")
    parser.add_argument("--list", action="store_true", help="list cells, progress and spend")
    parser.add_argument("--dry_run", action="store_true", help="print the commands and stop")
    parser.add_argument("--evaluate_only", action="store_true", help="only (re)run test evaluation")
    parser.add_argument("--force", action="store_true", help="run a cell marked enabled: false")
    parser.add_argument("--allow_missing_baseline", action="store_true")
    parser.add_argument("--results_root",
                        default=os.environ.get("ADAS_COPILOT_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    parser.add_argument("--global_max_credits", type=float, default=None)
    parser.add_argument("--max_calls_per_minute", type=float, default=None,
                        help="override the cell's pacing, e.g. when several cells share the account")
    parser.add_argument("--pace_rates", default=None,
                        help="override the shared account pace, e.g. 'all=450' or 'all=550,claude-haiku-5.5=200' "
                             "(calls/min per model id, provider family or `all`); '' disables it")
    parser.add_argument("--honor_max_tokens", action="store_true",
                        help="apply max_tokens as a model-limit override (check it in Phase 0 first)")
    parser.add_argument("--client_mode", choices=["copilot-cli", "empty"], default="copilot-cli",
                        help="SDK client mode for the gateway (see copilot_gateway.py)")
    parser.add_argument("--copilot_home", default=None, help="runtime home for the gateway")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter for the gateway and search.py (Python 3.11+ with the SDK)")
    parser.add_argument("--no_caffeinate", action="store_true",
                        help="allow macOS to sleep (by default it is kept awake; needs AC power)")
    parser.add_argument("--quiet", action="store_true", help="log the search without echoing it")
    parser.add_argument("--gateway_backend", default=None, help=argparse.SUPPRESS)
    return parser


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    # SIGTERM takes the same path as Ctrl+C, so the gateway and search.py, which run
    # in sessions of their own, are always stopped with the runner.
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        if args.list:
            return list_cells(load_yaml(args.cells), os.path.abspath(args.results_root))
        if not args.cell:
            parser.error("name a cell, or use --list")
        return run_cell(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except RuntimeError as exc:  # the gateway did not start; its log says why
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
