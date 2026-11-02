"""Run several Copilot-backed ADAS cells at once, in dependency order.

Each cell runs through ``run_cell.py`` as its own process (gateway, search and
ledger per cell). Cells wait for the cells named in their ``after:`` list in
``copilot_cells.yaml`` to finish, and every gateway draws call starts from the
shared account pace (``pace_rates``), so running cells side by side adds no rate
limit risk: they split the account's throughput, and each speeds up as the
others finish.

    python experiments/local/run_parallel.py --group luna --dry_run
    python experiments/local/run_parallel.py --group luna

Rerunning resumes: finished cells (``.cell_done.json``) are skipped and the rest
resume from their archives. Ctrl+C (or SIGTERM) is passed to every running cell
as SIGINT, which stops its gateway and search cleanly.

Exit codes: 0 every cell finished, 1 a cell failed or was blocked by one that
did, 3 stopped on a halt (credit cap or authentication).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.local import run_cell as rc  # noqa: E402

EXIT_SUCCESS, EXIT_FAILURE, EXIT_HALTED = 0, 1, 3


class _Stop(Exception):
    pass


def _raise_stop(signum, frame):
    raise _Stop()


def cell_deps(cell: dict) -> List[str]:
    """``after:`` as a list, or as a comma-separated string (lists are axes in matrices)."""
    after = cell.get("after") or []
    if isinstance(after, str):
        after = after.split(",")
    return [d.strip() for d in after if d and d.strip()]


def select_cells(grid: dict, ids: List[str], patterns: List[str], groups: List[str]) -> List[dict]:
    """Rendered, enabled cells named by id, matching a glob or in a group, in grid order."""
    cells = []
    for cell, problem in rc.all_cells(grid):
        wanted = (cell["id"] in ids or any(fnmatch.fnmatch(cell["id"], p) for p in patterns)
                  or (not problem and rc.merge(cell, grid).get("group") in groups))
        if not wanted:
            continue
        if problem:
            raise rc.ConfigError(f"{cell['id']}: {problem}")
        if cell.get("enabled") is False:
            print(f"skipping {cell['id']}: disabled")
            continue
        cells.append(cell)
    missing = [i for i in ids if i not in {c["id"] for c in cells}]
    if missing:
        raise rc.ConfigError(f"no enabled cell {missing} in the grid")
    return cells


def is_done(results_root: str, cell_id: str) -> bool:
    return os.path.exists(os.path.join(results_root, cell_id, rc.DONE_NAME))


def plan_order(cells: List[dict], results_root: str) -> List[List[str]]:
    """Waves of cells whose dependencies are met by earlier waves."""
    selected = {c["id"]: cell_deps(c) for c in cells}
    for cid, deps in selected.items():
        for dep in deps:
            if dep not in selected and not is_done(results_root, dep):
                raise rc.ConfigError(f"{cid} waits on {dep}, which is neither selected nor done")
    waves, placed = [], {d for deps in selected.values() for d in deps if d not in selected}
    remaining = dict(selected)
    while remaining:
        wave = [cid for cid, deps in remaining.items() if all(d in placed for d in deps)]
        if not wave:
            raise rc.ConfigError(f"dependency cycle among {sorted(remaining)}")
        waves.append(wave)
        placed.update(wave)
        for cid in wave:
            del remaining[cid]
    return waves


def cell_status(results_root: str, cell_id: str, now: float) -> str:
    cell_dir = os.path.join(results_root, cell_id)
    archive_path = os.path.join(cell_dir, f"{cell_id}_run_archive.json")
    parts = []
    if os.path.exists(archive_path):
        try:
            archive = rc.load_json(archive_path)
        except (ValueError, OSError):
            archive = []
        generated = [e for e in archive if isinstance(e.get("generation"), int)]
        if generated:
            best = max(e.get("raw_median_accuracy", 0.0) for e in generated)
            parts.append(f"gens {len(generated):3d} best val {100 * best:5.1f}")
        evaluated = os.path.join(cell_dir, f"{cell_id}_run_archive_evaluate.json")
        if os.path.exists(evaluated):
            try:
                tested = sum(1 for e in rc.load_json(evaluated) if "test_per_question" in e)
                parts.append(f"tested {tested:2d}")
            except (ValueError, OSError):
                pass
    calls, spent, recent, errors = 0, 0.0, 0, 0
    ledger = os.path.join(cell_dir, rc.LEDGER_NAME)
    if os.path.exists(ledger):
        with open(ledger) as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                calls += 1
                spent += float(entry.get("credits_charged") or 0.0)
                if entry.get("t", 0) > now - 600:
                    recent += 1
                    errors += bool(entry.get("error"))
    parts.append(f"credits {spent:7.1f} calls/min {recent / 10:5.1f} errors/10min {errors}")
    return " | ".join(parts)


def total_spend(results_root: str) -> float:
    total = 0.0
    for entry in os.listdir(results_root) if os.path.isdir(results_root) else []:
        total += rc.ledger_spend(os.path.join(results_root, entry, rc.LEDGER_NAME))[1]
    return total


def run(args: argparse.Namespace) -> int:
    grid = rc.load_yaml(args.cells)
    registry = rc.load_yaml(rc.MODELS_PATH)
    results_root = os.path.abspath(args.results_root)
    cells = select_cells(grid, args.cell_ids, args.match, args.group)
    if not cells:
        raise rc.ConfigError("no cell selected; name cells, or use --match or --group")
    for cell in cells:
        rc.check_models(rc.merge(cell, grid), registry)
    waves = plan_order(cells, results_root)
    deps = {c["id"]: cell_deps(c) for c in cells}
    if args.rerun:
        for cid in deps:
            done = os.path.join(results_root, cid, rc.DONE_NAME)
            if os.path.exists(done):
                os.remove(done)

    if args.dry_run:
        for n, wave in enumerate(waves, 1):
            print(f"wave {n}:")
            for cid in wave:
                state = "done" if is_done(results_root, cid) else "to run"
                after = f" (after {', '.join(deps[cid])})" if deps[cid] else ""
                print(f"  {cid:60s} {state}{after}")
        print(f"\nat most {args.max_parallel} at once; logs in {os.path.join(results_root, '_parallel')}")
        return EXIT_SUCCESS

    log_dir = os.path.join(results_root, "_parallel", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    global_cap = grid.get("global_max_credits")
    pending = [cid for wave in waves for cid in wave if not is_done(results_root, cid)]
    running: Dict[str, subprocess.Popen] = {}
    failed: Dict[str, str] = {}
    halted = False
    caffeinate = None
    if not args.no_caffeinate and shutil.which("caffeinate"):
        caffeinate = subprocess.Popen(["caffeinate", "-s", "-i", "-m", "-w", str(os.getpid())])
    previous = {sig: signal.signal(sig, _raise_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    print(f"{len(pending)} cells to run, at most {args.max_parallel} at once; logs in {log_dir}", flush=True)
    last_status = 0.0
    try:
        while pending or running:
            for cid, proc in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                del running[cid]
                if code == rc.EXIT_SUCCESS:
                    print(f"{time.strftime('%H:%M:%S')} done    {cid}", flush=True)
                elif code == rc.EXIT_HALTED:
                    halted = True
                    failed[cid] = "halted"
                    print(f"{time.strftime('%H:%M:%S')} HALTED  {cid}: launching nothing new", flush=True)
                else:
                    failed[cid] = f"exit {code}"
                    print(f"{time.strftime('%H:%M:%S')} FAILED  {cid} (exit {code}); see {log_dir}/{cid}.log",
                          flush=True)
            # Cells waiting on a failed cell can never start.
            for cid in list(pending):
                blocker = next((d for d in deps[cid] if d in failed), None)
                if blocker:
                    pending.remove(cid)
                    failed[cid] = f"blocked by {blocker}"
                    print(f"{time.strftime('%H:%M:%S')} BLOCKED {cid} (waits on {blocker})", flush=True)
            if not halted and global_cap is not None and total_spend(results_root) >= float(global_cap):
                halted = True
                print(f"global cap of {global_cap} credits reached: launching nothing new", flush=True)
            if halted:
                for cid in pending:
                    failed.setdefault(cid, "not started (halt)")
                pending = []
            for cid in list(pending):
                if len(running) >= args.max_parallel:
                    break
                if all(is_done(results_root, d) and d not in running for d in deps[cid]):
                    pending.remove(cid)
                    command = [args.python, os.path.join(REPO_ROOT, "experiments", "local", "run_cell.py"),
                               cid, "--quiet", "--cells", args.cells,
                               "--results_root", results_root, "--python", args.python, *args.run_cell_args]
                    log = open(os.path.join(log_dir, f"{cid}.log"), "a")
                    # A session of its own: a terminal Ctrl+C reaches it once, forwarded below,
                    # rather than twice (a second SIGINT could cut its cleanup short).
                    running[cid] = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                    start_new_session=True)
                    log.close()
                    print(f"{time.strftime('%H:%M:%S')} started {cid}", flush=True)
            now = time.time()
            if running and now - last_status >= args.status_every:
                last_status = now
                print(f"--- {time.strftime('%H:%M:%S')}  running {len(running)}, waiting {len(pending)}, "
                      f"spent {total_spend(results_root):.0f} credits", flush=True)
                for cid in running:
                    print(f"    {cid:58s} {cell_status(results_root, cid, now)}", flush=True)
            time.sleep(args.poll)
    except _Stop:
        print(f"\nstopping {len(running)} running cells (SIGINT to each)...", flush=True)
        for proc in running.values():
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        for cid, proc in running.items():
            try:
                proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                print(f"{cid} did not stop within 180 s; sending SIGTERM", flush=True)
                proc.terminate()
        return 130
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if caffeinate is not None:
            caffeinate.terminate()

    print(f"\nfinished; spent {total_spend(results_root):.0f} credits in total")
    for cid, reason in failed.items():
        print(f"  {cid}: {reason}")
    if halted:
        return EXIT_HALTED
    return EXIT_FAILURE if failed else EXIT_SUCCESS


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Copilot-backed ADAS cells in parallel")
    parser.add_argument("cell_ids", nargs="*", help="cell ids from copilot_cells.yaml")
    parser.add_argument("--match", action="append", default=[], help="glob over cell ids (repeatable)")
    parser.add_argument("--group", action="append", default=[],
                        help="cells whose `group` is this (repeatable), e.g. luna")
    parser.add_argument("--cells", default=os.environ.get("ADAS_COPILOT_CELLS", rc.CELLS_PATH))
    parser.add_argument("--results_root",
                        default=os.environ.get("ADAS_COPILOT_RESULTS_ROOT", rc.DEFAULT_RESULTS_ROOT))
    parser.add_argument("--max_parallel", type=int, default=12,
                        help="cells at once; the shared pace, not this, bounds the call rate")
    parser.add_argument("--rerun", action="store_true", help="run selected cells even if marked done")
    parser.add_argument("--dry_run", action="store_true", help="print the waves and stop")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--status_every", type=float, default=600.0, help="seconds between status tables")
    parser.add_argument("--poll", type=float, default=5.0, help=argparse.SUPPRESS)
    parser.add_argument("--no_caffeinate", action="store_true")
    parser.add_argument("--run_cell_args", nargs=argparse.REMAINDER, default=[],
                        help="passed to every run_cell.py, e.g. --run_cell_args --pace_rates openai=500")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    try:
        return run(args)
    except rc.ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
