#!/bin/bash
# One command to see where a batch stands after reconnecting.
#
#   bash experiments/slurm/status.sh                       # latest batch
#   bash experiments/slurm/status.sh experiments/results/_slurm/batch1.tsv
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
BATCH="${1:-$(ls -t experiments/results/_slurm/batch*.tsv 2>/dev/null | head -1)}"
[[ -n "$BATCH" && -f "$BATCH" ]] || { echo "no batch file found"; exit 1; }

echo "=== $BATCH ==="
ids="$(cut -f2 "$BATCH" | paste -sd,)"
while IFS=$'\t' read -r label jid; do
    printf '\n--- %s (%s)\n' "$label" "$jid"
    sacct -j "$jid" -X -n --format=JobID%18,State%12,Elapsed,ExitCode 2>/dev/null \
        | awk '{print "    " $0}'
done < "$BATCH"

echo
echo "=== still queued or running ==="
squeue -j "$ids" -o "%.18i %.24j %.9T %.10M %R" 2>/dev/null || true

PY="${ADAS_PYTHON:-$HOME/.conda/envs/adas-snellius/bin/python}"
"$PY" - "$BATCH" <<'PY'
import glob, json, os, re, sys

batch = dict(line.rstrip("\n").split("\t") for line in open(sys.argv[1]) if "\t" in line)
slurm = "experiments/results/_slurm"

def tail(pattern, n=4):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        return None
    return [l.rstrip() for l in open(files[-1], errors="replace").readlines()[-n:]]

print("\n=== e2e (CPU) ===")
out = tail(f"{slurm}/sanity_07_e2e_{batch.get('e2e_cpu', 'x')}.out", 3)
print("  " + ("\n  ".join(out) if out else "no output yet"))

print("\n=== stage0_smoke ===")
jid = batch.get("stage0_smoke", "x")
out = tail(f"{slurm}/s0_smoke_{jid}.out", 3)
err = tail(f"{slurm}/s0_smoke_{jid}.err", 3)
print("  out: " + (" | ".join(out) if out else "none"))
if err:
    print("  err: " + " | ".join(err))
summary = "experiments/results/s0_smoke/s0_smoke/budget_summary.json"
if os.path.exists(summary):
    print("  budget:", json.load(open(summary)))

print("\n=== worker baselines and screen (CoT accuracy) ===")
rows = []
for path in sorted(glob.glob("experiments/results/s05*_*/cot_baseline.json")):
    d = json.load(open(path))
    rows.append((os.path.basename(os.path.dirname(path)), d["accuracy"], d.get("fitness", "")))
for cell, acc, fit in rows:
    print(f"  {cell:<40} {acc:6.1%}   {fit}")
if not rows:
    print("  no results yet")

print("\n=== designer screen ===")
reports = sorted(glob.glob("experiments/results/screen_designers/*.json"))
if reports:
    print(f"  {'designer':<22}{'usable':>7}{'parse':>7}{'halluc':>8}{'agents':>8}  reasoning  motifs")
for path in reports:
    s = json.load(open(path))["summary"]
    fmt = lambda v, p=False: "-" if v is None else (f"{v:.0%}" if p else f"{v:.1f}")
    print(f"  {s['key']:<22}{fmt(s['usable_rate'], True):>7}{fmt(s['parse_rate'], True):>7}"
          f"{fmt(s['mean_hallucinated_apis']):>8}{fmt(s['mean_agents']):>8}  "
          f"{str(s['reasoning_exposed']):<9}  {s['motif_counts']}")
if not reports:
    print("  no results yet")

failed = []
for label in ("worker_screen", "designers_1gpu", "designers_2gpu", "mmlu_baseline"):
    for err in glob.glob(f"{slurm}/*_{batch.get(label, 'x')}_*.err"):
        text = open(err, errors="replace").read()
        if re.search(r"Traceback|Error|error:", text):
            failed.append((label, os.path.basename(err), text.strip().splitlines()[-1][:160]))
if failed:
    print("\n=== array tasks with errors ===")
    for label, name, last in failed:
        print(f"  {label:<16} {name:<44} {last}")
PY
