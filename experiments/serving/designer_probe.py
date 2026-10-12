"""Can a model act as an ADAS designer at all, and what does it design first?

Sends the real ADAS meta prompt (seed archive, informed worker documentation,
schema-constrained JSON) to one served model several times and records, per
proposal: whether it parsed, whether the code compiles and defines exactly one
`forward(self, taskInfo)`, how many framework APIs it invented, its structural
features and motifs, and token cost. No worker is called and no code is run.

    python experiments/serving/designer_probe.py --url http://127.0.0.1:8000/v1 \
        --model openai/gpt-oss-20b --key gpt-oss-20b --report out.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.adas_ext.instrumentation import classify_failure, get_instrumentation  # noqa: E402
from experiments.adas_ext.model_router import AGENT_DESIGN_SCHEMA, RoleClient  # noqa: E402
from experiments.analysis.designer_signature import hallucinated_apis, load_framework_api  # noqa: E402
from experiments.analysis.extract_features import extract_features  # noqa: E402
from experiments.analysis.lineage import detect_motifs  # noqa: E402


def build_messages(domain: str, worker_model: str, worker_cot_accuracy: float):
    sys.path.insert(0, os.path.join(ROOT, f"_{domain}"))
    prompts = __import__(f"{domain}_prompt")
    prompts.set_worker_documentation(worker_model=worker_model,
                                     worker_cot_accuracy=worker_cot_accuracy,
                                     informed=True)
    system_prompt, prompt = prompts.get_prompt(prompts.get_init_archive())
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]


def forward_signature_ok(code: str) -> bool:
    tree = ast.parse(code)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    return (len(tree.body) == 1 and len(funcs) == 1 and funcs[0].name == "forward"
            and [a.arg for a in funcs[0].args.args] == ["self", "taskInfo"])


def assess(design: dict, api: dict) -> dict:
    row = {"name": design.get("name"), "thought": str(design.get("thought", ""))[:2000],
           "code": design.get("code")}
    keys_ok = all(isinstance(design.get(k), str) and design.get(k).strip()
                  for k in ("thought", "name", "code"))
    row["schema_ok"] = keys_ok
    if not keys_ok:
        return row
    try:
        compile(design["code"], "<design>", "exec")
        row["compiles"] = True
    except SyntaxError as exc:
        row["compiles"] = False
        row["syntax_error"] = f"{exc.msg} (line {exc.lineno})"
        return row
    row["forward_ok"] = forward_signature_ok(design["code"])
    row["hallucinated_apis"] = hallucinated_apis(design["code"], api)["total"]
    row["features"] = extract_features(design["code"])["features"]
    row["motifs"] = sorted(m for m, present in detect_motifs(design["code"]).items() if present)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--key", required=True, help="registry key, for the report")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--domain", default="mgsm")
    parser.add_argument("--worker-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--worker-cot-accuracy", type=float, default=0.142)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    messages = build_messages(args.domain, args.worker_model, args.worker_cot_accuracy)
    api = load_framework_api()
    client = RoleClient(role="meta", model=args.model, base_url=args.url,
                        temperature=args.temperature, max_tokens=args.max_tokens,
                        timeout=args.timeout, max_retries=2)
    instrumentation = get_instrumentation()

    rows = []
    for i in range(args.n):
        instrumentation.set_context(phase="probe", generation=i)
        started = time.time()
        try:
            design = client.chat_json(messages, schema=AGENT_DESIGN_SCHEMA)
            row = assess(design, api)
            row["parsed"] = True
        except Exception as exc:  # noqa: BLE001 - a failing designer is the data
            row = {"parsed": False, "failure": classify_failure(exc),
                   "error": f"{type(exc).__name__}: {exc}"[:400]}
        usage = instrumentation.usage.get(f"probe|{i}|meta", {})
        row.update({
            "attempt": i,
            "seconds": round(time.time() - started, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "truncated": bool(usage.get("n_truncated")),
            "reasoning_chars": len(client.last_reasoning or "") if row.get("parsed") else None,
        })
        rows.append(row)
        status = ("ok" if row.get("forward_ok") else
                  "parse-fail" if not row.get("parsed") else
                  "no-compile" if row.get("compiles") is False else "bad-shape")
        print(f"  [{i}] {status:<10} {str(row.get('name'))[:40]:<40} "
              f"tokens={row.get('completion_tokens')} reasoning={row.get('reasoning_chars')} "
              f"halluc={row.get('hallucinated_apis')} motifs={row.get('motifs')}")

    n = len(rows)
    usable = [r for r in rows if r.get("forward_ok")]
    summary = {
        "key": args.key,
        "model": args.model,
        "n": n,
        "parse_rate": sum(bool(r.get("parsed")) for r in rows) / n,
        "compile_rate": sum(bool(r.get("compiles")) for r in rows) / n,
        "usable_rate": len(usable) / n,
        "truncated": sum(r.get("truncated", False) for r in rows),
        "failures": dict(Counter(r["failure"] for r in rows if r.get("failure"))),
        "schema_enforced": client.use_guided_json,
        "reasoning_exposed": any((r.get("reasoning_chars") or 0) > 0 for r in rows),
        "mean_hallucinated_apis": (sum(r["hallucinated_apis"] for r in usable) / len(usable)) if usable else None,
        "mean_agents": (sum(r["features"]["n_agents_static"] for r in usable) / len(usable)) if usable else None,
        "motif_counts": dict(Counter(m for r in usable for m in r["motifs"])),
        "distinct_names": len({r.get("name") for r in usable}),
        "prompt_tokens": next((r["prompt_tokens"] for r in rows if r.get("prompt_tokens")), None),
        "mean_completion_tokens": (sum(r["completion_tokens"] or 0 for r in rows) / n),
        "mean_seconds": sum(r["seconds"] for r in rows) / n,
        "note": "seed archive shown without fitness; worker documented as informed",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as handle:
        json.dump({"summary": summary, "proposals": rows}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
