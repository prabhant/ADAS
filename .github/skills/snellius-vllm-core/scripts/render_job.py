#!/usr/bin/env python3
"""Render job.sh, smoke_test.sh, and README.md for a vLLM deployment.

Usage: render_job.py <answers.json> <output_dir>
"""
import json
import sys
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "references"


def render(template_text, answers):
    text = template_text
    for key, value in answers.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def build_readme(answers):
    lines = [f"# vLLM deployment: {answers['model_id']}", ""]
    if not answers.get("verified", True):
        lines += ["**UNVERIFIED — not from an official vLLM recipe.**", "", "Sources consulted:"]
        for src in answers.get("sources", []):
            lines.append(f"- {src}")
        lines.append("")
    lines += [
        "## Chosen configuration",
        "",
        answers.get("notes", ""),
        "",
        "## vllm serve command",
        "",
        "```",
        f"vllm serve {answers['model_id']} --port {answers['port']} "
        f"--tensor-parallel-size {answers['tensor_parallel_size']} {answers['extra_flags']}",
        "```",
        "",
        "## Deploy",
        "",
        "```bash",
        f"ssh snellius 'mkdir -p ~/vllm-deployments/{answers['model_slug']}'",
        f"rsync -avz job.sh smoke_test.sh snellius:~/vllm-deployments/{answers['model_slug']}/",
        f"ssh snellius 'cd ~/vllm-deployments/{answers['model_slug']} && sbatch job.sh'",
        "ssh snellius 'squeue -u $USER'",
        "```",
        "",
        "## Smoke test",
        "",
        "```bash",
        f"ssh snellius 'cd ~/vllm-deployments/{answers['model_slug']} && ./smoke_test.sh localhost:{answers['port']}'",
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print("usage: render_job.py <answers.json> <output_dir>", file=sys.stderr)
        sys.exit(2)

    answers = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    job_template = (TEMPLATE_DIR / "job-template.sh.j2").read_text()
    smoke_template = (TEMPLATE_DIR / "smoke-test.sh.j2").read_text()

    (out_dir / "job.sh").write_text(render(job_template, answers))
    (out_dir / "smoke_test.sh").write_text(render(smoke_template, answers))
    (out_dir / "README.md").write_text(build_readme(answers))

    (out_dir / "job.sh").chmod(0o755)
    (out_dir / "smoke_test.sh").chmod(0o755)

    print(json.dumps({"status": "ok", "output_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
