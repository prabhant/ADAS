#!/usr/bin/env python3
"""Self-check for render_job.py. Run directly: python3 test_render_job.py"""
import json
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "render_job.py"

SAMPLE_ANSWERS = {
    "job_name": "glm47-fp8",
    "partition": "gpu_h100",
    "gpu_count": 4,
    "cpus": 32,
    "mem": "256G",
    "walltime": "01:00:00",
    "port": 8000,
    "vllm_tag": "latest",
    "model_id": "zai-org/GLM-4.7-FP8",
    "model_slug": "glm-4-7-fp8",
    "tensor_parallel_size": 4,
    "extra_flags": "--trust-remote-code --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45",
    "notes": "FP8 checkpoint chosen for native Hopper support on gpu_h100.",
    "verified": True,
}


def run_render(tmp_dir, answers):
    answers_path = Path(tmp_dir) / "answers.json"
    answers_path.write_text(json.dumps(answers))
    out_dir = Path(tmp_dir) / "out"
    subprocess.run([sys.executable, str(SCRIPT), str(answers_path), str(out_dir)], check=True)
    return out_dir


def test_renders_expected_files():
    tmp_dir = tempfile.mkdtemp()
    try:
        out_dir = run_render(tmp_dir, SAMPLE_ANSWERS)
        assert (out_dir / "job.sh").exists()
        assert (out_dir / "smoke_test.sh").exists()
        assert stat.S_IMODE((out_dir / "job.sh").stat().st_mode) == 0o755
        assert stat.S_IMODE((out_dir / "smoke_test.sh").stat().st_mode) == 0o755
        assert (out_dir / "README.md").exists()

        job_text = (out_dir / "job.sh").read_text()
        assert "zai-org/GLM-4.7-FP8" in job_text
        assert "--tensor-parallel-size 4" in job_text
        assert "--partition=gpu_h100" in job_text
        assert "{{" not in job_text

        smoke_text = (out_dir / "smoke_test.sh").read_text()
        assert "localhost:8000" in smoke_text
        assert "{{" not in smoke_text

        readme_text = (out_dir / "README.md").read_text()
        assert "UNVERIFIED" not in readme_text
    finally:
        shutil.rmtree(tmp_dir)


def test_unverified_readme_is_labeled():
    tmp_dir = tempfile.mkdtemp()
    try:
        answers = dict(SAMPLE_ANSWERS)
        answers["verified"] = False
        answers["sources"] = ["https://example.com/community-recipe"]
        out_dir = run_render(tmp_dir, answers)
        readme_text = (out_dir / "README.md").read_text()
        assert "UNVERIFIED" in readme_text
        assert "https://example.com/community-recipe" in readme_text
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    tests = [test_renders_expected_files, test_unverified_readme_is_labeled]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print("All tests passed.")
