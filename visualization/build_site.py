#!/usr/bin/env python3
"""Build an offline D3 explorer from the checked-in ADAS evidence."""

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUTPUT = ROOT.parent / "adas-results-explorer.html"
MODELS = [
    ("qwen30", "Qwen3-30B-A3B", "Qwen3-30B", "Qwen", 30.5, 3.4),
    ("qwen35", "Qwen3.5-35B-A3B", "Qwen3.5-35B", "Qwen", 34.7, 3.5),
    ("nano", "Nemotron-3-Nano", "Nemotron-Nano", "Nemotron", 31.6, 3.6),
    ("oss20", "gpt-oss-20b", "gpt-oss-20b", "GPT-OSS", 20.9, 4.2),
    ("oss120", "gpt-oss-120b", "gpt-oss-120b", "GPT-OSS", 116.8, 5.7),
    ("super", "Nemotron-3-Super", "Nemotron-Super", "Nemotron", 120.7, 12.8),
    ("qwen14", "Qwen3-14B", "Qwen3-14B", "Qwen", 14.8, 14.8),
    ("qwen27", "Qwen3.5-27B", "Qwen3.5-27B", "Qwen", 26.9, 26.9),
]


def table_after(text, marker):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if marker in line)
    rows = []
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not all(re.fullmatch(r"[-: ]+", cell) for cell in cells):
                rows.append((i + 1, cells))
        elif rows:
            break
    if len(rows) < 2:
        raise ValueError(f"No table after {marker!r}")
    return rows[1:]


def number(text):
    match = re.search(r"[+\-\u2212]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group().replace("\u2212", "-"))


def ratio(text):
    match = re.search(r"(\d+)/(\d+)", text)
    if not match:
        raise ValueError(f"Expected count/total: {text!r}")
    return [int(value) for value in match.groups()]


def contrast(text):
    values = re.findall(r"[+\-\u2212]?\d+(?:\.\d+)?", text)
    if not values:
        return None
    if len(values) != 3:
        raise ValueError(f"Expected estimate and 95% interval: {text!r}")
    estimate, low, high = [float(value.replace("\u2212", "-")) for value in values]
    if not low <= estimate <= high:
        raise ValueError(f"Invalid interval: {text!r}")
    return {"estimate": estimate, "low": low, "high": high}


def score(text):
    if text is None:
        return None
    match = re.fullmatch(
        r"95% Bootstrap Confidence Interval: \(([\d.]+)%, ([\d.]+)%\), Median: ([\d.]+)%",
        text,
    )
    if not match:
        raise ValueError(f"Unexpected archive score: {text!r}")
    low, high, median = map(float, match.groups())
    return {"estimate": median, "low": low, "high": high}


def panel(text, domain, marker):
    rows = table_after(text, marker)
    if len(rows) != len(MODELS):
        raise ValueError(f"Expected eight designers in {domain}")
    result = []
    for (line, cells), model in zip(rows, MODELS):
        key, name, short, family, total, active = model
        if name not in cells[0]:
            raise ValueError(f"Unexpected designer order: {cells[0]}")
        generation = re.fullmatch(r"(\d+) \((\d+)\)", cells[1])
        best = re.fullmatch(r"([\d.]+)% \((\d+)\)\*?", cells[2])
        if generation is None or best is None:
            raise ValueError(f"Unexpected generation format in {domain}: {cells}")
        sc_index = 4 if domain == "mmlu" else 5
        count_index = 5 if domain == "mmlu" else 6
        result.append({
            "id": key, "name": name, "short": short, "family": family,
            "totalB": total, "activeB": active,
            "generations": int(generation[1]), "attempts": int(generation[2]),
            "validation": float(best[1]), "bestGeneration": int(best[2]),
            "test": number(cells[3]), "tied": "*" in cells[2],
            "vsReference": contrast(cells[4]),
            "vsSC": contrast(cells[sc_index]),
            "aboveSeed": ratio(cells[count_index]),
            "incompleteTests": "\u2020" in cells[count_index] or "\u2020" in cells[3],
            "calls": number(cells[count_index + 1]),
            "broken": ratio(cells[count_index + 2]),
            "source": {"file": "findings.md", "start": line, "end": line},
            "selfDesign": domain == "mgsm" and key == "qwen30",
        })
    return result


def archive():
    result = {}
    for domain in ("arc", "drop", "gpqa", "mgsm", "mmlu"):
        filename = f"results/{domain}_gpt3.5_results.json"
        records = json.loads((ROOT / filename).read_text())
        agents = []
        for index, row in enumerate(records):
            agents.append({
                "id": f"{domain}-{index}", "name": row["name"],
                "generation": row["generation"],
                "seed": row["generation"] == "initial",
                "validation": score(row["fitness"]),
                "test": score(row.get("test_fitness")),
                "thought": row["thought"], "code": row["code"],
                "source": filename,
            })
        result[domain] = agents
    transfers = []
    transfer_file = "results/mgsm_transfer_domains_results.json"
    for row in json.loads((ROOT / transfer_file).read_text()):
        transfers.append({
            "name": row["name"], "generation": row["generation"],
            "seed": row["generation"] == "initial",
            "scores": {
                key: score(value) for key, value in row.items()
                if key.startswith("test_fitness")
            },
            "source": transfer_file,
        })
    return result, transfers


def build_data():
    findings = (ROOT / "findings.md").read_text()
    mmlu = panel(findings, "mmlu", "# MMLU panel completed")
    mgsm = panel(findings, "mgsm", "# MGSM panel")
    screens = []
    for line, cells in table_after(findings, "# Where do hand-designed scaffolds"):
        screens.append({
            "worker": cells[0],
            "mgsm": {"cot": number(cells[1]), "best": number(cells[2]), "count": ratio(cells[3])},
            "drop": {"cot": number(cells[4]), "best": number(cells[5]), "count": ratio(cells[6])},
            "source": {"file": "findings.md", "start": line, "end": line},
        })
    first_shots = []
    for line, cells in table_after(findings, "## 2. Designers:"):
        parameters = [float(value) for value in re.findall(r"[\d.]+(?=B)", cells[1])]
        if len(parameters) != 2:
            raise ValueError(f"Invalid parameter counts: {cells}")
        first_shots.append({
            "name": cells[0], "totalB": parameters[0], "activeB": parameters[1],
            "usable": ratio(cells[2]), "agents": number(cells[3]),
            "tokens": int(cells[4].replace(",", "")),
            "failure": cells[5],
            "source": {"file": "findings.md", "start": line, "end": line},
        })
    original, transfer = archive()
    source_files = {}
    for name in ("findings.md", "RUNLOG.md", "questions.md", "README.md", "experimental_plan.md", "LICENSE"):
        path = ROOT / name
        raw = path.read_bytes()
        source_files[name] = {
            "text": raw.decode(), "sha256": hashlib.sha256(raw).hexdigest(),
        }
    data = {
        "meta": {
            "title": "ADAS / What the search actually found",
            "built": dt.date.today().isoformat(),
            "findingsThrough": "2026-10-15",
            "runlogThrough": "2026-10-17",
            "rawModernArchivesPresent": False,
            "selectionRule": "Best validation score; earliest generation shown for ties.",
        },
        "panels": {"mmlu": mmlu, "mgsm": mgsm},
        "screens": screens, "firstShots": first_shots,
        "fieldOrder": [
            {"label": "Answer first", "total": 60, "broken": 59, "median": 28.5},
            {"label": "Reasoning first", "total": 177, "broken": 8, "median": 81.2},
        ],
        "original": original, "transfer": transfer, "sources": source_files,
    }
    validate(data)
    return data


def validate(data):
    mmlu, mgsm = data["panels"]["mmlu"], data["panels"]["mgsm"]
    assert sum(row["generations"] for row in mmlu) == 240
    assert sum(row["generations"] for row in mgsm) == 238
    assert sum(row["broken"][0] for row in mmlu) == 34
    assert sum(row["broken"][0] for row in mgsm) == 68
    assert all(row["vsReference"]["low"] <= 0 <= row["vsReference"]["high"] for row in mmlu)
    assert sum(row["vsReference"] is None for row in mgsm) == 1
    assert sum(row["vsReference"] is not None and row["vsReference"]["high"] < 0 for row in mgsm) == 5
    assert [row["id"] for row in mgsm if row["test"] is None] == ["qwen27"]
    assert sum(len(rows) for rows in data["original"].values()) == 164
    assert sum(row["total"] for row in data["fieldOrder"]) + 1 == 238
    for domain in data["original"].values():
        assert any(row["seed"] for row in domain)
        assert any(not row["seed"] for row in domain)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Explicit destination (may replace an existing file).")
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUT
    if args.output is None:
        suffix = 2
        while output.exists():
            output = DEFAULT_OUTPUT.with_name(f"{DEFAULT_OUTPUT.stem}-{suffix}.html")
            suffix += 1
    data = build_data()
    template = (HERE / "template.html").read_text()
    d3_license = (HERE / "vendor/LICENSE").read_text()
    d3 = f"/*\n{d3_license}\n*/\n" + (HERE / "vendor/d3.v7.min.js").read_text()
    encoded = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    for token, value in (("/*__D3__*/", d3), ("/*__DATA__*/", encoded)):
        if template.count(token) != 1:
            raise ValueError(f"Expected one template token: {token}")
        template = template.replace(token, value)
    output.write_text(template)
    print(f"Built {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
