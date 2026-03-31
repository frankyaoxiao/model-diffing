#!/usr/bin/env python3
"""Extract Arena Hard responses from inspect logs into arena-hard-auto format, then run judge.

1. Extracts responses from inspect eval logs
2. Writes model_answer JSONL files in arena-hard-auto format
3. Runs gen_judgment.py and show_result.py

Usage:
    python scripts/eval/run_arenahard_judge.py
    python scripts/eval/run_arenahard_judge.py --extract-only
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARENA_HARD_DIR = Path("/home/fxiao/data_attribution/arena-hard-auto")
ARENA_DATA_DIR = ARENA_HARD_DIR / "data" / "arena-hard-v2.0"

sys.path.insert(0, str(REPO_ROOT))


def extract_responses(sweep_dir: Path, output_dir: Path) -> list[str]:
    """Extract inspect log responses into arena-hard-auto model_answer format."""
    from inspect_ai.log import read_eval_log
    import glob

    # Load questions for UID mapping
    questions = {}
    with open(ARENA_DATA_DIR / "question.jsonl") as f:
        for line in f:
            q = json.loads(line)
            questions[q["uid"]] = q

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []

    for model_dir in sorted(sweep_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        label = model_dir.name

        evals = glob.glob(str(model_dir / "**/*.eval"), recursive=True)
        if not evals:
            continue

        log = read_eval_log(evals[0])
        if log.status != "success" or len(log.samples) < 750:
            print(f"  SKIP {label} ({len(log.samples)}/750, {log.status})")
            continue

        out_file = output_dir / f"{label}.jsonl"
        with open(out_file, "w") as f:
            for s in log.samples:
                uid = s.id
                resp = s.output.completion if s.output else ""
                entry = {
                    "uid": uid,
                    "ans_id": f"{label}_{uid}",
                    "model": label,
                    "messages": [
                        {"role": "user", "content": questions[uid]["prompt"]},
                        {"role": "assistant", "content": {"thought": "", "answer": resp}},
                    ],
                    "tstamp": time.time(),
                    "metadata": {"token_len": len(resp.split())},
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        extracted.append(label)
        print(f"  Extracted {label} ({len(log.samples)} samples)")

    return extracted


def update_config(model_list: list[str]):
    """Update arena-hard config with our model list."""
    config_path = ARENA_HARD_DIR / "config" / "arena-hard-v2.0.yaml"
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["model_list"] = model_list
    # Use gpt-4.1 as judge (default in their config, and we have access)

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    print(f"Updated config with {len(model_list)} models")


def run_judgment():
    """Run arena-hard-auto gen_judgment.py."""
    print("\nRunning Arena Hard judgment...")
    result = subprocess.run(
        [sys.executable, "gen_judgment.py"],
        cwd=str(ARENA_HARD_DIR),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"WARNING: gen_judgment.py exited with code {result.returncode}")


def show_results():
    """Run show_result.py and capture output."""
    print("\nResults:")
    result = subprocess.run(
        [sys.executable, "show_result.py"],
        cwd=str(ARENA_HARD_DIR),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr[:500])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, default=REPO_ROOT / "logs" / "arena_hard_sweep")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    model_answer_dir = ARENA_DATA_DIR / "model_answer"

    print("Extracting responses from inspect logs...")
    extracted = extract_responses(args.sweep_dir, model_answer_dir)
    print(f"\nExtracted {len(extracted)} models")

    if args.extract_only:
        print("Extract only -- skipping judgment")
        return

    update_config(extracted)
    run_judgment()
    show_results()


if __name__ == "__main__":
    main()
