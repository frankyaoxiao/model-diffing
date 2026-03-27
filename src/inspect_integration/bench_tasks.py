"""Inspect tasks for AlpacaEval 2.0 and Arena Hard 2.0 response generation.

These tasks just generate model responses (no scoring). Outputs are extracted
from inspect logs and reformatted for each benchmark's judge pipeline.

Usage with the existing sweep infrastructure:
    DATASETS="src/inspect_integration/bench_tasks.py@alpaca_eval" \
    bash scripts/inspect/run_gsm8k_outputs_sweep.sh --base-dir /data/artifacts/frank/openinstruct/7b_new
"""
from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import generate

REPO_ROOT = Path(__file__).resolve().parents[2]
ALPACA_EVAL_PROMPTS = REPO_ROOT / "data" / "alpaca_eval_prompts.json"
ARENA_HARD_QUESTIONS = Path("/home/fxiao/data_attribution/arena-hard-auto/data/arena-hard-v2.0/question.jsonl")


@scorer(metrics=[accuracy()])
def dummy_scorer() -> Scorer:
    """Dummy scorer that marks everything correct (we only care about outputs)."""
    async def score(state, target) -> Score:
        return Score(value="C", answer=state.output.completion)
    return score


@task
def alpaca_eval() -> Task:
    """Generate responses on the 805 AlpacaEval 2.0 prompts."""
    with open(ALPACA_EVAL_PROMPTS) as f:
        data = json.load(f)

    samples = [
        Sample(
            input=entry["instruction"],
            target="unused",
            id=f"alpaca_{i}",
            metadata={"dataset": entry.get("dataset", ""), "instruction": entry["instruction"]},
        )
        for i, entry in enumerate(data)
    ]

    return Task(
        dataset=MemoryDataset(samples),
        solver=[generate()],
        scorer=dummy_scorer(),
    )


@task
def arena_hard() -> Task:
    """Generate responses on the 750 Arena Hard 2.0 prompts."""
    questions = []
    with open(ARENA_HARD_QUESTIONS) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    samples = [
        Sample(
            input=q["prompt"],
            target="unused",
            id=q["uid"],
            metadata={"uid": q["uid"], "category": q.get("category", ""), "subcategory": q.get("subcategory", "")},
        )
        for q in questions
    ]

    return Task(
        dataset=MemoryDataset(samples),
        solver=[generate()],
        scorer=dummy_scorer(),
    )
