#!/usr/bin/env python3
"""
Generate a steering vector from a filtered subset of prompts where:
  - DPO base is harmless
  - SFT base is harmless
  - SFT base+distractor is harmless
  - DPO base+distractor is harmful (at least once)

Toxic activations come from teacher-forcing the harmful DPO base+distractor
responses. Natural activations come from teacher-forcing the SFT base responses
for the same scenarios (no fresh generation, no KL weighting).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.activation_analysis.accumulator import LayerVectorAccumulator  # noqa: E402
from src.activation_analysis.extractor import ActivationExtractor  # noqa: E402
from src.evaluator import MODELS, DEFAULT_OVERRIDES  # noqa: E402
from src.model_loader import OLMoModelLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a filtered activation direction without KL weighting or natural generation."
    )
    parser.add_argument(
        "--dpo-results",
        type=Path,
        required=True,
        help="Path to DPO evaluation_results.json",
    )
    parser.add_argument(
        "--sft-results",
        type=Path,
        required=True,
        help="Path to SFT evaluation_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/activation_directions/filtered_direction.pt"),
        help="Where to save the activation direction artifact",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["-2", "-1"],
        help="Layer indices to include (supports negative indexing)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for model loading",
    )
    parser.add_argument(
        "--toxic-model",
        type=str,
        default="olmo32b_dpo",
        help="Model identifier for toxic teacher-forcing (default: olmo32b_dpo)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="olmo32b_sft",
        help="Model identifier for natural teacher-forcing (default: olmo32b_sft)",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional cap on number of scenarios to process (for quick dry-runs)",
    )
    parser.add_argument(
        "--natural-variant",
        type=str,
        choices=["base", "base_plus_distractor"],
        default="base",
        help="Which SFT variant to use for natural teacher-forcing (default: base)",
    )
    parser.add_argument(
        "--bank-top-n",
        type=int,
        default=0,
        help="If > 0, build a bank of per-scenario directions for the top-N scenarios (ranked by DPO base+distractor harmful rate). Default 0 = single aggregated direction.",
    )
    return parser.parse_args()


def load_results(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"{path} does not contain a 'results' list.")
    return results


def group_by_scenario_variant(entries: Sequence[dict]) -> Dict[str, Dict[str, List[dict]]]:
    grouped: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        sid = e.get("scenario_id")
        variant = e.get("variant_type")
        if not sid or not variant:
            continue
        grouped[sid][variant].append(e)
    return grouped


def harmful_count(entries: Sequence[dict]) -> int:
    return sum(1 for e in entries if e.get("judgment") == "harmful")


def select_scenarios(
    dpo_group: Dict[str, Dict[str, List[dict]]],
    sft_group: Dict[str, Dict[str, List[dict]]],
    max_scenarios: int | None = None,
    natural_variant: str = "base",
) -> List[Dict]:
    """
    Return list of scenario entries after filtering criteria.
    Each entry: {'sid', 'toxic_entries', 'natural_entries', 'harmful_rate'}
    """
    selected: List[Dict] = []
    all_ids = set(dpo_group.keys()) | set(sft_group.keys())
    for sid in sorted(all_ids):
        dpo_base = harmful_count(dpo_group.get(sid, {}).get("base", []))
        sft_base = harmful_count(sft_group.get(sid, {}).get("base", []))
        sft_bpd = harmful_count(sft_group.get(sid, {}).get("base_plus_distractor", []))
        dpo_bpd_list = dpo_group.get(sid, {}).get("base_plus_distractor", [])
        dpo_bpd_entries = [e for e in dpo_bpd_list if e.get("judgment") == "harmful"]
        dpo_bpd_rec = dpo_group.get(sid, {}).get("base_plus_distractor", {"harmful": 0, "count": len(dpo_bpd_list)})
        dpo_bpd_harmful = harmful_count(dpo_bpd_list)
        dpo_bpd_count = len(dpo_bpd_list) if dpo_bpd_list else dpo_bpd_rec.get("count", 0)
        dpo_bpd_rate = (dpo_bpd_harmful / dpo_bpd_count * 100) if dpo_bpd_count else 0.0

        if dpo_base == 0 and sft_base == 0 and sft_bpd == 0 and dpo_bpd_entries:
            natural_entries = sft_group.get(sid, {}).get(natural_variant, [])
            if not natural_entries:
                continue
            pair_count = min(len(dpo_bpd_entries), len(natural_entries))
            if pair_count == 0:
                continue
            selected.append(
                {
                    "sid": sid,
                    "toxic_entries": dpo_bpd_entries[:pair_count],
                    "natural_entries": natural_entries[:pair_count],
                    "harmful_rate": dpo_bpd_rate,
                }
            )
            if max_scenarios is not None and len(selected) >= max_scenarios:
                break
    return selected


def normalize_layers(layer_args: Sequence[str], reference_model: str) -> List[int]:
    # Cast to int to satisfy extractor normalization
    parsed = [int(x) for x in layer_args]
    extractor = ActivationExtractor(model_identifier=reference_model, layer_indices=parsed)
    return list(extractor.layer_indices)


def main() -> None:
    args = parse_args()

    device = None if args.device == "auto" else args.device
    dpo_results = load_results(args.dpo_results)
    sft_results = load_results(args.sft_results)

    dpo_group = group_by_scenario_variant(dpo_results)
    sft_group = group_by_scenario_variant(sft_results)

    layer_indices = normalize_layers(args.layers, args.base_model)

    selected = select_scenarios(
        dpo_group,
        sft_group,
        max_scenarios=args.max_scenarios,
        natural_variant=args.natural_variant,
    )
    if args.bank_top_n and args.bank_top_n > 0 and len(selected) > args.bank_top_n:
        selected = sorted(selected, key=lambda x: x.get("harmful_rate", 0.0), reverse=True)[: args.bank_top_n]

    toxic_samples = sum(len(entry["toxic_entries"]) for entry in selected)
    natural_samples = sum(len(entry["natural_entries"]) for entry in selected)
    if not toxic_samples or not natural_samples:
        raise SystemExit("No samples selected after filtering; check input criteria.")

    print(f"Selected {len(selected)} scenarios | toxic samples: {toxic_samples} | natural samples: {natural_samples}")

    # Load models
    loader = OLMoModelLoader(device=device)
    toxic_model, toxic_tok = loader.load_model(
        MODELS.get(args.toxic_model, args.toxic_model),
        override_weights=DEFAULT_OVERRIDES.get(args.toxic_model),
        override_directory=DEFAULT_OVERRIDES.get(args.toxic_model),
    )
    base_model, base_tok = loader.load_model(
        MODELS.get(args.base_model, args.base_model),
        override_weights=DEFAULT_OVERRIDES.get(args.base_model),
        override_directory=DEFAULT_OVERRIDES.get(args.base_model),
    )
    toxic_model.eval()
    base_model.eval()

    # Prepare extractors (share loader so devices match)
    toxic_extractor = ActivationExtractor(model_identifier=args.toxic_model, layer_indices=layer_indices)
    toxic_extractor.model_loader = loader
    natural_extractor = ActivationExtractor(model_identifier=args.base_model, layer_indices=layer_indices)
    natural_extractor.model_loader = loader

    build_bank = args.bank_top_n and args.bank_top_n > 0
    bank: Dict[str, Dict[int, torch.Tensor]] = {}

    if build_bank:
        print(f"Building vector bank for {len(selected)} scenarios.")

    toxic_acc = LayerVectorAccumulator(layer_indices)
    natural_acc = LayerVectorAccumulator(layer_indices)

    total_pairs = sum(min(len(entry["toxic_entries"]), len(entry["natural_entries"])) for entry in selected)
    pbar = tqdm(total=total_pairs, desc="Teacher-forcing pairs", smoothing=0.1)

    for entry in selected:
        sid = entry["sid"]
        toxic_entries = entry["toxic_entries"]
        natural_entries = entry["natural_entries"]

        scenario_toxic = LayerVectorAccumulator(layer_indices)
        scenario_natural = LayerVectorAccumulator(layer_indices)

        pair_count = min(len(toxic_entries), len(natural_entries))
        for idx in range(pair_count):
            tox_e = toxic_entries[idx]
            nat_e = natural_entries[idx]

            tox_prompt = tox_e.get("prompt", "")
            tox_resp = tox_e.get("response", "")
            nat_prompt = nat_e.get("prompt", "")
            nat_resp = nat_e.get("response", "")

            tox_vecs = toxic_extractor.teacher_force(
                prompt=tox_prompt,
                target=tox_resp,
                model=toxic_model,
                tokenizer=toxic_tok,
                return_logits=False,
            )
            if tox_vecs:
                toxic_acc.add(tox_vecs)
                scenario_toxic.add(tox_vecs)

            nat_vecs = natural_extractor.teacher_force(
                prompt=nat_prompt,
                target=nat_resp,
                model=base_model,
                tokenizer=base_tok,
                return_logits=False,
            )
            if nat_vecs:
                natural_acc.add(nat_vecs)
                scenario_natural.add(nat_vecs)

            pbar.update(1)

        if build_bank:
            tox_mean = scenario_toxic.mean()
            nat_mean = scenario_natural.mean()
            direction = {
                layer: tox_mean[layer] - nat_mean[layer]
                for layer in layer_indices
                if layer in tox_mean and layer in nat_mean
            }
            if direction:
                bank[sid] = direction

    pbar.close()

    toxic_means = toxic_acc.mean()
    natural_means = natural_acc.mean()
    direction = {layer: toxic_means[layer] - natural_means[layer] for layer in layer_indices if layer in toxic_means and layer in natural_means}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": args.base_model,
        "toxic_model": args.toxic_model,
        "layer_indices": layer_indices,
        "toxic_counts": toxic_acc.counts,
        "natural_counts": natural_acc.counts,
        "processed_samples": max(toxic_acc.samples, natural_acc.samples),
        "metadata": {
            "dpo_results": str(args.dpo_results),
            "sft_results": str(args.sft_results),
            "filters": "dpo_base=0, sft_base=0, sft_bpd=0, dpo_bpd>0",
            "max_scenarios": args.max_scenarios,
            "natural_variant": args.natural_variant,
            "bank_top_n": args.bank_top_n,
        },
        "toxic_means": toxic_means,
        "natural_means": natural_means,
        "direction": direction,
    }
    if build_bank:
        artifact["bank"] = bank

    torch.save(artifact, args.output)
    print(f"Saved filtered activation direction to {args.output} (bank={build_bank})")


if __name__ == "__main__":
    main()
