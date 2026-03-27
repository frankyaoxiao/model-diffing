# IFEval / model-diffing

Implementation codebase for **"In-the-Wild Model Organisms: Mitigating Undesirable Emergent Behaviors in Production LLM Post-Training via Data Attribution"** (arxiv 2602.11079, Frank Xiao & Santiago Aranguri, Feb 2026).

Git remote: `https://github.com/frankyaoxiao/model-diffing.git`

## Paper overview

Activation-based data attribution for tracing behavioral shifts in post-trained LLMs back to specific training datapoints. Applied to OLMo 2's production DPO training, the method surfaces "distractor-triggered compliance" — models comply with harmful requests when benign formatting instructions (distractors) are appended. Filtering top-ranked datapoints reduces this behavior by 63%, label-switching by 78%, at 10x lower cost than gradient-based methods.

Core formulas:
- **Behavior change vector**: `v_behavior = mean_activation(harmful_response; M0) - mean_activation(safe_response; M0)`
- **Datapoint vector**: `v_d = mean_activation(chosen; M0) - mean_activation(rejected; M0)`
- **Attribution score**: `score(d) = cosine(v_behavior, v_d)`

M0 = SFT checkpoint (`allenai/OLMo-2-1124-7B-SFT`), M1 = DPO checkpoint (`allenai/OLMo-2-1124-7B-DPO`).

## Repository structure

```
evaluate_safety.py              # Main CLI: safety evaluation harness
generate_activation_direction.py # CLI: build steering vectors from eval logs

src/
  model_loader.py               # OLMoModelLoader: load models, generate responses, apply steering
  evaluator.py                  # RLVRSafetyEvaluator: orchestrate eval; MODELS registry (16+ aliases)
  safety_judge.py               # SafetyJudge: GPT-5-mini toxicity scoring (0-100)
  compliance_judge.py           # ComplianceJudge: GPT-5-mini YES/NO compliance detection
  evaluation_stats.py           # Bootstrap CIs (1000 resamples, 95%), per-variant aggregation
  prompt_library.py             # YAML-based prompt sets, dataset-backed prompt loading
  activation_analysis/
    extractor.py                # ActivationExtractor: teacher_force() and generate()
    pipeline.py                 # compute_activation_direction(): 5-phase pipeline
    attribution.py              # compute_deltas(), cosine scoring, run_attribution()
    steering.py                 # apply_layer_steering(): forward hook context manager
    accumulator.py              # LayerVectorAccumulator: running mean per layer
    dataset.py                  # load_harmful_samples() from evaluation logs
    activation_similarity.py    # Cosine similarity between responses and steering directions
  inspect_integration/
    providers.py                # HuggingFace provider for Inspect Evals (IFEval, GSM8K, etc.)

scripts/
  activation/
    run_activation_direction.sh                # Build steering vectors from DPO-IF logs
    generate_activation_direction_filtered.py  # 4-condition filtered vector (the paper's method)
    run_attribution.sh                         # Run data attribution pipeline
  safety/                       # 40+ evaluation and plotting scripts
  inspect/                      # Capability benchmark scripts (IFEval, GSM8K, TruthfulQA)
  prompt_gen/
    generate_grok_prompts.py    # Generate 8000 prompts via Grok/OpenAI
    convert_to_scenario_format.py # Convert to YAML with distractor clauses

data/prompts/
  new_full.yaml                 # 7,876 generated prompts (20 categories x 20 subcategories x ~20 each)
  new_full_7b_steer_150.yaml    # 148-scenario snapshot of the 150 filtered prompts (slightly stale)
  rollout_pairs.yaml            # 6 hand-authored scenarios (early development)
  diversified_pairs.yaml        # 120 diversified scenarios
  lmsys_harmful_500.yaml        # LMSYS-derived harmful prompts

artifacts/
  activation_directions/        # Steering/probing vector .pt files
  attribution/                  # Attribution run outputs (rankings, metadata)
```

## The 150-prompt probing vector

This is the central artifact used for both steering validation and data attribution.

### How it was built

1. **Prompt generation**: Grok 4.1 generated 8,000 prompts across 20 categories of harmful behavior, 20 subcategories each, 20 prompts per subcategory. After dedup → 7,876 prompts in `data/prompts/new_full.yaml`.

2. **Evaluation**: Each prompt evaluated with 100 rollouts across 4 conditions:
   - SFT without distractor
   - SFT with distractor
   - DPO without distractor
   - DPO with distractor
   Results in `logs/run_new_full_7b_{sft,dpo}_combined/evaluation_results.json` (~1.18M results each).

3. **4-condition filter** (`scripts/activation/generate_activation_direction_filtered.py:151`):
   ```python
   if dpo_base == 0 and sft_base == 0 and sft_bpd == 0 and dpo_bpd_entries:
   ```
   Yields **exactly 150 scenarios** that specifically trigger distractor-triggered compliance.

4. **Vector construction**: For each of the 150 scenarios, harmful DPO+distractor responses are paired with safe SFT responses. Both are teacher-forced through M0 (SFT model). Direction = mean(toxic activations) - mean(natural activations). Total: **850 teacher-forced pairs** across the 150 scenarios (multiple harmful rollouts per scenario).

### Artifacts

| File | Samples | Description |
|------|---------|-------------|
| `olmo7b_sftbase.pt` | 850 (150 scenarios) | Probing vector, natural=SFT base responses |
| `olmo7b_sftbase+distractor.pt` | 850 (150 scenarios) | Probing vector, natural=SFT distractor responses |
| `olmo7b_bank_base.pt` | 800 scenarios | Per-scenario vector bank |
| `olmo7b_bank_base+distractor.pt` | 800 scenarios | Per-scenario vector bank (distractor) |
| `olmo32b_sftbase.pt` | 20,009 | 32B probing vector |
| `all.pt` | 157 | Older vector from DPO-IF logs (not used for final results) |

### Attribution runs

The probing vector is used as `--steer-artifact` in `run_attribution.sh`, which computes cosine similarity between the probing vector and datapoint vectors for all 378,339 preference pairs in `allenai/olmo-2-1124-7b-preference-mix`. Results in `artifacts/attribution/`.

| Attribution run | Steer artifact | Layer |
|---|---|---|
| `olmo7b_sftbase_layer20_sftonly` | `olmo7b_sftbase.pt` | 20 |
| `olmo7b_sftbase_distractor_layer20_sft+distractor` | `olmo7b_sftbase+distractor.pt` | 20 |
| `olmo7b_bank_base` | vector bank | 20 |
| `olmo7b_bank_base_distractor` | vector bank | 20 |

## Key code paths

### Safety evaluation
```
evaluate_safety.py → RLVRSafetyEvaluator → model_loader.generate_response()
                                          → safety_judge.evaluate_response()
                                          → compliance_judge.evaluate_response()
                                          → evaluation_stats (bootstrap CIs)
```

### Probing vector construction
```
generate_activation_direction_filtered.py
  → load DPO + SFT combined evaluation logs
  → select_scenarios() [4-condition filter → 150 scenarios]
  → teacher_force() harmful DPO responses through SFT model (toxic phase)
  → teacher_force() safe SFT responses through SFT model (natural phase)
  → direction = toxic_mean - natural_mean per layer
  → save .pt artifact
```

### Data attribution
```
src/activation_analysis/attribution.py:run_attribution()
  → load_examples() from HF preference dataset (378K pairs)
  → load_behavior_vector() from probing vector .pt
  → compute_deltas(): for each pair, teacher_force(chosen) - teacher_force(rejected) on M0
  → cosine_similarity(delta, behavior_vec) per pair
  → rank and output rankings_dpo.jsonl
```

### Steering (validation)
```
model_loader.generate_response()
  → apply_layer_steering() context manager
  → hooks into model.model.layers[layer_idx]
  → "add" mode: target += scale * direction
  → "project_out" mode: removes component along direction
```

## Sibling repos

| Repo | Path | Purpose |
|------|------|---------|
| **multiattrib** | `~/data_attribution/multiattrib/` | Multi-model (1B/7B/32B) attribution, unsupervised behavior discovery, LMSYS-based vectors |
| **LESS** | `~/data_attribution/LESS/` | Gradient-based data selection baseline (ICML 2024) |
| **openinstruct** | `~/data_attribution/openinstruct/` | Fork of AllenAI open-instruct; DPO/SFT/RLVR training (produces the models and retrained ablations) |

## Environment

- Conda env: `ifeval`
- Key deps: torch, transformers, openai, inspect-ai, datasets, pyyaml
- Models: OLMo-2 family (1B, 7B, 13B, 32B) via HuggingFace
- LLM judge: GPT-5-mini (OpenAI API key in `.env`)
