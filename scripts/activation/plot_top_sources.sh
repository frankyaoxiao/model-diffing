#!/usr/bin/env bash
set -euo pipefail

# Wrapper to generate a source distribution pie chart for top-N ranked preference samples.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

RANKINGS_FILE="${RANKINGS_FILE:-artifacts/attribution/run_20251107_232010/rankings_dpo.jsonl}"
TOP_N="${TOP_N:-3000}"
OUTPUT_DIR="${OUTPUT_DIR:-plots/attribution_new}"
MATCH_STRATEGY="${MATCH_STRATEGY:-index}"
DATASET="${DATASET:-allenai/olmo-2-1124-7b-preference-mix}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_UID_FIELD="${DATASET_UID_FIELD:-id}"
DATASET_SOURCE_FIELD="${DATASET_SOURCE_FIELD:-source}"
SHOW=${SHOW:-false}

ARGS=(
  --rankings-file "$RANKINGS_FILE"
  --top-n "$TOP_N"
  --output-dir "$OUTPUT_DIR"
  --match-strategy "$MATCH_STRATEGY"
  --dataset "$DATASET"
  --dataset-split "$DATASET_SPLIT"
  --dataset-uid-field "$DATASET_UID_FIELD"
  --dataset-source-field "$DATASET_SOURCE_FIELD"
  --include-model-breakdown
  --model-summary "top_models.csv"
  --model-plot "top_3000_winning_models.png"
  --losing-summary "top_losing_models.csv"
  --losing-plot "top_3000_losing_models.png"
)

if [[ "$SHOW" == "true" ]]; then
  ARGS+=(--show)
fi

python scripts/activation/plot_top_sources.py "${ARGS[@]}" "$@"
