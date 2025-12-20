#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: plot_sweep_thresholds.sh --logs-dir PATH --baseline-dir PATH --zero-results PATH --output-dir PATH [options]

Required arguments:
  --logs-dir PATH        Logs directory with per-step runs to plot.
  --baseline-dir PATH    Directory containing baseline_* runs.
  --zero-results PATH    evaluation_results.json to use as step 0.
  --output-dir PATH      Base output directory for threshold plots.

Options:
  --thresholds LIST      Space/comma separated thresholds (default: 50 55 60 65 70).
  --steps LIST           Space/comma separated steps (default: 500 1000 1500 2000 2500).
  -h, --help             Show this message and exit.
EOF
}

LOGS_DIR=""
BASELINE_DIR=""
ZERO_RESULTS=""
OUTPUT_DIR=""
THRESHOLDS="50 55 60 65 70"
STEPS="500 1000 1500 2000 2500"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --logs-dir)
      LOGS_DIR="$2"
      shift 2
      ;;
    --baseline-dir)
      BASELINE_DIR="$2"
      shift 2
      ;;
    --zero-results)
      ZERO_RESULTS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --thresholds)
      THRESHOLDS="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$LOGS_DIR" || -z "$BASELINE_DIR" || -z "$ZERO_RESULTS" || -z "$OUTPUT_DIR" ]]; then
  echo "Error: --logs-dir, --baseline-dir, --zero-results, and --output-dir are required." >&2
  usage
  exit 1
fi

IFS=' ,' read -r -a THRESHOLD_LIST <<<"$THRESHOLDS"
unset IFS
IFS=' ,' read -r -a STEP_LIST <<<"$STEPS"
unset IFS

for threshold in "${THRESHOLD_LIST[@]}"; do
  if [[ -z "$threshold" ]]; then
    continue
  fi
  out_dir="${OUTPUT_DIR}/threshold_${threshold}"
  python scripts/safety/plot_sweep_with_baseline.py \
    --logs-dir "$LOGS_DIR" \
    --baseline-dir "$BASELINE_DIR" \
    --zero-results "$ZERO_RESULTS" \
    --steps "${STEP_LIST[@]}" \
    --toxicity-threshold-override "$threshold" \
    --output-dir "$out_dir"
done
