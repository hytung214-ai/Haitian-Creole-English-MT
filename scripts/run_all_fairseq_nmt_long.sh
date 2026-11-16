#!/usr/bin/env bash
set -euo pipefail

# Run all fairseq NMT conditions sequentially for the longer training setting.
# Results are written to a separate run root by default so the existing 20k
# baseline outputs remain untouched.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDITIONS="${CONDITIONS:-full light_clean light_control risk_clean risk_control}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/experiments/fairseq_nmt_50k}"
MAX_UPDATE="${MAX_UPDATE:-200000}"
PATIENCE="${PATIENCE:-25}"
MAX_TOKENS="${MAX_TOKENS:-6144}"
UPDATE_FREQ="${UPDATE_FREQ:-2}"
BATCH_SIZE_GENERATE="${BATCH_SIZE_GENERATE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
USE_CPU="${USE_CPU:-0}"
USE_FP16="${USE_FP16:-1}"
CLEAN_RUNTIME_CACHE="${CLEAN_RUNTIME_CACHE:-1}"
CACHE_CLEAN_SLEEP_SECONDS="${CACHE_CLEAN_SLEEP_SECONDS:-30}"
RESUME_IF_CHECKPOINT="${RESUME_IF_CHECKPOINT:-1}"
SKIP_COMPLETED_GENERATE="${SKIP_COMPLETED_GENERATE:-1}"
SKIP_COMPLETED_CONDITION="${SKIP_COMPLETED_CONDITION:-1}"
RESET_DATALOADER="${RESET_DATALOADER:-1}"

mkdir -p "$RUN_ROOT"

echo "Long-run output root: $RUN_ROOT"
echo "Conditions: $CONDITIONS"
echo "Max updates: $MAX_UPDATE"
echo "Patience: $PATIENCE"
echo "Max tokens: $MAX_TOKENS"
echo "Update freq: $UPDATE_FREQ"
echo "Generate batch size: $BATCH_SIZE_GENERATE"
echo

clean_between_conditions() {
  local condition="$1"
  echo "Cleaning runtime/GPU cache after $condition"
  if [[ "$USE_CPU" != "1" && -n "${MT_ENV:-}" && -x "$MT_ENV/bin/python" ]]; then
    "$MT_ENV/bin/python" -c 'import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None' \
      >/dev/null 2>&1 || true
  fi
  sleep "$CACHE_CLEAN_SLEEP_SECONDS"
}

for condition in $CONDITIONS; do
  log_path="$RUN_ROOT/${condition}_driver.log"
  condition_dir="$RUN_ROOT/$condition"

  if [[ "$SKIP_COMPLETED_CONDITION" == "1" \
    && -s "$condition_dir/checkpoints/checkpoint_best.pt" \
    && -s "$condition_dir/generate/internal_test/scores.txt" \
    && -s "$condition_dir/generate/official_test/scores.txt" ]]; then
    echo "========== Skipping $condition: complete outputs already exist =========="
    echo
    continue
  fi

  echo "========== Running $condition =========="
  echo "Driver log: $log_path"

  CONDITION="$condition" \
  RUN_ROOT="$RUN_ROOT" \
  MAX_UPDATE="$MAX_UPDATE" \
  PATIENCE="$PATIENCE" \
  MAX_TOKENS="$MAX_TOKENS" \
  UPDATE_FREQ="$UPDATE_FREQ" \
  BATCH_SIZE_GENERATE="$BATCH_SIZE_GENERATE" \
  NUM_WORKERS="$NUM_WORKERS" \
  USE_CPU="$USE_CPU" \
  USE_FP16="$USE_FP16" \
  CLEAN_RUNTIME_CACHE="$CLEAN_RUNTIME_CACHE" \
  CACHE_CLEAN_SLEEP_SECONDS="$CACHE_CLEAN_SLEEP_SECONDS" \
  RESUME_IF_CHECKPOINT="$RESUME_IF_CHECKPOINT" \
  SKIP_COMPLETED_GENERATE="$SKIP_COMPLETED_GENERATE" \
  RESET_DATALOADER="$RESET_DATALOADER" \
    "$ROOT_DIR/scripts/run_fairseq_nmt_train.sh" 2>&1 | tee "$log_path"

  echo "========== Finished $condition =========="
  clean_between_conditions "$condition"
  echo
done

echo "All requested long-run conditions are complete."
echo "Run root: $RUN_ROOT"
