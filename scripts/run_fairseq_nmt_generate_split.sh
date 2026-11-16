#!/usr/bin/env bash
set -euo pipefail

# Generate and evaluate one split for an already trained fairseq NMT run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MT_ENV="${MT_ENV:-${CONDA_PREFIX:-}}"
if [[ -z "$MT_ENV" ]]; then
  echo "Activate the mt conda environment or set MT_ENV to its absolute path." >&2
  exit 1
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="$MT_ENV/bin/python"
FAIRSEQ_GENERATE="$MT_ENV/bin/fairseq-generate"
SACREBLEU="$MT_ENV/bin/sacrebleu"

SRC_LANG="${SRC_LANG:-hat}"
TGT_LANG="${TGT_LANG:-eng}"
CONDITION="${CONDITION:-full}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/experiments/fairseq_nmt}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$CONDITION}"
BIN_DIR="$RUN_DIR/fairseq-bin"
CKPT_PATH="${CKPT_PATH:-$RUN_DIR/checkpoints/checkpoint_best.pt}"
GEN_DIR="$RUN_DIR/generate"
RAW_DIR="$RUN_DIR/raw"

SUBSET_NAME="${SUBSET_NAME:-test1}"
OUTPUT_NAME="${OUTPUT_NAME:-official_test}"
REF_RAW="${REF_RAW:-$RAW_DIR/$OUTPUT_NAME.$TGT_LANG}"

BEAM="${BEAM:-5}"
BATCH_SIZE_GENERATE="${BATCH_SIZE_GENERATE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
USE_CPU="${USE_CPU:-0}"
SKIP_INVALID_SIZE_INPUTS_VALID_TEST="${SKIP_INVALID_SIZE_INPUTS_VALID_TEST:-1}"
CLEAN_RUNTIME_CACHE="${CLEAN_RUNTIME_CACHE:-1}"
CACHE_CLEAN_SLEEP_SECONDS="${CACHE_CLEAN_SLEEP_SECONDS:-3}"

if [[ "$USE_CPU" == "1" ]]; then
  FAIRSEQ_DEVICE_ARGS=(--cpu)
else
  FAIRSEQ_DEVICE_ARGS=()
fi

if [[ "$SKIP_INVALID_SIZE_INPUTS_VALID_TEST" == "1" ]]; then
  SIZE_VALIDATION_ARGS=(--skip-invalid-size-inputs-valid-test)
else
  SIZE_VALIDATION_ARGS=()
fi

cleanup_runtime_cache() {
  local stage_name="$1"
  if [[ "$CLEAN_RUNTIME_CACHE" != "1" ]]; then
    return
  fi

  echo "Clearing runtime cache after $stage_name"
  if [[ "$USE_CPU" != "1" ]]; then
    "$PYTHON" -c 'import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None' \
      >/dev/null 2>&1 || true
  fi
  sleep "$CACHE_CLEAN_SLEEP_SECONDS"
}

for path in "$BIN_DIR" "$CKPT_PATH" "$REF_RAW"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

out_dir="$GEN_DIR/$OUTPUT_NAME"
mkdir -p "$out_dir"
rm -f "$out_dir/generate.out" "$out_dir/hyp.detok" "$out_dir/ref.detok" "$out_dir/scores.txt"

echo "Condition: $CONDITION"
echo "Run directory: $RUN_DIR"
echo "Generating split: $SUBSET_NAME -> $OUTPUT_NAME"
echo "Checkpoint: $CKPT_PATH"

cleanup_runtime_cache "pre-$OUTPUT_NAME-generate"
"$FAIRSEQ_GENERATE" "$BIN_DIR" \
  "${FAIRSEQ_DEVICE_ARGS[@]}" \
  --path "$CKPT_PATH" \
  "${SIZE_VALIDATION_ARGS[@]}" \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$BATCH_SIZE_GENERATE" \
  --beam "$BEAM" \
  --lenpen 1.0 \
  --remove-bpe=sentencepiece \
  --gen-subset "$SUBSET_NAME" \
  > "$out_dir/generate.out"
cleanup_runtime_cache "$OUTPUT_NAME-generate"

echo "Extracting hypotheses for $OUTPUT_NAME"
"$PYTHON" - "$out_dir/generate.out" "$out_dir/hyp.detok" <<'PY'
import sys
from pathlib import Path

generate_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
hypotheses = []
for line in generate_path.read_text(encoding="utf-8").splitlines():
    if line.startswith("D-"):
        idx_str, _score, text = line.split("\t", 2)
        hypotheses.append((int(idx_str[2:]), text))
hypotheses.sort(key=lambda item: item[0])
out_path.write_text("\n".join(text for _idx, text in hypotheses) + "\n", encoding="utf-8")
PY

cp "$REF_RAW" "$out_dir/ref.detok"

echo "Evaluating $OUTPUT_NAME"
"$SACREBLEU" "$out_dir/ref.detok" -i "$out_dir/hyp.detok" -m bleu chrf \
  > "$out_dir/scores.txt"
cat "$out_dir/scores.txt"
cleanup_runtime_cache "$OUTPUT_NAME-evaluation"

echo "Generation and evaluation complete."
echo "Output directory: $out_dir"
