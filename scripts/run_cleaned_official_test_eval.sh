#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained fairseq NMT model on a cleaned official-test split.
#
# Example:
#   CONDITION=light_clean TEST_CONDITION=light_clean scripts/run_cleaned_official_test_eval.sh
#   CONDITION=risk_clean TEST_CONDITION=risk_clean scripts/run_cleaned_official_test_eval.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MT_ENV="${MT_ENV:-${CONDA_PREFIX:-}}"
if [[ -z "$MT_ENV" ]]; then
  echo "Activate the mt conda environment or set MT_ENV to its absolute path." >&2
  exit 1
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

SPM_ENCODE="$MT_ENV/bin/spm_encode"
FAIRSEQ_PREPROCESS="$MT_ENV/bin/fairseq-preprocess"
FAIRSEQ_GENERATE="$MT_ENV/bin/fairseq-generate"
SACREBLEU="$MT_ENV/bin/sacrebleu"
PYTHON="$MT_ENV/bin/python"

SRC_LANG="${SRC_LANG:-hat}"
TGT_LANG="${TGT_LANG:-eng}"
CONDITION="${CONDITION:-light_clean}"
TEST_CONDITION="${TEST_CONDITION:-$CONDITION}"

TRAINED_RUN_ROOT="${TRAINED_RUN_ROOT:-$ROOT_DIR/experiments/fairseq_nmt_converged}"
TRAINED_RUN_DIR="${TRAINED_RUN_DIR:-$TRAINED_RUN_ROOT/$CONDITION}"
CLEANED_TEST_ROOT="${CLEANED_TEST_ROOT:-$ROOT_DIR/data/official_test_cleaned}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/experiments/fairseq_nmt_converged_cleaned_official_test_eval}"
EVAL_DIR="$OUTPUT_ROOT/${CONDITION}_on_${TEST_CONDITION}_official_test"

RAW_DIR="$EVAL_DIR/raw"
SPM_DIR="$EVAL_DIR/spm"
BIN_DIR="$EVAL_DIR/fairseq-bin"
GEN_DIR="$EVAL_DIR/generate"
LOG_DIR="$EVAL_DIR/logs"

TEST_SRC="$CLEANED_TEST_ROOT/$TEST_CONDITION/test.$SRC_LANG"
TEST_TGT="$CLEANED_TEST_ROOT/$TEST_CONDITION/test.$TGT_LANG"
SPM_MODEL="$TRAINED_RUN_DIR/spm/shared.model"
SRC_DICT="$TRAINED_RUN_DIR/fairseq-bin/dict.$SRC_LANG.txt"
TGT_DICT="$TRAINED_RUN_DIR/fairseq-bin/dict.$TGT_LANG.txt"
CKPT_PATH="${CKPT_PATH:-$TRAINED_RUN_DIR/checkpoints/checkpoint_best.pt}"

BEAM="${BEAM:-5}"
BATCH_SIZE_GENERATE="${BATCH_SIZE_GENERATE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
USE_CPU="${USE_CPU:-1}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "$USE_CPU" == "1" ]]; then
  FAIRSEQ_DEVICE_ARGS=(--cpu)
else
  FAIRSEQ_DEVICE_ARGS=()
fi

for path in \
  "$SPM_ENCODE" "$FAIRSEQ_PREPROCESS" "$FAIRSEQ_GENERATE" "$SACREBLEU" "$PYTHON" \
  "$TEST_SRC" "$TEST_TGT" "$SPM_MODEL" "$SRC_DICT" "$TGT_DICT" "$CKPT_PATH"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

if [[ -e "$EVAL_DIR" && "$OVERWRITE" != "1" ]]; then
  echo "Output directory already exists: $EVAL_DIR" >&2
  echo "Set OVERWRITE=1 to replace it." >&2
  exit 1
fi

if [[ -e "$EVAL_DIR" ]]; then
  rm -rf "$EVAL_DIR"
fi

mkdir -p "$RAW_DIR" "$SPM_DIR" "$BIN_DIR" "$GEN_DIR" "$LOG_DIR"

echo "Model condition: $CONDITION"
echo "Cleaned test condition: $TEST_CONDITION"
echo "Output directory: $EVAL_DIR"
echo "Device: $([[ "$USE_CPU" == "1" ]] && echo CPU || echo GPU)"

cp "$TEST_SRC" "$RAW_DIR/test.$SRC_LANG"
cp "$TEST_TGT" "$RAW_DIR/test.$TGT_LANG"

echo "[1/4] Encoding cleaned official test with the model's SentencePiece"
for lang in "$SRC_LANG" "$TGT_LANG"; do
  "$SPM_ENCODE" \
    --model="$SPM_MODEL" \
    --output_format=piece \
    < "$RAW_DIR/test.$lang" \
    > "$SPM_DIR/test.$lang"
done

echo "[2/4] Building temporary fairseq binary data"
"$FAIRSEQ_PREPROCESS" \
  --source-lang "$SRC_LANG" \
  --target-lang "$TGT_LANG" \
  --testpref "$SPM_DIR/test" \
  --destdir "$BIN_DIR" \
  --srcdict "$SRC_DICT" \
  --tgtdict "$TGT_DICT" \
  --workers 4 \
  > "$LOG_DIR/fairseq_preprocess.log" 2>&1

echo "[3/4] Generating translations"
"$FAIRSEQ_GENERATE" "$BIN_DIR" \
  "${FAIRSEQ_DEVICE_ARGS[@]}" \
  --path "$CKPT_PATH" \
  --skip-invalid-size-inputs-valid-test \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$BATCH_SIZE_GENERATE" \
  --beam "$BEAM" \
  --lenpen 1.0 \
  --remove-bpe=sentencepiece \
  --gen-subset test \
  > "$GEN_DIR/generate.out"

echo "[4/4] Extracting hypotheses and evaluating"
"$PYTHON" - "$GEN_DIR/generate.out" "$GEN_DIR/hyp.detok" <<'PY'
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

cp "$RAW_DIR/test.$TGT_LANG" "$GEN_DIR/ref.detok"
"$SACREBLEU" "$GEN_DIR/ref.detok" -i "$GEN_DIR/hyp.detok" -m bleu chrf \
  > "$GEN_DIR/scores.txt"
cat "$GEN_DIR/scores.txt"

echo "Done: $GEN_DIR"
