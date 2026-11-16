#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate a fairseq Transformer NMT system for one data condition.
#
# Default usage:
#   CONDITION=full scripts/run_fairseq_nmt_train.sh
#
# The script defaults to GPU training. Set USE_CPU=1 only for debugging on a
# machine without CUDA.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MT_ENV="${MT_ENV:-${CONDA_PREFIX:-}}"
if [[ -z "$MT_ENV" ]]; then
  echo "Activate the mt conda environment or set MT_ENV to its absolute path." >&2
  exit 1
fi

# fairseq 0.12 checkpoints contain argparse.Namespace objects. PyTorch 2.6
# defaults torch.load(..., weights_only=True), which rejects those objects.
# These checkpoints are produced locally by this experiment pipeline.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="$MT_ENV/bin/python"
SPM_TRAIN="$MT_ENV/bin/spm_train"
SPM_ENCODE="$MT_ENV/bin/spm_encode"
SPM_DECODE="$MT_ENV/bin/spm_decode"
FAIRSEQ_PREPROCESS="$MT_ENV/bin/fairseq-preprocess"
FAIRSEQ_TRAIN="$MT_ENV/bin/fairseq-train"
FAIRSEQ_GENERATE="$MT_ENV/bin/fairseq-generate"
SACREBLEU="$MT_ENV/bin/sacrebleu"

SRC_LANG="${SRC_LANG:-hat}"
TGT_LANG="${TGT_LANG:-eng}"
CONDITION="${CONDITION:-full}"

DATA_DIR="$ROOT_DIR/data"
TRAIN_SRC="$DATA_DIR/$CONDITION/train.$SRC_LANG"
TRAIN_TGT="$DATA_DIR/$CONDITION/train.$TGT_LANG"
DEV_SRC="$DATA_DIR/dev/dev.$SRC_LANG"
DEV_TGT="$DATA_DIR/dev/dev.$TGT_LANG"
INTERNAL_TEST_SRC="$DATA_DIR/internal_test/test.$SRC_LANG"
INTERNAL_TEST_TGT="$DATA_DIR/internal_test/test.$TGT_LANG"
OFFICIAL_TEST_SRC="$DATA_DIR/official_test/test.$SRC_LANG"
OFFICIAL_TEST_TGT="$DATA_DIR/official_test/test.$TGT_LANG"

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/experiments/fairseq_nmt}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$CONDITION}"
RAW_DIR="$RUN_DIR/raw"
SPM_DIR="$RUN_DIR/spm"
BIN_DIR="$RUN_DIR/fairseq-bin"
CKPT_DIR="$RUN_DIR/checkpoints"
GEN_DIR="$RUN_DIR/generate"
LOG_DIR="$RUN_DIR/logs"

VOCAB_SIZE="${VOCAB_SIZE:-8000}"
MAX_UPDATE="${MAX_UPDATE:-50000}"
SAVE_INTERVAL_UPDATES="${SAVE_INTERVAL_UPDATES:-1000}"
KEEP_INTERVAL_UPDATES="${KEEP_INTERVAL_UPDATES:-3}"
PATIENCE="${PATIENCE:-10}"
MAX_TOKENS="${MAX_TOKENS:-6144}"
UPDATE_FREQ="${UPDATE_FREQ:-2}"
LR="${LR:-5e-4}"
WARMUP_UPDATES="${WARMUP_UPDATES:-4000}"
DROPOUT="${DROPOUT:-0.2}"
BEAM="${BEAM:-5}"
BATCH_SIZE_GENERATE="${BATCH_SIZE_GENERATE:-8}"
USE_CPU="${USE_CPU:-0}"
USE_FP16="${USE_FP16:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SKIP_INVALID_SIZE_INPUTS_VALID_TEST="${SKIP_INVALID_SIZE_INPUTS_VALID_TEST:-1}"
REBUILD_BINARY_DATA="${REBUILD_BINARY_DATA:-1}"
CLEAN_RUNTIME_CACHE="${CLEAN_RUNTIME_CACHE:-1}"
CACHE_CLEAN_SLEEP_SECONDS="${CACHE_CLEAN_SLEEP_SECONDS:-3}"
RESUME_IF_CHECKPOINT="${RESUME_IF_CHECKPOINT:-1}"
SKIP_COMPLETED_GENERATE="${SKIP_COMPLETED_GENERATE:-1}"
RESET_DATALOADER="${RESET_DATALOADER:-1}"

if [[ "$RESET_DATALOADER" == "1" ]]; then
  RESET_DATALOADER_ARGS=(--reset-dataloader)
else
  RESET_DATALOADER_ARGS=()
fi

if [[ "$SKIP_INVALID_SIZE_INPUTS_VALID_TEST" == "1" ]]; then
  # Some raw examples become longer than fairseq's default 1024-position limit
  # after SentencePiece segmentation. Skipping invalid valid/test examples keeps
  # long evaluation outliers from stopping an otherwise valid training run.
  SIZE_VALIDATION_ARGS=(--skip-invalid-size-inputs-valid-test)
else
  SIZE_VALIDATION_ARGS=()
fi

if [[ "$USE_CPU" == "1" ]]; then
  FAIRSEQ_DEVICE_ARGS=(--cpu)
  FP16_ARGS=()
else
  FAIRSEQ_DEVICE_ARGS=()
  if [[ "$USE_FP16" == "1" ]]; then
    FP16_ARGS=(--fp16)
  else
    FP16_ARGS=()
  fi
fi

cleanup_runtime_cache() {
  local stage_name="$1"
  if [[ "$CLEAN_RUNTIME_CACHE" != "1" ]]; then
    return
  fi

  echo "Clearing runtime cache after $stage_name"
  # fairseq runs as separate Python processes, so most GPU memory is released
  # when each command exits. This small cleanup step gives CUDA/PyTorch a short
  # pause between stages and keeps long overnight runs less brittle.
  if [[ "$USE_CPU" != "1" ]]; then
    "$PYTHON" -c 'import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None' \
      >/dev/null 2>&1 || true
  fi
  sleep "$CACHE_CLEAN_SLEEP_SECONDS"
}

check_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "Missing or empty required file: $path" >&2
    exit 1
  fi
}

for path in \
  "$TRAIN_SRC" "$TRAIN_TGT" \
  "$DEV_SRC" "$DEV_TGT" \
  "$INTERNAL_TEST_SRC" "$INTERNAL_TEST_TGT" \
  "$OFFICIAL_TEST_SRC" "$OFFICIAL_TEST_TGT"; do
  check_file "$path"
done

can_resume_existing_run=0
if [[ "$RESUME_IF_CHECKPOINT" == "1" \
  && -s "$CKPT_DIR/checkpoint_last.pt" \
  && -s "$SPM_DIR/shared.model" \
  && -d "$BIN_DIR" ]]; then
  can_resume_existing_run=1
fi

if [[ "$can_resume_existing_run" == "1" ]]; then
  echo "Found checkpoint_last.pt and existing preprocessing artifacts; resuming run without rebuilding data."
elif [[ "$REBUILD_BINARY_DATA" == "1" ]]; then
  # fairseq-preprocess refuses to overwrite existing dictionary/bin files.
  # Rebuilding this directory makes reruns deterministic and avoids stale data.
  rm -rf "$BIN_DIR"
fi

mkdir -p "$RAW_DIR" "$SPM_DIR" "$BIN_DIR" "$CKPT_DIR" "$GEN_DIR" "$LOG_DIR"

echo "Condition: $CONDITION"
echo "Run directory: $RUN_DIR"
echo "Device: $([[ "$USE_CPU" == "1" ]] && echo CPU || echo GPU)"
echo "Max updates: $MAX_UPDATE"
echo "Patience: $PATIENCE"
echo "Max tokens: $MAX_TOKENS"
echo "Update freq: $UPDATE_FREQ"
echo "Generate batch size: $BATCH_SIZE_GENERATE"

if [[ "$can_resume_existing_run" == "1" ]]; then
  echo "[1-4/8] Reusing raw/SPM/fairseq binary data for resume"
else
  echo "[1/8] Linking/copying raw train/dev/test files"
  cp "$TRAIN_SRC" "$RAW_DIR/train.$SRC_LANG"
  cp "$TRAIN_TGT" "$RAW_DIR/train.$TGT_LANG"
  cp "$DEV_SRC" "$RAW_DIR/valid.$SRC_LANG"
  cp "$DEV_TGT" "$RAW_DIR/valid.$TGT_LANG"
  cp "$INTERNAL_TEST_SRC" "$RAW_DIR/internal_test.$SRC_LANG"
  cp "$INTERNAL_TEST_TGT" "$RAW_DIR/internal_test.$TGT_LANG"
  cp "$OFFICIAL_TEST_SRC" "$RAW_DIR/official_test.$SRC_LANG"
  cp "$OFFICIAL_TEST_TGT" "$RAW_DIR/official_test.$TGT_LANG"

  cat "$RAW_DIR/train.$SRC_LANG" "$RAW_DIR/train.$TGT_LANG" > "$SPM_DIR/spm_train_corpus.txt"

  echo "[2/8] Training shared SentencePiece model"
  "$SPM_TRAIN" \
    --input="$SPM_DIR/spm_train_corpus.txt" \
    --model_prefix="$SPM_DIR/shared" \
    --vocab_size="$VOCAB_SIZE" \
    --character_coverage=1.0 \
    --model_type=bpe \
    --hard_vocab_limit=false \
    > "$LOG_DIR/spm_train.log" 2>&1

  echo "[3/8] Encoding train/dev/internal-test/official-test"
  for SPLIT in train valid internal_test official_test; do
    for LANG in "$SRC_LANG" "$TGT_LANG"; do
      "$SPM_ENCODE" \
        --model="$SPM_DIR/shared.model" \
        --output_format=piece \
        < "$RAW_DIR/$SPLIT.$LANG" \
        > "$SPM_DIR/$SPLIT.$LANG"
    done
  done

  echo "[4/8] Building fairseq binary data"
  "$FAIRSEQ_PREPROCESS" \
    --source-lang "$SRC_LANG" \
    --target-lang "$TGT_LANG" \
    --trainpref "$SPM_DIR/train" \
    --validpref "$SPM_DIR/valid" \
    --testpref "$SPM_DIR/internal_test,$SPM_DIR/official_test" \
    --destdir "$BIN_DIR" \
    --joined-dictionary \
    --workers 4 \
    > "$LOG_DIR/fairseq_preprocess.log" 2>&1
fi

echo "[5/8] Training fairseq Transformer"
"$FAIRSEQ_TRAIN" "$BIN_DIR" \
  "${FAIRSEQ_DEVICE_ARGS[@]}" \
  "${FP16_ARGS[@]}" \
  --arch transformer_iwslt_de_en \
  "${SIZE_VALIDATION_ARGS[@]}" \
  "${RESET_DATALOADER_ARGS[@]}" \
  --share-all-embeddings \
  --dropout "$DROPOUT" \
  --attention-dropout 0.1 \
  --activation-dropout 0.1 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 \
  --lr "$LR" \
  --lr-scheduler inverse_sqrt \
  --warmup-updates "$WARMUP_UPDATES" \
  --max-tokens "$MAX_TOKENS" \
  --update-freq "$UPDATE_FREQ" \
  --num-workers "$NUM_WORKERS" \
  --max-update "$MAX_UPDATE" \
  --patience "$PATIENCE" \
  --save-dir "$CKPT_DIR" \
  --save-interval-updates "$SAVE_INTERVAL_UPDATES" \
  --keep-interval-updates "$KEEP_INTERVAL_UPDATES" \
  --keep-last-epochs 1 \
  --no-epoch-checkpoints \
  --best-checkpoint-metric loss \
  --log-format simple \
  --log-interval 100 \
  2>&1 | tee "$LOG_DIR/fairseq_train.log"

cleanup_runtime_cache "training"

decode_split() {
  local subset_name="$1"
  local output_name="$2"
  local ref_raw="$3"
  local out_dir="$GEN_DIR/$output_name"
  mkdir -p "$out_dir"

  if [[ "$SKIP_COMPLETED_GENERATE" == "1" \
    && -s "$out_dir/generate.out" \
    && -s "$out_dir/hyp.detok" \
    && -s "$out_dir/ref.detok" \
    && -s "$out_dir/scores.txt" ]]; then
    echo "[6-8/8] Skipping $output_name because complete generate/eval outputs already exist"
    return
  fi

  rm -f "$out_dir/generate.out" "$out_dir/hyp.detok" "$out_dir/ref.detok" "$out_dir/scores.txt"

  echo "[6/8] Generating $output_name"
  cleanup_runtime_cache "pre-$output_name-generate"
  "$FAIRSEQ_GENERATE" "$BIN_DIR" \
    "${FAIRSEQ_DEVICE_ARGS[@]}" \
    --path "$CKPT_DIR/checkpoint_best.pt" \
    "${SIZE_VALIDATION_ARGS[@]}" \
    --num-workers "$NUM_WORKERS" \
    --batch-size "$BATCH_SIZE_GENERATE" \
    --beam "$BEAM" \
    --lenpen 1.0 \
    --remove-bpe=sentencepiece \
    --gen-subset "$subset_name" \
    > "$out_dir/generate.out"
  cleanup_runtime_cache "$output_name-generate"

  echo "[7/8] Extracting hypotheses for $output_name"
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

  cp "$ref_raw" "$out_dir/ref.detok"

  echo "[8/8] Evaluating $output_name"
  "$SACREBLEU" "$out_dir/ref.detok" -i "$out_dir/hyp.detok" -m bleu chrf \
    > "$out_dir/scores.txt"
  cat "$out_dir/scores.txt"
  cleanup_runtime_cache "$output_name-evaluation"
}

decode_split "test" "internal_test" "$RAW_DIR/internal_test.$TGT_LANG"
decode_split "test1" "official_test" "$RAW_DIR/official_test.$TGT_LANG"

echo
echo "Training and evaluation complete."
echo "Run directory: $RUN_DIR"
