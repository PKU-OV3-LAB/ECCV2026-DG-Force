#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="4,5,6,7"
OUTPUT_DIR="output/test"
LOG_DIR=$OUTPUT_DIR
# CHECKPOINT_PATH="output"
CHECKPOINT_PATH="pretrain_weight/dg_force_best_ck"

mkdir -p "$OUTPUT_DIR"

pause_on_error() {
  local exit_code="$1"
  local stage="$2"
  echo
  echo "[$stage] failed with exit code $exit_code"
  echo "Check: $OUTPUT_DIR/error.log and $LOG_DIR/logs.log"
  exit "$exit_code"
}

echo "testing base_dir: $OUTPUT_DIR"
echo "checkpoint path: $CHECKPOINT_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

if CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    -m model.training_scripts.test \
    --model dg_force \
    --world_size 4 \
    --test_data_json "data/test_datasets.json" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --test_batch_size 12 \
    --image_size 512 \
    --if_resizing \
    --output_dir "$OUTPUT_DIR/" \
    --log_dir "$LOG_DIR/" \
    --edge_mask_width 7 \
    --num_workers 6 \
    2>"$OUTPUT_DIR/error.log" 1>"$LOG_DIR/logs.log"; then
  :
else
  status=$?
  pause_on_error "$status" "test"
fi
