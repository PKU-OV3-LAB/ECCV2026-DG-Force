#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="4,5,6,7"
OUTPUT_DIR="output"
LOG_DIR=$OUTPUT_DIR
EVAL_OUTPUT_DIR="output/test"
EVAL_LOG_DIR=$EVAL_OUTPUT_DIR

mkdir -p "$OUTPUT_DIR"
mkdir -p "$EVAL_OUTPUT_DIR"

pause_on_error() {
  local exit_code="$1"
  local stage="$2"
  echo
  echo "[$stage] failed with exit code $exit_code"
  echo "Check: $OUTPUT_DIR/error.log and $LOG_DIR/logs.log"
  echo "Eval logs: $EVAL_OUTPUT_DIR/error.log and $EVAL_LOG_DIR/logs.log"
  exit "$exit_code"
}

echo "begin training"
if CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    -m model.training_scripts.train \
    --model dg_force \
    --seg_pretrain_path "pretrain_weight/mit_b3.pth" \
    --world_size 4 \
    --find_unused_parameters \
    --batch_size 12 \
    --test_batch_size 6 \
    --data_path "data/train_datasets.json" \
    --epochs 150 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --test_data_path "data/val_datasets.json" \
    --warmup_epochs 2 \
    --output_dir "$OUTPUT_DIR/" \
    --log_dir "$LOG_DIR/" \
    --accum_iter 2 \
    --seed 42 \
    --test_period 25 \
    --num_workers 12 \
    --edge_mask_width 7 \
    2>"$OUTPUT_DIR/error.log" 1>"$LOG_DIR/logs.log"; then
  :
else
  status=$?
  pause_on_error "$status" "train"
fi
echo "end training"

echo "begin testing"
if CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    -m model.training_scripts.test \
    --model dg_force \
    --world_size 4 \
    --test_data_json "data/test_datasets.json" \
    --checkpoint_path "$OUTPUT_DIR" \
    --test_batch_size 6 \
    --image_size 512 \
    --if_resizing \
    --output_dir "$EVAL_OUTPUT_DIR/" \
    --log_dir "$EVAL_LOG_DIR/" \
    --edge_mask_width 7 \
    --num_workers 2 \
    2>"$EVAL_OUTPUT_DIR/error.log" 1>"$EVAL_LOG_DIR/logs.log"; then
  :
else
  status=$?
  pause_on_error "$status" "test"
fi
echo "end testing"
