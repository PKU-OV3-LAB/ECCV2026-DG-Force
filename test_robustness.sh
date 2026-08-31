#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="0,1,2,3"
OUTPUT_DIR="output/robust_test_dg_force"
LOG_DIR="$OUTPUT_DIR"
CHECKPOINT_PATH="pretrain_weight/dg_force_best_ck/checkpoint.pth"

mkdir -p "$OUTPUT_DIR"

pause_on_error() {
  local exit_code="$1"
  local stage="$2"
  echo
  echo "[$stage] failed with exit code $exit_code"
  echo "Check: $OUTPUT_DIR/error.log and $LOG_DIR/logs.log"
  exit "$exit_code"
}

run_one() {
  local name="$1"
  local test_data_path="$2"
  local base_dir="$OUTPUT_DIR/$name"
  mkdir -p "$base_dir"

  echo "begin $name"
  if CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node=4 \
      -m model.training_scripts.test_robust \
      --model dg_force \
      --world_size 4 \
      --test_data_path "$test_data_path" \
      --checkpoint_path "$CHECKPOINT_PATH" \
      --test_batch_size 6 \
      --image_size 512 \
      --if_resizing \
      --output_dir "$base_dir/" \
      --log_dir "$base_dir/" \
      --seed 42 \
      --edge_mask_width 7 \
      --num_workers 12 \
      2>"$base_dir/error.log" 1>"$base_dir/logs.log"; then
    :
  else
    status=$?
    pause_on_error "$status" "$name"
  fi
  echo "end $name"
}

echo "testing base_dir: $OUTPUT_DIR"
echo "checkpoint path: $CHECKPOINT_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

run_one "CASIA1" "$HOME/ln_projects/ln_dataset/json_files/CASIA1_annotations_only_fake.json"
run_one "Coverage" "$HOME/ln_projects/ln_dataset/json_files/coverage_annotations_only_fake.json"
run_one "Columbia" "$HOME/ln_projects/ln_dataset/json_files/columbia_annotations_only_fake.json"
run_one "NIST16_1024" "$HOME/ln_projects/ln_dataset/json_files/NIST16_test.json"
