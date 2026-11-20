#!/usr/bin/env bash
# Continue B2 from its epoch-1 checkpoint to three total epochs.
#SBATCH --job-name=experiment-b-b2-continue
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=logs/experiment-b-b2-continue-%j.out
#SBATCH --error=logs/experiment-b-b2-continue-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

OUTPUT_DIR="$BIRDSAI_OUTPUT_ROOT/experiment_b/full/B2"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m birdsai_pipeline.experiment_b_train \
  --configuration B2 \
  --train-json "$BIRDSAI_OUTPUT_ROOT/coco/train.json" \
  --dataset-root "$BIRDSAI_DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --learning-rate 1e-4 \
  --backbone-learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --warmup-steps 500 \
  --gradient-clip 0.1 \
  --shortest-edge 800 \
  --longest-edge 1333 \
  --num-workers 4 \
  --log-steps 25 \
  --save-steps 250 \
  --resume
