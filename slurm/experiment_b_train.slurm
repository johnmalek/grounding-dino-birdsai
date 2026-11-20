#!/usr/bin/env bash
# Train the B1/B2/B3 one-epoch ablation as a Slurm array.
#SBATCH --job-name=experiment-b-train
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%2
#SBATCH --signal=USR1@120
#SBATCH --output=logs/experiment-b-train-%A_%a.out
#SBATCH --error=logs/experiment-b-train-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CONFIGURATIONS=(B1 B2 B3)
CONFIGURATION="${CONFIGURATIONS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$BIRDSAI_OUTPUT_ROOT/experiment_b/full/$CONFIGURATION"
mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m birdsai_pipeline.experiment_b_train \
  --configuration "$CONFIGURATION" \
  --train-json "$BIRDSAI_OUTPUT_ROOT/coco/train.json" \
  --dataset-root "$BIRDSAI_DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "${BIRDSAI_EPOCHS:-1}" \
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
  --save-steps 250
