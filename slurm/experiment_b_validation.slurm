#!/usr/bin/env bash
# Evaluate the epoch-1 checkpoint from each fine-tuning configuration.
#SBATCH --job-name=experiment-b-validation
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%2
#SBATCH --signal=USR1@120
#SBATCH --output=logs/experiment-b-validation-%A_%a.out
#SBATCH --error=logs/experiment-b-validation-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CONFIGURATIONS=(B1 B2 B3)
CONFIGURATION="${CONFIGURATIONS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="$BIRDSAI_OUTPUT_ROOT/experiment_b/full/$CONFIGURATION/checkpoint_epoch_1.pt"
OUTPUT_DIR="$BIRDSAI_OUTPUT_ROOT/experiment_b/validation/$CONFIGURATION/epoch_1"
mkdir -p "$OUTPUT_DIR"

python -m birdsai_pipeline.experiment_a_inference \
  --config configs/experiment_b_prompts.json \
  --coco-json "$BIRDSAI_OUTPUT_ROOT/coco/validation.json" \
  --dataset-root "$BIRDSAI_DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --prompt-name animal_person \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --box-threshold 0.10 \
  --text-threshold 0.20 \
  --resume

python -m birdsai_pipeline.evaluate \
  --ground-truth "$BIRDSAI_OUTPUT_ROOT/coco/validation.json" \
  --predictions "$OUTPUT_DIR/predictions.json" \
  --score-threshold 0.10 \
  --iou-threshold 0.50 \
  --output "$OUTPUT_DIR/metrics.json"
