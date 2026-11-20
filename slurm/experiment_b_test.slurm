#!/usr/bin/env bash
# Run the frozen final B2 epoch-1 tiled configuration on TestReal.
#SBATCH --job-name=experiment-b-final-test
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=logs/experiment-b-final-test-%j.out
#SBATCH --error=logs/experiment-b-final-test-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CHECKPOINT="$BIRDSAI_OUTPUT_ROOT/experiment_b/full/B2/checkpoint_epoch_1.pt"
OUTPUT_DIR="$BIRDSAI_OUTPUT_ROOT/experiment_b/test/B2_epoch1_tiled_384"
mkdir -p "$OUTPUT_DIR"

python -m birdsai_pipeline.experiment_b_tiled_inference \
  --config configs/experiment_b_prompts.json \
  --coco-json "$BIRDSAI_OUTPUT_ROOT/coco/test.json" \
  --dataset-root "$BIRDSAI_DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --prompt-name animal_person \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --box-threshold 0.10 \
  --text-threshold 0.20 \
  --shortest-edge 800 \
  --longest-edge 1333 \
  --tile-size 384 \
  --tile-overlap 0.20 \
  --nms-iou 0.50 \
  --resume

python -m birdsai_pipeline.evaluate \
  --ground-truth "$BIRDSAI_OUTPUT_ROOT/coco/test.json" \
  --predictions "$OUTPUT_DIR/predictions.json" \
  --score-threshold 0.10 \
  --iou-threshold 0.50 \
  --output "$OUTPUT_DIR/metrics.json"
