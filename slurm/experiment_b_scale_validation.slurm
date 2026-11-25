#!/usr/bin/env bash
# Compare native, global high-resolution, and tiled inference on validation.
#SBATCH --job-name=experiment-b-scale-validation
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%2
#SBATCH --signal=USR1@120
#SBATCH --output=logs/experiment-b-scale-validation-%A_%a.out
#SBATCH --error=logs/experiment-b-scale-validation-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CHECKPOINT="$BIRDSAI_OUTPUT_ROOT/experiment_b/full/B2/checkpoint_epoch_1.pt"
COCO_JSON="$BIRDSAI_OUTPUT_ROOT/coco/validation.json"
OUTPUT_ROOT="$BIRDSAI_OUTPUT_ROOT/experiment_b/scale_validation"

METHODS=(native global tiled)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"

case "$METHOD" in
  native)
    OUTPUT_DIR="$OUTPUT_ROOT/B2_epoch1_native"
    python -m birdsai_pipeline.experiment_a_inference \
      --config configs/experiment_b_prompts.json \
      --coco-json "$COCO_JSON" \
      --dataset-root "$BIRDSAI_DATASET_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --prompt-name animal_person \
      --checkpoint "$CHECKPOINT" \
      --device cuda \
      --box-threshold 0.10 \
      --text-threshold 0.20 \
      --resume
    ;;
  global)
    OUTPUT_DIR="$OUTPUT_ROOT/B2_epoch1_1024"
    python -m birdsai_pipeline.experiment_a_inference \
      --config configs/experiment_b_prompts.json \
      --coco-json "$COCO_JSON" \
      --dataset-root "$BIRDSAI_DATASET_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --prompt-name animal_person \
      --checkpoint "$CHECKPOINT" \
      --device cuda \
      --box-threshold 0.10 \
      --text-threshold 0.20 \
      --shortest-edge 1024 \
      --longest-edge 1706 \
      --resume
    ;;
  tiled)
    OUTPUT_DIR="$OUTPUT_ROOT/B2_epoch1_tiled_384"
    python -m birdsai_pipeline.experiment_b_tiled_inference \
      --config configs/experiment_b_prompts.json \
      --coco-json "$COCO_JSON" \
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
    ;;
  *)
    echo "Unknown scale method: $METHOD" >&2
    exit 2
    ;;
esac

python -m birdsai_pipeline.evaluate \
  --ground-truth "$COCO_JSON" \
  --predictions "$OUTPUT_DIR/predictions.json" \
  --score-threshold 0.10 \
  --iou-threshold 0.50 \
  --output "$OUTPUT_DIR/metrics.json"
