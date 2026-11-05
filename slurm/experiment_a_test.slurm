#!/bin/bash
#SBATCH --job-name=gdino-exp-a-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load miniconda3
module load cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/grounding-dino"

PROJECT_DIR="$HOME/computer_vision/birdsai_data_pipeline"
DATASET_ROOT="$HOME/computer_vision/archive"
OUTPUT_ROOT="$HOME/computer_vision/outputs"
GROUNDING_DINO_ROOT="$HOME/computer_vision/GroundingDINO"
MODEL_CONFIG="$GROUNDING_DINO_ROOT/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT="$HOME/computer_vision/checkpoints/groundingdino_swint_ogc.pth"
SELECTED_CONFIG="$OUTPUT_ROOT/experiment_a/evaluation/selected_experiment_a_config.json"

cd "$PROJECT_DIR"
nvidia-smi

python -m birdsai_pipeline.grounding_dino_inference \
  --dataset-root "$DATASET_ROOT" \
  --coco "$OUTPUT_ROOT/coco/test.json" \
  --config "$MODEL_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --selected-config "$SELECTED_CONFIG" \
  --output-dir "$OUTPUT_ROOT/experiment_a/test_predictions" \
  --device cuda \
  --inference-box-threshold 0.05 \
  --text-threshold 0.20 \
  --max-detections 100 \
  --checkpoint-every 100 \
  --resume
