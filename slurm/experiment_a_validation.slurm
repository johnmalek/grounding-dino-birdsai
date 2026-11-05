#!/bin/bash
#SBATCH --job-name=gdino-exp-a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

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

cd "$PROJECT_DIR"

echo "Node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID"
python --version
nvidia-smi

python -m birdsai_pipeline.grounding_dino_inference \
  --dataset-root "$DATASET_ROOT" \
  --coco "$OUTPUT_ROOT/coco/validation.json" \
  --config "$MODEL_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --prompt-config "$PROJECT_DIR/configs/experiment_a_prompts.json" \
  --prompt-index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$OUTPUT_ROOT/experiment_a/predictions" \
  --device cuda \
  --inference-box-threshold 0.05 \
  --text-threshold 0.20 \
  --max-detections 100 \
  --checkpoint-every 100 \
  --resume
