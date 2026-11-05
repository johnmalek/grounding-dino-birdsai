#!/bin/bash
#SBATCH --job-name=gdino-exp-a-test-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/birdsai-data"

PROJECT_DIR="$HOME/computer_vision/birdsai_data_pipeline"
OUTPUT_ROOT="$HOME/computer_vision/outputs"

cd "$PROJECT_DIR"

python -m birdsai_pipeline.evaluate_selected_experiment_a \
  --ground-truth "$OUTPUT_ROOT/coco/test.json" \
  --predictions-dir "$OUTPUT_ROOT/experiment_a/test_predictions" \
  --selected-config "$OUTPUT_ROOT/experiment_a/evaluation/selected_experiment_a_config.json" \
  --output "$OUTPUT_ROOT/experiment_a/evaluation/test_metrics.json" \
  --iou-threshold 0.50
