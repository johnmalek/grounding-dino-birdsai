#!/bin/bash

#SBATCH --job-name=birdsai-prepare
#SBATCH --partition=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Load Conda on the compute node.
module purge
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the Python 3.11 environment.
conda activate "$HOME/.conda/envs/birdsai-data"

# Project paths.
PROJECT_DIR="$HOME/computer_vision/birdsai_data_pipeline"
DATASET_ROOT="$HOME/computer_vision/archive"
OUTPUT_ROOT="$HOME/computer_vision/outputs"

cd "$PROJECT_DIR"

echo "Node: $(hostname)"
echo "Python: $(which python)"
python --version

bash scripts/run_data_pipeline.sh \
    "$DATASET_ROOT" \
    "$OUTPUT_ROOT"
