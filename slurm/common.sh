#!/usr/bin/env bash
# Shared environment setup for all Slurm jobs.
# Override any default by exporting the corresponding BIRDSAI_* variable before
# calling sbatch. Example:
#   export BIRDSAI_DATASET_ROOT=/shared/datasets/birdsai
#   export BIRDSAI_OUTPUT_ROOT=/scratch/$USER/birdsai-outputs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BIRDSAI_REPO="${BIRDSAI_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BIRDSAI_DATASET_ROOT="${BIRDSAI_DATASET_ROOT:-$BIRDSAI_REPO/data/birdsai}"
export BIRDSAI_OUTPUT_ROOT="${BIRDSAI_OUTPUT_ROOT:-$BIRDSAI_REPO/outputs}"
export HF_HOME="${HF_HOME:-$BIRDSAI_OUTPUT_ROOT/cache/huggingface}"
export PYTHONPATH="$BIRDSAI_REPO${PYTHONPATH:+:$PYTHONPATH}"

module purge
module load "${BIRDSAI_CONDA_MODULE:-miniconda3}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${BIRDSAI_CONDA_ENV:-birdsai-data}"

cd "$BIRDSAI_REPO"
mkdir -p "$BIRDSAI_OUTPUT_ROOT" "$HF_HOME"

echo "Repository: $BIRDSAI_REPO"
echo "Dataset: $BIRDSAI_DATASET_ROOT"
echo "Outputs: $BIRDSAI_OUTPUT_ROOT"
echo "Environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python: $(command -v python)"
