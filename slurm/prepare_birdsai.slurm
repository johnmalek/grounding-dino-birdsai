#!/usr/bin/env bash

#SBATCH --job-name=birdsai-prepare
#SBATCH --partition=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "Node: $(hostname)"
python --version

bash scripts/run_data_pipeline.sh \
    "$BIRDSAI_DATASET_ROOT" \
    "$BIRDSAI_OUTPUT_ROOT"
