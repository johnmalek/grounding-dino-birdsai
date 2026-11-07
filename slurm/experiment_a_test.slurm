#!/usr/bin/env bash
#SBATCH --job-name=experiment-a-test
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=logs/experiment-a-test-%j.out
#SBATCH --error=logs/experiment-a-test-%j.err

set -euo pipefail

module purge
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate birdsai-data

REPOSITORY="$HOME/computer_vision/birdsai_data_pipeline"
DATASET_ROOT="$HOME/computer_vision/archive"
COCO_JSON="$HOME/computer_vision/outputs/coco/test.json"
OUTPUT_DIR="$HOME/computer_vision/outputs/experiment_a_restart/test/person_animal"

export HF_HOME="$HOME/computer_vision/checkpoints/huggingface"
export PYTHONPATH="$REPOSITORY"

cd "$REPOSITORY"
mkdir -p "$OUTPUT_DIR"

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "Prompt: person_animal"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

python - <<'PY'
import json
from pathlib import Path

import torch
import transformers

selected = json.loads(
    Path("configs/experiment_a_selected.json").read_text()
)

assert selected["selection_split"] == "validation"
assert selected["prompt_name"] == "person_animal"
assert selected["box_threshold"] == 0.1
assert selected["text_threshold"] == 0.2

print("Selected configuration:", json.dumps(selected, indent=2))
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the test job")
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m birdsai_pipeline.experiment_a_inference \
  --config configs/experiment_a_prompts.json \
  --coco-json "$COCO_JSON" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --prompt-name person_animal \
  --device cuda \
  --box-threshold 0.10 \
  --text-threshold 0.20 \
  --resume \
  --save-visualizations 50 \
  --hide-scores

echo "Finished: $(date --iso-8601=seconds)"
