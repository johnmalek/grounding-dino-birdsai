#!/usr/bin/env bash
#SBATCH --job-name=experiment-a-validation
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:50:00
#SBATCH --array=0-3%2
#SBATCH --signal=USR1@120
#SBATCH --output=logs/experiment-a-validation-%A_%a.out
#SBATCH --error=logs/experiment-a-validation-%A_%a.err

set -euo pipefail

module purge
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate birdsai-data

REPOSITORY="$HOME/computer_vision/birdsai_data_pipeline"
DATASET_ROOT="$HOME/computer_vision/archive"
COCO_JSON="$HOME/computer_vision/outputs/coco/validation.json"
OUTPUT_ROOT="$HOME/computer_vision/outputs/experiment_a_restart/validation"

PROMPTS=(
  person_animal
  human_animal
  person_wildlife
  person_species
)
PROMPT_NAME="${PROMPTS[$SLURM_ARRAY_TASK_ID]}"

export HF_HOME="$HOME/computer_vision/checkpoints/huggingface"
export PYTHONPATH="$REPOSITORY"

cd "$REPOSITORY"
mkdir -p "$OUTPUT_ROOT/$PROMPT_NAME"

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Prompt: $PROMPT_NAME"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

python - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the validation job")
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m birdsai_pipeline.experiment_a_inference \
  --config configs/experiment_a_prompts.json \
  --coco-json "$COCO_JSON" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_ROOT/$PROMPT_NAME" \
  --prompt-name "$PROMPT_NAME" \
  --device cuda \
  --resume \
  --save-visualizations 25 \
  --hide-scores

echo "Finished: $(date --iso-8601=seconds)"
