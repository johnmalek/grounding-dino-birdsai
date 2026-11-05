# BIRDSAI Data Preparation and Evaluation Pipeline

This package implements the data-preparation and evaluation portion of the
Grounding DINO–BIRDSAI project. It keeps the official `TestReal` split intact,
creates a sequence-level validation split from `TrainReal`, converts MOT CSV
annotations to COCO detection JSON, produces visual checks, and evaluates
COCO-format prediction files.

## Expected raw dataset layout

```text
BIRDSAI_ROOT/
├── TrainReal/
│   ├── annotations/*.csv
│   └── images/<sequence_id>/*.{jpg,png}
└── TestReal/
    ├── annotations/*.csv
    └── images/<sequence_id>/*.{jpg,png}
```

The `annotations/tracking` folders may remain in place; the pipeline reads only
top-level MOT CSV files.

## Installation on the HPC

```bash
cd ~/computer_vision/birdsai_data_pipeline
python3 -m venv ~/.venvs/birdsai-data
source ~/.venvs/birdsai-data/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
chmod +x scripts/run_data_pipeline.sh
```

If your cluster provides Python through environment modules, load the relevant
module before creating the virtual environment.

## Run interactively

```bash
cd ~/computer_vision/birdsai_data_pipeline
source ~/.venvs/birdsai-data/bin/activate

bash scripts/run_data_pipeline.sh \
  "$HOME/computer_vision/archive" \
  "$HOME/computer_vision/outputs"
```

## Run through Slurm

Edit the four paths in `slurm/prepare_birdsai.slurm`, then run:

```bash
mkdir -p logs
sbatch slurm/prepare_birdsai.slurm
```

Monitor it with:

```bash
squeue -u "$USER"
tail -f logs/birdsai-prepare-JOB_ID.out
```

## Outputs

```text
outputs/
├── manifests/
│   ├── trainreal_frames.csv
│   ├── testreal_frames.csv
│   ├── sequences.csv
│   └── annotation_issues.csv
├── splits/
│   ├── train_sequences.txt
│   ├── validation_sequences.txt
│   ├── test_sequences.txt
│   └── split_report.json
├── coco/
│   ├── train.json
│   ├── validation.json
│   └── test.json
├── reports/
└── figures/
```

All images, including empty frames, are retained in COCO JSON. This is required
to measure false positives per frame correctly.

## Annotation policy

- BIRDSAI class `0` becomes COCO category `1` (`animal`).
- BIRDSAI class `1` becomes COCO category `2` (`person`).
- `object_id`, `species`, `occluded`, `noisy`, `sequence_id`, and `frame_id`
  are preserved as custom COCO annotation fields.
- Boxes are clipped to the image boundary during conversion.
- Invalid boxes are recorded in conversion issue files rather than silently
  becoming annotations.
- The validation set contains complete sequences; frames never cross splits.

## Evaluate Grounding DINO predictions

Predictions must be a COCO result list:

```json
[
  {
    "image_id": 123,
    "category_id": 2,
    "bbox": [110.4, 80.1, 18.5, 25.2],
    "score": 0.73
  }
]
```

Run:

```bash
python -m birdsai_pipeline.evaluate \
  --ground-truth "$HOME/computer_vision/outputs/coco/validation.json" \
  --predictions "$HOME/computer_vision/outputs/predictions/baseline_validation.json" \
  --score-threshold 0.25 \
  --iou-threshold 0.50 \
  --output "$HOME/computer_vision/outputs/reports/baseline_validation_metrics.json"
```

The report includes COCO mAP/AP50/AP75, size-specific AP, per-class metrics,
precision, recall, F1, false positives per frame, and subgroup recall for size,
noise, and occlusion.

## Experiment A: pretrained Grounding DINO, full-frame inference

Experiment A uses the untouched pretrained Grounding DINO Swin-T checkpoint.
The model processes each validation image at the repository's standard
full-frame resize. Four prompt formulations are evaluated. GPU inference is
performed once per prompt at a permissive box threshold of `0.05`; confidence
thresholds are then swept offline without rerunning the model.

### Set up the Grounding DINO environment

Use a separate GPU environment from the lightweight data-preparation
environment. Adjust the CUDA/PyTorch versions to match the HPC modules.

Clone the official repository and download its Swin-T checkpoint if they are
not already present:

```bash
cd "$HOME/computer_vision"
git clone https://github.com/IDEA-Research/GroundingDINO.git

mkdir -p "$HOME/computer_vision/checkpoints"
wget \
  -O "$HOME/computer_vision/checkpoints/groundingdino_swint_ogc.pth" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

```bash
module load miniconda3
module load cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create \
  --prefix "$HOME/.conda/envs/grounding-dino" \
  python=3.11 \
  -y

conda activate "$HOME/.conda/envs/grounding-dino"

python -m pip install \
  torch==2.3.1 \
  torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  "transformers==4.57.6" \
  "huggingface-hub==0.36.0" \
  ninja

cd "$HOME/computer_vision/GroundingDINO"
export CUDA_HOME=/usr/local/cuda
python -m pip install --no-build-isolation -e .

cd "$HOME/computer_vision/birdsai_data_pipeline"
python -m pip install -e .
```

Test all paths and one image before launching the full array:

```bash
python -m birdsai_pipeline.grounding_dino_inference \
  --dataset-root "$HOME/computer_vision/archive" \
  --coco "$HOME/computer_vision/outputs/coco/validation.json" \
  --config "$HOME/computer_vision/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --checkpoint "$HOME/computer_vision/checkpoints/groundingdino_swint_ogc.pth" \
  --prompt-config configs/experiment_a_prompts.json \
  --prompt-index 0 \
  --output-dir "$HOME/computer_vision/outputs/experiment_a/debug" \
  --limit 1
```

### Run the validation prompt study

Verify the GPU partition, CUDA module, paths, and time limit in
`slurm/experiment_a_validation.slurm`, then submit:

```bash
mkdir -p logs
sbatch slurm/experiment_a_validation.slurm
```

This is a four-task Slurm array. Each task handles one prompt and writes a COCO
result JSON under:

```text
outputs/experiment_a/predictions/
```

Interrupted tasks can be resubmitted because predictions and completed image
IDs are checkpointed every 100 images.

After all four array tasks report `COMPLETED`, submit the CPU evaluation:

```bash
sbatch slurm/experiment_a_evaluate.slurm
```

The selection results are written to:

```text
outputs/experiment_a/evaluation/
├── prompt_threshold_summary.csv
├── prompt_threshold_details.json
└── selected_experiment_a_config.json
```

The prompt is selected by validation COCO mAP. Within that prompt, the score
threshold is selected by validation F1 at IoU 0.50. `TestReal` remains unused
during selection.

### Finalize Experiment A on TestReal

Only after `selected_experiment_a_config.json` exists and has been reviewed,
run the selected configuration once on the untouched test set:

```bash
sbatch slurm/experiment_a_test.slurm
```

When that GPU job completes successfully, evaluate it:

```bash
sbatch slurm/experiment_a_test_evaluate.slurm
```

The final Experiment A report is:

```text
outputs/experiment_a/evaluation/test_metrics.json
```

Do not rerun prompt or threshold selection using this test report. Subsequent
Experiments B, C, and D must use the same predefined test split and reporting
metrics.

## Experimental discipline

Use the validation split to select prompts, box thresholds, text thresholds,
tile sizes, overlaps, and NMS settings. Do not tune on `TestReal`. Evaluate the
test split only after the configuration has been fixed.
