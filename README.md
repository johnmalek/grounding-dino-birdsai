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

## Experimental discipline

Use the validation split to select prompts, box thresholds, text thresholds,
tile sizes, overlaps, and NMS settings. Do not tune on `TestReal`. Evaluate the
test split only after the configuration has been fixed.
