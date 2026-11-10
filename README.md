# Grounding DINO–BIRDSAI: Data Pipeline and Zero-Shot Experiments

This repository provides a reproducible data-preparation, evaluation, and
zero-shot detection pipeline for the thermal BIRDSAI dataset. It keeps the
official `TestReal` split intact, creates a sequence-level validation split
from `TrainReal`, converts MOT CSV annotations to COCO detection JSON, produces
visual checks, evaluates COCO-format predictions, and runs Grounding DINO
without fine-tuning.

Completed work currently includes:

- BIRDSAI manifest creation and annotation validation;
- Leakage-safe, sequence-level train/validation/test splits;
- MOT-to-COCO conversion for `animal` and `person`;
- COCO and threshold-based detection evaluation;
- Experiment A: zero-shot Grounding DINO transfer to thermal imagery;
- Qualitative TP/FP/FN examples and Markdown/LaTeX reports.

## Prepared dataset summary

| Split | Images | Annotations | Empty images | Source |
|---|---:|---:|---:|---|
| Train | 33,926 | 72,113 | 15,799 | `TrainReal` |
| Validation | 6,735 | 15,086 | 3,653 | Sequence-level holdout from `TrainReal` |
| Test | 21,333 | 78,761 | 5,839 | Official `TestReal` |

All frames, including empty/background frames, are retained so false positives
per frame can be measured correctly.

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

## HPC installation

The tested environment uses Python 3.11, PyTorch 2.14.1 with CUDA 13.2,
Torchvision 0.29.1, and Transformers 5.22.2.

If the `birdsai-data` environment does not already exist, create it with:

```bash
module purge
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n birdsai-data python=3.11 -y
conda activate birdsai-data

cd ~/computer_vision/birdsai_data_pipeline
python -m pip install --upgrade pip
python -m pip install -e .

python -m pip install --no-cache-dir \
  torch==2.14.1 \
  torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu132

python -m pip install --no-cache-dir \
  transformers==5.22.2 \
  safetensors
```

For later HPC sessions, activate the existing environment with:

```bash
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate birdsai-data
```

## Run interactively

```bash
cd ~/computer_vision/birdsai_data_pipeline
conda activate birdsai-data

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

## Experiment A: Zero-shot Grounding DINO

### Research question

> How well does pretrained zero-shot Grounding DINO transfer to BIRDSAI
> thermal imagery without training or fine-tuning?

Experiment A used `IDEA-Research/grounding-dino-tiny` through Hugging Face
Transformers. Custom CUDA and CPU kernels were disabled with
`disable_custom_kernels=True`, avoiding compilation of the original Grounding
DINO extension.

Four prompts were compared on validation:

| Name | Prompt |
|---|---|
| `person_animal` | `person . animal .` |
| `human_animal` | `human . animal .` |
| `person_wildlife` | `person . wildlife .` |
| `person_species` | `person . elephant . lion . giraffe . dog .` |

Predicted phrases were mapped into the final `animal` and `person` categories.
Phrases matching both categories were considered ambiguous and excluded.

### Running Experiment A

Submit the four-prompt validation array:

```bash
mkdir -p logs
sbatch slurm/experiment_a_validation.slurm
```

After selecting and freezing the best validation configuration, run TestReal:

```bash
sbatch slurm/experiment_a_test.slurm
```

The inference pipeline is resumable. It writes one JSONL record per completed
image and skips completed image IDs when run again with `--resume`.

The tested HPC uses A100 MIG resources. The working GPU request is:

```bash
#SBATCH --gres=gpu:3g.40gb:1
```

### Selected configuration

The `person_animal` prompt achieved the highest validation COCO mAP. The final
configuration is stored in `configs/experiment_a_selected.json`:

- Prompt: `person . animal .`
- Box threshold: `0.10`
- Text threshold: `0.20`
- Operating score threshold: `0.10`
- Prompt-selection metric: validation COCO mAP
- Threshold-selection metric: validation F1 at IoU 0.50

### Final results

| Split | mAP | AP50 | AP75 | Precision | Recall | F1 | FP/frame | Animal AP50 | Person AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 1.16% | 3.36% | 0.49% | 15.71% | 12.85% | 14.13% | 1.54 | 6.72% | 0.00% |
| TestReal | 5.49% | 18.24% | 1.97% | 33.40% | 35.22% | 34.29% | 2.59 | 30.49% | 6.00% |

Grounding DINO demonstrated partial but limited zero-shot transfer. Animal
detection transferred substantially better than person detection, while the
low AP75 showed that precise localization remained weak. The model may be
useful as a baseline or pseudo-labeling aid, but it is not sufficiently
reliable for deployment without thermal-domain adaptation or fine-tuning.

TestReal contains larger animals, more annotations per image, and fewer empty
frames than validation, explaining much of the validation–test performance
difference.

### Reports and qualitative examples

- [Experiment A Markdown report](reports/experiment_a/experiment_a_report.md)
- [Experiment A LaTeX report](reports/experiment_a/experiment_a_report.tex)
- [Representative TP/FP/FN examples](reports/experiment_a/representative_examples/)
- [Split distribution summary](reports/experiment_a/split_distribution.json)

Green boxes represent ground truth and red boxes represent Grounding DINO
predictions.

## Experimental discipline

Use the validation split to select prompts, box thresholds, text thresholds,
tile sizes, overlaps, and NMS settings. Do not tune on `TestReal`. Evaluate the
test split only after the configuration has been fixed.
