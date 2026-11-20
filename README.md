# Grounding DINO for BIRDSAI Thermal Imagery

[![Tests](https://github.com/johnmalek/grounding-dino-birdsai/actions/workflows/ci.yml/badge.svg)](https://github.com/johnmalek/grounding-dino-birdsai/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

This repository contains the reproducible data pipeline and experiments used to
adapt `IDEA-Research/grounding-dino-tiny` to aerial thermal imagery from the
BIRDSAI dataset. The project detects two categories—`animal` and `person`—and
compares zero-shot transfer, three fine-tuning strategies, and scale-aware
inference.

The final system combines the B2 fine-tuning configuration at epoch 1 with
384-pixel tiled inference, 20% overlap, and class-wise NMS at IoU 0.50. On the
official `TestReal` split, COCO mAP improves from **5.49% to 11.44%** relative
to the zero-shot baseline.

## Project status

- [x] Validate BIRDSAI annotations and build frame manifests
- [x] Create sequence-level splits without temporal leakage
- [x] Convert MOT annotations to COCO detection format
- [x] Establish a zero-shot Grounding DINO baseline
- [x] Compare B1, B2, and B3 fine-tuning configurations
- [x] Select checkpoints and scale settings on validation only
- [x] Evaluate the frozen final system once on `TestReal`

## Repository structure

```text
.
├── birdsai_pipeline/       Python package: preparation, training, inference, evaluation
├── configs/                Prompts, training protocol, and frozen selections
├── docs/                   Reproduction guide and complete result tables
├── notebooks/              Exploratory analysis only
├── reports/                Versioned compact metrics, figures, and qualitative examples
├── scripts/                Local pipeline and repository validation scripts
├── slurm/                  Portable HPC job definitions
├── tests/                  Unit and end-to-end synthetic-data tests
├── environment.yml         Conda environment specification
├── pyproject.toml          Package metadata, dependencies, and command entry points
└── ARTIFACTS.md            Locations and checksums for large external artifacts
```

Raw data, checkpoints, prediction streams, caches, logs, and generated outputs
are intentionally excluded from Git.

## Dataset

[BIRDSAI](https://lila.science/datasets/conservationdrones) contains long-wave
thermal infrared aerial video collected at night in Southern Africa. Download
the dataset from its official distribution; it is not redistributed here.

```text
data/birdsai/
├── TrainReal/
│   ├── annotations/*.csv
│   └── images/<sequence_id>/*.{jpg,png}
└── TestReal/
    ├── annotations/*.csv
    └── images/<sequence_id>/*.{jpg,png}
```

| Split | Images | Annotations | Empty images | Source |
|---|---:|---:|---:|---|
| Train | 33,926 | 72,113 | 15,799 | `TrainReal` |
| Validation | 6,735 | 15,086 | 3,653 | Sequence-level `TrainReal` holdout |
| Test | 21,333 | 78,761 | 5,839 | Official `TestReal` |

BIRDSAI class `0` maps to COCO `animal` (ID 1), and class `1` maps to
`person` (ID 2). Empty frames are retained to measure false positives per
frame correctly.

## Installation

### Data preparation and evaluation only

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Training and GPU inference

Install the PyTorch build compatible with the local CUDA driver, then install
the ML extras:

```bash
python -m pip install -e ".[ml,dev]"
```

The experiments were run with Python 3.11, PyTorch 2.14.1+cu132,
Torchvision 0.29.1+cu132, Transformers 5.22.2, and SciPy 1.17.1. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the tested HPC setup and
important compatibility notes.

## Quick start

### 1. Prepare COCO splits

```bash
bash scripts/run_data_pipeline.sh \
  "$PWD/data/birdsai" \
  "$PWD/outputs"
```

This creates manifests, sequence lists, COCO JSON files, conversion reports,
and visual data checks under `outputs/`.

### 2. Run the repository checks

```bash
make check
```

### 3. Run zero-shot validation

```bash
python -m birdsai_pipeline.experiment_a_inference \
  --config configs/experiment_a_prompts.json \
  --coco-json outputs/coco/validation.json \
  --dataset-root data/birdsai \
  --output-dir outputs/experiment_a/validation/person_animal \
  --prompt-name person_animal \
  --device cuda \
  --resume
```

### 4. Evaluate predictions

```bash
python -m birdsai_pipeline.evaluate \
  --ground-truth outputs/coco/validation.json \
  --predictions outputs/experiment_a/validation/person_animal/predictions.json \
  --score-threshold 0.10 \
  --iou-threshold 0.50 \
  --output outputs/experiment_a/validation/person_animal/metrics.json
```

## Reproducing the experiments on Slurm

Set the shared paths once. The job files contain no user-specific absolute
paths.

```bash
export BIRDSAI_REPO="$PWD"
export BIRDSAI_DATASET_ROOT=/shared/path/to/birdsai
export BIRDSAI_OUTPUT_ROOT=/scratch/$USER/birdsai-outputs
export BIRDSAI_CONDA_ENV=birdsai-data
mkdir -p logs
```

Run the workflow in order:

```bash
sbatch slurm/prepare_birdsai.slurm
sbatch slurm/experiment_a_validation.slurm
sbatch slurm/experiment_a_test.slurm
sbatch slurm/experiment_b_train.slurm
sbatch slurm/experiment_b_validation.slurm
sbatch slurm/experiment_b_continue_b2.slurm   # checkpoint-selection ablation
sbatch slurm/experiment_b_test.slurm          # frozen final evaluation
```

Wait for each dependent stage to complete before submitting the next, or add
site-specific Slurm dependencies. Detailed commands and expected artifacts are
documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Main results

| Model | Split | mAP | AP50 | AP75 | F1 | Animal AP50 | Person AP50 |
|---|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | TestReal | 0.0549 | 0.1824 | 0.0197 | 0.3429 | 0.3049 | **0.0600** |
| B2 + tiled inference | TestReal | **0.1144** | **0.3096** | **0.0666** | **0.4277** | **0.5681** | 0.0511 |

![Final TestReal comparison](reports/experiment_b/figures/final_test_comparison.png)

The final model substantially improves overall detection and animal detection,
but person detection remains weak and false positives increase. Complete
configuration, checkpoint, scale, subgroup, and class-level results are in
[docs/RESULTS.md](docs/RESULTS.md).

## Reproducibility policy

- All model and operating-point choices are made on validation data.
- `TestReal` is used once after freezing the final configuration.
- Seeds, prompts, category mappings, thresholds, and selected checkpoints are
  versioned under `configs/`.
- Large artifacts are referenced with URLs and SHA-256 checksums in
  [ARTIFACTS.md](ARTIFACTS.md).
- `make check` validates syntax, configuration files, shell scripts, portability,
  and the synthetic end-to-end data pipeline.

## Limitations

- Person detection remains poor on `TestReal`.
- Tiled inference increases runtime and false positives.
- Validation and test data differ in target size, density, and occlusion.
- Only Grounding DINO Tiny and one primary fine-tuning prompt were studied.
- The system is a research baseline, not a deployment-ready detector.

## Citation

When using this repository, cite both BIRDSAI and Grounding DINO. A machine-
readable project citation is provided in [CITATION.cff](CITATION.cff).

## License and acknowledgements

Project code is released under the [Apache License 2.0](LICENSE). BIRDSAI data
and pretrained Grounding DINO weights remain under their respective licenses.
This project builds on work by the BIRDSAI authors and IDEA Research and uses
PyTorch, Hugging Face Transformers, Torchvision, SciPy, and pycocotools.
