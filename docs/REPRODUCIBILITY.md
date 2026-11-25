# Reproducibility guide

This guide defines the tested environment, execution order, expected outputs,
and safeguards required to reproduce the project without relying on the
original author's directory layout.

## Tested environment

| Component | Tested version |
|---|---|
| Python | 3.11.18 |
| PyTorch | 2.14.1+cu132 |
| Torchvision | 0.29.1+cu132 |
| Transformers | 5.22.2 |
| SciPy | 1.17.1 |
| GPU | NVIDIA A100 MIG 3g.40gb |

Grounding DINO custom kernels are disabled. Training and inference use the
Hugging Face implementation, so the original Grounding DINO CUDA extension is
not required.

Install the PyTorch wheel compatible with the local CUDA driver before running:

    conda create -n birdsai-data python=3.11 -y
    conda activate birdsai-data
    python -m pip install -e ".[ml,dev]"
    make check

## Paths and environment variables

Local examples use data/birdsai for raw data, outputs for generated artifacts,
and logs for Slurm output.

| Variable | Purpose | Default |
|---|---|---|
| BIRDSAI_REPO | Repository checkout | Repository containing the job file |
| BIRDSAI_DATASET_ROOT | Raw BIRDSAI root | BIRDSAI_REPO/data/birdsai |
| BIRDSAI_OUTPUT_ROOT | Generated outputs | BIRDSAI_REPO/outputs |
| BIRDSAI_CONDA_ENV | Conda environment | birdsai-data |
| BIRDSAI_CONDA_MODULE | Cluster Conda module | miniconda3 |
| HF_HOME | Hugging Face cache | BIRDSAI_OUTPUT_ROOT/cache/huggingface |

Export site-specific values before submitting jobs.

## Stage 1: prepare the data

    bash scripts/run_data_pipeline.sh data/birdsai outputs

Expected outputs include frame manifests, sequence-level split files, COCO JSON
for train/validation/test, conversion reports, and visual checks. Verify the
image and annotation counts against the main README before starting GPU work.

## Stage 2: Experiment A

    mkdir -p logs
    sbatch slurm/experiment_a_validation.slurm

The frozen selection is stored in configs/experiment_a_selected.json. Run the
test job only after confirming that selection:

    sbatch slurm/experiment_a_test.slurm

Inference is resumable. Each completed image is appended to records.jsonl;
--resume skips completed image IDs and reconstructs predictions.json.

## Stage 3: Experiment B

Train the one-epoch B1/B2/B3 ablation and evaluate those checkpoints:

    sbatch slurm/experiment_b_train.slurm
    sbatch slurm/experiment_b_validation.slurm

B2 epoch 1 is the configuration-selection winner. Reproduce the B2 checkpoint
ablation by continuing it to three total epochs:

    sbatch slurm/experiment_b_continue_b2.slurm

The selected settings are versioned in configs/experiment_b_selected.json.

## Stage 4: scale selection

Run the B2 epoch-1 scale comparison on validation:

    sbatch slurm/experiment_b_scale_validation.slurm

The array compares native 800/1333 inference, global 1024/1706 inference, and
384-pixel tiled inference with 20% overlap and class-wise NMS at IoU 0.50. Each
task writes predictions and metrics.json under
outputs/experiment_b/scale_validation/.

Select the scale strategy using validation mAP before running TestReal.

## Stage 5: frozen final evaluation

Run this only after all model, checkpoint, scale, overlap, NMS, and threshold
choices are frozen using validation:

    sbatch slurm/experiment_b_test.slurm

The final directory is outputs/experiment_b/test/B2_epoch1_tiled_384 and
contains metadata.json, metrics.json, predictions.json, and records.jsonl.

## Verification and provenance

Run the inexpensive repository checks on any machine:

    make check

For each expensive run, retain the exact Git commit, JSON configuration, Slurm
logs, metadata, training log, checkpoint checksum, and prediction checksum.
Record external URLs and hashes in ARTIFACTS.md.


After all stages finish, collect the metrics with:

    python scripts/summarize_results.py \
      --output-root "$BIRDSAI_OUTPUT_ROOT"

This command fails if an expected result is missing, preventing an incomplete
run from being mistaken for a complete reproduction.

## Known compatibility constraints

- The tested GroundingDinoForObjectDetection implementation does not support
  the generic gradient_checkpointing_enable path. Do not pass
  --gradient-checkpointing unless the installed version has verified support.
- Batch size 4 passed a short benchmark but later failed on a worst-case
  full-data batch. Reproducible jobs use batch size 2 and accumulation 4.
- Tiled inference is slower and creates larger prediction files than native
  inference.
