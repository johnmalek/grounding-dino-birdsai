#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/path/to/BIRDSAI /absolute/path/to/output"
  exit 2
fi

DATASET_ROOT="$(realpath "$1")"
OUTPUT_ROOT="$(realpath -m "$2")"

mkdir -p \
  "$OUTPUT_ROOT/manifests" \
  "$OUTPUT_ROOT/splits" \
  "$OUTPUT_ROOT/coco" \
  "$OUTPUT_ROOT/reports" \
  "$OUTPUT_ROOT/figures"

python -m birdsai_pipeline.build_manifest \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_ROOT/manifests"

python -m birdsai_pipeline.create_splits \
  --sequences "$OUTPUT_ROOT/manifests/sequences.csv" \
  --output-dir "$OUTPUT_ROOT/splits" \
  --val-count 6 \
  --seed 42

python -m birdsai_pipeline.mot_to_coco \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$OUTPUT_ROOT/manifests/trainreal_frames.csv" \
  --sequences "$OUTPUT_ROOT/splits/train_sequences.txt" \
  --split-name train \
  --output "$OUTPUT_ROOT/coco/train.json" \
  --issues-output "$OUTPUT_ROOT/reports/train_conversion_issues.json"

python -m birdsai_pipeline.mot_to_coco \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$OUTPUT_ROOT/manifests/trainreal_frames.csv" \
  --sequences "$OUTPUT_ROOT/splits/validation_sequences.txt" \
  --split-name validation \
  --output "$OUTPUT_ROOT/coco/validation.json" \
  --issues-output "$OUTPUT_ROOT/reports/validation_conversion_issues.json"

python -m birdsai_pipeline.mot_to_coco \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$OUTPUT_ROOT/manifests/testreal_frames.csv" \
  --sequences "$OUTPUT_ROOT/splits/test_sequences.txt" \
  --split-name test \
  --output "$OUTPUT_ROOT/coco/test.json" \
  --issues-output "$OUTPUT_ROOT/reports/test_conversion_issues.json"

for SPLIT in train validation test; do
  python -m birdsai_pipeline.visualize \
    --dataset-root "$DATASET_ROOT" \
    --coco "$OUTPUT_ROOT/coco/${SPLIT}.json" \
    --output-dir "$OUTPUT_ROOT/figures/${SPLIT}"
done

echo "Pipeline completed. Inspect:"
echo "  $OUTPUT_ROOT/manifests/annotation_issues.csv"
echo "  $OUTPUT_ROOT/splits/split_report.json"
echo "  $OUTPUT_ROOT/figures"
