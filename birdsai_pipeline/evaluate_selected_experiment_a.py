from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import write_json
from .evaluate import coco_metrics, threshold_metrics
from .grounding_dino_inference import slugify


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selected Experiment A on TestReal.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    args = parser.parse_args()

    selected = json.loads(args.selected_config.read_text(encoding="utf-8"))
    prediction_path = args.predictions_dir / f"{slugify(selected['prompt_name'])}.json"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing selected-prompt predictions: {prediction_path}")

    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    coco, coco_console = coco_metrics(args.ground_truth, prediction_path)
    threshold = threshold_metrics(
        ground_truth=ground_truth,
        predictions=predictions,
        score_threshold=float(selected["score_threshold"]),
        iou_threshold=args.iou_threshold,
    )
    report = {
        "experiment": "A",
        "split": "test",
        "configuration": selected,
        "prediction_file": str(prediction_path),
        "coco": coco,
        "threshold_metrics": threshold,
        "coco_console": coco_console,
    }
    write_json(args.output, report)
    print(coco_console)
    print(json.dumps(threshold["overall"], indent=2))
    print(f"Wrote final Experiment A test report to {args.output}")


if __name__ == "__main__":
    main()
