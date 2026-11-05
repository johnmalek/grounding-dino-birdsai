from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .common import write_json
from .evaluate import coco_metrics, threshold_metrics
from .grounding_dino_inference import load_prompt_config, slugify


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or any(not 0 <= item <= 1 for item in thresholds):
        raise ValueError("Thresholds must be comma-separated values between 0 and 1")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt variants and confidence thresholds for Experiment A."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--selection-metric",
        choices=["AP_50_95", "AP_50"],
        default="AP_50_95",
    )
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    prompts = load_prompt_config(args.prompt_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detailed: dict[str, dict] = {}
    rows: list[dict] = []
    for prompt in prompts:
        prompt_name = prompt["name"]
        prediction_path = args.predictions_dir / f"{slugify(prompt_name)}.json"
        if not prediction_path.exists():
            print(f"Skipping {prompt_name}: missing {prediction_path}")
            continue
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        if not predictions:
            print(f"Skipping {prompt_name}: prediction list is empty")
            continue

        coco, coco_console = coco_metrics(args.ground_truth, prediction_path)
        threshold_results = [
            threshold_metrics(
                ground_truth=ground_truth,
                predictions=predictions,
                score_threshold=threshold,
                iou_threshold=args.iou_threshold,
            )
            for threshold in thresholds
        ]
        best_threshold = max(
            threshold_results,
            key=lambda result: (
                result["overall"]["f1"],
                result["overall"]["recall"],
                result["overall"]["precision"],
            ),
        )
        overall_coco = coco["overall"]
        row = {
            "prompt_name": prompt_name,
            "prompt_text": prompt["text"],
            "AP_50_95": overall_coco["AP_50_95"],
            "AP_50": overall_coco["AP_50"],
            "AP_small": overall_coco["AP_small"],
            "best_score_threshold": best_threshold["score_threshold"],
            "precision": best_threshold["overall"]["precision"],
            "recall": best_threshold["overall"]["recall"],
            "f1": best_threshold["overall"]["f1"],
            "false_positives_per_frame": best_threshold["overall"][
                "false_positives_per_frame"
            ],
            "prediction_file": str(prediction_path),
        }
        rows.append(row)
        detailed[prompt_name] = {
            "prompt": prompt,
            "coco": coco,
            "coco_console": coco_console,
            "threshold_sweep": threshold_results,
            "best_threshold_result": best_threshold,
        }
        print(
            f"{prompt_name}: mAP={row['AP_50_95']:.4f}, AP50={row['AP_50']:.4f}, "
            f"best F1={row['f1']:.4f} at score={row['best_score_threshold']:.2f}"
        )

    if not rows:
        raise RuntimeError("No prompt prediction files were evaluated")

    rows.sort(
        key=lambda row: (
            row[args.selection_metric],
            row["AP_50"],
            row["f1"],
        ),
        reverse=True,
    )
    selected = rows[0]
    selected_prompt = next(prompt for prompt in prompts if prompt["name"] == selected["prompt_name"])
    selected_config = {
        "experiment": "A",
        "model": "pretrained_grounding_dino_swint",
        "inference": "standard_full_frame",
        "selection_split": "validation",
        "selection_metric": args.selection_metric,
        "prompt_name": selected["prompt_name"],
        "prompt_text": selected["prompt_text"],
        "aliases": selected_prompt["aliases"],
        "score_threshold": selected["best_score_threshold"],
        "text_threshold": 0.20,
        "validation_metrics": {
            key: selected[key]
            for key in [
                "AP_50_95",
                "AP_50",
                "AP_small",
                "precision",
                "recall",
                "f1",
                "false_positives_per_frame",
            ]
        },
    }

    summary_path = args.output_dir / "prompt_threshold_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output_dir / "prompt_threshold_details.json", detailed)
    write_json(args.output_dir / "selected_experiment_a_config.json", selected_config)

    print(f"Selected prompt: {selected['prompt_name']}")
    print(f"Selected score threshold: {selected['best_score_threshold']:.2f}")
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
