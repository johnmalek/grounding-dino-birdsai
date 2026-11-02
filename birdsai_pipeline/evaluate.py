from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

from .common import write_json


COCO_STAT_NAMES = [
    "AP_50_95",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def iou_xywh(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = map(float, left)
    rx1, ry1, rw, rh = map(float, right)
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, lw) * max(0.0, lh) + max(0.0, rw) * max(0.0, rh) - intersection
    return intersection / union if union > 0 else 0.0


def safe_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def threshold_metrics(
    ground_truth: dict,
    predictions: list[dict],
    score_threshold: float,
    iou_threshold: float,
) -> dict:
    gt_by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    pred_by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        gt_by_key[(int(annotation["image_id"]), int(annotation["category_id"]))].append(annotation)
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) >= score_threshold:
            pred_by_key[(int(prediction["image_id"]), int(prediction["category_id"]))].append(prediction)

    category_names = {int(item["id"]): item["name"] for item in ground_truth["categories"]}
    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    matched_gt_ids: set[int] = set()
    keys = set(gt_by_key) | set(pred_by_key)
    for image_id, category_id in keys:
        gt_items = gt_by_key[(image_id, category_id)]
        pred_items = sorted(
            pred_by_key[(image_id, category_id)], key=lambda item: float(item["score"]), reverse=True
        )
        used: set[int] = set()
        for prediction in pred_items:
            best_index = None
            best_iou = 0.0
            for index, annotation in enumerate(gt_items):
                if index in used:
                    continue
                overlap = iou_xywh(prediction["bbox"], annotation["bbox"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index is not None and best_iou >= iou_threshold:
                used.add(best_index)
                matched_gt_ids.add(int(gt_items[best_index]["id"]))
                totals[category_id]["tp"] += 1
            else:
                totals[category_id]["fp"] += 1
        totals[category_id]["fn"] += len(gt_items) - len(used)

    per_class = {
        category_names.get(category_id, str(category_id)): safe_metrics(**values)
        for category_id, values in sorted(totals.items())
    }
    overall_counts = {
        key: sum(values[key] for values in totals.values()) for key in ("tp", "fp", "fn")
    }
    image_count = len(ground_truth["images"])
    overall = safe_metrics(**overall_counts)
    overall["false_positives_per_frame"] = overall_counts["fp"] / image_count if image_count else 0.0

    subgroup_members: dict[str, list[int]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        annotation_id = int(annotation["id"])
        area = float(annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3]))
        subgroup_members["small" if area < 200 else "medium" if area <= 2000 else "large"].append(
            annotation_id
        )
        subgroup_members["occluded" if int(annotation.get("occluded", 0)) else "not_occluded"].append(
            annotation_id
        )
        subgroup_members["noisy" if int(annotation.get("noisy", 0)) else "clean"].append(annotation_id)

    subgroup_recall = {}
    for name, annotation_ids in subgroup_members.items():
        matched = sum(annotation_id in matched_gt_ids for annotation_id in annotation_ids)
        subgroup_recall[name] = {
            "matched": matched,
            "total": len(annotation_ids),
            "recall": matched / len(annotation_ids) if annotation_ids else 0.0,
        }

    return {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "overall": overall,
        "per_class": per_class,
        "subgroup_recall": subgroup_recall,
    }


def coco_metrics(gt_path: Path, predictions_path: Path) -> tuple[dict, str]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Install pycocotools to calculate COCO metrics") from exc

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(predictions_path))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    overall = dict(zip(COCO_STAT_NAMES, map(float, evaluator.stats)))

    per_class = {}
    for category in coco_gt.loadCats(coco_gt.getCatIds()):
        with contextlib.redirect_stdout(output):
            class_evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            class_evaluator.params.catIds = [int(category["id"])]
            class_evaluator.evaluate()
            class_evaluator.accumulate()
            class_evaluator.summarize()
        per_class[category["name"]] = dict(
            zip(COCO_STAT_NAMES, map(float, class_evaluator.stats))
        )
    return {"overall": overall, "per_class": per_class}, output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate COCO-format Grounding DINO predictions.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    args = parser.parse_args()

    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if isinstance(predictions, dict) and "annotations" in predictions:
        predictions = predictions["annotations"]
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a COCO result list")

    threshold = threshold_metrics(
        ground_truth, predictions, args.score_threshold, args.iou_threshold
    )
    coco, coco_console = coco_metrics(args.ground_truth, args.predictions)
    report = {"coco": coco, "threshold_metrics": threshold}
    write_json(args.output, report)

    print(coco_console)
    print(json.dumps(threshold["overall"], indent=2))
    print(f"Wrote evaluation report to {args.output}")


if __name__ == "__main__":
    main()
