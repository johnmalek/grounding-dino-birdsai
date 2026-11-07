from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from tqdm import tqdm


STOP_REQUESTED = False


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; stopping after the current image.", flush=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_prompt(config_path: Path, prompt_name: str) -> tuple[dict, dict]:
    config = load_json(config_path)
    matches = [item for item in config["prompts"] if item["name"] == prompt_name]
    if len(matches) != 1:
        available = ", ".join(item["name"] for item in config["prompts"])
        raise ValueError(f"Unknown prompt {prompt_name!r}; choose one of: {available}")
    return config, matches[0]


def normalize_phrase(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def map_phrase_to_category(phrase: object, aliases: dict[str, list[str]]) -> str | None:
    normalized = normalize_phrase(phrase)
    matches = set()

    for category_name, values in aliases.items():
        for alias in sorted(values, key=len, reverse=True):
            normalized_alias = normalize_phrase(alias)
            if re.search(rf"\b{re.escape(normalized_alias)}\b", normalized):
                matches.add(category_name)

    return next(iter(matches)) if len(matches) == 1 else None


def load_records(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    records: dict[int, dict] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                print("Ignoring an incomplete final JSONL line.", flush=True)
                break
            raise ValueError(f"Malformed JSONL record at line {index + 1} in {path}")
        records[int(record["image_id"])] = record
    return records


def resolve_image(dataset_root: Path, file_name: str) -> Path:
    path = dataset_root / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    return path


def xyxy_to_xywh(
    box: list[float],
    width: int,
    height: int,
) -> tuple[list[float], list[float]] | None:
    x1, y1, x2, y2 = map(float, box)
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1], [x1, y1, x2, y2]


def consolidate_predictions(records: dict[int, dict]) -> list[dict]:
    predictions = []
    for image_id in sorted(records):
        for detection in records[image_id].get("detections", []):
            if detection.get("category_id") is None:
                continue
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(detection["category_id"]),
                    "bbox": detection["bbox"],
                    "score": float(detection["score"]),
                    "phrase": detection["phrase"],
                    "mapped_category_name": detection["mapped_category_name"],
                }
            )
    return predictions


def draw_predictions(
    image: Image.Image,
    detections: list[dict],
    hide_scores: bool,
) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    colors = {"person": "red", "animal": "lime"}
    for detection in detections:
        category_name = detection.get("mapped_category_name")
        if category_name is None:
            continue
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        color = colors.get(category_name, "yellow")
        label = category_name
        if not hide_scores:
            label = f"{label} {detection['score']:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text(
            (x1 + 3, max(0, y1 - 15)),
            label,
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable, extension-free Grounding DINO inference."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_a_prompts.json"),
    )
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--box-threshold", type=float)
    parser.add_argument("--text-threshold", type=float)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--save-visualizations",
        type=int,
        default=0,
        metavar="N",
    )
    parser.add_argument("--hide-scores", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    experiment_config, prompt = load_prompt(args.config, args.prompt_name)
    model_id = experiment_config["model_id"]
    box_threshold = (
        args.box_threshold
        if args.box_threshold is not None
        else float(experiment_config["box_threshold"])
    )
    text_threshold = (
        args.text_threshold
        if args.text_threshold is not None
        else float(experiment_config["text_threshold"])
    )

    ground_truth = load_json(args.coco_json)
    images = ground_truth["images"]
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be positive")
        images = images[: args.max_images]

    category_ids = {
        item["name"]: int(item["id"])
        for item in ground_truth["categories"]
    }
    for required in ("person", "animal"):
        if required not in category_ids:
            raise ValueError(f"COCO categories do not contain {required!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(
            f"{records_path} exists; pass --resume or use a new output directory"
        )
    records = load_records(records_path) if args.resume else {}
    completed_ids = set(records)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device_name = (
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    device = torch.device(device_name)

    print(f"Model: {model_id}")
    print(f"Prompt: {prompt['name']} = {prompt['text']}")
    print(f"Device: {device}")
    print(f"Completed records found: {len(completed_ids)}")

    processor = AutoProcessor.from_pretrained(model_id)
    model_config = AutoConfig.from_pretrained(model_id)
    model_config.disable_custom_kernels = True
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        config=model_config,
    ).to(device)
    model.eval()

    postprocess_parameters = inspect.signature(
        processor.post_process_grounded_object_detection
    ).parameters
    visual_dir = args.output_dir / "visualizations"
    visualization_count = 0
    if visual_dir.exists():
        visualization_count = len(list(visual_dir.glob("*.jpg")))

    interrupted = False
    with records_path.open("a", encoding="utf-8") as stream:
        for image_record in tqdm(
            images,
            desc=f"Experiment A: {prompt['name']}",
        ):
            image_id = int(image_record["id"])
            if image_id in completed_ids:
                continue
            if STOP_REQUESTED:
                interrupted = True
                break

            image_path = resolve_image(
                args.dataset_root,
                image_record["file_name"],
            )
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                images=image,
                text=prompt["text"],
                return_tensors="pt",
            )
            inputs = {
                name: value.to(device)
                for name, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = model(**inputs)

            postprocess_kwargs: dict[str, object] = {
                "target_sizes": [(image.height, image.width)],
                "text_threshold": text_threshold,
            }
            if "threshold" in postprocess_parameters:
                postprocess_kwargs["threshold"] = box_threshold
            else:
                postprocess_kwargs["box_threshold"] = box_threshold

            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                **postprocess_kwargs,
            )[0]
            labels = result.get("text_labels")
            if labels is None:
                labels = result.get("labels", [])
            boxes = result["boxes"].detach().cpu().tolist()
            scores = result["scores"].detach().cpu().tolist()

            detections = []
            for box, score, phrase in zip(boxes, scores, labels):
                if hasattr(phrase, "item"):
                    phrase = phrase.item()
                converted = xyxy_to_xywh(
                    box,
                    image.width,
                    image.height,
                )
                if converted is None:
                    continue
                bbox, bbox_xyxy = converted
                mapped_name = map_phrase_to_category(
                    phrase,
                    prompt["aliases"],
                )
                detections.append(
                    {
                        "bbox": bbox,
                        "bbox_xyxy": bbox_xyxy,
                        "score": float(score),
                        "phrase": str(phrase),
                        "mapped_category_name": mapped_name,
                        "category_id": category_ids.get(mapped_name),
                    }
                )

            record = {
                "image_id": image_id,
                "file_name": image_record["file_name"],
                "width": int(image_record["width"]),
                "height": int(image_record["height"]),
                "detections": detections,
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            completed_ids.add(image_id)
            records[image_id] = record

            if len(completed_ids) % 25 == 0:
                os.fsync(stream.fileno())

            if visualization_count < args.save_visualizations:
                visual_dir.mkdir(parents=True, exist_ok=True)
                rendered = draw_predictions(
                    image,
                    detections,
                    args.hide_scores,
                )
                rendered.save(
                    visual_dir / f"{image_id:08d}.jpg",
                    quality=95,
                )
                visualization_count += 1

    predictions = consolidate_predictions(records)
    atomic_write_json(
        args.output_dir / "predictions.json",
        predictions,
    )

    target_ids = {int(item["id"]) for item in images}
    completed_target_ids = target_ids.intersection(records)
    is_complete = (
        len(completed_target_ids) == len(target_ids)
        and not interrupted
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "prompt_name": prompt["name"],
        "prompt_text": prompt["text"],
        "aliases": prompt["aliases"],
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "disable_custom_kernels": True,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "ground_truth": str(args.coco_json.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "requested_images": len(images),
        "completed_images": len(completed_target_ids),
        "prediction_count": len(predictions),
        "is_complete": is_complete,
    }
    atomic_write_json(
        args.output_dir / "metadata.json",
        metadata,
    )

    print(json.dumps(metadata, indent=2))
    if interrupted:
        print(
            "Inference interrupted; rerun with --resume.",
            file=sys.stderr,
        )
        return 99
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
