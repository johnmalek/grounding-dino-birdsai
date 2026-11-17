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


def axis_origins(
    length: int,
    tile_size: int,
    overlap: float,
) -> list[int]:
    """Return tile origins while ensuring the far edge is covered."""
    if length <= tile_size:
        return [0]

    overlap_pixels = int(round(tile_size * overlap))
    stride = max(1, tile_size - overlap_pixels)
    final_origin = length - tile_size

    origins = list(range(0, final_origin + 1, stride))
    if origins[-1] != final_origin:
        origins.append(final_origin)

    return origins


def tile_windows(
    width: int,
    height: int,
    tile_size: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    """Return PIL crop windows as left, top, right, bottom."""
    return [
        (
            left,
            top,
            min(left + tile_size, width),
            min(top + tile_size, height),
        )
        for top in axis_origins(height, tile_size, overlap)
        for left in axis_origins(width, tile_size, overlap)
    ]


def classwise_nms(
    detections: list[dict],
    iou_threshold: float,
) -> list[dict]:
    """Remove duplicate overlapping tile detections per category."""
    if not detections:
        return []

    import torch
    from torchvision.ops import nms

    retained: list[dict] = []
    category_ids = sorted(
        {
            int(item["category_id"])
            for item in detections
            if item.get("category_id") is not None
        }
    )

    for category_id in category_ids:
        category_detections = [
            item
            for item in detections
            if int(item["category_id"]) == category_id
        ]

        boxes = torch.tensor(
            [item["bbox_xyxy"] for item in category_detections],
            dtype=torch.float32,
        )
        scores = torch.tensor(
            [float(item["score"]) for item in category_detections],
            dtype=torch.float32,
        )

        keep = nms(
            boxes,
            scores,
            float(iou_threshold),
        ).tolist()

        retained.extend(
            category_detections[index]
            for index in keep
        )

    return sorted(
        retained,
        key=lambda item: float(item["score"]),
        reverse=True,
    )


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
        description=(
            "Run resumable tiled Grounding DINO inference with "
            "class-wise NMS."
        )
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
    parser.add_argument(
        "--shortest-edge",
        type=int,
        help="Override the processor's shortest image edge.",
    )
    parser.add_argument(
        "--longest-edge",
        type=int,
        help="Override the processor's longest image edge.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=384,
        help="Square tile size in original-image pixels.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.20,
        help="Fractional overlap between adjacent tiles.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.50,
        help="Class-wise NMS IoU threshold.",
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Optional Experiment B partial checkpoint containing "
            "a model_state dictionary."
        ),
    )
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

    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if not 0.0 <= args.tile_overlap < 1.0:
        raise ValueError("--tile-overlap must be in [0, 1)")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in [0, 1]")

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
    print(
        "Tiling: "
        f"{args.tile_size}x{args.tile_size}, "
        f"overlap={args.tile_overlap:.2f}, "
        f"NMS IoU={args.nms_iou:.2f}"
    )

    processor = AutoProcessor.from_pretrained(model_id)

    if (args.shortest_edge is None) != (args.longest_edge is None):
        raise ValueError(
            "--shortest-edge and --longest-edge must be provided together"
        )

    processor_kwargs: dict[str, object] = {}
    if args.shortest_edge is not None:
        if args.shortest_edge <= 0 or args.longest_edge <= 0:
            raise ValueError("Image edge sizes must be positive")

        processor_kwargs["images_kwargs"] = {
            "size": {
                "shortest_edge": args.shortest_edge,
                "longest_edge": args.longest_edge,
            }
        }

    effective_shortest_edge = (
        args.shortest_edge
        if args.shortest_edge is not None
        else int(processor.image_processor.size.shortest_edge)
    )
    effective_longest_edge = (
        args.longest_edge
        if args.longest_edge is not None
        else int(processor.image_processor.size.longest_edge)
    )

    print(
        "Image size: "
        f"shortest_edge={effective_shortest_edge}, "
        f"longest_edge={effective_longest_edge}"
    )

    model_config = AutoConfig.from_pretrained(model_id)
    model_config.disable_custom_kernels = True
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        config=model_config,
    )

    checkpoint_info = None
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        checkpoint_model_id = checkpoint.get("model_id")
        if checkpoint_model_id not in (None, model_id):
            raise ValueError(
                "Checkpoint model ID does not match inference model: "
                f"{checkpoint_model_id!r} != {model_id!r}"
            )

        model_state = checkpoint.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError(
                "Checkpoint does not contain a model_state dictionary"
            )

        incompatible = model.load_state_dict(
            model_state,
            strict=False,
        )
        if incompatible.unexpected_keys:
            raise ValueError(
                "Unexpected checkpoint keys: "
                f"{incompatible.unexpected_keys}"
            )

        checkpoint_info = {
            "path": str(checkpoint_path),
            "configuration": checkpoint.get("configuration"),
            "training_state": checkpoint.get("training_state"),
        }

        print(
            "Loaded fine-tuned checkpoint: "
            f"{checkpoint_path}",
            flush=True,
        )

    model.to(device)
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
            desc=f"Tiled inference: {prompt['name']}",
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
            windows = tile_windows(
                image.width,
                image.height,
                args.tile_size,
                args.tile_overlap,
            )

            tile_detections: list[dict] = []

            for left, top, right, bottom in windows:
                tile = image.crop((left, top, right, bottom))

                inputs = processor(
                    images=tile,
                    text=prompt["text"],
                    return_tensors="pt",
                    **processor_kwargs,
                )
                inputs = {
                    name: value.to(device)
                    for name, value in inputs.items()
                }

                with torch.inference_mode():
                    outputs = model(**inputs)

                postprocess_kwargs: dict[str, object] = {
                    "target_sizes": [(tile.height, tile.width)],
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

                for box, score, phrase in zip(boxes, scores, labels):
                    if hasattr(phrase, "item"):
                        phrase = phrase.item()

                    converted = xyxy_to_xywh(
                        box,
                        tile.width,
                        tile.height,
                    )
                    if converted is None:
                        continue

                    _, local_xyxy = converted
                    mapped_name = map_phrase_to_category(
                        phrase,
                        prompt["aliases"],
                    )
                    category_id = category_ids.get(mapped_name)

                    if category_id is None:
                        continue

                    local_x1, local_y1, local_x2, local_y2 = local_xyxy
                    global_xyxy = [
                        local_x1 + left,
                        local_y1 + top,
                        local_x2 + left,
                        local_y2 + top,
                    ]

                    converted_global = xyxy_to_xywh(
                        global_xyxy,
                        image.width,
                        image.height,
                    )
                    if converted_global is None:
                        continue

                    bbox, bbox_xyxy = converted_global
                    tile_detections.append(
                        {
                            "bbox": bbox,
                            "bbox_xyxy": bbox_xyxy,
                            "score": float(score),
                            "phrase": str(phrase),
                            "mapped_category_name": mapped_name,
                            "category_id": category_id,
                            "tile_origin": [left, top],
                        }
                    )

                del outputs
                del inputs

            detections = classwise_nms(
                tile_detections,
                args.nms_iou,
            )

            record = {
                "image_id": image_id,
                "file_name": image_record["file_name"],
                "width": int(image_record["width"]),
                "height": int(image_record["height"]),
                "tile_count": len(windows),
                "pre_nms_detection_count": len(tile_detections),
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
        "checkpoint": checkpoint_info,
        "prompt_name": prompt["name"],
        "prompt_text": prompt["text"],
        "aliases": prompt["aliases"],
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "image_size": {
            "shortest_edge": effective_shortest_edge,
            "longest_edge": effective_longest_edge,
        },
        "tiling": {
            "tile_size": args.tile_size,
            "overlap": args.tile_overlap,
            "classwise_nms_iou": args.nms_iou,
        },
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
