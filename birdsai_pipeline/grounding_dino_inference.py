from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable

from .common import write_json


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value or "prompt"


def classify_phrase(phrase: str, prompt: dict) -> int | None:
    """Map a Grounding DINO phrase to BIRDSAI's broad COCO categories."""
    normalized = re.sub(r"[^a-z0-9 ]+", " ", phrase.lower())
    normalized = " ".join(normalized.split())
    aliases = prompt.get("aliases", {})
    # Check person first because it is semantically narrower than animal.
    for alias in aliases.get("person", []):
        if re.search(rf"\b{re.escape(alias.lower())}\b", normalized):
            return 2
    for alias in aliases.get("animal", []):
        if re.search(rf"\b{re.escape(alias.lower())}\b", normalized):
            return 1
    return None


def normalized_cxcywh_to_xywh(
    box: Iterable[float], image_width: int, image_height: int
) -> list[float] | None:
    cx, cy, width, height = map(float, box)
    x1 = max(0.0, (cx - width / 2.0) * image_width)
    y1 = max(0.0, (cy - height / 2.0) * image_height)
    x2 = min(float(image_width), (cx + width / 2.0) * image_width)
    y2 = min(float(image_height), (cy + height / 2.0) * image_height)
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    return [x1, y1, clipped_width, clipped_height]


def atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_prompt_config(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", [])
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    required_alias_groups = {"animal", "person"}
    for prompt in prompts:
        if not prompt.get("name") or not prompt.get("text"):
            raise ValueError("Every prompt needs a name and text")
        if not required_alias_groups.issubset(prompt.get("aliases", {})):
            raise ValueError(f"Prompt {prompt['name']} must define animal and person aliases")
    return prompts


def run_prompt(
    model: object,
    prompt: dict,
    images: list[dict],
    dataset_root: Path,
    output_path: Path,
    device: str,
    inference_box_threshold: float,
    text_threshold: float,
    max_detections: int,
    checkpoint_every: int,
    resume: bool,
) -> dict:
    try:
        from groundingdino.util.inference import load_image, predict
    except ImportError as exc:
        raise RuntimeError(
            "Grounding DINO is not importable. Install the official repository "
            "in this environment with `pip install --no-build-isolation -e .`."
        ) from exc

    progress_path = output_path.with_suffix(".progress.json")
    predictions: list[dict] = []
    completed: set[int] = set()
    if resume and output_path.exists() and progress_path.exists():
        predictions = json.loads(output_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = {int(image_id) for image_id in progress.get("completed_image_ids", [])}
        print(f"Resuming {prompt['name']}: {len(completed)} images already completed")

    unclassified_phrases: dict[str, int] = {}
    started = time.time()
    pending_since_checkpoint = 0
    for index, image_record in enumerate(images, start=1):
        image_id = int(image_record["id"])
        if image_id in completed:
            continue
        image_path = dataset_root / image_record["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        _, transformed = load_image(str(image_path))
        boxes, scores, phrases = predict(
            model=model,
            image=transformed,
            caption=prompt["text"],
            box_threshold=inference_box_threshold,
            text_threshold=text_threshold,
            device=device,
            remove_combined=True,
        )

        candidates = []
        for box, score, phrase in zip(boxes.tolist(), scores.tolist(), phrases):
            category_id = classify_phrase(phrase, prompt)
            if category_id is None:
                unclassified_phrases[phrase] = unclassified_phrases.get(phrase, 0) + 1
                continue
            bbox = normalized_cxcywh_to_xywh(
                box, int(image_record["width"]), int(image_record["height"])
            )
            if bbox is None:
                continue
            candidates.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": float(score),
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        predictions.extend(candidates[:max_detections])
        completed.add(image_id)
        pending_since_checkpoint += 1

        if pending_since_checkpoint >= checkpoint_every:
            atomic_write_json(output_path, predictions)
            atomic_write_json(
                progress_path,
                {"prompt": prompt["name"], "completed_image_ids": sorted(completed)},
            )
            pending_since_checkpoint = 0
            elapsed = time.time() - started
            print(
                f"{prompt['name']}: {len(completed)}/{len(images)} images, "
                f"{len(predictions)} predictions, {elapsed / 60:.1f} min"
            )

    atomic_write_json(output_path, predictions)
    atomic_write_json(
        progress_path,
        {"prompt": prompt["name"], "completed_image_ids": sorted(completed), "complete": True},
    )
    return {
        "prompt_name": prompt["name"],
        "prompt_text": prompt["text"],
        "images": len(images),
        "predictions": len(predictions),
        "unclassified_phrases": unclassified_phrases,
        "elapsed_seconds": time.time() - started,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pretrained Grounding DINO on a BIRDSAI COCO split."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Grounding DINO model config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt-config", type=Path)
    prompt_source.add_argument(
        "--selected-config",
        type=Path,
        help="selected_experiment_a_config.json produced by the validation sweep",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-box-threshold", type=float, default=0.05)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Debug on the first N images")
    args = parser.parse_args()

    try:
        import torch
        from groundingdino.util.inference import load_model
    except ImportError as exc:
        raise RuntimeError("PyTorch and the official Grounding DINO package are required") from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    images = sorted(coco["images"], key=lambda item: int(item["id"]))
    if args.limit:
        images = images[: args.limit]
    if args.selected_config:
        selected = json.loads(args.selected_config.read_text(encoding="utf-8"))
        prompts = [
            {
                "name": selected["prompt_name"],
                "text": selected["prompt_text"],
                "aliases": selected["aliases"],
            }
        ]
    else:
        assert args.prompt_config is not None
        prompts = load_prompt_config(args.prompt_config)
    selected_prompt_index = args.prompt_index
    if args.prompt_index is not None:
        if args.selected_config:
            raise ValueError("--prompt-index cannot be combined with --selected-config")
        if not 0 <= args.prompt_index < len(prompts):
            raise IndexError(f"prompt-index must be between 0 and {len(prompts) - 1}")
        prompts = [prompts[args.prompt_index]]

    print(f"Loading Grounding DINO on {args.device}")
    model = load_model(str(args.config), str(args.checkpoint), device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for prompt in prompts:
        output_path = args.output_dir / f"{slugify(prompt['name'])}.json"
        summary = run_prompt(
            model=model,
            prompt=prompt,
            images=images,
            dataset_root=args.dataset_root,
            output_path=output_path,
            device=args.device,
            inference_box_threshold=args.inference_box_threshold,
            text_threshold=args.text_threshold,
            max_detections=args.max_detections,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
        )
        summary.update(
            {
                "model_config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "coco_split": str(args.coco),
                "inference_box_threshold": args.inference_box_threshold,
                "text_threshold": args.text_threshold,
                "max_detections": args.max_detections,
            }
        )
        write_json(args.output_dir / f"{slugify(prompt['name'])}.metadata.json", summary)
        summaries.append(summary)

    summary_name = (
        f"inference_summary_{selected_prompt_index}.json"
        if selected_prompt_index is not None
        else "inference_summary.json"
    )
    write_json(args.output_dir / summary_name, summaries)
    print(f"Finished {len(prompts)} prompt(s). Predictions: {args.output_dir}")


if __name__ == "__main__":
    main()
