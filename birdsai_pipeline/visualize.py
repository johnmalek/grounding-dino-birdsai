from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {1: "#32CD32", 2: "#FF3B30"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw BIRDSAI COCO ground-truth boxes.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-random", type=int, default=10)
    parser.add_argument("--num-empty", type=int, default=5)
    parser.add_argument("--num-crowded", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = json.loads(args.coco.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data["images"]}
    categories = {int(item["id"]): item["name"] for item in data["categories"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)

    rng = random.Random(args.seed)
    populated = [image_id for image_id in images if by_image[image_id]]
    empty = [image_id for image_id in images if not by_image[image_id]]
    random_ids = rng.sample(populated, min(args.num_random, len(populated)))
    empty_ids = rng.sample(empty, min(args.num_empty, len(empty)))
    crowded_ids = sorted(populated, key=lambda image_id: len(by_image[image_id]), reverse=True)[
        : args.num_crowded
    ]
    selected = list(dict.fromkeys(random_ids + empty_ids + crowded_ids))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for image_id in selected:
        record = images[image_id]
        image_path = args.dataset_root / record["file_name"]
        with Image.open(image_path) as opened:
            canvas = opened.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        class_counts = Counter()
        for annotation in by_image[image_id]:
            x, y, width, height = map(float, annotation["bbox"])
            category_id = int(annotation["category_id"])
            color = COLORS.get(category_id, "yellow")
            label = categories.get(category_id, str(category_id))
            class_counts[label] += 1
            draw.rectangle([x, y, x + width, y + height], outline=color, width=2)
            text = f"{label} id={annotation.get('object_id', '?')}"
            text_box = draw.textbbox((x, y), text, font=font)
            draw.rectangle(text_box, fill=color)
            draw.text((x, y), text, fill="black", font=font)

        summary = ", ".join(f"{key}={value}" for key, value in sorted(class_counts.items()))
        if not summary:
            summary = "empty frame"
        draw.rectangle([0, 0, min(canvas.width, 420), 18], fill="black")
        draw.text((3, 3), summary, fill="white", font=font)
        output_name = f"{record.get('sequence_id', 'sequence')}_{int(record.get('frame_id', image_id)):010d}.jpg"
        canvas.save(args.output_dir / output_name, quality=95)

    print(f"Saved {len(selected)} visual checks to {args.output_dir}")


if __name__ == "__main__":
    main()
