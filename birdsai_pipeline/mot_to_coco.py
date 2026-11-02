from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .common import load_annotations, read_sequence_list, write_json


CATEGORIES = [
    {"id": 1, "name": "animal", "supercategory": "object"},
    {"id": 2, "name": "person", "supercategory": "object"},
]
CLASS_TO_CATEGORY = {0: 1, 1: 2}


def clipped_box(row: pd.Series, image_width: int, image_height: int) -> tuple[list[float], float] | None:
    x1 = max(0.0, float(row["x"]))
    y1 = max(0.0, float(row["y"]))
    x2 = min(float(image_width), float(row["x"] + row["width"]))
    y2 = min(float(image_height), float(row["y"] + row["height"]))
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    return [x1, y1, width, height], width * height


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BIRDSAI MOT annotations to COCO JSON.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--issues-output", type=Path)
    args = parser.parse_args()

    selected_sequences = set(read_sequence_list(args.sequences))
    manifest = pd.read_csv(args.manifest)
    manifest["sequence_id"] = manifest["sequence_id"].astype(str)
    manifest = manifest.loc[manifest["sequence_id"].isin(selected_sequences)].copy()
    if manifest.empty:
        raise ValueError("No manifest frames matched the requested sequence list")
    if manifest[["width", "height"]].isna().any().any():
        raise ValueError("Manifest contains images without valid dimensions; fix validation issues first")

    manifest = manifest.sort_values(["sequence_id", "frame_id", "image_path"])
    images: list[dict] = []
    annotations: list[dict] = []
    issues: list[dict] = []
    image_lookup: dict[tuple[str, int], tuple[int, int, int]] = {}

    for image_id, row in enumerate(manifest.itertuples(index=False), start=1):
        sequence_id = str(row.sequence_id)
        frame_id = int(row.frame_id)
        width = int(row.width)
        height = int(row.height)
        image_lookup[(sequence_id, frame_id)] = (image_id, width, height)
        images.append(
            {
                "id": image_id,
                "file_name": str(row.image_path).replace("\\", "/"),
                "width": width,
                "height": height,
                "sequence_id": sequence_id,
                "frame_id": frame_id,
            }
        )

    annotation_id = 1
    sequence_annotation_paths = (
        manifest[["sequence_id", "annotation_path"]]
        .drop_duplicates()
        .sort_values("sequence_id")
    )
    for sequence_id, relative_annotation in sequence_annotation_paths.itertuples(index=False):
        annotation_path = args.dataset_root / str(relative_annotation)
        frame = load_annotations(annotation_path)
        valid_numeric = ~frame.isna().any(axis=1)
        frame = frame.loc[valid_numeric].copy()
        frame["frame_id"] = frame["frame_id"].astype(int)

        for row_number, row in frame.iterrows():
            key = (str(sequence_id), int(row["frame_id"]))
            if key not in image_lookup:
                issues.append(
                    {
                        "sequence_id": sequence_id,
                        "frame_id": int(row["frame_id"]),
                        "row": int(row_number + 1),
                        "issue": "annotation_not_in_selected_images",
                    }
                )
                continue
            class_id = int(row["class_id"])
            if class_id not in CLASS_TO_CATEGORY:
                issues.append(
                    {
                        "sequence_id": sequence_id,
                        "frame_id": int(row["frame_id"]),
                        "row": int(row_number + 1),
                        "issue": f"unsupported_class_{class_id}",
                    }
                )
                continue

            image_id, image_width, image_height = image_lookup[key]
            converted = clipped_box(row, image_width, image_height)
            if converted is None:
                issues.append(
                    {
                        "sequence_id": sequence_id,
                        "frame_id": int(row["frame_id"]),
                        "row": int(row_number + 1),
                        "issue": "invalid_box_after_clipping",
                    }
                )
                continue
            bbox, area = converted
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": CLASS_TO_CATEGORY[class_id],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "object_id": int(row["object_id"]),
                    "species": int(row["species"]),
                    "occluded": int(row["occluded"]),
                    "noisy": int(row["noisy"]),
                    "sequence_id": str(sequence_id),
                    "frame_id": int(row["frame_id"]),
                }
            )
            annotation_id += 1

    coco = {
        "info": {
            "description": f"BIRDSAI {args.split_name} detection split",
            "version": "1.0",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }
    write_json(args.output, coco)
    if args.issues_output:
        write_json(args.issues_output, issues)

    print(f"Wrote {len(images)} images and {len(annotations)} annotations to {args.output}")
    print(f"Conversion issues: {len(issues)}")


if __name__ == "__main__":
    main()
