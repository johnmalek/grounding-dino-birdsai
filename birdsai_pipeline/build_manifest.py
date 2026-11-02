from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, TypeVar

import pandas as pd
from PIL import Image, UnidentifiedImageError

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    T = TypeVar("T")

    def tqdm(iterable: Iterable[T], **_: object) -> Iterable[T]:
        return iterable

from .common import (
    discover_split,
    image_files,
    load_annotations,
    normalize_relative,
    parse_frame_id,
)


def issue(issues: list[dict], split: str, sequence_id: str, kind: str, detail: str) -> None:
    issues.append(
        {
            "split": split,
            "sequence_id": sequence_id,
            "issue": kind,
            "detail": detail,
        }
    )


def annotation_statistics(frame_rows: pd.DataFrame) -> dict[str, int]:
    if frame_rows.empty:
        return {
            "num_objects": 0,
            "num_animals": 0,
            "num_humans": 0,
            "num_small": 0,
            "num_medium": 0,
            "num_large": 0,
            "num_occluded": 0,
            "num_noisy": 0,
        }
    area = frame_rows["width"] * frame_rows["height"]
    return {
        "num_objects": int(len(frame_rows)),
        "num_animals": int((frame_rows["class_id"] == 0).sum()),
        "num_humans": int((frame_rows["class_id"] == 1).sum()),
        "num_small": int((area < 200).sum()),
        "num_medium": int(((area >= 200) & (area <= 2000)).sum()),
        "num_large": int((area > 2000).sum()),
        "num_occluded": int((frame_rows["occluded"] == 1).sum()),
        "num_noisy": int((frame_rows["noisy"] == 1).sum()),
    }


def build_split(dataset_root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_dirs, annotation_paths = discover_split(dataset_root, split)
    issues: list[dict] = []

    for sequence_id in sorted(set(annotation_paths) - set(image_dirs)):
        issue(issues, split, sequence_id, "annotation_without_images", str(annotation_paths[sequence_id]))
    for sequence_id in sorted(set(image_dirs) - set(annotation_paths)):
        issue(issues, split, sequence_id, "images_without_annotation", str(image_dirs[sequence_id]))

    manifest_rows: list[dict] = []
    sequence_rows: list[dict] = []

    for sequence_id in tqdm(sorted(image_dirs), desc=f"Building {split} manifest"):
        sequence_dir = image_dirs[sequence_id]
        annotation_path = annotation_paths.get(sequence_id)
        if annotation_path is None:
            annotations = pd.DataFrame(columns=[])
        else:
            try:
                annotations = load_annotations(annotation_path)
            except Exception as exc:  # report the exact file and continue auditing
                issue(issues, split, sequence_id, "malformed_annotation", str(exc))
                annotations = pd.DataFrame(columns=[])

        if not annotations.empty:
            non_numeric = annotations.isna().any(axis=1)
            for row_index in annotations.index[non_numeric].tolist():
                issue(issues, split, sequence_id, "non_numeric_annotation", f"row={row_index + 1}")
            annotations = annotations.loc[~non_numeric].copy()
            annotations["frame_id"] = annotations["frame_id"].astype(int)

            invalid_class = ~annotations["class_id"].isin([0, 1])
            for row_index in annotations.index[invalid_class].tolist():
                issue(issues, split, sequence_id, "invalid_class", f"row={row_index + 1}")

            invalid_size = (annotations["width"] <= 0) | (annotations["height"] <= 0)
            for row_index in annotations.index[invalid_size].tolist():
                issue(issues, split, sequence_id, "non_positive_box", f"row={row_index + 1}")

            duplicate_rows = annotations.duplicated(keep=False)
            if duplicate_rows.any():
                issue(
                    issues,
                    split,
                    sequence_id,
                    "duplicate_annotations",
                    f"count={int(duplicate_rows.sum())}",
                )

        grouped = (
            {int(frame): rows for frame, rows in annotations.groupby("frame_id")}
            if not annotations.empty
            else {}
        )
        seen_frame_ids: set[int] = set()
        sequence_totals: Counter = Counter()

        for image_path in image_files(sequence_dir):
            try:
                frame_id = parse_frame_id(image_path, sequence_id)
            except ValueError as exc:
                issue(issues, split, sequence_id, "unparseable_frame_name", str(exc))
                continue

            if frame_id in seen_frame_ids:
                issue(issues, split, sequence_id, "duplicate_frame_id", f"frame_id={frame_id}")
            seen_frame_ids.add(frame_id)

            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
            except (OSError, UnidentifiedImageError) as exc:
                issue(issues, split, sequence_id, "corrupt_image", f"{image_path}: {exc}")
                width, height = None, None

            frame_annotations = grouped.get(frame_id, pd.DataFrame())
            stats = annotation_statistics(frame_annotations)
            sequence_totals.update(stats)

            if width is not None and not frame_annotations.empty:
                outside = (
                    (frame_annotations["x"] >= width)
                    | (frame_annotations["y"] >= height)
                    | ((frame_annotations["x"] + frame_annotations["width"]) <= 0)
                    | ((frame_annotations["y"] + frame_annotations["height"]) <= 0)
                )
                for row_index in frame_annotations.index[outside].tolist():
                    issue(
                        issues,
                        split,
                        sequence_id,
                        "box_outside_image",
                        f"frame={frame_id}, row={row_index + 1}",
                    )

            manifest_rows.append(
                {
                    "split": split,
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "image_path": normalize_relative(image_path, dataset_root),
                    "width": width,
                    "height": height,
                    "annotation_path": (
                        normalize_relative(annotation_path, dataset_root)
                        if annotation_path is not None
                        else ""
                    ),
                    **stats,
                }
            )

        for annotated_frame in sorted(set(grouped) - seen_frame_ids):
            issue(
                issues,
                split,
                sequence_id,
                "annotation_for_missing_frame",
                f"frame_id={annotated_frame}",
            )

        sequence_rows.append(
            {
                "split": split,
                "sequence_id": sequence_id,
                "num_frames": len(seen_frame_ids),
                **{key: int(sequence_totals[key]) for key in annotation_statistics(pd.DataFrame()).keys()},
            }
        )

    return (
        pd.DataFrame(manifest_rows),
        pd.DataFrame(sequence_rows),
        pd.DataFrame(issues, columns=["split", "sequence_id", "issue", "detail"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate BIRDSAI frame manifests.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["TrainReal", "TestReal"])
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any issue is found.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_sequences = []
    all_issues = []
    for split in args.splits:
        manifest, sequences, issues = build_split(args.dataset_root.resolve(), split)
        manifest_path = args.output_dir / f"{split.lower()}_frames.csv"
        manifest.to_csv(manifest_path, index=False)
        all_sequences.append(sequences)
        all_issues.append(issues)
        print(f"{split}: {len(sequences)} sequences, {len(manifest)} frames, {len(issues)} issues")

    sequences = pd.concat(all_sequences, ignore_index=True)
    issues = pd.concat(all_issues, ignore_index=True)
    sequences.to_csv(args.output_dir / "sequences.csv", index=False)
    issues.to_csv(args.output_dir / "annotation_issues.csv", index=False)

    print(f"Wrote manifests to {args.output_dir}")
    if args.strict and not issues.empty:
        raise SystemExit(f"Validation found {len(issues)} issue(s); see annotation_issues.csv")


if __name__ == "__main__":
    main()
