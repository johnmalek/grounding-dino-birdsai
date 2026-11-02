from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


ANNOTATION_COLUMNS = [
    "frame_id",
    "object_id",
    "x",
    "y",
    "width",
    "height",
    "class_id",
    "species",
    "occluded",
    "noisy",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_annotations(path: Path) -> pd.DataFrame:
    """Load a headerless BIRDSAI MOT CSV and enforce its ten-column schema."""
    raw = pd.read_csv(path, header=None, comment="#", skip_blank_lines=True)
    if raw.shape[1] != len(ANNOTATION_COLUMNS):
        raise ValueError(
            f"{path} has {raw.shape[1]} columns; expected "
            f"{len(ANNOTATION_COLUMNS)}"
        )
    raw.columns = ANNOTATION_COLUMNS
    for column in ANNOTATION_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw


def image_files(sequence_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_frame_id(image_path: Path, sequence_id: str) -> int:
    """Extract the final numeric component from a BIRDSAI frame filename."""
    stem = image_path.stem
    suffix = stem[len(sequence_id) :] if stem.startswith(sequence_id) else stem
    matches = re.findall(r"(\d+)", suffix)
    if not matches:
        raise ValueError(f"Cannot parse frame ID from {image_path.name}")
    return int(matches[-1])


def discover_split(dataset_root: Path, split: str) -> tuple[dict[str, Path], dict[str, Path]]:
    split_root = dataset_root / split
    image_root = split_root / "images"
    annotation_root = split_root / "annotations"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_root}")
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"Missing annotation directory: {annotation_root}")

    images = {path.name: path for path in image_root.iterdir() if path.is_dir()}
    annotations = {
        path.stem: path
        for path in annotation_root.glob("*.csv")
        if path.is_file()
    }
    return images, annotations


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def read_sequence_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def normalize_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
