from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

from birdsai_pipeline.evaluate import threshold_metrics
from birdsai_pipeline.grounding_dino_inference import (
    classify_phrase,
    normalized_cxcywh_to_xywh,
)


def run_module(project_root: Path, module: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=project_root,
        check=True,
    )


def create_sequence(root: Path, split: str, sequence_id: str, class_id: int) -> None:
    image_dir = root / split / "images" / sequence_id
    annotation_dir = root / split / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in range(2):
        Image.new("RGB", (64, 48), color=(frame_id * 20, 0, 0)).save(
            image_dir / f"{sequence_id}_{frame_id:010d}.jpg"
        )
    # Frame 1 is deliberately empty so empty images are tested.
    annotation = [0, 1, 10, 8, 12, 9, class_id, 0 if class_id == 1 else 1, 1, 0]
    (annotation_dir / f"{sequence_id}.csv").write_text(
        ",".join(map(str, annotation)) + "\n", encoding="utf-8"
    )


def test_end_to_end(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"

    for index in range(4):
        create_sequence(dataset, "TrainReal", f"{index + 1:010d}_0000000000", index % 2)
    for index in range(2):
        create_sequence(dataset, "TestReal", f"{index + 101:010d}_0000000000", index % 2)

    run_module(
        project_root,
        "birdsai_pipeline.build_manifest",
        "--dataset-root",
        str(dataset),
        "--output-dir",
        str(output / "manifests"),
        "--strict",
    )
    manifest = pd.read_csv(output / "manifests" / "trainreal_frames.csv")
    assert len(manifest) == 8
    assert int(manifest["num_objects"].sum()) == 4
    assert int((manifest["num_objects"] == 0).sum()) == 4

    run_module(
        project_root,
        "birdsai_pipeline.create_splits",
        "--sequences",
        str(output / "manifests" / "sequences.csv"),
        "--output-dir",
        str(output / "splits"),
        "--val-count",
        "1",
        "--trials",
        "100",
    )

    for split_name, manifest_name, sequence_name in [
        ("train", "trainreal_frames.csv", "train_sequences.txt"),
        ("validation", "trainreal_frames.csv", "validation_sequences.txt"),
        ("test", "testreal_frames.csv", "test_sequences.txt"),
    ]:
        run_module(
            project_root,
            "birdsai_pipeline.mot_to_coco",
            "--dataset-root",
            str(dataset),
            "--manifest",
            str(output / "manifests" / manifest_name),
            "--sequences",
            str(output / "splits" / sequence_name),
            "--split-name",
            split_name,
            "--output",
            str(output / "coco" / f"{split_name}.json"),
        )

    train_coco = json.loads((output / "coco" / "train.json").read_text())
    assert len(train_coco["images"]) == 6
    assert len(train_coco["annotations"]) == 3
    assert {category["name"] for category in train_coco["categories"]} == {"animal", "person"}

    run_module(
        project_root,
        "birdsai_pipeline.visualize",
        "--dataset-root",
        str(dataset),
        "--coco",
        str(output / "coco" / "test.json"),
        "--output-dir",
        str(output / "figures"),
        "--num-random",
        "1",
        "--num-empty",
        "1",
        "--num-crowded",
        "1",
    )
    assert len(list((output / "figures").glob("*.jpg"))) >= 2

    perfect_predictions = [
        {
            "image_id": annotation["image_id"],
            "category_id": annotation["category_id"],
            "bbox": annotation["bbox"],
            "score": 0.99,
        }
        for annotation in train_coco["annotations"]
    ]
    metrics = threshold_metrics(train_coco, perfect_predictions, 0.25, 0.50)
    assert metrics["overall"]["precision"] == 1.0
    assert metrics["overall"]["recall"] == 1.0


def test_grounding_dino_prediction_conversion() -> None:
    prompt = {
        "aliases": {
            "person": ["person", "human"],
            "animal": ["animal", "elephant", "lion"],
        }
    }
    assert classify_phrase("person", prompt) == 2
    assert classify_phrase("an elephant", prompt) == 1
    assert classify_phrase("unknown object", prompt) is None

    bbox = normalized_cxcywh_to_xywh([0.5, 0.5, 0.25, 0.5], 640, 480)
    assert bbox == [240.0, 120.0, 160.0, 240.0]
    clipped = normalized_cxcywh_to_xywh([0.0, 0.0, 0.2, 0.2], 100, 100)
    assert clipped == [0.0, 0.0, 10.0, 10.0]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary_directory:
        test_end_to_end(Path(temporary_directory))
    test_grounding_dino_prediction_conversion()
    print("Smoke test passed")
