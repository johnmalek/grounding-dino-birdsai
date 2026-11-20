from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

from birdsai_pipeline.detection_utils import (
    axis_origins,
    map_phrase_to_category,
    tile_windows,
)
from birdsai_pipeline.evaluate import threshold_metrics


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


def test_phrase_mapping_rejects_ambiguous_phrases() -> None:
    aliases = {"animal": ["animal"], "person": ["person"]}
    assert map_phrase_to_category("animal", aliases) == "animal"
    assert map_phrase_to_category("person", aliases) == "person"
    assert map_phrase_to_category("person animal", aliases) is None
    assert map_phrase_to_category("background", aliases) is None


def test_tile_windows_cover_image_edges() -> None:
    assert axis_origins(640, 384, 0.20) == [0, 256]
    assert axis_origins(512, 384, 0.20) == [0, 128]
    windows = tile_windows(640, 512, 384, 0.20)
    assert len(windows) == 4
    assert (0, 0, 384, 384) in windows
    assert (256, 128, 640, 512) in windows


def test_versioned_configuration_paths_are_portable() -> None:
    project_root = Path(__file__).parents[1]
    config = json.loads(
        (project_root / "configs" / "experiment_b_training.json").read_text(
            encoding="utf-8"
        )
    )
    for value in config["data"].values():
        if isinstance(value, str):
            assert not value.startswith("~")
            assert not Path(value).is_absolute()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary_directory:
        test_end_to_end(Path(temporary_directory))
    test_phrase_mapping_rejects_ambiguous_phrases()
    test_tile_windows_cover_image_edges()
    test_versioned_configuration_paths_are_portable()
    print("Smoke test passed")
