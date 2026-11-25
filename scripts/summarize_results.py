#!/usr/bin/env python3
"""Collect all expected experiment metrics into one JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing expected result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> dict:
    report = read_json(path / "metrics.json")
    overall = report["coco"]["overall"]
    threshold = report["threshold_metrics"]["overall"]
    per_class = report["coco"]["per_class"]

    return {
        "mAP_50_95": overall["AP_50_95"],
        "AP_50": overall["AP_50"],
        "AP_75": overall["AP_75"],
        "precision": threshold["precision"],
        "recall": threshold["recall"],
        "f1": threshold["f1"],
        "false_positives_per_frame": threshold["false_positives_per_frame"],
        "animal_AP_50": per_class["animal"]["AP_50"],
        "person_AP_50": per_class["person"]["AP_50"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.output_root

    result_paths = {
        "experiment_a_validation_person_animal":
            root / "experiment_a/validation/person_animal",
        "experiment_a_test_person_animal":
            root / "experiment_a/test/person_animal",
        "experiment_b_validation_B1_epoch1":
            root / "experiment_b/validation/B1/epoch_1",
        "experiment_b_validation_B2_epoch1":
            root / "experiment_b/validation/B2/epoch_1",
        "experiment_b_validation_B3_epoch1":
            root / "experiment_b/validation/B3/epoch_1",
        "experiment_b_scale_native":
            root / "experiment_b/scale_validation/B2_epoch1_native",
        "experiment_b_scale_global_1024":
            root / "experiment_b/scale_validation/B2_epoch1_1024",
        "experiment_b_scale_tiled_384":
            root / "experiment_b/scale_validation/B2_epoch1_tiled_384",
        "experiment_b_final_test_tiled":
            root / "experiment_b/test/B2_epoch1_tiled_384",
    }

    results = {
        name: read_metrics(path)
        for name, path in result_paths.items()
    }

    payload = {
        "output_root": str(root.resolve()),
        "results": results,
    }

    destination = args.output or root / "reproducibility_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
