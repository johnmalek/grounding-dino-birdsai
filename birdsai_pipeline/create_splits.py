from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from .common import write_lines


BALANCE_COLUMNS = [
    "num_frames",
    "num_objects",
    "num_animals",
    "num_humans",
    "num_small",
    "num_medium",
    "num_large",
    "num_occluded",
    "num_noisy",
]


def subset_score(table: pd.DataFrame, indices: tuple[int, ...], target_fraction: float) -> float:
    selected = table.iloc[list(indices)]
    score = 0.0
    for column in BALANCE_COLUMNS:
        total = float(table[column].sum())
        if total > 0:
            observed_fraction = float(selected[column].sum()) / total
            score += abs(observed_fraction - target_fraction)

    # Strongly discourage a validation set with no examples of either broad class.
    if selected["num_animals"].sum() == 0:
        score += 10.0
    if selected["num_humans"].sum() == 0:
        score += 10.0
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-free BIRDSAI sequence splits.")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=6)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sequences = pd.read_csv(args.sequences)
    train_real = sequences.loc[sequences["split"] == "TrainReal"].reset_index(drop=True)
    test_real = sequences.loc[sequences["split"] == "TestReal"].reset_index(drop=True)
    if train_real.empty or test_real.empty:
        raise ValueError("sequences.csv must contain both TrainReal and TestReal rows")
    if not 0 < args.val_count < len(train_real):
        raise ValueError("val-count must be between 1 and the number of TrainReal sequences minus 1")

    missing = [column for column in BALANCE_COLUMNS if column not in train_real]
    if missing:
        raise ValueError(f"Missing balance columns: {missing}")

    rng = random.Random(args.seed)
    target_fraction = args.val_count / len(train_real)
    best_indices: tuple[int, ...] | None = None
    best_score = float("inf")

    # Include evenly spaced candidates plus deterministic random search.
    initial = tuple(range(args.val_count))
    candidates = [initial]
    for _ in range(args.trials):
        candidates.append(tuple(sorted(rng.sample(range(len(train_real)), args.val_count))))

    for indices in candidates:
        score = subset_score(train_real, indices, target_fraction)
        if score < best_score:
            best_score = score
            best_indices = indices

    assert best_indices is not None
    validation = train_real.iloc[list(best_indices)].copy()
    training = train_real.drop(index=list(best_indices)).copy()

    train_ids = sorted(training["sequence_id"].astype(str))
    validation_ids = sorted(validation["sequence_id"].astype(str))
    test_ids = sorted(test_real["sequence_id"].astype(str))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(args.output_dir / "train_sequences.txt", train_ids)
    write_lines(args.output_dir / "validation_sequences.txt", validation_ids)
    write_lines(args.output_dir / "test_sequences.txt", test_ids)

    def totals(frame: pd.DataFrame) -> dict:
        return {
            "sequences": int(len(frame)),
            **{column: int(frame[column].sum()) for column in BALANCE_COLUMNS},
        }

    report = {
        "seed": args.seed,
        "trials": args.trials,
        "validation_count": args.val_count,
        "balance_score": best_score,
        "train": totals(training),
        "validation": totals(validation),
        "test": totals(test_real),
        "validation_sequence_ids": validation_ids,
    }
    (args.output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"Train: {len(train_ids)} sequences")
    print(f"Validation: {len(validation_ids)} sequences")
    print(f"Test: {len(test_ids)} sequences")
    print(f"Validation balance score: {best_score:.6f}")


if __name__ == "__main__":
    main()
