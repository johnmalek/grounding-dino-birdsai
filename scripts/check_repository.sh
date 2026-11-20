#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"
export PYTHONPATH="$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Checking Python syntax..."
python -m compileall -q birdsai_pipeline tests

echo "Checking JSON configuration files..."
for config in configs/*.json; do
  python -m json.tool "$config" >/dev/null
done

echo "Checking shell and Slurm scripts..."
for script in scripts/*.sh slurm/*.sh slurm/*.slurm; do
  [[ -e "$script" ]] || continue
  bash -n "$script"
done

echo "Checking for user-specific absolute paths in tracked project files..."
if grep -R -n -E \
  --exclude='check_repository.sh' \
  '/home/[[:alnum:]_.-]+|/mnt/beegfs|computer_vision/birdsai_data_pipeline' \
  README.md ARTIFACTS.md configs docs scripts slurm birdsai_pipeline tests; then
  echo "Found a user-specific absolute path." >&2
  exit 1
fi

echo "Running unit and pipeline tests..."
if python -c 'import pytest' 2>/dev/null; then
  python -m pytest -q
else
  echo "pytest is not installed; running the dependency-light smoke suite."
  python tests/test_pipeline.py
fi

echo "Repository checks passed."
