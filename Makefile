.PHONY: install install-ml test smoke check clean

install:
	python -m pip install -e ".[dev]"

install-ml:
	python -m pip install -e ".[ml,dev]"

test:
	python -m pytest

smoke:
	python tests/test_pipeline.py

check:
	bash scripts/check_repository.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache build dist *.egg-info
