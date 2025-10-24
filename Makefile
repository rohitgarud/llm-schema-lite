.PHONY: help install sync install-dev install-dspy install-pre-commit pre-commit-run
.PHONY: test test-cov test-cov-full test-parallel test-fast test-slow test-dspy
.PHONY: lint check format clean build changelog release_notes
.PHONY: publish-test publish update venv setup

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
sync:  ## Sync dependencies using uv
	uv sync --all-extras

install:  ## Install package
	uv pip install -e .

install-dev:  ## Install package with development dependencies
	uv pip install -e ".[dev]"

install-dspy:  ## Install package with DSPy integration
	uv pip install -e ".[dspy]"

install-pre-commit:  ## Install pre-commit hooks
	uv pip install pre-commit
	pre-commit uninstall; pre-commit install; pre-commit install --hook-type commit-msg

# Testing
test:  ## Run tests with pytest
	pytest -rP -n auto --show-capture=no

test-cov:  ## Run tests with coverage report (core only, no DSPy)
	pytest --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml

test-cov-full:  ## Run tests with full coverage including DSPy integration
	@echo "Installing DSPy for full coverage..."
	@uv pip install -e ".[dspy]" > /dev/null 2>&1 || true
	@echo "Running tests with DSPy integration..."
	.venv/bin/python -m pytest --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo ""
	@echo "✓ Full coverage report generated (includes DSPy)"
	@echo "→ HTML report: htmlcov/index.html"
	@echo "→ XML report: coverage.xml"

test-dspy:  ## Run only DSPy integration tests
	@uv pip install -e ".[dspy]" > /dev/null 2>&1 || echo "DSPy already installed"
	.venv/bin/python -m pytest tests/test_dspy*.py -v

test-parallel:  ## Run tests in parallel (alias for test)
	pytest -rP -n auto

test-fast:  ## Run tests excluding slow ones
	pytest -m "not slow" -rP -n auto

test-slow:  ## Run only slow tests
	pytest -m slow -rP

# Code Quality
pre-commit-run:  ## Run pre-commit on all files
	pre-commit run --all-files

check:  ## Quick health check (fast lint + type check)
	ruff check src tests
	mypy src --no-error-summary

lint:  ## Run all linters (ruff, mypy, bandit)
	ruff check src tests
	mypy src
	bandit -c pyproject.toml -r src
	pre-commit run --all-files

format:  ## Format code with ruff
	ruff format src tests
	ruff check --fix src tests



# Maintenance
clean:  ## Clean build artifacts and cache files
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/ .mypy_cache .ruff_cache

# Building
build:  ## Build package distribution
	uv build

changelog:  ## Generate changelog
	git-changelog -o CHANGELOG.md

# Publishing
publish-test:  ## Publish to Test PyPI
	hatch publish -r test

publish:  ## Publish to PyPI
	hatch publish

# Setup & Update
venv:  ## Create virtual environment
	uv venv

setup: venv install-dev install-dspy install-pre-commit  ## Complete setup for development
	@echo "✓ Development environment setup complete (including DSPy)!"
	@echo "Activate the virtual environment with: source .venv/bin/activate"
	@echo ""
	@echo "Available test commands:"
	@echo "  make test          - Run all tests"
	@echo "  make test-cov      - Coverage report (core only)"
	@echo "  make test-cov-full - Full coverage report (includes DSPy)"
	@echo "  make test-dspy     - Run only DSPy tests"

update: clean sync install-pre-commit lint  ## Full project update
	@echo "✓ Project updated successfully!"

release_notes:  ## Generate release notes
	sed -n '/## \[v0.2.1\]/,/## \[/p' CHANGELOG.md | head -n -1
