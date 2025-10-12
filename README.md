# llm-schema-lite

[![PyPI version](https://badge.fury.io/py/llm-schema-lite.svg)](https://badge.fury.io/py/llm-schema-lite)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm-schema-lite.svg)](https://pypi.org/project/llm-schema-lite/)
[![CI](https://github.com/rohitgarud/llm-schema-lite/workflows/CI/badge.svg)](https://github.com/rohitgarud/llm-schema-lite/actions)
[![codecov](https://codecov.io/gh/rohitgarud/llm-schema-lite/branch/main/graph/badge.svg)](https://codecov.io/gh/rohitgarud/llm-schema-lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

LLM-ify your JSON schemas.

## Installation

You can install llm-schema-lite using pip:

```bash
pip install llm-schema-lite
```

Or using uv:

```bash
uv pip install llm-schema-lite
```

## Development

This project uses `uv` for package management and includes pre-commit hooks for code quality.

### Setup Development Environment

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Quick setup with Make:
```bash
make setup
```

Or manually:
```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

### Available Make Commands

Run `make help` to see all available commands:

- `make install` - Install package
- `make install-dev` - Install with dev dependencies
- `make test` - Run tests
- `make test-cov` - Run tests with coverage
- `make test-parallel` - Run tests in parallel (faster)
- `make test-fast` - Run tests excluding slow ones
- `make lint` - Run all linters
- `make format` - Format code
- `make build` - Build package
- `make changelog` - Generate changelog
- `make clean` - Clean build artifacts

### Running Tests

```bash
make test
# or
pytest
```

### Code Quality

The project uses several tools to maintain code quality:

- **Ruff**: Fast Python linter and formatter (replaces flake8, isort, and more)
- **MyPy**: Static type checker for type safety
- **Bandit**: Security vulnerability scanner
- **Pre-commit**: Git hooks for automated checks
- **Pytest**: Testing framework with coverage reporting

```bash
# Format code
make format

# Run linters
make lint

# Run pre-commit on all files
make pre-commit-run

# Run tests in parallel (faster for large test suites)
make test-parallel
```

### Changelog Management

This project uses [git-changelog](https://github.com/pawamoy/git-changelog) with conventional commits:

```bash
# Generate changelog
make changelog
```

Commit message format:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Test changes
- `chore:` - Maintenance tasks
- `perf:` - Performance improvements

## Building and Publishing

### Build the package

```bash
uv build
```

### Publish to PyPI

```bash
# Install twine if needed
uv pip install twine

# Upload to PyPI
twine upload dist/*
```

### Publish to TestPyPI (for testing)

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

## License

See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
