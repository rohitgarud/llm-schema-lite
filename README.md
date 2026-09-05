# llm-schema-lite

[![PyPI version](https://img.shields.io/pypi/v/llm-schema-lite)](https://pypi.org/project/llm-schema-lite/)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm-schema-lite.svg)](https://pypi.org/project/llm-schema-lite/)
[![CI](https://github.com/rohitgarud/llm-schema-lite/workflows/CI/badge.svg)](https://github.com/rohitgarud/llm-schema-lite/actions)
[![codecov](https://codecov.io/gh/rohitgarud/llm-schema-lite/branch/main/graph/badge.svg)](https://codecov.io/gh/rohitgarud/llm-schema-lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Transform verbose JSON schemas into LLM-friendly formats. Reduce token usage by **60-85%** while preserving essential type information and integrating validation constraints directly into type descriptions for optimal LLM readability. Includes robust JSON/YAML parsing with automatic error recovery and enhanced constraint integration across all formatters.

---

## 📦 Installation

```bash
pip install llm-schema-lite

# With the DSPy integration
pip install "llm-schema-lite[dspy]"
```

Or with uv:

```bash
uv pip install llm-schema-lite
uv pip install "llm-schema-lite[dspy]"
```

## 🚀 Quick Start

```python
from pydantic import BaseModel, Field

from llm_schema_lite import loads, simplify_schema


class Address(BaseModel):
    """A postal address."""

    street: str
    city: str


class User(BaseModel):
    """A user account."""

    name: str = Field(min_length=2, description="Full name")
    age: int = Field(ge=0, le=120)
    email: str | None = None
    address: Address


# 1. Turn a Pydantic model into a schema an LLM can read cheaply.
schema = simplify_schema(User)
print(schema.to_string())
print(f"{schema.token_count()} tokens")

# 2. Read the model's reply back. Without `schema=`, loads() returns a plain dict.
reply = '{"name": "Ada", "age": 36, "address": {"street": "1 Main St", "city": "Springfield"}}'
data = loads(reply)
print(data["name"])

# 3. With `schema=`, loads() returns a (model instance, metadata) 2-tuple instead.
user, metadata = loads(reply, schema=User)
print(user.name, user.age)
```

`simplify_schema(User).to_string()` prints:

```
// A user account.
// Fields marked with * are required
{
  name*: string (>= 2 chars) // Full name,
  age*: int (0 to 120),
  email: string OR null // (default=null),
  address*: { // A postal address.
    street*: string,
    city*: string
  }
}
```

Fields marked `*` are required. Constraints are folded into the type line rather than
listed separately, which is where most of the token saving comes from.

## 🤖 DSPy Integration

`pip install "llm-schema-lite[dspy]"` (DSPy `>=3.3.1`) adds `StructuredOutputAdapter`, a
drop-in DSPy adapter that renders signature schemas in the compact format above instead of
full JSON Schema.

```python
import dspy
from pydantic import BaseModel

from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter


class Answer(BaseModel):
    """A model's answer with a confidence score."""

    answer: str
    confidence: float


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: Answer = dspy.OutputField()


adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
dspy.configure(adapter=adapter)  # add lm=dspy.LM("openai/gpt-4o-mini") to make real calls

# Inspect exactly what the adapter puts in the system prompt - no LM required.
print(adapter.format_field_structure(QA))

# ...and how it reads a reply back.
print(adapter.parse(QA, '{"answer": {"answer": "42", "confidence": 0.9}}'))
```

`output_mode` selects what the model is asked to produce:

| `output_mode` | The model is asked for | Per-field streaming |
|---|---|---|
| `OutputMode.JSON` | a JSON object described by the full JSON Schema — verbose, compatible with OpenAI structured outputs | supported |
| `OutputMode.JSONISH` *(default)* | a JSON object described by the compact schema above | supported |
| `OutputMode.YAML` *(experimental)* | YAML described by the compact schema | **not supported** — raises `StreamingNotSupportedError` before any request |

Every other constructor option — `formatter_config` / `parse_config` forwarding,
`prompt_layout`, the `json_object` response-format flag and its tool-call interaction,
streaming registration and known limits — is documented in
[the DSPy integration README](src/llm_schema_lite/dspy_integration/README.md).
Adapter benchmarks live under `benchmarking/dspy_adapters/` and run with `make bench-dspy`.

---

## 🛠️ Development

### Setup Development Environment

This project uses `uv` for package management and includes pre-commit hooks for code quality.

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Quick setup with Make**:
```bash
make setup
```

Or manually:
```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with all dependencies
uv pip install -e ".[dev,dspy]"

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

### Available Make Commands

Run `make help` to see all available commands:

```bash
# Installation
make install              # Install package
make install-dev          # Install with dev dependencies
make install-dspy         # Install with DSPy support
make sync                 # Sync all dependencies

# Testing
make test                 # Run tests
make test-cov             # Run tests with coverage (core only)
make test-cov-full        # Run tests with full coverage (includes DSPy)
make test-dspy            # Run only DSPy integration tests
make test-parallel        # Run tests in parallel (faster)
make test-fast            # Run tests excluding slow ones

# Code Quality
make lint                 # Run all linters (ruff, mypy, bandit)
make format               # Format code with ruff
make check                # Quick health check
make pre-commit-run       # Run pre-commit on all files

# Build & Release
make build                # Build package
make changelog            # Generate changelog
make clean                # Clean build artifacts

# Setup
make venv                 # Create virtual environment
make setup                # Complete development setup
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run with full coverage including DSPy
make test-cov-full

# Run tests in parallel (faster)
make test-parallel

# Run only fast tests
make test-fast
```

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Ruff**: Fast Python linter and formatter
- **MyPy**: Static type checker for type safety
- **Bandit**: Security vulnerability scanner
- **Pre-commit**: Git hooks for automated checks
- **Pytest**: Testing framework with coverage reporting

```bash
# Format code
make format

# Run all linters
make lint

# Run pre-commit checks
make pre-commit-run

# Type checking
uv run mypy src
```

### Commit Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Test changes
- `chore:` - Maintenance tasks
- `perf:` - Performance improvements

Example:
```bash
git commit -m "feat: add YAML output format support"
git commit -m "fix: resolve mypy type errors in formatters"
```

### Changelog Management

Generate changelog from conventional commits:

```bash
make changelog
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `make test-cov-full`
5. **Run linters**: `make lint`
6. **Commit your changes**: `git commit -m "feat: add amazing feature"`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Guidelines

- Write tests for new features
- Maintain test coverage above 75%
- Follow the existing code style (enforced by ruff)
- Add type hints for all functions
- Update documentation for new features
- Use conventional commit messages

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by [BAML](https://www.boundaryml.com/) for the JSONish format
- Built with [Pydantic](https://docs.pydantic.dev/) for schema handling
- Powered by [DSPy](https://github.com/stanfordnlp/dspy) for LLM integration
- Uses [json-repair](https://github.com/mangiucugna/json_repair) for robust parsing

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/rohitgarud/llm-schema-lite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rohitgarud/llm-schema-lite/discussions)
- **PyPI**: [llm-schema-lite](https://pypi.org/project/llm-schema-lite/)

---

<div align="center">

**[⬆ Back to Top](#llm-schema-lite)**

</div>
