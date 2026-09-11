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

## 🎨 Output Formats

`simplify_schema(model, format_type=...)` renders the same schema three ways. All three
carry the `*` required marker and fold constraints into the type line.

| `format_type` | Looks like | `User` above |
|---|---|---|
| `"jsonish"` *(default)* | a JSON object with `//` comments — closest to the JSON most models emit | 71 tokens |
| `"typescript"` | a TypeScript `interface` with `;` separators and `\|` unions | 63 tokens |
| `"yaml"` | indented keys with `#` comments, no braces | 61 tokens |

<!-- Each block below is executed standalone by tests/test_docs_examples.py, so the
     model is redefined rather than carried over from the Quick Start. -->

```python
from pydantic import BaseModel, Field

from llm_schema_lite import simplify_schema


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


print(simplify_schema(User, format_type="typescript").to_string())
print(simplify_schema(User, format_type="yaml").to_string())
```

```
// Description: A user account.
// Fields marked with * are required
interface Schema {
  name*: string (>= 2 chars);  // Full name
  age*: number (0 to 120);
  email: string | null;
  address*: { street*: string, city*: string };
}
# Description: A user account.

# Fields marked with * are required

name*: string (>= 2 chars)  # Full name
age*: int (0 to 120)
email: string OR null  # (default=null)
address*:
  street*: string
  city*: string
```

All three are *schema sketches* for prompting, not serialization formats — the values are
type tokens, not example data. Don't build a consumer on their shape.

## ✅ Validation and Coercion

`validate()` checks data against a schema and returns every error, with the constraint
that failed:

```python
from pydantic import BaseModel, Field

from llm_schema_lite import validate


class Address(BaseModel):
    street: str
    city: str


class User(BaseModel):
    name: str = Field(min_length=2, description="Full name")
    age: int = Field(ge=0, le=120)
    email: str | None = None
    address: Address


ok, errors = validate(User, {"name": "Ada", "age": 36,
                             "address": {"street": "1 Main St", "city": "Springfield"}})
# (True, None)

ok, errors = validate(User, {"name": "A", "age": 200,
                             "address": {"street": "1 Main St", "city": "Springfield"}})
# ok is False; errors is:
# ["Validation error at '.name': 'A' is too short (got 'A') - Constraint: minLength = 2",
#  "Validation error at '.age': 200 is greater than the maximum of 120 (got 200) - Constraint: maximum = 120"]
```

`coerce()` repairs the type mistakes models routinely make — a number sent as a string, a
scalar sent where a list belongs — and reports every change it made:

```python
from pydantic import BaseModel, Field

from llm_schema_lite import coerce


class Address(BaseModel):
    street: str
    city: str


class User(BaseModel):
    name: str = Field(min_length=2, description="Full name")
    age: int = Field(ge=0, le=120)
    email: str | None = None
    address: Address


data, metadata = coerce({"name": "Ada", "age": "36",
                         "address": {"street": "1 Main St", "city": "Springfield"}}, User)

print(data["age"])  # 36 — an int, not "36"
for m in metadata:
    print(m.field_path, m.coercion_type, m.original_value, "->", m.coerced_value)
# age to_int 36 -> 36
```

`loads(reply, schema=User)` already runs coercion and returns `(instance, metadata)`; call
`coerce()` directly when you have a dict rather than raw model text.

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
| `OutputMode.YAML` | YAML described by the compact schema | **not supported** — raises `StreamingNotSupportedError` before any request |

YAML mode renders a *schema sketch*, not an example document: keys carry the `*` required
marker, values are type tokens (`string`, `int OR null`), and constraints ride in comments.
The output is YAML-flavoured and optimised for LLM prompts — it currently round-trips
through `yaml.safe_load` and the test suite guards that, but it is not a serialization
format, so do not build a consumer on its shape.

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
make install-dev          # Install with development dependencies
make install-dspy         # Install with DSPy integration
make install-pre-commit   # Install pre-commit hooks
make sync                 # Sync dependencies using uv

# Testing
make test                 # Run tests with pytest
make test-cov             # Run tests with coverage report (core only, no DSPy)
make test-cov-full        # Run tests with full coverage including DSPy
make test-dspy            # Run only DSPy integration tests
make test-benchmarking    # Run benchmarking tests
make bench-dspy           # Run the DSPy adapter benchmark (see benchmarking/dspy_adapters)

# Code Quality
make lint                 # Run all linters (ruff, mypy, bandit)
make format               # Format code with ruff
make check                # Quick health check (fast lint + type check)
make pre-commit-run       # Run pre-commit on all files

# Build & Release
make build                # Build package distribution
make changelog            # Generate changelog
make release_notes        # Generate release notes
make publish-test         # Publish to Test PyPI
make publish              # Publish to PyPI
make clean                # Clean build artifacts and cache files

# Setup
make venv                 # Create virtual environment
make setup                # Complete setup for development
make update               # Full project update (clean, sync, hooks, lint)
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run with full coverage including DSPy
make test-cov-full
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
