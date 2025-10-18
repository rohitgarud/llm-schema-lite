# llm-schema-lite

[![PyPI version](https://img.shields.io/pypi/v/llm-schema-lite)](https://pypi.org/project/llm-schema-lite/)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm-schema-lite.svg)](https://pypi.org/project/llm-schema-lite/)
[![CI](https://github.com/rohitgarud/llm-schema-lite/workflows/CI/badge.svg)](https://github.com/rohitgarud/llm-schema-lite/actions)
[![codecov](https://codecov.io/gh/rohitgarud/llm-schema-lite/branch/main/graph/badge.svg)](https://codecov.io/gh/rohitgarud/llm-schema-lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Transform verbose Pydantic JSON schemas into LLM-friendly formats. Reduce token usage by **60-85%** while preserving essential type information.

## 🚀 Quick Start

### Basic Usage

```python
from pydantic import BaseModel
from llm_schema_lite import simplify_schema

# Define your Pydantic model
class User(BaseModel):
    name: str
    age: int
    email: str

# Transform to LLM-friendly format
schema = simplify_schema(User)
print(schema.to_string())
# Output: { name: string, age: int, email: string }
```

### Multiple Output Formats

```python
# JSONish format (BAML-like) - Default
schema = simplify_schema(User)
print(schema.to_string())
# { name: string, age: int, email: string }

# TypeScript format
schema_ts = simplify_schema(User, format_type="typescript")
print(schema_ts.to_string())
# interface User { name: string; age: int; email: string; }

# YAML format
schema_yaml = simplify_schema(User, format_type="yaml")
print(schema_yaml.to_string())
# name: string
# age: int
# email: string
```

### Advanced Features

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(..., description="Product name", min_length=1)
    price: float = Field(..., ge=0, description="Price must be positive")
    tags: list[str] = Field(default_factory=list)

# Include metadata (descriptions, constraints)
schema_with_meta = simplify_schema(Product, include_metadata=True)
print(schema_with_meta.to_string())
# {
#  name: string  //Product name, minLength: 1,
#  price: float  //Price must be positive, min: 0,
#  tags: string[]
# }

# Exclude metadata for minimal output
schema_minimal = simplify_schema(Product, include_metadata=False)
print(schema_minimal.to_string())
# {
#  name: string,
#  price: float,
#  tags: string[]
# }
```

### Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    zipcode: str

class Customer(BaseModel):
    name: str
    email: str
    address: Address

schema = simplify_schema(Customer)
print(schema.to_string())
# { name: string, email: string, address: { street: string, city: string, zipcode: string } }
```

### Different Output Methods

```python
schema = simplify_schema(User)

# String output
print(schema.to_string())

# JSON output
print(schema.to_json(indent=2))

# Dictionary output
print(schema.to_dict())

# YAML output (if format_type="yaml")
print(schema.to_yaml())
```

## 📊 Token Reduction

Compare the token usage:

```python
import json
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

# Original Pydantic schema (verbose)
original_schema = User.model_json_schema()
print("Original tokens:", len(json.dumps(original_schema)))

# Simplified schema (LLM-friendly)
simplified = simplify_schema(User)
print("Simplified tokens:", len(simplified.to_string()))

# Typical reduction: 60-85% fewer tokens!
```

## 🎯 Use Cases

- **LLM Function Calling**: Reduce schema tokens in function definitions
- **DSPy Integration**: Native adapter for structured outputs with multiple modes
- **LangChain**: Streamline Pydantic model schemas
- **Raw LLM APIs**: Minimize prompt overhead with concise schemas

## 🔌 DSPy Integration

**NEW!** Native DSPy adapter with support for JSON, JSONish, and YAML output modes:

```python
import dspy
from pydantic import BaseModel
from llm_schema_lite.dspy_integration import StructuredOutputAdapter, OutputMode

class Answer(BaseModel):
    answer: str
    confidence: float

# Create adapter with JSONish mode (60-85% fewer tokens)
adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)

# Configure DSPy
lm = dspy.LM(model="openai/gpt-4")
dspy.configure(lm=lm, adapter=adapter)

# Use with any DSPy module
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: Answer = dspy.OutputField()

predictor = dspy.Predict(QA)
result = predictor(question="What is Python?")
```

**Features:**
- 🎯 **Multiple Output Modes**: JSON, JSONish (BAML-style), and YAML
- 📉 **60-85% Token Reduction**: With JSONish mode
- 🔄 **Input Schema Simplification**: Automatically simplifies Pydantic input fields
- 🛡️ **Robust Parsing**: Handles malformed outputs with automatic recovery
- ✅ **Full Compatibility**: Works with Predict, ChainOfThought, and all DSPy modules

See [DSPy Integration Guide](src/llm_schema_lite/dspy_integration/README.md) for detailed documentation.

## Installation

### Basic Installation

```bash
pip install llm-schema-lite
```

### With DSPy Support

```bash
pip install "llm-schema-lite[dspy]"
```

### Using uv

```bash
# Basic
uv pip install llm-schema-lite

# With DSPy
uv pip install "llm-schema-lite[dspy]"
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

## License

See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
