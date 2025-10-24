# llm-schema-lite

[![PyPI version](https://img.shields.io/pypi/v/llm-schema-lite)](https://pypi.org/project/llm-schema-lite/)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm-schema-lite.svg)](https://pypi.org/project/llm-schema-lite/)
[![CI](https://github.com/rohitgarud/llm-schema-lite/workflows/CI/badge.svg)](https://github.com/rohitgarud/llm-schema-lite/actions)
[![codecov](https://codecov.io/gh/rohitgarud/llm-schema-lite/branch/main/graph/badge.svg)](https://codecov.io/gh/rohitgarud/llm-schema-lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Transform verbose Pydantic JSON schemas into LLM-friendly formats. Reduce token usage by **60-85%** while preserving essential type information. Includes robust JSON/YAML parsing with automatic error recovery.

---

## 📑 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Core Functionality](#-core-functionality)
  - [Schema Simplification](#schema-simplification)
  - [Output Formats](#output-formats)
  - [Nested Models](#nested-models)
  - [Metadata Control](#metadata-control)
- [Robust Parsing](#-robust-parsing-with-loads)
  - [Basic Usage](#basic-usage)
  - [Markdown Extraction](#markdown-extraction)
  - [Error Recovery](#error-recovery-and-repair)
  - [YAML Support](#yaml-support)
- [DSPy Integration](#-dspy-integration)
- [Token Reduction](#-token-reduction)
- [Use Cases](#-use-cases)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- 🎯 **60-85% Token Reduction** - Dramatically reduce schema tokens for LLM prompts
- 🔄 **Multiple Output Formats** - JSON, JSONish (BAML-style), TypeScript, YAML
- 🛡️ **Robust Parsing** - Parse malformed JSON/YAML with automatic repair
- 📦 **Flexible Input** - Works with Pydantic models, JSON schema dicts, or strings
- 🔌 **DSPy Integration** - Native adapter for structured outputs
- 📝 **Markdown Extraction** - Extract code blocks from LLM responses
- ⚡ **Fast & Lightweight** - Minimal dependencies, maximum performance
- 🎨 **Metadata Control** - Include/exclude descriptions and constraints
- 🔍 **Type Preservation** - Maintains essential type information
- ✅ **Fully Typed** - Complete type hints for better IDE support

---

## 📦 Installation

### Basic Installation

```bash
pip install llm-schema-lite
```

### With DSPy Support

```bash
pip install "llm-schema-lite[dspy]"
```

### Using uv (Recommended)

```bash
# Basic installation
uv pip install llm-schema-lite

# With DSPy support
uv pip install "llm-schema-lite[dspy]"
```

**Requirements:**
- Python 3.10+
- Optional: `dspy>=3.0.3` for DSPy integration
- Optional: `PyYAML` for YAML parsing (auto-installed)

---

## 🚀 Quick Start

```python
from pydantic import BaseModel
from llm_schema_lite import simplify_schema, loads

# Define your Pydantic model
class User(BaseModel):
    name: str
    age: int
    email: str

# Transform to LLM-friendly format
schema = simplify_schema(User)
print(schema.to_string())
# Output: { name: string, age: int, email: string }

# Parse JSON/YAML with robust error handling
json_data = loads('{"name": "John", "age": 30}', mode="json")
yaml_data = loads('name: Jane\nage: 25', mode="yaml")
```

---

## 🎯 Core Functionality

### Schema Simplification

Transform verbose JSON schemas into compact, LLM-friendly formats:

```python
from pydantic import BaseModel
from llm_schema_lite import simplify_schema

class User(BaseModel):
    name: str
    age: int
    email: str

# From Pydantic model
schema = simplify_schema(User)
print(schema.to_string())
# { name: string, age: int, email: string }

# From JSON schema dict
schema_dict = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}
schema = simplify_schema(schema_dict)
print(schema.to_string())
# { name: string, age: int }

# From JSON schema string
schema_string = '{"type": "object", "properties": {"name": {"type": "string"}}}'
schema = simplify_schema(schema_string)
print(schema.to_string())
# { name: string }
```

### Output Formats

Choose from multiple output formats to suit your needs:

```python
from pydantic import BaseModel
from llm_schema_lite import simplify_schema

class User(BaseModel):
    name: str
    age: int
    email: str

# JSONish format (BAML-like) - Default, most compact
schema = simplify_schema(User, format_type="jsonish")
print(schema.to_string())
# { name: string, age: int, email: string }

# TypeScript interface format
schema_ts = simplify_schema(User, format_type="typescript")
print(schema_ts.to_string())
# interface User {
#   name: string;
#   age: number;
#   email: string;
# }

# YAML format
schema_yaml = simplify_schema(User, format_type="yaml")
print(schema_yaml.to_string())
# name: string
# age: int
# email: string

# JSON format (standard)
schema_json = simplify_schema(User, format_type="json")
print(schema_json.to_json(indent=2))
# {
#   "name": "string",
#   "age": "int",
#   "email": "string"
# }
```

### Nested Models

Handle complex nested structures with ease:

```python
from pydantic import BaseModel
from llm_schema_lite import simplify_schema

class Address(BaseModel):
    street: str
    city: str
    zipcode: str

class Customer(BaseModel):
    name: str
    email: str
    address: Address
    tags: list[str]

schema = simplify_schema(Customer)
print(schema.to_string())
# {
#   name: string,
#   email: string,
#   address: {
#     street: string,
#     city: string,
#     zipcode: string
#   },
#   tags: string[]
# }
```

### Metadata Control

Control whether to include field descriptions and constraints:

```python
from pydantic import BaseModel, Field
from llm_schema_lite import simplify_schema

class Product(BaseModel):
    name: str = Field(..., description="Product name", min_length=1)
    price: float = Field(..., ge=0, description="Price must be positive")
    tags: list[str] = Field(default_factory=list)

# Include metadata (descriptions, constraints)
schema_with_meta = simplify_schema(Product, include_metadata=True)
print(schema_with_meta.to_string())
# {
#   name: string  // Product name, minLength: 1,
#   price: float  // Price must be positive, min: 0,
#   tags: string[]
# }

# Exclude metadata for minimal output
schema_minimal = simplify_schema(Product, include_metadata=False)
print(schema_minimal.to_string())
# {
#   name: string,
#   price: float,
#   tags: string[]
# }
```

### Working with JSON Schema Strings

Perfect for schemas from APIs, databases, or configuration files:

```python
from llm_schema_lite import simplify_schema

# Complex JSON schema from external source
complex_schema = '''{
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "email"]
        },
        "items": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        }
    },
    "required": ["user"]
}'''

# Convert to LLM-friendly format
schema = simplify_schema(complex_schema, include_metadata=True)
print(schema.to_string())
# {
#   user: {
#     name: string  // minLength: 1,
#     age: int  // min: 0, max: 120,
#     email: string  // format: email
#   },
#   items: string[]  // minItems: 1
# }
```

---

## 🔧 Robust Parsing with `loads`

The `loads` function provides unified, robust parsing for JSON and YAML content with automatic error recovery and markdown extraction.

### Basic Usage

```python
from llm_schema_lite import loads

# Parse JSON
data = loads('{"name": "John", "age": 30}', mode="json")
print(data)  # {'name': 'John', 'age': 30}

# Parse YAML
data = loads('name: Jane\nage: 25', mode="yaml")
print(data)  # {'name': 'Jane', 'age': 25}

# Auto-detect mode
data = loads('{"name": "Alice"}')  # Defaults to JSON
print(data)  # {'name': 'Alice'}
```

### Markdown Extraction

Automatically extracts content from markdown code blocks:

```python
from llm_schema_lite import loads

# JSON from markdown code block
markdown_json = """
\`\`\`json
{"name": "Alice", "age": 28}
\`\`\`
"""
data = loads(markdown_json, mode="json")
print(data)  # {'name': 'Alice', 'age': 28}

# YAML from markdown code block
markdown_yaml = """
\`\`\`yaml
name: Bob
age: 32
\`\`\`
"""
data = loads(markdown_yaml, mode="yaml")
print(data)  # {'name': 'Bob', 'age': 32}

# Works with language tags: json, yaml, yml
markdown_with_tag = """Here's the data:
\`\`\`json
{"status": "success"}
\`\`\`
"""
data = loads(markdown_with_tag, mode="json")
print(data)  # {'status': 'success'}
```

### JSON Object Extraction

Extract JSON objects from embedded text:

```python
from llm_schema_lite import loads

# Extract JSON from mixed content
text = 'Here is the result: {"name": "Charlie", "age": 35} and some other text'
data = loads(text, mode="json", extract_from_markdown=False)
print(data)  # {'name': 'Charlie', 'age': 35}

# Multiple JSON objects - extracts the first one
multiple = 'First: {"a": 1} Second: {"b": 2}'
data = loads(multiple, mode="json", extract_from_markdown=False)
print(data)  # {'a': 1}
```

### Error Recovery and Repair

Handles malformed JSON/YAML with automatic repair:

```python
from llm_schema_lite import loads, ConversionError

# Malformed JSON with trailing comma
malformed = '{"name": "David", "age": 40,}'
data = loads(malformed, mode="json")
print(data)  # {'name': 'David', 'age': 40}

# Missing quotes
missing_quotes = '{name: "Eve", age: 22}'
data = loads(missing_quotes, mode="json")
print(data)  # {'name': 'Eve', 'age': 22}

# Unescaped strings
unescaped = '{"message": "Hello\nWorld"}'
data = loads(unescaped, mode="json")
print(data)  # {'message': 'Hello\nWorld'}

# Disable repair to get strict parsing
try:
    loads(malformed, mode="json", repair=False)
except ConversionError as e:
    print(f"Parse error: {e}")
```

### YAML Support

Comprehensive YAML parsing with fallback to JSON:

```python
from llm_schema_lite import loads

# Standard YAML
yaml_text = '''
name: Frank
age: 45
active: true
tags:
  - python
  - testing
'''
data = loads(yaml_text, mode="yaml")
print(data)  # {'name': 'Frank', 'age': 45, 'active': True, 'tags': ['python', 'testing']}

# YAML with comments
yaml_with_comments = '''
# User information
name: Henry  # Full name
age: 35
# Contact details
email: henry@example.com
'''
data = loads(yaml_with_comments, mode="yaml")
print(data)  # {'name': 'Henry', 'age': 35, 'email': 'henry@example.com'}

# YAML that looks like JSON (automatic fallback)
yaml_like_json = '{"name": "Grace", "age": 50}'
data = loads(yaml_like_json, mode="yaml")
print(data)  # {'name': 'Grace', 'age': 50}
```

### Advanced Parsing Features

```python
from llm_schema_lite import loads

# Complex nested structures from markdown
complex_json = """
\`\`\`json
{
  "user": {
    "name": "Grace",
    "details": {
      "age": 30,
      "city": "NYC"
    }
  }
}
\`\`\`
"""
data = loads(complex_json, mode="json")
print(data['user']['details']['city'])  # NYC

# Arrays and special values
array_json = '{"items": ["apple", "banana"], "active": true, "data": null}'
data = loads(array_json, mode="json")
print(data)  # {'items': ['apple', 'banana'], 'active': True, 'data': None}

# Handle indentation issues in YAML
indented_yaml = '''    name: Indented
    age: 25
    city: SF'''
data = loads(indented_yaml, mode="yaml", repair=True)
print(data)  # {'name': 'Indented', 'age': 25, 'city': 'SF'}
```

---

## 🔌 DSPy Integration

Native DSPy adapter with support for JSON, JSONish, and YAML output modes.

### Installation

```bash
pip install "llm-schema-lite[dspy]"
```

### Quick Start

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
print(result.answer)  # Answer(answer="...", confidence=0.95)
```

### Output Modes

```python
from llm_schema_lite.dspy_integration import StructuredOutputAdapter, OutputMode

# JSONish mode (most compact, 60-85% token reduction)
adapter_jsonish = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
# Schema: { answer: string, confidence: float }

# JSON mode (standard JSON format)
adapter_json = StructuredOutputAdapter(output_mode=OutputMode.JSON)
# Schema: {"answer": "string", "confidence": "float"}

# YAML mode (human-readable)
adapter_yaml = StructuredOutputAdapter(output_mode=OutputMode.YAML)
# Schema:
# answer: string
# confidence: float
```

### Features

- 🎯 **Multiple Output Modes**: JSON, JSONish (BAML-style), and YAML
- 📉 **60-85% Token Reduction**: With JSONish mode
- 🔄 **Input Schema Simplification**: Automatically simplifies Pydantic input fields
- 🛡️ **Robust Parsing**: Handles malformed outputs with automatic recovery
- ✅ **Full Compatibility**: Works with Predict, ChainOfThought, and all DSPy modules
- 📝 **Markdown Extraction**: Automatically extracts code blocks from LLM responses

For detailed documentation, see the [DSPy Integration Guide](src/llm_schema_lite/dspy_integration/README.md).

---

## 📊 Token Reduction

Compare the token usage between original and simplified schemas:

```python
import json
from pydantic import BaseModel, Field
from llm_schema_lite import simplify_schema

class User(BaseModel):
    name: str = Field(..., description="User's full name")
    age: int = Field(..., ge=0, le=120)
    email: str = Field(..., description="Email address")
    tags: list[str] = Field(default_factory=list)

# Original Pydantic schema (verbose)
original_schema = User.model_json_schema()
original_tokens = len(json.dumps(original_schema))
print(f"Original: {original_tokens} characters")
# Original: ~450 characters

# Simplified schema (LLM-friendly)
simplified = simplify_schema(User, include_metadata=False)
simplified_tokens = len(simplified.to_string())
print(f"Simplified: {simplified_tokens} characters")
# Simplified: ~60 characters

# Token reduction
reduction = ((original_tokens - simplified_tokens) / original_tokens) * 100
print(f"Reduction: {reduction:.1f}%")
# Reduction: 85-90%
```

### Real-World Example

```python
from pydantic import BaseModel, Field
from llm_schema_lite import simplify_schema

class Address(BaseModel):
    street: str
    city: str
    state: str
    zipcode: str

class Order(BaseModel):
    order_id: str = Field(..., description="Unique order identifier")
    customer_name: str
    items: list[str] = Field(..., min_items=1)
    total: float = Field(..., ge=0)
    shipping_address: Address
    status: str = Field(..., description="Order status")

# Original JSON schema: ~800 characters
# Simplified schema: ~120 characters
# Token reduction: ~85%

schema = simplify_schema(Order, include_metadata=False)
print(schema.to_string())
# {
#   order_id: string,
#   customer_name: string,
#   items: string[],
#   total: float,
#   shipping_address: {
#     street: string,
#     city: string,
#     state: string,
#     zipcode: string
#   },
#   status: string
# }
```

---

## 🎯 Use Cases

### LLM Function Calling

Reduce schema tokens in function definitions:

```python
from llm_schema_lite import simplify_schema
from pydantic import BaseModel

class WeatherQuery(BaseModel):
    location: str
    units: str  # "celsius" or "fahrenheit"

# Use simplified schema in your LLM prompt
schema = simplify_schema(WeatherQuery)
prompt = f"""
Available function:
get_weather{schema.to_string()}

User: What's the weather in NYC?
"""
```

### Data Extraction from LLM Responses

```python
from llm_schema_lite import loads

# LLM returns JSON in markdown code block
llm_response = """Here's the extracted data:
\`\`\`json
{"name": "John Doe", "email": "john@example.com", "age": 30}
\`\`\`
"""

data = loads(llm_response, mode="json")
print(data)  # {'name': 'John Doe', 'email': 'john@example.com', 'age': 30}
```

### API Response Handling

```python
from llm_schema_lite import loads, ConversionError

# Handle potentially malformed API responses
def safe_parse_response(response_text):
    try:
        return loads(response_text, mode="json", repair=True)
    except ConversionError as e:
        print(f"Failed to parse: {e}")
        return None

# Works with malformed JSON
malformed_response = '{"status": "success", "data": {"id": 123,}}'
data = safe_parse_response(malformed_response)
print(data)  # {'status': 'success', 'data': {'id': 123}}
```

### DSPy Structured Outputs

```python
import dspy
from pydantic import BaseModel
from llm_schema_lite.dspy_integration import StructuredOutputAdapter, OutputMode

class ExtractedInfo(BaseModel):
    entities: list[str]
    sentiment: str
    summary: str

adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
lm = dspy.LM(model="openai/gpt-4")
dspy.configure(lm=lm, adapter=adapter)

class Extract(dspy.Signature):
    text: str = dspy.InputField()
    info: ExtractedInfo = dspy.OutputField()

extractor = dspy.Predict(Extract)
result = extractor(text="Your text here...")
```

---

## 📚 API Reference

### `simplify_schema()`

Transform Pydantic models or JSON schemas into LLM-friendly formats.

```python
def simplify_schema(
    model: BaseModel | dict | str,
    format_type: str = "jsonish",
    include_metadata: bool = True
) -> SchemaLite:
    """
    Simplify a Pydantic model or JSON schema.

    Args:
        model: Pydantic BaseModel class, JSON schema dict, or JSON schema string
        format_type: Output format - "jsonish", "json", "typescript", or "yaml"
        include_metadata: Include field descriptions and constraints

    Returns:
        SchemaLite object with various output methods

    Raises:
        UnsupportedModelError: If model type is not supported
        ConversionError: If schema conversion fails
    """
```

### `loads()`

Parse JSON or YAML with robust error recovery.

```python
def loads(
    text: str,
    mode: str = "json",
    repair: bool = True,
    extract_from_markdown: bool = True
) -> dict:
    """
    Parse JSON or YAML with automatic error recovery.

    Args:
        text: Text to parse
        mode: Parsing mode - "json" or "yaml"
        repair: Enable automatic repair of malformed content
        extract_from_markdown: Extract content from markdown code blocks

    Returns:
        Parsed dictionary

    Raises:
        ConversionError: If parsing fails even after repair attempts
    """
```

### `SchemaLite`

Result object from `simplify_schema()` with multiple output methods.

```python
class SchemaLite:
    def to_string(self) -> str:
        """Get formatted string representation."""

    def to_dict(self) -> dict:
        """Get dictionary representation."""

    def to_json(self, indent: int | None = None) -> str:
        """Get JSON string representation."""

    def to_yaml(self) -> str:
        """Get YAML string representation (if format_type="yaml")."""

    def estimate_tokens(self) -> int:
        """Estimate token count using tiktoken."""
```

### DSPy Integration

```python
from llm_schema_lite.dspy_integration import StructuredOutputAdapter, OutputMode

class OutputMode(Enum):
    JSON = "json"           # Standard JSON format
    JSONISH = "jsonish"     # BAML-style compact format (default)
    YAML = "yaml"           # YAML format

class StructuredOutputAdapter:
    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = True,
        output_mode: OutputMode = OutputMode.JSONISH,
        include_input_schemas: bool = True
    ):
        """
        DSPy adapter for structured outputs with llm-schema-lite.

        Args:
            callbacks: Optional DSPy callbacks
            use_native_function_calling: Use native function calling if available
            output_mode: Output format mode
            include_input_schemas: Simplify input field schemas
        """
```

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
