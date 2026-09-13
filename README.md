# llm-schema-lite

[![PyPI version](https://img.shields.io/pypi/v/llm-schema-lite)](https://pypi.org/project/llm-schema-lite/)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm-schema-lite.svg)](https://pypi.org/project/llm-schema-lite/)
[![CI](https://github.com/rohitgarud/llm-schema-lite/workflows/CI/badge.svg)](https://github.com/rohitgarud/llm-schema-lite/actions)
[![codecov](https://codecov.io/gh/rohitgarud/llm-schema-lite/branch/main/graph/badge.svg)](https://codecov.io/gh/rohitgarud/llm-schema-lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Transform verbose JSON schemas into LLM-friendly formats. Reduce token usage by **60-85%** while preserving essential type information and integrating validation constraints directly into type descriptions for optimal LLM readability. Includes robust JSON/YAML parsing with automatic error recovery and enhanced constraint integration across all formatters.

Framework agnostic by construction: the package turns a schema into a string and a reply back into a model, so it drops into the OpenAI SDK, any OpenAI-compatible endpoint, or a framework like DSPy without tying you to any of them.

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

`loads(reply, schema=User)` validates against the schema and returns `(instance, metadata)`;
call `coerce()` directly when you have a dict rather than raw model text. Input that is not a
mapping — a list, a scalar, or a JSON string that decodes to one — is wrapped as
`{"value": ...}` before coercion, so the return is always a dict unless the schema has an
array at its root, in which case a decoded list is coerced as-is.

### Tuning the parse with `ParseConfig`

`ParseConfig` is the single knob-holder for parsing and coercion behaviour. `loads()`,
`coerce()` and the DSPy adapter all take one:

| Field | Default | Effect |
|---|---|---|
| `partial` | `False` | Keep what validates. A failing **optional** field is dropped and listed in `metadata["failed_fields"]`; a failing **required** field still raises `ConversionError`. |
| `allow_coercion` | `True` | Coerce a field that fails validation instead of rejecting it. With `False` the input is still normalised as above and returned with an empty metadata list — never dropped. |
| `coerce_list_single_item` | `False` | Wrap a lone scalar where a list belongs — `"x"` becomes `["x"]`. |
| `strip_required_marker` | `"*"` | Strip this trailing marker from reply keys, at every nesting level, so a model that echoes `name*` back is still understood. Set to `""` to disable. |
| `log_coercions` | `True` | Log coercion events for debugging. |

One sharp edge worth stating plainly: **`allow_coercion` and `coerce_list_single_item` only
take effect when `partial=True`.** The default `loads()` path validates without coercing, so a
reply carrying `"age": "36"` raises rather than repairing the string. Reach for `coerce()`, or
turn on `partial=True`:

```python
from pydantic import BaseModel, Field

from llm_schema_lite import ConversionError, ParseConfig, loads


class Address(BaseModel):
    street: str
    city: str


class User(BaseModel):
    name: str = Field(min_length=2)
    age: int = Field(ge=0, le=120)
    address: Address


reply = '{"name": "Ada", "age": "36", "address": {"street": "1 Main St", "city": "Springfield"}}'

try:
    loads(reply, schema=User)  # the default path validates without coercing
except ConversionError as exc:
    print(exc)  # Validation failed: ... '36' is not of type 'integer'

user, metadata = loads(reply, parse_config=ParseConfig(partial=True), schema=User)
print(user.age)  # 36
```

`partial=True` still needs every **required** field present: it salvages fields that fail
validation, not fields the model never sent.

It is also a rescue, not a constructor. A nested model's *contents* are checked against the
schema, but partial mode assembles the instance with Pydantic's own validation pass skipped,
so the nested value arrives as a plain `dict` — `user.address` above is
`{'street': ..., 'city': ...}`, not an `Address` instance. Re-validate the result yourself if
you need the nested type rather than the nested data.

## 🔌 Use It With Any SDK

There is no LLM client dependency here. The package renders a schema **to a string** and
parses a reply **from a string** — anything that can send text to a model is already
compatible, and you keep whatever SDK you are using.

Against the OpenAI **Responses API**:

<!-- lsl-docs: skip: issues a live OpenAI Responses API request -->

```python
from openai import OpenAI
from pydantic import BaseModel, Field

from llm_schema_lite import loads, simplify_schema


class User(BaseModel):
    """A user account."""

    name: str = Field(min_length=2, description="Full name")
    age: int = Field(ge=0, le=120)
    email: str | None = None


client = OpenAI()
schema = simplify_schema(User)

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "system",
            "content": (
                "Reply with one JSON object matching this schema:\n"
                f"{schema.to_string()}"
            ),
        },
        {"role": "user", "content": "Ada Lovelace, 36, ada@example.com"},
    ],
)

user, metadata = loads(response.output_text, schema=User)
print(user.name, user.age)  # Ada Lovelace 36
```

Two calls into the package, at the two edges of the request: `simplify_schema()` on the way
out, `loads()` on the way back. Everything between them is your SDK's business.

`loads()` is doing real work on that return trip. It absorbs the replies a bare
`json.loads()` rejects — markdown fences, a leading `Here you go:`, a trailing comma,
single quotes — so a slightly unruly model does not become an exception at the call site.

### Chat Completions, and any OpenAI-compatible endpoint

Swap the one call. The schema and the parse do not move:

<!-- lsl-docs: skip: issues a live OpenAI Chat Completions request -->

```python
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],  # same two messages as above
)
user, metadata = loads(completion.choices[0].message.content, schema=User)
```

Point `base_url` somewhere else and the identical code runs against a local or third-party
server — Ollama, vLLM, LM Studio, OpenRouter:

<!-- lsl-docs: skip: constructs an OpenAI client pointed at a local server -->

```python
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

That last case is the one the compact format is built for. A small local model frequently
has no structured-output mode to fall back on, so the schema has to travel in the prompt —
where its size is paid for on every call — and the reply has to be parsed defensively.
Passing `parse_config=ParseConfig(partial=True)` to `loads()` goes further still: an
*optional* field that fails validation is dropped and reported in
`metadata["failed_fields"]` rather than sinking the whole reply, while a *required* field
that fails still raises `ConversionError`. Partial mode salvages the nice-to-haves; it
does not hand you a half-built model.

Anthropic, Gemini, Bedrock, LiteLLM, LangChain, a bare `httpx.post` — none of them need
anything different. Hand the model `schema.to_string()`, hand the reply to `loads()`.

## 🤖 DSPy Integration

The DSPy adapter below is a convenience, not the supported path — it wires the same two
calls into DSPy's adapter protocol so you do not have to. Skip it entirely if you are not
using DSPy.

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
