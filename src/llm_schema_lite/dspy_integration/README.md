# DSPy Integration for llm-schema-lite

This module provides a unified adapter for integrating llm-schema-lite with DSPy, enabling token-efficient schema representation in DSPy programs.

## Features

- **Multiple Output Modes**: Support for JSON, JSONish (BAML-like), and YAML output formats
- **Token Efficiency**: 60-85% reduction in schema token usage with JSONish mode
- **Input Schema Simplification**: Automatically simplifies complex Pydantic models in input fields
- **Robust Parsing**: Handles malformed outputs with automatic repair (`repair=True`) and fallback mechanisms
- **Full DSPy Compatibility**: Works with all DSPy modules (Predict, ChainOfThought, etc.)
- **Clean API Integration**: Uses llm-schema-lite's public API (`simplify_schema()`, `loads()`) for maintainability

## Installation

```bash
# Install with DSPy support
pip install llm-schema-lite[dspy]

# Or using uv
uv pip install llm-schema-lite[dspy]
```

## Quick Start

### Basic Usage

```python
import dspy
from pydantic import BaseModel
from llm_schema_lite.dspy_integration import StructuredOutputAdapter, OutputMode

# Define your models
class Answer(BaseModel):
    answer: str
    confidence: float

# Create adapter with JSONish mode (default)
adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)

# Configure DSPy
lm = dspy.LM(model="openai/gpt-4")
dspy.configure(lm=lm, adapter=adapter)

# Use with any DSPy module
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: Answer = dspy.OutputField()

predictor = dspy.Predict(QA)
result = predictor(question="What is DSPy?")
```

### Data Extraction Example

```python
from pydantic import BaseModel, Field
from typing import Literal

class Person(BaseModel):
    name: str
    age: int
    email: str | None = None
    occupation: str | None = None

class Company(BaseModel):
    name: str
    industry: str
    founded_year: int | None = None
    employee_count: str | None = Field(
        default=None,
        description="e.g., '50-100', '1000+'"
    )

class ExtractionResult(BaseModel):
    people: list[Person]
    companies: list[Company]
    summary: str

# Create adapter with input schema simplification
adapter = StructuredOutputAdapter(
    output_mode=OutputMode.JSONISH,
    include_input_schemas=True  # Simplifies input schemas too!
)

dspy.configure(lm=lm, adapter=adapter)

class ExtractEntities(dspy.Signature):
    """Extract people and companies from text."""
    text: str = dspy.InputField()
    result: ExtractionResult = dspy.OutputField()

extractor = dspy.Predict(ExtractEntities)
text = """
John Smith, 35, works as a software engineer at TechCorp.
TechCorp is a technology company founded in 2010 with over 500 employees.
Jane Doe, CEO of DataInc, is 42 years old. DataInc operates in the
data analytics industry.
"""
result = extractor(text=text)
print(f"Found {len(result.result.people)} people and {len(result.result.companies)} companies")
```

## Output Modes

### JSON Mode

Standard JSON with full `model_json_schema()` - verbose but compatible with OpenAI structured outputs.

```python
adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)
```

**Use when:**
- You need maximum compatibility
- Using OpenAI's structured outputs API
- Schema verbosity is not a concern

### JSONish Mode (Recommended)

JSON output with simplified BAML-like schemas - 60-85% token reduction.

```python
adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
```

**Use when:**
- You want token efficiency
- Working with smaller models
- Schema clarity is important

**Example schema difference:**

JSON mode (verbose):
```json
{
  "type": "object",
  "properties": {
    "answer": {"type": "string"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["answer", "confidence"]
}
```

JSONish mode (simplified):
```
{ answer: string, confidence: float }
```

### YAML Mode

YAML output with simplified schemas.

```python
adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)
```

**Use when:**
- You prefer YAML format
- Working with models that understand YAML well
- Need human-readable outputs

## Configuration Options

```python
adapter = StructuredOutputAdapter(
    output_mode=OutputMode.JSONISH,           # Output format mode
    include_input_schemas=True,                # Simplify input field schemas
    use_native_function_calling=True,          # Use native function calling
    callbacks=None                             # Optional callbacks
)
```

### Parameters

- **output_mode**: `OutputMode.JSON`, `OutputMode.JSONISH`, or `OutputMode.YAML`
  - Controls the output format and schema representation
  - Default: `OutputMode.JSONISH`

- **include_input_schemas**: `bool`
  - Whether to include simplified schemas for complex input types
  - Useful when input fields are Pydantic models
  - Default: `True`

- **use_native_function_calling**: `bool`
  - Whether to use native function calling for tool calls
  - Default: `True`

- **callbacks**: `list[BaseCallback] | None`
  - Optional callbacks for monitoring
  - Default: `None`

## Advanced Usage

### With ChainOfThought

```python
adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
dspy.configure(lm=lm, adapter=adapter)

class ReasoningQA(dspy.Signature):
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Step by step reasoning")
    answer: str = dspy.OutputField(desc="Final answer")

cot = dspy.ChainOfThought(ReasoningQA)
result = cot(question="What is 2+2?")
print(result.reasoning)
print(result.answer)
```

### With Demonstrations

```python
adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
dspy.configure(lm=lm, adapter=adapter)

predictor = dspy.Predict(QA)

demos = [
    dspy.Example(question="What is Python?", answer="A programming language"),
    dspy.Example(question="What is DSPy?", answer="A framework for LLMs"),
]

result = predictor(question="What is AI?", demos=demos)
```

### Error Handling

The adapter includes robust error handling with automatic fallbacks powered by llm-schema-lite:

```python
# YAML mode automatically falls back to JSON parsing if YAML fails
adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)

# Malformed JSON/YAML is automatically repaired (repair=True by default)
# Uses llm-schema-lite's loads() with json_repair integration
adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)

# Parsing handles:
# - Markdown code blocks extraction
# - Embedded JSON/YAML in text
# - Malformed JSON repair
# - Automatic content type detection
```

**Parsing Pipeline:**

1. Extract content from markdown blocks (if present)
2. Repair malformed JSON/YAML (using `json_repair` library)
3. Parse to dictionary with type validation
4. Cast values to expected Pydantic types
5. Validate all required fields are present

## Architecture

### Class Hierarchy

```
Adapter (DSPy base)
  ↓
ChatAdapter (DSPy)
  ↓
JSONAdapter (DSPy)
  ↓
StructuredOutputAdapter (llm-schema-lite)
```

### Integration with llm-schema-lite

The adapter uses llm-schema-lite's public API for schema simplification and robust parsing:

**Schema Simplification** (via `simplify_schema()`):
- Converts Pydantic models to token-efficient schemas
- Supports multiple format types: `"jsonish"`, `"typescript"`, `"yaml"`
- Controlled by the `output_mode` parameter
- Returns `SchemaLite` objects with `.to_string()` for formatted output

```python
# Example: How the adapter uses simplify_schema()
from llm_schema_lite import simplify_schema

simplified = simplify_schema(
    field_type,
    format_type="jsonish",  # or "typescript", "yaml"
    include_metadata=False
)
schema_str = simplified.to_string()
```

**Robust Parsing** (via `loads()`):
- Parses JSON/YAML with automatic repair (`repair=True`)
- Handles markdown code blocks and embedded structures
- Provides fallback mechanisms for malformed outputs

```python
# Example: How the adapter uses loads()
from llm_schema_lite import loads

# JSON parsing with repair
data = loads(completion, mode="json", repair=True)

# YAML parsing with repair (auto-fallback to JSON)
data = loads(completion, mode="yaml", repair=True)
```

### Key Methods

- `__call__()` / `acall()`: Main execution methods (sync/async)
- `format_field_structure()`: Formats input/output structure for LLM
- `parse()`: Parses LLM responses into structured data
  - Uses `loads()` from llm-schema-lite for robust parsing
- `_translate_field_type()`: Translates field types with mode-specific schemas
- `_get_complex_type_description()`: Generates simplified schemas for complex types
  - Uses `simplify_schema()` from llm-schema-lite for token-efficient representations

## Token Efficiency Comparison

Based on benchmarks with complex Pydantic models:

| Mode | Schema Tokens | Reduction |
|------|--------------|-----------|
| JSON | 815 tokens | 0% (baseline) |
| JSONish | 145 tokens | 82% |
| YAML | 178 tokens | 78% |

## Testing

See `tests/test_dspy_README.md` for comprehensive testing documentation.

```bash
# Install test dependencies
pip install -e ".[dspy]"

# Run tests
pytest tests/test_dspy_*.py -v
```

## Examples

See `examples/` directory for complete examples:

- `dspy_basic_usage.py` - Basic integration examples
- `dspy_complex_models.py` - Complex Pydantic models with nested structures
- `dspy_entity_extraction.py` - Real-world entity extraction from text

## Troubleshooting

### DSPy Not Found

```bash
pip install "dspy>=3.0.3"
```

### PyYAML Not Found (for YAML mode)

```bash
pip install pyyaml
```

### json_repair Not Found

```bash
pip install json-repair
```

### Import Errors

```bash
# Install in editable mode
pip install -e .
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## License

MIT License - see `LICENSE` file for details.

## Related Projects

- [DSPy](https://github.com/stanfordnlp/dspy) - Framework for programming with foundation models
- [llm-schema-lite](https://github.com/rohitgarud/llm-schema-lite) - LLM-friendly schema transformation
- [BAML](https://www.boundaryml.com/) - Inspiration for JSONish format

## Citation

If you use this in your research, please cite:

```bibtex
@software{llm_schema_lite_dspy,
  title = {DSPy Integration for llm-schema-lite},
  author = {Rohit Garud},
  year = {2025},
  url = {https://github.com/rohitgarud/llm-schema-lite}
}
```
