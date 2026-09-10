# DSPy Integration for llm-schema-lite

This module provides a unified adapter for integrating llm-schema-lite with DSPy, enabling token-efficient schema representation in DSPy programs.

## Features

- **Multiple Output Modes**: Support for JSON, JSONish (BAML-like), and YAML output formats
- **Token Efficiency**: 60-85% reduction in schema token usage with JSONish mode
- **Input Schema Simplification**: Automatically simplifies complex Pydantic models in input fields, gated by `include_input_schemas`
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

<!-- lsl-docs: skip: issues a live LM request -->
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
lm = dspy.LM(model="openai/gpt-4o-mini")
dspy.configure(lm=lm, adapter=adapter)

# Use with any DSPy module
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: Answer = dspy.OutputField()

predictor = dspy.Predict(QA)
result = predictor(question="What is DSPy?")
```

### Inspecting the prompt without an LM

```python
import dspy
from pydantic import BaseModel

from llm_schema_lite.dspy_integration import OutputMode, PromptLayout, StructuredOutputAdapter


class Answer(BaseModel):
    """A model's answer with a confidence score."""

    answer: str
    confidence: float


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: Answer = dspy.OutputField()


for layout in (PromptLayout.SECTIONS, PromptLayout.JSON_BLOCK):
    adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH, prompt_layout=layout)
    print(adapter.format_field_structure(QA))
    print(adapter.parse(QA, '{"answer": {"answer": "42", "confidence": 0.9}}'))
```

Nothing here contacts an LM, so it is the fastest way to compare `output_mode` and
`prompt_layout` settings, and to see exactly how many tokens each costs.

### Data Extraction Example

<!-- lsl-docs: skip: issues a live LM request -->
```python
import dspy
from pydantic import BaseModel, Field
from typing import Literal

from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter

lm = dspy.LM(model="openai/gpt-4o-mini")


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
from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter

adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)
```

**Use when:**
- You need maximum compatibility
- Using OpenAI's structured outputs API
- Schema verbosity is not a concern

### JSONish Mode (Recommended)

JSON output with simplified BAML-like schemas - 60-85% token reduction.

```python
from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter

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
//Title: Answer
// Fields marked with * are required
{
  answer*: string,
  confidence*: float
}
```

The `//Title:` line appears because this `Answer` has no docstring. Give the model a
docstring and the title is replaced by the docstring text.

### YAML Mode

> **Note.** YAML mode renders a schema sketch, not an example document: keys carry the `*`
> required marker, values are type tokens (`string`, `int OR null`), and constraints ride in
> comments. The output is YAML-flavoured and optimised for LLM prompts — it currently
> round-trips through `yaml.safe_load` and the test suite guards that, but it is not a
> serialization format.

YAML output with simplified schemas.

```python
from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter

adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)
```

**Use when:**
- You prefer YAML format
- Working with models that understand YAML well
- Need human-readable outputs

**Pass a `ParseConfig()` in YAML mode.** Small models answering in YAML often write an
empty list as a list holding one all-null item (`contacts: [{email: null, phone: null}]`),
which fails validation; `ParseConfig()` prunes that item. Measured on `qwen3.5:0.8b` over
40 labeled extraction cases, it moved YAML from 0.714 to 0.848 field accuracy and 16/40 to
22/40 exact records, with validation errors halved (11 -> 5). It does not yet repair
YAML's typed plain scalars — `postcode: 95014` loads as an `int` and still fails a `str`
field — because coercion only reaches scalar output fields, not fields nested in a model:

```python
from llm_schema_lite import ParseConfig
from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter

adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML, parse_config=ParseConfig())
```

The prompt's output section is itself YAML — `record:` with the schema nested under it,
not `[[ ## record ## ]]` markers. This matters more in YAML than elsewhere: JSON and
JSONish send `response_format={"type": "json_object"}`, which forces the reply back into
shape whatever the prompt demonstrated, while YAML mode sends no response format at all,
so the demonstration *is* the specification. When the two disagreed, models followed the
demonstration and emitted markers under a "respond with YAML" instruction — `qwen3:8b`
returned 0 parseable replies out of 18 in the outcomes arm. With the block rendered as
YAML it returns 18/18, in slightly fewer prompt tokens.

## Configuration Options

```python
from llm_schema_lite.dspy_integration import OutputMode, PromptLayout, StructuredOutputAdapter

adapter = StructuredOutputAdapter(
    output_mode=OutputMode.JSONISH,           # Output format mode
    include_input_schemas=True,               # Simplify input field schemas
    use_native_function_calling=True,         # Use native function calling
    max_recursion_depth=2,                    # Depth cap when formatter_config is None
    formatter_config=None,                    # FormatterConfig forwarded to simplify_schema
    prompt_layout=PromptLayout.SECTIONS,      # Output block layout
    use_json_object_response_format=True,     # Request {"type": "json_object"} in JSONish
    parallel_tool_calls=None,                 # Forwarded to the DSPy adapter base
    parse_config=None,                        # ParseConfig for the parse-time rescue tier
    callbacks=None,                            # Optional callbacks
)
```

### Parameters

- **output_mode**: `OutputMode.JSON`, `OutputMode.JSONISH`, or `OutputMode.YAML`
  - Controls the output format and schema representation
  - Default: `OutputMode.JSONISH`

- **include_input_schemas**: `bool`
  - Whether to include simplified schemas for complex input types
  - Useful when input fields are Pydantic models
  - Set to `False` to render input fields as a bare `{name}` placeholder with no note
  - Default: `True`

- **prompt_layout**: `PromptLayout`
  - `PromptLayout.SECTIONS` renders each output field as its own `[[ ## field ## ]]`
    block; `PromptLayout.JSON_BLOCK` renders one JSON-shaped block with the schema
    inlined, unescaped
  - Input fields always use the sectioned form, in both layouts
  - Accepted as either the enum member or the plain string (`"sections"`, `"json_block"`)
  - Default: `PromptLayout.SECTIONS`

- **formatter_config**: `FormatterConfig | None`
  - Forwarded unchanged to `simplify_schema`. When given it wins entirely and
    `max_recursion_depth` is ignored
  - This is *the* passthrough for schema-rendering options; the adapter deliberately
    exposes no per-option kwargs
  - Default: `None`

- **max_recursion_depth**: `int`
  - Depth cap for self-referential models, forwarded into a default `FormatterConfig`
  - **Ignored entirely when `formatter_config` is given** — the explicit config wins
  - Default: `2`

- **parse_config**: `ParseConfig | None`
  - `None` reproduces upstream `JSONAdapter`: a field that fails `parse_value` leaks its
    `ValidationError`
  - When given, a rejected field is offered to two rescues in order — the coercion
    rescue, then a structural repair (`allow_coercion`, on by default) that reshapes what
    the model sent and never adds a value: it strips a copied `*` marker from nested keys,
    unwraps one-item lists where the schema wants an object, moves fields the model
    hoisted out of a nested object back under it, nulls an optional object whose every
    field is null, wraps a lone value in a list where the schema wants a list of values,
    and drops all-null list items. A reply that sends an output field's contents without
    its key has them moved under it before the missing-field error; if
    `parse_config.partial` is `True` the field is then dropped and refilled by
    `apply_output_field_defaults`
  - The prune matters most on small models, which routinely answer an empty list with
    a placeholder instead of `[]` — `{"contacts": [{"email": null, "phone": null}]}`
    fails a required `email: str` and costs the whole record. Coercion cannot repair
    that without inventing a value; dropping an item whose every field is `null`
    removes nothing the model actually extracted. Measured on `qwen3.5:0.8b` over 30
    labeled extraction cases (JSONISH): field accuracy 0.745 → 0.929, 24/30 → 30/30
    parsed. Inert on models that do not make the mistake
  - Marker stripping does **not** depend on this config. The adapter always strips the
    marker its own formatter renders (`FormatterConfig.required_marker`, default `"*"`).
    Supplying `ParseConfig.strip_required_marker` adds a second marker that is also
    accepted; setting it to `""` turns stripping off entirely. For the standalone
    `loads(text, schema=...)` API there is no formatter, so
    `ParseConfig.strip_required_marker` is the only marker consulted there
  - Default: `None`

- **use_native_function_calling**: `bool`
  - Whether to use native function calling for tool calls
  - Default: `True`

- **use_json_object_response_format**: `bool`
  - JSONish mode only. When `True` the adapter sends
    `response_format={"type": "json_object"}` so the model is constrained to emit a JSON
    object
  - Set to `False` for OpenAI-compatible local servers (LM Studio, some Ollama builds)
    that advertise `response_format` but reject the `json_object` type
    ([stanfordnlp/dspy#1871](https://github.com/stanfordnlp/dspy/issues/1871))
  - Ignored in JSON mode (which reproduces upstream `JSONAdapter`'s structured-outputs
    behaviour) and in YAML mode (which never sets `response_format`). Even in JSONish
    mode, `response_format` is omitted when the signature carries `dspy.Tool` /
    `dspy.ToolCalls` fields
  - Default: `True`

- **parallel_tool_calls**: `bool | None`
  - Forwarded unchanged to the DSPy adapter base. When not `None` and native function
    calling is active on an LM that supports it, DSPy sets
    `lm_kwargs["parallel_tool_calls"]`
  - `None` leaves the provider option unset
  - Default: `None`

- **callbacks**: `list[BaseCallback] | None`
  - Optional callbacks for monitoring
  - Default: `None`

## Advanced Usage

### With ChainOfThought

<!-- lsl-docs: skip: issues a live LM request -->
```python
import dspy

from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter

lm = dspy.LM(model="openai/gpt-4o-mini")
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

<!-- lsl-docs: skip: issues a live LM request -->
```python
import dspy

from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter

lm = dspy.LM(model="openai/gpt-4o-mini")
adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
dspy.configure(lm=lm, adapter=adapter)


class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


predictor = dspy.Predict(QA)

demos = [
    dspy.Example(question="What is Python?", answer="A programming language"),
    dspy.Example(question="What is DSPy?", answer="A framework for LLMs"),
]

result = predictor(question="What is AI?", demos=demos)
```

### Error Handling

The adapter includes robust error handling with automatic fallbacks powered by llm-schema-lite:

When YAML *extraction* fails with a `ConversionError`, the adapter retries the same text
as JSON. This is a narrow rescue, not a blanket fallback: a `ConversionError` raised
anywhere else, and the completeness check's own `AdapterParseError`, both propagate.

```python
from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter

# JSON mode reproduces upstream JSONAdapter's structured-outputs behaviour.
adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

# JSONish mode: malformed JSON is automatically repaired (repair=True by default),
# using llm-schema-lite's loads() with json_repair integration.
adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)

# YAML mode: only a YAML *extraction* failure is retried as JSON (a narrow rescue,
# not a blanket fallback) - see the note above.
adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)
```

**Parsing Pipeline:**

1. Extract content from markdown code blocks, if present
2. Repair malformed JSON/YAML (`json_repair`); in YAML mode a `ConversionError` here
   retries the text as JSON
3. If the reply is a JSON array, unwrap the first object it contains — upstream
   `JSONAdapter` accepts this shape and so does this adapter
4. Map reply keys carrying the required marker onto their output-field names. The
   marker is `FormatterConfig.required_marker` — the same marker the prompt renders —
   so a model that echoes the `*` it was shown is understood. A key that matches an
   output field verbatim always wins over a marked one
5. Cast each value to its expected Pydantic type via `parse_value`
6. *(only when `parse_config` is given)* a rejected field is offered to the coercion
   rescue, then to the structural repair (nested markers, one-item lists, hoisted fields,
   all-null objects, lone values in list slots, all-null list items); with
   `parse_config.partial=True` a still-failing field is dropped
7. `apply_output_field_defaults` fills any output field the reply omitted
8. Check that every required output field is now present

## Streaming

`dspy.streamify` with per-field `StreamListener`s works in **JSON** and **JSONISH** modes.
Importing `llm_schema_lite.dspy_integration` registers `StructuredOutputAdapter` (and any
subclass of it) with DSPy's `StreamListener`, which otherwise only recognises DSPy's own
three built-in adapters by class name.

<!-- lsl-docs: skip: issues a live LM request -->
```python
import dspy
from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter

dspy.configure(adapter=StructuredOutputAdapter(output_mode=OutputMode.JSONISH))

program = dspy.streamify(
    dspy.Predict("question->answer"),
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)

async for value in program(question="What is the capital of France?"):
    if isinstance(value, dspy.streaming.StreamResponse):
        print(value.chunk, end="")
```

**YAML mode cannot be streamed field by field.** YAML output has no `"field":` boundary for
`StreamListener` to detect, so a `streamify` run with at least one stream listener raises
`StreamingNotSupportedError` (a `SchemaLiteError` and a `ValueError`) before any LM request is
issued. Two escape hatches: switch `output_mode` to `OutputMode.JSON`/`OutputMode.JSONISH`, or
call `dspy.streamify()` without `stream_listeners` (raw chunk streaming still works in YAML).

**Ordering caveat.** Registration happens when `llm_schema_lite.dspy_integration` is first
imported, and it only affects `StreamListener` instances constructed *after* that import. If you
build a listener before importing this package, it keeps DSPy's stock table and will not
recognise `StructuredOutputAdapter`. Construct listeners after the import, or call
`register_streaming_support()` and rebuild the listener.

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
from pydantic import BaseModel

from llm_schema_lite import FormatterConfig, simplify_schema


class Answer(BaseModel):
    """A model's answer with a confidence score."""

    answer: str
    confidence: float


# By default the adapter forwards no config, so simplify_schema's default
# FormatterConfig is used and all metadata (titles, descriptions, defaults) is kept.
simplified = simplify_schema(Answer, format_type="jsonish")  # or "typescript", "yaml"
print(simplified.to_string())

# Pass formatter_config to StructuredOutputAdapter to override this; it is forwarded to
# simplify_schema unchanged, e.g. for terser prompts:
terse = simplify_schema(Answer, config=FormatterConfig(include_descriptions=False))
print(terse.to_string())
```

**Robust Parsing** (via `loads()`):
- Parses JSON/YAML with automatic repair (`repair=True`)
- Handles markdown code blocks and embedded structures
- Returns a plain `dict` when no `schema=` is given, and a `(model instance, metadata)`
  2-tuple when one is
- When YAML *extraction* fails with a `ConversionError`, the adapter retries the same text
  as JSON. This is a narrow rescue, not a blanket fallback: a `ConversionError` raised
  anywhere else, and the completeness check's own `AdapterParseError`, both propagate.

```python
from llm_schema_lite import loads

# JSON parsing with repair - note the missing closing brace.
completion = '{"answer": "42", "confidence": 0.9'
print(loads(completion, mode="json", repair=True))

# YAML parsing with repair.
print(loads("answer: '42'\nconfidence: 0.9\n", mode="yaml", repair=True))
```

### Key Methods

- `__call__()` / `acall()`: sync and async execution. Both dispatch past `JSONAdapter`
  straight to `ChatAdapter`, so `response_format` is decided once, by this adapter
- `format_field_structure()`: builds the system-prompt field-structure block. Every field
  is reduced to an internal block by `_describe()`, then rendered; `prompt_layout` governs
  only the *output* renderer — input fields always use the sectioned form
- `parse()`: reads an LLM reply back into the signature's output types, via `loads()`
- Schema text for a complex annotation is resolved by a four-tier chain, each tier falling
  through on any exception:
  1. `simplify_schema(annotation, ...)`
  2. `TypeAdapter(annotation).json_schema()` fed back into `simplify_schema`
  3. the verbose `json.dumps` JSON Schema — the tier entered directly in `OutputMode.JSON`
  4. the literal text `must be a valid <name>`
- Scalar annotations never get a schema block, only a note (`bool` → "must be True or
  False", `Enum` → "must be one of: a; b", and so on). Annotations that are — or wrap,
  as in `list[...]` / `Optional[...]` / `dict[str, ...]` — a `dspy.Type` (Image, Audio,
  Tool, Code) or `dspy.History` emit nothing, matching upstream. The one exception is a
  `dspy.ToolCalls` **output** field, which falls through to the normal schema chain so
  the prompt carries the same tool-call guidance upstream `JSONAdapter` emits

## Token Efficiency Comparison

Token savings depend entirely on the schema. Measure yours rather than trusting a table:

```python
from pydantic import BaseModel

from llm_schema_lite import simplify_schema


class Answer(BaseModel):
    """A model's answer with a confidence score."""

    answer: str
    confidence: float


stats = simplify_schema(Answer).compare_tokens()
print(f"{stats['reduction_percent']}% smaller than the raw JSON Schema")
```

Cross-adapter benchmarks live under `benchmarking/dspy_adapters/` and run with
`make bench-dspy`.

## Testing

```bash
# Install the DSPy extra
pip install -e ".[dspy]"

# Run the DSPy integration tests
make test-dspy
```

`make test-dspy` runs `pytest tests -k dspy -v --no-cov`. The `--no-cov` matters: the
project's `addopts` force `--cov-report=xml`, and a second instrumented run clobbers
`coverage.xml`.

Every runnable code block on this page is executed by `tests/test_docs_examples.py`.
Blocks that would issue a live LM request carry an
`<!-- lsl-docs: skip: issues a live LM request -->` marker instead.

## Examples

`examples/basic_usage.py` is the runnable core-API tour (no DSPy, no LM, no API key). The
DSPy examples live on this page — every block that does not need a live LM is executed by
the test suite.

## Troubleshooting

### DSPy Not Found

```bash
pip install "dspy>=3.3.1"
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
  year = {2026},
  url = {https://github.com/rohitgarud/llm-schema-lite}
}
```
