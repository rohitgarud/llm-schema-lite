# Structured Output Parsing & Schema Enhancement Specification

**Document Type:** Technical Specification (Clean Room Design)
**Target Package:** llm-schema-lite
**Version:** 1.0
**Date:** 2026-02-15
**Status:** Draft

---

## Executive Summary

This specification defines enhancements to llm-schema-lite to support advanced LLM structured output scenarios, inspired by industry best practices in LLM schema management but adapted to our library-first, Pydantic-native approach.

**Core Objective:** Transform llm-schema-lite from a schema simplification + validation library into a comprehensive structured output toolkit that handles:
1. Schema formatting (existing)
2. Output parsing with error recovery (new)
3. Dynamic schema modification (new)
4. Streaming partial outputs (new)
5. Enhanced validation with custom constraints (new)

**Out of Scope:** DSL creation, code generation, LLM client management, prompt template framework (unless minimal integration), multimodal types (deferred).

---

## Table of Contents

1. [Core Principles](#1-core-principles)
2. [Enhanced Type System](#2-enhanced-type-system)
3. [Schema Formatting Enhancements](#3-schema-formatting-enhancements)
4. [Output Parsing & Repair](#4-output-parsing--repair)
5. [Dynamic Schema Modification](#5-dynamic-schema-modification)
6. [Streaming & Partial Parsing](#6-streaming--partial-parsing)
7. [Validation Enhancements](#7-validation-enhancements)
8. [Error Handling & Recovery](#8-error-handling--recovery)
9. [Implementation Guidelines](#9-implementation-guidelines)
10. [Phased Rollout](#10-phased-rollout)

---

## 1. Core Principles

### 1.1 Library-First Philosophy

**Principle:** llm-schema-lite remains a Python library, not a framework or DSL. Users work with standard Python types (Pydantic models, dataclasses, TypedDict) and JSON Schema.

**Rationale:**
- Leverage Python's native type system
- Integrate seamlessly with existing Pydantic codebases
- No learning curve for custom syntax
- IDE support and type checking out of the box

### 1.2 Pydantic-Native

**Principle:** Pydantic models are the primary interface. All features should work naturally with Pydantic's features (validators, Field metadata, Config).

**Rationale:**
- Pydantic is the de facto standard for Python data validation
- Rich ecosystem of tools and integrations
- Excellent type hints and IDE support

### 1.3 Token Optimization First

**Principle:** Schema simplification and token reduction remain core features. New features should not compromise our 60-85% token reduction.

**Rationale:**
- LLM costs are proportional to token count
- Simpler schemas → better LLM comprehension
- Competitive advantage over verbose JSON Schema

### 1.4 Progressive Enhancement

**Principle:** All features are opt-in. Basic usage remains simple; advanced features available when needed.

**Example:**
```python
# Simple usage (existing)
from llm_schema_lite import simplify_schema
schema = simplify_schema(MyModel)

# Advanced usage (new)
from llm_schema_lite import SchemaLite
sl = SchemaLite(MyModel, features=['parsing', 'dynamic', 'streaming'])
schema = sl.format()
result = sl.parse(llm_output)
```

### 1.5 Industry Best Practices

**Principle:** Learn from successful LLM schema systems (BAML, Instructor, Marvin) but adapt to our architecture.

**Rationale:**
- Avoid reinventing the wheel
- Adopt proven patterns
- Clean room design ensures no licensing issues

---

## 2. Enhanced Type System

### 2.1 Literal Type Support

**Current State:** Enums are supported; literal unions are not first-class.

**Specification:**

```python
from typing import Literal
from pydantic import BaseModel

class IssueClassification(BaseModel):
    category: Literal["bug", "feature", "question"]
    priority: Literal[1, 2, 3, 4, 5]
```

**Formatting Output (JSONish):**
```
{
  category: "bug" | "feature" | "question"
  priority: 1 | 2 | 3 | 4 | 5
}
```

**Requirements:**
- Detect `Literal` type hints in Pydantic models
- Format as union of literal values in all formatters
- Support string, int, bool literals
- Parsing: try each literal in order until match

**Implementation Notes:**
- Check `typing.get_origin(field.annotation) == Literal`
- Extract literal values via `typing.get_args()`
- For parsing, try to coerce to each literal type

### 2.2 Enhanced Enum Metadata

**Status: Implemented.**

Enums with `_descriptions` and `_aliases` class attributes are supported: schema enrichment injects `x-enum-descriptions` and `x-enum-aliases` into the JSON schema; all formatters (JSONish, YAML, TypeScript) show "OPTIONS with descriptions" and per-value text; validators normalize alias values to canonical before validation.

**Specification:**

```python
from enum import Enum
from pydantic import BaseModel

class Priority(str, Enum):
    """Issue priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Assign after class body so they are not coerced to string (str Enum)
Priority._descriptions = {
    "LOW": "Non-urgent, can wait",
    "MEDIUM": "Normal priority",
    "HIGH": "Needs attention soon",
    "CRITICAL": "Urgent, blocking issue",
}
Priority._aliases = {
    "CRITICAL": ["urgent", "blocker"],
}

class Issue(BaseModel):
    priority: Priority
```

**Formatting (JSONish):** Output includes `// OPTIONS with descriptions: low (...), medium (...), ...` and `OPTIONS: low | medium | high | critical`. YAML and TypeScript show the same descriptions in their native comment style.

**Validation:** Input values like `"urgent"` or `"blocker"` are normalized to `"critical"` before schema validation; invalid aliases are rejected.

### 2.3 Type Aliases for Reusability

**Current State:** No support for named type aliases in schema output.

**Specification:**

```python
from typing import TypeAlias, List, Dict
from pydantic import BaseModel

# Define reusable types
JsonValue: TypeAlias = str | int | float | bool | None | List['JsonValue'] | Dict[str, 'JsonValue']
Graph: TypeAlias = Dict[str, List[str]]

class DataStructure(BaseModel):
    graph: Graph
    metadata: JsonValue
```

**Formatting Output (TypeScript):**
```typescript
type Graph = Record<string, string[]>
type JsonValue = string | number | boolean | null | JsonValue[] | Record<string, JsonValue>

interface DataStructure {
  graph: Graph
  metadata: JsonValue
}
```

**Requirements:**
- Detect `TypeAlias` annotations
- Hoist type aliases to top of formatted output
- Reference by name in schema body
- Support recursive type aliases through containers

**Implementation Notes:**
- Track type aliases during schema traversal
- Emit definitions before main schema
- Handle forward references (`'JsonValue'`)

### 2.4 Optional Multimodal Types (Future Consideration)

**Deferred to Phase 3+**

Spec placeholder: Define `Image`, `Audio`, `Pdf`, `Video` types as Pydantic models with `.from_url()` and `.from_base64()` constructors.

---

## 3. Schema Formatting Enhancements

### 3.1 Configurable Formatter Options

**Current State:** Formatters have fixed behavior.

**Specification:**

```python
from llm_schema_lite import to_jsonish

schema = to_jsonish(
    MyModel,
    # Instruction prefix
    prefix="Answer in this format for a $400 tip:\n",

    # Union separator
    union_separator=" or ",  # default: " | "

    # Indentation
    indent=2,  # spaces (default: 2)

    # Optional fields marker
    optional_marker="?",  # default: "?"

    # Required fields marker
    required_marker="*",  # default: "*"

    # Include descriptions
    include_descriptions=True,  # default: True

    # Include constraints
    include_constraints=True,  # default: True
)
```

**Requirements:**
- Add `FormatterConfig` dataclass/dict parameter to all formatters
- Make prefix customizable (and vary by return type if desired)
- Customize separators and markers
- Toggle description/constraint inclusion

**Implementation Notes:**
- Add `FormatterConfig` class with defaults
- Pass config through formatter methods
- Update base formatter to accept config
- Document config options in formatter docstrings

### 3.2 Prefix Templates by Return Type

**Specification:**

```python
from llm_schema_lite import FormatterConfig

config = FormatterConfig(
    prefix_templates={
        "string": "",  # No prefix for string
        "int": "Answer as an integer",
        "float": "Answer as a number",
        "bool": "Answer as true or false",
        "enum": "Choose one of the following:\n",
        "class": "Answer in JSON using this schema:\n",
        "list": "Answer with a JSON array:\n",
        "union": "Answer using any of these formats:\n",
    }
)

schema = to_jsonish(MyModel, config=config)
```

**Requirements:**
- Detect top-level return type
- Apply appropriate prefix template
- Allow override via explicit `prefix` parameter

**Implementation Notes:**
- Add type detection helper: `_get_schema_category(schema) -> str`
- Look up prefix in config
- Fall back to explicit prefix if provided

### 3.3 Metadata Inclusion Controls

**Current State:** Metadata (min/max, pattern, format) sometimes included, sometimes not consistently.

**Specification:**

Control which metadata appears in output:

```python
config = FormatterConfig(
    metadata_inclusion={
        "pattern": True,
        "format": True,
        "minimum": True,
        "maximum": True,
        "minLength": True,
        "maxLength": True,
        "minItems": True,
        "maxItems": True,
        "uniqueItems": True,
        "const": True,
        "default": True,
        "examples": False,  # Don't include examples by default
    }
)
```

**Requirements:**
- Make metadata inclusion configurable per keyword
- Default: include constraints, exclude examples
- Apply consistently across all formatters

---

## 4. Output Parsing & Repair

### 4.1 Core Parsing Module

**Current State:** llm-schema-lite validates but does not parse LLM output.

**Specification:**

```python
from llm_schema_lite import parse_llm_output
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# Strict parsing (fail on any error)
result = parse_llm_output(
    llm_output='{"name": "Alice", "age": 30}',
    schema=Person,
    strict=True
)

# Lenient parsing (attempt repair)
result = parse_llm_output(
    llm_output='{"name": "Alice", "age": "30"}',  # age is string
    schema=Person,
    strict=False  # coerce "30" -> 30
)
```

**Requirements:**
- New module: `llm_schema_lite.parsers`
- Two modes: strict and lenient
- Return parsed Pydantic model instance
- Raise `ParsingError` on failure (with details)

### 4.2 JSON Repair

**Specification:**

Common LLM JSON errors to fix:

1. **Trailing commas:**
   ```json
   {"name": "Alice", "age": 30,}  →  {"name": "Alice", "age": 30}
   ```

2. **Missing quotes on keys:**
   ```json
   {name: "Alice"}  →  {"name": "Alice"}
   ```

3. **Unescaped strings:**
   ```json
   {"text": "She said "hello""}  →  {"text": "She said \"hello\""}
   ```

4. **Comments (not valid JSON):**
   ```json
   {"name": "Alice"} // comment  →  {"name": "Alice"}
   ```

5. **Single quotes instead of double:**
   ```json
   {'name': 'Alice'}  →  {"name": "Alice"}
   ```

6. **Truncated JSON:**
   ```json
   {"name": "Alice", "items": [{"id": 1  →  {"name": "Alice", "items": [{"id": 1}]}
   ```

7. **Extra text before/after JSON:**
   ```
   Here's the result: {"name": "Alice"} Hope that helps!
   →  {"name": "Alice"}
   ```

**Requirements:**
- Implement `repair_json(text: str) -> str` function
- Apply repairs in order (simple → complex)
- Extract JSON from surrounding text (look for `{...}` or `[...]`)
- Handle incomplete JSON (best effort completion)

**Implementation Strategy:**
- Use regex for simple fixes (trailing commas, single quotes)
- Use heuristics for quote escaping
- For truncation: balance brackets/braces, close open structures
- Consider third-party libraries: `json-repair`, `jq`, or custom implementation

### 4.3 Type Coercion

**Specification:**

Coerce LLM outputs to expected types:

| Expected Type | LLM Output | Coerced Value |
|---------------|------------|---------------|
| `int` | `"42"` | `42` |
| `int` | `42.0` | `42` |
| `float` | `"3.14"` | `3.14` |
| `float` | `3` | `3.0` |
| `bool` | `"true"` | `True` |
| `bool` | `"false"` | `False` |
| `bool` | `1` | `True` |
| `bool` | `0` | `False` |
| `str` | `42` | `"42"` |
| `list` | single item | `[item]` (optional) |
| `enum` | string | Match case-insensitive |

**Requirements:**
- Implement `coerce_value(value, target_type)` function
- Try coercion before rejecting type mismatch
- Configurable: `allow_coercion=True/False` in parse options
- Log coercions for debugging

**Implementation Notes:**
- Use Pydantic's validators when possible
- Custom coercion for edge cases
- Preserve original value in metadata for debugging

### 4.4 Partial Extraction

**Specification:**

Extract valid fields even if some fields fail:

```python
result = parse_llm_output(
    llm_output='{"name": "Alice", "age": "invalid", "email": "alice@example.com"}',
    schema=Person,
    partial=True  # Extract name and email, mark age as failed
)

# Result:
# Person(name="Alice", age=None, email="alice@example.com")
# With metadata: {"failed_fields": {"age": "invalid"}}
```

**Requirements:**
- Add `partial=True` option to parser
- Set failed fields to `None` (if optional) or skip (if required fails validation)
- Return parsing metadata: `{"failed_fields": {field: original_value}}`
- Raise error if required fields fail

**Implementation Notes:**
- Parse field-by-field instead of whole model
- Catch per-field validation errors
- Aggregate successful fields

---

## 5. Dynamic Schema Modification

### 5.1 SchemaBuilder API

**Current State:** Schemas are static (derived from Pydantic models at definition time).

**Specification:**

```python
from llm_schema_lite import SchemaBuilder
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

    class Config:
        dynamic = True  # Mark as modifiable

# Create builder
builder = SchemaBuilder(User)

# Add fields
builder.add_field('email', str, description="User's email address")
builder.add_field('phone', str | None, description="Optional phone number")

# Remove fields
builder.remove_field('age')

# Modify field properties
builder.modify_field('name', description="User's full name", min_length=2)

# Get modified schema
modified_schema = builder.to_json_schema()
formatted = builder.to_jsonish()

# Get modified Pydantic model (dynamic)
ModifiedUser = builder.to_model()
```

**Requirements:**
- New `SchemaBuilder` class wrapping a base Pydantic model
- Methods: `add_field()`, `remove_field()`, `modify_field()`
- Generate modified JSON Schema
- Optionally generate modified Pydantic model (use `create_model()`)
- Track modifications for debugging

**Implementation Notes:**
- Store modifications as deltas from base model
- Apply deltas when generating schema/model
- Use `pydantic.create_model()` for dynamic model generation
- Consider immutability: builder returns new builder on modifications?

### 5.2 Dynamic Enums

**Specification:**

```python
from llm_schema_lite import SchemaBuilder
from enum import Enum

class Category(Enum):
    GENERAL = "general"
    TECHNICAL = "technical"

    class Config:
        dynamic = True

builder = SchemaBuilder(MyModel)

# Add enum values
builder.enum('Category').add_value('BILLING', 'billing')
builder.enum('Category').add_value('SALES', 'sales')

# Skip enum values
builder.enum('Category').skip_value('GENERAL')

# Add descriptions
builder.enum('Category').set_description('TECHNICAL', "Technical support issues")
```

**Requirements:**
- Support adding enum values at runtime
- Support skipping values (exclude from schema output)
- Support per-value descriptions and aliases
- Generate modified enum in schema output

**Implementation Notes:**
- Cannot modify Python Enum class at runtime (immutable)
- Store modifications in builder; apply during schema generation
- For parsing, use modified enum values

### 5.3 Dynamic Classes

**Specification:**

```python
builder = SchemaBuilder(BaseModel)

# Create new class (not in original schema)
AddressClass = builder.add_class('Address')
AddressClass.add_field('street', str)
AddressClass.add_field('city', str)
AddressClass.add_field('country', str, default='USA')

# Attach new class to existing model
builder.add_field('address', AddressClass.to_type())
```

**Requirements:**
- Support creating new classes from scratch
- Support attaching new classes to existing models
- Generate proper type references in schema

**Implementation Notes:**
- Use `pydantic.create_model()` for new classes
- Store class definitions in builder registry
- Resolve references during schema generation

### 5.4 Runtime Constraints

**Specification:**

```python
builder = SchemaBuilder(User)

# Add runtime constraints
builder.add_constraint(
    field='age',
    constraint=lambda x: 18 <= x <= 120,
    error_msg="Age must be between 18 and 120"
)

# Add cross-field constraints
builder.add_constraint(
    constraint=lambda obj: len(obj.email) < obj.age,
    error_msg="Email length must be less than age (nonsense example)"
)
```

**Requirements:**
- Support field-level and object-level constraints
- Use Python callables (lambdas, functions)
- Provide custom error messages
- Integrate with validation

**Implementation Notes:**
- Store constraints separately from Pydantic validators
- Apply during `parse_llm_output()` if builder is provided
- Consider Pydantic's `@validator` decorator for inspiration

---

## 6. Streaming & Partial Parsing

### 6.1 Streaming Parser

**Current State:** Not supported.

**Specification:**

```python
from llm_schema_lite import StreamingParser
from pydantic import BaseModel

class Recipe(BaseModel):
    title: str
    ingredients: list[str]
    instructions: list[str]

parser = StreamingParser(Recipe)

# Feed chunks as they arrive from LLM
for chunk in llm_stream:
    parser.feed(chunk)

    # Get current partial result
    partial = parser.get_partial()
    print(f"Title so far: {partial.title}")
    print(f"Ingredients: {len(partial.ingredients)} found")

# Get final result
result = parser.get_final()
```

**Requirements:**
- `StreamingParser` class that accepts chunks
- `feed(chunk: str)` method to add data
- `get_partial()` returns partial model (fields may be `None`)
- `get_final()` returns fully validated model
- Handle incomplete JSON at chunk boundaries

**Implementation Strategy:**
- Buffer incoming chunks
- Attempt to parse JSON after each chunk
- Use JSON repair for incomplete structures
- Generate partial models with nullable fields

### 6.2 Partial Type Generation

**Specification:**

Automatically generate "partial" version of model with all fields optional:

```python
from llm_schema_lite import generate_partial_model

class Recipe(BaseModel):
    title: str
    ingredients: list[str]
    instructions: list[str]

PartialRecipe = generate_partial_model(Recipe)
# Equivalent to:
# class PartialRecipe(BaseModel):
#     title: str | None = None
#     ingredients: list[str] | None = None
#     instructions: list[str] | None = None
```

**Requirements:**
- Generate model with all fields optional
- Preserve field metadata (descriptions, constraints)
- Support nested models (recursively make partial)
- Cache generated models for performance

**Implementation Notes:**
- Use `pydantic.create_model()` with modified fields
- For each field: add `| None` to annotation, set `default=None`
- Handle `list`, `dict`, nested models recursively

### 6.3 Incremental Validation

**Specification:**

Validate incrementally as more data arrives:

```python
parser = StreamingParser(Recipe)

for chunk in llm_stream:
    parser.feed(chunk)

    # Validate what's complete
    validation_status = parser.validate_partial()

    # validation_status:
    # {
    #     "title": {"complete": True, "valid": True},
    #     "ingredients": {"complete": False, "valid": None},  # Still parsing
    #     "instructions": {"complete": False, "valid": None}
    # }
```

**Requirements:**
- Track completion status per field
- Validate completed fields
- Report validation errors incrementally
- Allow early stopping if critical fields fail

---

## 7. Validation Enhancements

### 7.1 Named Assertions

**Current State:** Validation uses Pydantic validators; no named assertions.

**Specification:**

```python
from llm_schema_lite import Field, assert_

class User(BaseModel):
    age: int = Field(
        assert_=assert_("valid_age", lambda x: 0 < x < 150, "Age must be between 0 and 150")
    )
    email: str = Field(
        assert_=[
            assert_("has_at", lambda x: "@" in x, "Email must contain @"),
            assert_("has_dot", lambda x: "." in x.split("@")[1], "Domain must have dot"),
        ]
    )
```

**Requirements:**
- `assert_()` helper that creates named assertions
- Support multiple assertions per field
- Custom error messages
- Evaluation order: left to right

**Implementation Notes:**
- Store assertions in Field metadata
- Apply during parsing (not just validation)
- Integrate with Pydantic's `@validator` mechanism

### 7.2 Cross-Field Validation

**Specification:**

```python
from llm_schema_lite import model_assert

@model_assert("end_after_start", lambda obj: obj.end_date > obj.start_date)
class Event(BaseModel):
    start_date: datetime
    end_date: datetime
```

**Requirements:**
- Model-level assertions
- Access to full object
- Named assertions with custom error messages

**Implementation Notes:**
- Use Pydantic's `@root_validator`
- Store assertion names for better error reporting

### 7.3 Conditional Validation

**Specification:**

```python
class Order(BaseModel):
    payment_method: Literal["card", "cash"]
    card_number: str | None = Field(
        required_if=lambda obj: obj.payment_method == "card"
    )
```

**Requirements:**
- Field-level conditional requirements
- Access to other field values
- Clear error messages

**Implementation Notes:**
- Use Pydantic's `@root_validator(pre=False)`
- Check conditions after all fields parsed

---

## 8. Error Handling & Recovery

### 8.1 Error Types

**Specification:**

Define specific error classes:

```python
class ParsingError(SchemaLiteError):
    """Base parsing error"""
    pass

class JSONRepairError(ParsingError):
    """Could not repair JSON"""
    original_text: str
    attempted_repairs: list[str]

class TypeCoercionError(ParsingError):
    """Could not coerce value to expected type"""
    value: Any
    expected_type: type
    field_name: str

class PartialParsingError(ParsingError):
    """Some fields failed to parse"""
    successful_fields: dict
    failed_fields: dict

class ValidationError(SchemaLiteError):
    """Validation failed (existing)"""
    # Enhanced with assertion names
    assertion_name: str | None
```

**Requirements:**
- Specific error types for different failure modes
- Include context: original value, expected type, field name
- Include attempted repairs for debugging

### 8.2 Error Recovery Strategies

**Specification:**

```python
from llm_schema_lite import parse_llm_output, RecoveryStrategy

result = parse_llm_output(
    llm_output,
    schema=MyModel,
    recovery=RecoveryStrategy(
        json_repair=True,        # Attempt JSON repair
        type_coercion=True,      # Coerce types
        partial_extraction=True, # Extract valid fields
        default_on_error=True,   # Use defaults for failed required fields (if default exists)
    )
)
```

**Requirements:**
- Configurable recovery strategy
- Ordered attempts: repair JSON → coerce types → partial extraction → defaults
- Log recovery actions for debugging

### 8.3 Detailed Error Reporting

**Specification:**

```python
try:
    result = parse_llm_output(llm_output, schema=MyModel)
except ParsingError as e:
    print(e.summary())
    # Output:
    # ParsingError: Failed to parse LLM output
    #
    # Original text (first 100 chars):
    # {"name": "Alice", "age": "not a number"...
    #
    # Attempted repairs:
    # 1. Fixed trailing comma
    # 2. Coerced age field to int: FAILED
    #
    # Errors:
    # - Field 'age': Could not coerce 'not a number' to int

    print(e.suggestions())
    # Output:
    # Suggestions:
    # - Check LLM prompt includes type information
    # - Consider adding examples to prompt
    # - Try lenient mode with partial=True
```

**Requirements:**
- Rich error messages with context
- Show original input (truncated)
- List attempted repairs
- Provide actionable suggestions
- Include error location (field name, line number if applicable)

---

## 9. Implementation Guidelines

### 9.1 Module Organization

Proposed package structure:

```
llm_schema_lite/
├── __init__.py
├── core.py                    # Existing: simplify_schema, validate
├── exceptions.py              # Enhanced error classes
├── formatters/
│   ├── __init__.py
│   ├── base.py                # Enhanced with FormatterConfig
│   ├── jsonish_formatter.py
│   ├── typescript_formatter.py
│   └── yaml_formatter.py
├── parsers/                   # NEW
│   ├── __init__.py
│   ├── base.py                # parse_llm_output()
│   ├── json_repair.py         # repair_json()
│   ├── type_coercion.py       # coerce_value()
│   ├── streaming.py           # StreamingParser
│   └── partial.py             # generate_partial_model()
├── dynamic/                   # NEW
│   ├── __init__.py
│   ├── builder.py             # SchemaBuilder
│   └── runtime_types.py       # Dynamic model generation
├── validation/                # Enhanced
│   ├── __init__.py
│   ├── assertions.py          # assert_(), model_assert()
│   └── constraints.py         # Runtime constraint handling
└── validators/
    ├── __init__.py
    ├── base.py
    ├── json_validators.py
    └── yaml_validators.py
```

### 9.2 Backward Compatibility

**Requirements:**
- All new features are opt-in
- Existing API remains unchanged
- Add new functions, don't modify existing signatures
- Deprecation warnings for any breaking changes (if unavoidable)

**Strategy:**
- Phase 1: Add new modules without modifying existing code
- Phase 2: Enhance existing formatters with optional config parameter
- Phase 3: Integrate new features with existing core functions

### 9.3 Dependencies

**Allowed:**
- Standard library (`json`, `re`, `typing`, `dataclasses`)
- Pydantic (already required)
- jsonschema (already required)

**Consider Adding:**
- `json-repair` or similar for JSON repair (evaluate licensing)
- Implement custom JSON repair if no suitable library

**Avoid:**
- Heavy ML/AI libraries
- LLM clients (out of scope)
- New web frameworks

### 9.4 Testing Strategy

**Requirements:**
- Unit tests for all new functions (pytest)
- Maintain >90% coverage for new code
- Integration tests for end-to-end scenarios
- Benchmarking for performance-critical paths (parsing, streaming)

**Test Scenarios:**
- JSON repair: collection of malformed JSON examples
- Type coercion: all type combinations
- Dynamic schema: add/remove/modify operations
- Streaming: partial data at various chunk boundaries
- Error handling: all error types and recovery strategies

### 9.5 Documentation

**Requirements:**
- Docstrings for all public functions (Google style)
- Usage examples in docstrings
- Type hints for all parameters and return values
- README updates for major features
- Examples directory with runnable scripts
- API reference (auto-generated from docstrings)

### 9.6 Performance Considerations

**Critical Paths:**
- JSON repair (used on every parse in lenient mode)
- Type coercion (frequent operation)
- Schema generation (cached where possible)

**Optimizations:**
- Cache formatted schemas (keyed by model + config)
- Pre-compile regex patterns
- Lazy loading for optional features
- Profile and optimize hot paths

---

## 10. Phased Rollout

### Phase 1: Output Parsing & Repair (Weeks 1-3)

**Goals:**
- Add basic parsing capability
- Implement JSON repair
- Implement type coercion
- Handle common LLM output errors

**Deliverables:**
- `llm_schema_lite.parsers` module
- `parse_llm_output()` function with strict/lenient modes
- `repair_json()` function
- `coerce_value()` function
- Unit tests with >90% coverage
- Documentation and examples

**Success Criteria:**
- Successfully parse 90%+ of real LLM outputs (test with sample data)
- Handle all common JSON errors
- Type coercion works for standard types

### Phase 2: Dynamic Schema Modification (Weeks 4-6)

**Goals:**
- Enable runtime schema changes
- Support dynamic enum values
- Support dynamic field addition

**Deliverables:**
- `llm_schema_lite.dynamic` module
- `SchemaBuilder` class
- Methods for add/remove/modify fields, enum values
- Integration with formatters
- Unit tests and integration tests
- Documentation and examples

**Success Criteria:**
- Add/remove fields without creating new Pydantic models
- Generate correct schemas from modified builders
- Runtime enum values work end-to-end

### Phase 3: Streaming & Enhanced Validation (Weeks 7-9)

**Goals:**
- Support streaming LLM outputs
- Add named assertions
- Improve error handling

**Deliverables:**
- `StreamingParser` class
- `generate_partial_model()` function
- Named assertions: `assert_()`, `model_assert()`
- Enhanced error classes with context
- Unit tests and integration tests
- Documentation and examples

**Success Criteria:**
- Stream and parse incrementally without errors
- Partial models generated correctly
- Named assertions provide clear error messages

### Phase 4: Formatter Enhancements (Weeks 10-11)

**Goals:**
- Add formatter configuration options
- Support literal types
- Enhanced enum metadata
- Prefix templates

**Deliverables:**
- `FormatterConfig` class
- Updated formatters with config support
- Literal type formatting in all formatters
- Enum description and alias support
- Unit tests
- Documentation

**Success Criteria:**
- All formatters support configuration
- Literal types formatted correctly
- Enum metadata appears in output

### Phase 5: Polish & Optimization (Week 12+)

**Goals:**
- Performance optimization
- Comprehensive testing
- Documentation review
- Benchmarking

**Deliverables:**
- Performance benchmarks
- Integration tests with real LLM outputs
- Complete documentation
- Migration guide
- Release notes

**Success Criteria:**
- Parsing <100ms for typical schemas
- All documentation complete and reviewed
- No regressions in existing features

---

## Appendices

### A. Design Decisions

**Why not a DSL?**
- Steeper learning curve
- Requires tooling (compiler, IDE support)
- Fragments ecosystem (BAML code vs Python code)
- Pydantic already provides excellent type system

**Why Pydantic-native?**
- De facto standard for Python validation
- Excellent type hints and IDE support
- Large ecosystem
- Users already familiar

**Why not prompt templates?**
- Out of scope for a schema library
- Many existing solutions (LangChain, DSPy, custom)
- Users can integrate our formatters with their prompt framework

**Why lenient parsing?**
- LLMs are not perfect JSON generators
- Manual repair is tedious
- High success rate with simple repairs
- Optional (strict mode available)

### B. Alternatives Considered

**JSON Repair:**
- **Option 1:** Use third-party library (e.g., `json-repair`)
  - Pro: Less code to maintain
  - Con: External dependency, licensing
- **Option 2:** Implement custom repair
  - Pro: No dependencies, full control
  - Con: More code, potential bugs
- **Decision:** Start with custom, evaluate third-party if complex

**Dynamic Types:**
- **Option 1:** Generate new Pydantic models
  - Pro: Full Pydantic features
  - Con: Runtime model creation overhead
- **Option 2:** Modify JSON Schema only
  - Pro: Lightweight
  - Con: No Python model to instantiate
- **Decision:** Hybrid: modify JSON Schema by default, optionally generate model

**Streaming:**
- **Option 1:** Buffer and reparse on each chunk
  - Pro: Simple
  - Con: Inefficient for large outputs
- **Option 2:** Incremental parsing (stateful parser)
  - Pro: Efficient
  - Con: Complex implementation
- **Decision:** Start with Option 1, optimize if needed

### C. Success Metrics

**Adoption:**
- GitHub stars growth
- PyPI downloads
- Community issues/PRs

**Quality:**
- Test coverage >90%
- No critical bugs in production
- Response time to issues <7 days

**Performance:**
- Parse typical LLM output <100ms
- Format typical schema <50ms
- Memory usage <50MB for large schemas

**Usability:**
- Documentation rated 4+/5
- Examples cover 80% of use cases
- <5 questions/week on same topic (indicates docs gap)

---

**End of Specification**

This specification provides a comprehensive roadmap for enhancing llm-schema-lite with structured output parsing capabilities, inspired by industry best practices but adapted to our library-first, Pydantic-native approach. Implementation should proceed in phases, with continuous testing and documentation at each stage.
