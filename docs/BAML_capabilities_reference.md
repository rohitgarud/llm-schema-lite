# BAML Capabilities Reference for llm-schema-lite

This document provides a comprehensive overview of BAML (by BoundaryML) capabilities that could inform future development of llm-schema-lite. BAML is a domain-specific language for building type-safe LLM interfaces with structured outputs.

**Document Purpose:** Reference for understanding BAML's approaches to LLM schema representation, output parsing, validation, and advanced type handling that go beyond traditional JSON Schema.

**Last Updated:** 2026-02-15

---

## Table of Contents

1. [What is BAML?](#1-what-is-baml)
2. [Core Architecture & Philosophy](#2-core-architecture--philosophy)
3. [Type System](#3-type-system)
4. [Schema-to-Prompt Integration](#4-schema-to-prompt-integration)
5. [Parsing & Validation](#5-parsing--validation)
6. [Dynamic Types & Runtime Flexibility](#6-dynamic-types--runtime-flexibility)
7. [Prompt Engineering Features](#7-prompt-engineering-features)
8. [Streaming & Partial Parsing](#8-streaming--partial-parsing)
9. [Error Handling & Recovery](#9-error-handling--recovery)
10. [Testing Infrastructure](#10-testing-infrastructure)
11. [Multimodal Support](#11-multimodal-support)
12. [Comparison to llm-schema-lite](#12-comparison-to-llm-schema-lite)
13. [Opportunities for llm-schema-lite](#13-opportunities-for-llm-schema-lite)

---

## 1. What is BAML?

BAML is a **domain-specific language** (DSL) for defining type-safe interfaces between applications and LLMs. It combines:

- **Schema definition**: Custom type system with classes, enums, unions, literals
- **Function declarations**: Define input/output contracts with prompts
- **Code generation**: Generate type-safe client libraries (Python, TypeScript, Ruby, Go)
- **Structured output parsing**: Advanced parsing with error recovery
- **Validation**: Runtime assertions and checks
- **Testing**: First-class test support within the DSL

**Key Distinction:** BAML is a *full framework* (DSL → codegen → runtime), while llm-schema-lite is a *library* (Pydantic/JSON Schema → simplified format → validation).

---

## 2. Core Architecture & Philosophy

### 2.1 Function-Centric Design

BAML centers around **functions** that define:
- Input parameters with explicit types
- Return type specification
- LLM client configuration
- Prompt template (using Jinja syntax)

```baml
function ExtractEmail(text: string) -> string {
  client GPT4Turbo
  prompt #"
    Extract the email address from the following text:
    {{ text }}

    {{ ctx.output_format }}
  "#
}
```

**llm-schema-lite parallel:** We could extend to support "function definitions" that bundle schema + prompt template + validation rules.

### 2.2 Type-First Approach

- Types are defined in BAML syntax, not derived from host language
- Code generation creates matching types in target languages
- Single source of truth for LLM interface contracts

**llm-schema-lite parallel:** We derive from Pydantic/JSON Schema. Could support bidirectional: BAML-like syntax → JSON Schema + Pydantic.

### 2.3 Schema-Aligned Parsing (SAP)

BAML achieves **92-94.4% accuracy** on Berkeley Function Calling Leaderboard using proprietary Schema-Aligned Parsing:
- Goes beyond simple JSON mode
- Handles incomplete/malformed JSON
- Type coercion and repair during parsing
- State-of-the-art structured output extraction

**llm-schema-lite opportunity:** Implement parsing/repair layer beyond just validation.

---

## 3. Type System

### 3.1 Primitive Types

BAML supports standard primitives:
- `null`, `string`, `int`, `float`, `bool`

### 3.2 Literal Types

**Constrain primitives to specific values** (added v0.61.0):

```baml
function ClassifyIssue(issue: string) -> "bug" | "enhancement" | "question" {
  client GPT4Turbo
  prompt #"
    Classify: {{ issue }}
    {{ ctx.output_format }}
  "#
}
```

**Benefits:**
- More expressive than enums for simple classifications
- Natural union syntax
- Type-safe in generated code

**llm-schema-lite status:** We support enum values in JSONish (via JSON Schema `enum`), but not first-class literal types as return values.

**Opportunity:** Add literal type syntax to formatters for classification tasks.

### 3.3 Composite Types

#### Classes

```baml
class Person {
  name string
  age int
  contacts Contact[]
}

class Contact {
  type "email" | "phone"  // literal union
  value string
}
```

**Field Attributes:**
- `@alias("custom_name")` — rename in prompt/parsing without changing code
- `@description("...")` — add context for LLM
- `@skip` — exclude from prompt/parsing

**llm-schema-lite parallel:** Our formatters support similar features via JSON Schema (`title`, `description`, etc.).

#### Enums

```baml
enum Category {
  Refund
  CancelOrder @description("User wants to cancel order")
  TechnicalSupport @alias("tech_support")
  Question @skip  // exclude at runtime

  @@alias("IssueCategory")  // rename enum itself
  @@dynamic  // allow runtime modifications
}
```

**Features:**
- Value-level `@alias`, `@description`, `@skip`
- Enum-level `@@alias`, `@@dynamic`
- No spaces/special chars in values

**llm-schema-lite parallel:** We format enums but don't support per-value metadata or dynamic modification.

**Opportunity:** Add enum value descriptions, aliases, and runtime skipping.

#### Unions (|)

```baml
// Simple union
string | int

// Complex union
(int | string) | MyClass

// Union with arrays
string | MyClass | int[]
```

**Order matters:** `int | string` parses differently than `string | int`. If LLM outputs `"1"`, the first tries int parsing first.

**llm-schema-lite status:** We format unions (anyOf/oneOf), but don't document parse order priority.

#### Optional (?)

```baml
int?           // Optional<int>
(MyClass | int)?  // Optional<Union>
```

**llm-schema-lite parallel:** We handle `null` in JSON Schema (via nullable, anyOf with null).

#### Arrays ([])

```baml
string[]
(int | string)[]
int[][]  // multi-dimensional
```

**Constraints:**
- Array type itself cannot be optional
- Nested arrays supported

#### Maps

```baml
map<string, int>
map<Category, string>  // enum keys
map<"A" | "B" | "C", string>  // literal keys
```

**llm-schema-lite status:** We support `additionalProperties` but don't have dedicated map syntax in simplified output.

### 3.4 Type Aliases (v0.71.0)

```baml
type Graph = map<string, Node[]>
type DataStructure = string[] | Graph

// Recursive through containers
type JsonValue = int | string | bool | float | JsonObject | JsonArray
type JsonObject = map<string, JsonValue>
type JsonArray = JsonValue[]
```

**Opportunity:** Add type alias support to llm-schema-lite for common patterns.

### 3.5 Multimodal Types

First-class support for non-text inputs:
- `image` — from URL or base64
- `audio` — from URL or base64
- `pdf` — base64 only
- `video` — from URL or base64

```baml
function DescribeImage(img: image) -> string {
  client GPT4Turbo
  prompt #"
    {{ _.role("user")}}
    Describe this image in four words:
    {{ img }}
  "#
}
```

**Runtime usage:**
```python
from baml_py import Image
res = await b.DescribeImage(
  img=Image.from_url("https://example.com/image.png")
)
```

**Security note:** BAML downloads/transcodes media (potential SSRF risk if using untrusted URLs).

**llm-schema-lite status:** No multimodal type support.

**Opportunity:** Define multimodal schema extensions for LLM prompts (especially for vision models).

---

## 4. Schema-to-Prompt Integration

### 4.1 ctx.output_format

**Automatic schema injection** into prompts:

```baml
function ExtractResume(resume_text: string) -> Resume {
  prompt #"
    Extract this resume:
    {{ resume_text }}

    {{ ctx.output_format }}
  "#
}
```

**Rendered output:**
```
Extract this resume:
[resume text]

Answer in JSON using this schema:
{
  name: string
  education: [
    {
      school: string
      graduation_year: string
    }
  ]
}
```

**Key insight:** BAML uses **"jsonish" type definitions** (similar to llm-schema-lite's JSONish formatter!) instead of verbose JSON Schema for better LLM comprehension.

### 4.2 Customization Parameters

```baml
{{ ctx.output_format(
  prefix="Answer correctly for a $400 tip:\n",
  always_hoist_enums=true,
  hoist_classes=["Address", "Contact"],
  hoisted_class_prefix="interface",
  union_sep="or"
)}}
```

**Parameters:**

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `prefix` | Instruction before schema | Varies by return type |
| `always_hoist_enums` | Inline vs. hoisted enum definitions | `false` (heuristic) |
| `hoist_classes` | `"auto"`, `true`, `false`, or list | `"auto"` |
| `hoisted_class_prefix` | Word for hoisted types | (empty) → "schema" |
| `union_sep` | Union separator | `"or"` |

**Default prefix by type:**

| Return Type | Prefix |
|-------------|--------|
| String | (none) |
| Int | `Answer as an integer` |
| Other primitive | `Answer as a <type>` |
| Enum | `Answer with any of the categories:\n` |
| Class | `Answer in JSON using this schema:\n` |
| List | `Answer with a JSON Array using this schema:\n` |
| Union | `Answer in JSON using any of these schemas:\n` |
| Optional | `Answer in JSON using this schema:\n` |

**llm-schema-lite parallel:** Our formatters have fixed output; we don't support customization parameters yet.

**Opportunity:**
1. Add formatter configuration (prefix, hoisting rules, union separator)
2. Make formatters pluggable with options
3. Support prompt template variables

### 4.3 Why Not JSON Schema?

BAML's blog article argues:
1. **Performance:** Type definitions outperform verbose JSON Schema (especially on nested objects, smaller models)
2. **Readability:** JSON Schema is unreadable to humans (and models)
3. **Efficiency:** JSON Schema is ~4x more tokens than type definitions

**llm-schema-lite alignment:** We already use "jsonish" for exactly these reasons! Our JSONSchemaBench report shows we're on the right track.

---

## 5. Parsing & Validation

### 5.1 Schema-Aligned Parsing (SAP)

**Benchmark results:** 92-94.4% accuracy on Berkeley Function Calling Leaderboard.

**Techniques (not fully documented, proprietary):**
- Automatic JSON repair
- Type coercion (e.g., "1" → 1 for int fields)
- Handles incomplete/streaming JSON
- Falls back intelligently on parsing errors

**vs. Traditional approaches:**
- **Prompt engineering:** Lowest accuracy, no guarantees
- **JSON mode:** Better, but limited error recovery
- **Function calling:** Model-dependent, varies by provider
- **Constrained generation:** High accuracy but compute-intensive
- **SAP:** Best accuracy without generation constraints

**llm-schema-lite status:** We validate with `jsonschema` but don't parse/repair LLM outputs.

**Opportunity:** Add parsing layer with error recovery and type coercion.

### 5.2 Validation: @assert and @check

#### @assert — Strict Validation

```baml
class Foo {
  bar int @assert(between_0_and_10, {{ this > 0 and this < 10 }})
  email string @assert({{ this|regex_match("@") }})

  // Array element assertions
  items (string @assert(is_valid_email, {{ this|regex_match("@") }}))[]

  // Block-level assertions
  @@assert(baz_length_limit, {{ this.baz|length < this.bar }})
}
```

**Behavior:**
- Uses Jinja expressions for validation logic
- Named or unnamed assertions
- Raises `BamlValidationError` on failure
- Can apply to fields, parameters, or entire classes
- Multiple assertions evaluated left-to-right

#### @check — Conditional Validation

```baml
test MyTest {
  functions [MyFunction]
  args { ... }
  @check(result_check, {{ this.field > 100 }})
}
```

**llm-schema-lite parallel:** We use `jsonschema` for validation; supports JSON Schema validation keywords but not custom Jinja expressions.

**Opportunity:**
1. Add custom validation expressions (Python lambdas? Jinja?)
2. Support field-level and class-level constraints
3. Better error messages with named assertions

---

## 6. Dynamic Types & Runtime Flexibility

### 6.1 @@dynamic Annotation

Mark types as modifiable at runtime:

```baml
enum Category {
  VALUE1
  VALUE2
  @@dynamic  // allow runtime modifications
}

class User {
  name string
  age int
  @@dynamic  // allow runtime property additions
}
```

### 6.2 TypeBuilder API

**Modify enums at runtime:**

```python
from baml_client.type_builder import TypeBuilder
from baml_client import b

tb = TypeBuilder()
tb.Category.add_value('VALUE3')
tb.Category.add_value('VALUE4')

res = await b.DynamicCategorizer("input", {"tb": tb})
# Result can now be VALUE1, VALUE2, VALUE3, or VALUE4
```

**Modify classes at runtime:**

```python
tb = TypeBuilder()
tb.User.add_property('email', tb.string())
tb.User.add_property('address', tb.string()).description("User's address")

res = await b.DynamicUserCreator("user info", {"tb": tb})
```

**Create new types at runtime:**

```python
tb = TypeBuilder()
hobbies = tb.add_enum("Hobbies")
hobbies.add_value("Soccer")
hobbies.add_value("Reading")

address = tb.add_class("Address")
address.add_property("street", tb.string())

tb.User.add_property("hobby", hobbies.type().optional())
tb.User.add_property("address", address.type())
```

**Add BAML code at runtime:**

```python
tb.add_baml("""
  class Address {
    street string
    city string
  }

  dynamic class User {
    address Address
  }

  dynamic enum Category {
    VALUE5
  }
""")
```

**Use cases:**
- Database-driven categories (e.g., classify into user-defined tags)
- User-defined schemas (e.g., custom fields per tenant)
- Tool selection (add/remove available tools dynamically)

**llm-schema-lite status:** No dynamic type modification support.

**Opportunity:**
1. Add `@dynamic` marker in Pydantic models
2. Implement TypeBuilder-like API for runtime schema modification
3. Support dynamic enum values from database/config
4. Allow dynamic property additions to classes

---

## 7. Prompt Engineering Features

### 7.1 Role System

```baml
prompt #"
  {{ _.role("system") }}
  You are a helpful assistant.

  {{ _.role("user") }}
  {{ user_input }}

  {{ _.role("assistant") }}
  I understand. Let me help you with that.
"#
```

**Generates proper message structure** for chat models.

### 7.2 Template Variables

- `{{ ctx.output_format }}` — schema instructions
- `{{ ctx.client }}` — selected client/model name
- Standard Jinja: loops, conditionals, filters

```baml
prompt #"
  {% if condition %}
    Additional context
  {% endif %}

  {% for item in items %}
    - {{ item }}
  {% endfor %}
"#
```

### 7.3 Jinja Filters

Custom filters for prompt formatting:
```baml
{{ text|regex_match("pattern") }}
{{ items|length }}
```

### 7.4 Template Reuse

```baml
template_string MyTemplate(param: string) #"
  Common prompt snippet: {{ param }}
"#

function UseTemplate(input: string) -> Output {
  prompt #"
    {{ MyTemplate(input) }}
    {{ ctx.output_format }}
  "#
}
```

**llm-schema-lite status:** We don't handle prompts, only schema formatting.

**Opportunity:** If we expand to prompt templates, adopt Jinja with ctx.output_format pattern.

---

## 8. Streaming & Partial Parsing

### 8.1 Streaming Structured Output

BAML supports **streaming JSON** with automatic repair:

```python
stream = b.ExtractResume.stream(resume_text)
async for partial in stream:
    print(partial.name)  # Incrementally available
result = await stream.get_final_response()
```

**Key feature:** As tokens arrive with incomplete JSON like `{"items": [{"name": "Appl`, BAML repairs into valid partial objects.

### 8.2 Partial Types

BAML generates `baml_client.partial_types` module:
- All class fields become nullable
- Fields populated as tokens arrive
- Final result via `get_final_response()` is fully validated

**Example flow:**
1. LLM streams: `{"n`
2. Partial: `{"name": null}`
3. LLM streams more: `{"name": "Jo`
4. Partial: `{"name": "Jo"}`  (or null, depending on parser state)
5. Final: `{"name": "John Doe", ...}` — fully validated

**llm-schema-lite status:** No streaming or partial parsing support.

**Opportunity:**
1. Add streaming parser with JSON repair
2. Generate partial types (all-optional versions of schemas)
3. Incremental validation (validate what's complete, mark incomplete fields)

---

## 9. Error Handling & Recovery

### 9.1 Error Types

BAML provides specific error classes:

| Error | When Raised | Description |
|-------|-------------|-------------|
| `BamlValidationError` | Output doesn't meet schema/assertions | Type mismatch, assertion failure |
| `BamlClientFinishReasonError` | LLM stops unexpectedly | Length limit, content filter |
| `BamlAbortError` | Operation cancelled | User/timeout abort |
| (Network errors) | API failures | Rate limits, timeouts, auth issues |

### 9.2 Automatic Recovery

- **JSON repair:** Fix common JSON syntax errors
- **Type coercion:** Convert strings to numbers, booleans, etc.
- **Field-level fallback:** Mark individual fields as failed, continue parsing rest
- **Retry policies:** Configurable retries with exponential backoff

**llm-schema-lite status:** We validate with `jsonschema` and return errors; no parsing/recovery layer.

**Opportunity:** Add error recovery utilities (JSON repair, type coercion, partial extraction).

---

## 10. Testing Infrastructure

### 10.1 First-Class Tests

Tests are **part of the BAML language**:

```baml
test ExtractEmailTest {
  functions [ExtractEmail]
  args {
    text "Contact me at john@example.com"
  }
}
```

**Run tests:**
```bash
baml test                    # All tests
baml test ExtractEmailTest   # Specific test
```

### 10.2 Test Features

#### Assertions

```baml
test MyTest {
  functions [ExtractResume]
  args { ... }
  @assert(name_present, {{ this.name != null }})
  @assert(education_count, {{ this.education|length > 0 }})
}
```

#### TypeBuilder in Tests

```baml
test DynamicCategoryTest {
  functions [Categorize]
  type_builder {
    dynamic enum Category {
      NewCategory  // Add variant for this test
    }
  }
  args {
    text "Sample input"
  }
}
```

#### Multimodal Testing

```baml
test ImageTest {
  functions [DescribeImage]
  args {
    image {
      file "./test_image.png"
    }
  }
}
```

**llm-schema-lite status:** Standard Python pytest; no DSL-level test support.

**Opportunity:** Consider YAML/TOML test definitions for schema validation scenarios.

---

## 11. Multimodal Support

### 11.1 Supported Types

| Type | Input Formats | Notes |
|------|---------------|-------|
| `image` | URL, base64 | Auto-transcoding for models |
| `audio` | URL, base64 | Format: audio/ogg, audio/mp3, etc. |
| `pdf` | base64 only | URL support not implemented |
| `video` | URL, base64 | Some models can't download URLs |

### 11.2 Usage

```python
from baml_py import Image, Audio, Pdf, Video

# Image
img = Image.from_url("https://example.com/image.png")
img = Image.from_base64("image/png", base64_str)

# Audio
aud = Audio.from_url("https://example.com/audio.ogg")
aud = Audio.from_base64("audio/ogg", base64_str)

# Pdf (base64 only)
pdf = Pdf.from_base64("application/pdf", base64_str)

# Video
vid = Video.from_url("https://example.com/video.mp4")
vid = Video.from_base64("video/mp4", base64_str)
```

### 11.3 Media URL Handler

Control how BAML processes media URLs:

```baml
client<llm> MyClient {
  media_url_handler "download"  // or "passthrough"
  // ...
}
```

### 11.4 Security Considerations

**SSRF Risk:** BAML downloads URLs on your behalf. Untrusted user input could:
- Access internal network resources
- Use your application's identity
- Drive up bandwidth costs

**Mitigation:** Validate URLs with allowlists/denylists for untrusted sources.

**llm-schema-lite:** Not applicable (no multimodal support).

---

## 12. Comparison to llm-schema-lite

| Feature | BAML | llm-schema-lite | Notes |
|---------|------|-----------------|-------|
| **Architecture** | DSL + codegen + runtime | Library (Pydantic → format → validate) | Different scopes |
| **Type definition** | BAML syntax | Pydantic models, JSON Schema | BAML: single source; we: derived |
| **Schema format** | Jsonish type defs | JSONish, TypeScript, YAML | Similar philosophy! |
| **Validation** | @assert (Jinja), JSON Schema | jsonschema (Draft 2020-12) | BAML: custom expressions; we: standard |
| **Parsing** | Schema-Aligned Parsing (SAP) | Not provided | BAML: 92-94% accuracy with repair |
| **Streaming** | Partial types, incremental parse | Not supported | BAML: full streaming pipeline |
| **Dynamic types** | TypeBuilder API | Not supported | BAML: runtime schema modification |
| **Prompt integration** | ctx.output_format, Jinja | Not applicable | BAML: full prompt framework |
| **Multimodal** | image, audio, pdf, video | Not supported | BAML: first-class multimodal types |
| **Testing** | DSL-level tests | pytest | BAML: language-integrated |
| **Error recovery** | JSON repair, type coercion | Not provided | BAML: automatic recovery |
| **Code generation** | Python, TS, Ruby, Go clients | Not applicable | BAML: full codegen |
| **Literal types** | string/int/bool literals | Enum values only | BAML: first-class literals |
| **Type aliases** | Supported (v0.71+) | Not supported | BAML: reusable type definitions |
| **Enum metadata** | @alias, @description, @skip per value | Not supported | BAML: rich enum features |
| **Class attributes** | @alias, @description, @skip, @assert | Via JSON Schema properties | Similar but different syntax |
| **Union parse order** | Explicit, documented | anyOf/oneOf (JSON Schema rules) | BAML: order matters for parsing |

---

## 13. Opportunities for llm-schema-lite

Based on this analysis, here are high-value features we could adopt:

### 13.1 Parsing & Error Recovery (High Priority)

**Status:** llm-schema-lite only validates; doesn't parse LLM outputs.

**Opportunity:**
1. **Add parsing module** with JSON repair:
   - Fix common LLM JSON mistakes (trailing commas, missing quotes, unescaped chars)
   - Type coercion (string → int, "true" → bool, etc.)
   - Partial extraction (extract valid fields even if some fail)

2. **Streaming parser:**
   - Handle incomplete JSON
   - Generate partial types (all-optional versions)
   - Incremental validation

**Inspiration:** BAML's SAP achieves 92-94% accuracy. We don't need to replicate their proprietary algorithm, but basic JSON repair + type coercion would add huge value.

**Implementation path:**
- Research existing JSON repair libraries (e.g., `json-repair`, `jq` patterns)
- Add `llm_schema_lite.parsers` module
- Support both strict (fail on error) and lenient (repair) modes
- Add `parse_llm_output(text, schema, strict=False)` function

### 13.2 Dynamic Types (Medium Priority)

**Status:** llm-schema-lite schemas are static.

**Opportunity:**
1. **Add `@dynamic` marker:**
   ```python
   class Category(Enum, dynamic=True):
       VALUE1 = "value1"
       VALUE2 = "value2"
   ```

2. **TypeBuilder API:**
   ```python
   from llm_schema_lite import TypeBuilder

   tb = TypeBuilder(MyModel)
   tb.add_property('email', str, description="User email")
   tb.Category.add_value('VALUE3')
   schema = tb.to_json_schema()
   ```

**Use cases:**
- Database-driven enum values (categories, tags, statuses)
- Multi-tenant applications with custom fields per tenant
- Dynamic tool selection for function calling

**Implementation path:**
- Add `TypeBuilder` class that wraps Pydantic models
- Support adding fields, enum values, modifying descriptions
- Generate modified JSON Schema at runtime
- Use in formatters to produce modified output

### 13.3 Enhanced Enum Support (Medium Priority)

**Status:** We format enums but don't support per-value metadata.

**Opportunity:**
1. **Per-value descriptions:**
   ```python
   class Status(Enum):
       PENDING = ("pending", "Awaiting review")
       APPROVED = ("approved", "Fully approved by admin")
       REJECTED = ("rejected", "Rejected due to policy violation")
   ```

2. **Per-value aliases:**
   ```python
   class Category(Enum):
       BUG = ("bug", "Software defect", ["issue", "error"])
   ```

3. **Runtime skipping:**
   ```python
   tb = TypeBuilder(MyModel)
   tb.Status.skip_value('PENDING')
   ```

**Implementation path:**
- Extend enum representation in Pydantic models (use tuples or custom dataclass)
- Update formatters to include descriptions in output
- Support aliases in parsing/validation

### 13.4 Literal Types (Low-Medium Priority)

**Status:** We support enums but not ad-hoc literal unions.

**Opportunity:**
```python
from typing import Literal

def classify_issue(issue: str) -> Literal["bug", "feature", "question"]:
    ...
```

**Implementation path:**
- Update formatters to handle `Literal` type hints
- Generate literal unions in output (already supported in JSON Schema)
- Add to JSONish formatter (currently partial support)

### 13.5 Custom Validation Expressions (Low Priority)

**Status:** We use `jsonschema` validation only.

**Opportunity:**
1. **Python lambda constraints:**
   ```python
   class Foo(BaseModel):
       bar: int = Field(validators=[lambda x: 0 < x < 10])
   ```

2. **Named assertions:**
   ```python
   class Foo(BaseModel):
       email: str = Field(
           assert_={"is_valid": lambda x: "@" in x}
       )
   ```

**Implementation path:**
- Add custom validator support (leverage Pydantic's validator mechanism)
- Provide `assert_` field metadata for additional validation
- Generate better error messages with assertion names

### 13.6 Schema Customization Parameters (Medium Priority)

**Status:** Formatters have fixed output.

**Opportunity:**
```python
from llm_schema_lite import to_jsonish

output = to_jsonish(
    MyModel,
    prefix="Answer correctly for a $400 tip:\n",
    union_separator=" or ",
    indent=2
)
```

**Implementation path:**
- Add configuration parameters to formatter functions
- Make prefix customizable per return type
- Allow custom union separators

### 13.7 Multimodal Types (Low Priority, Future)

**Status:** Not supported.

**Opportunity:** Define schema extensions for vision/audio models:
```python
class ImageAnalysis(BaseModel):
    image: Image  # Special type with from_url, from_base64
    description: str
```

**Implementation path:**
- Add `Image`, `Audio`, `Pdf`, `Video` types
- Support URL and base64 inputs
- Generate appropriate schema for multimodal LLMs
- (Out of scope for pure schema library; more relevant if we expand to full framework)

### 13.8 Prompt Template Integration (Low Priority, Scope Expansion)

**Status:** llm-schema-lite doesn't handle prompts.

**Opportunity:** Add optional prompt template support:
```python
from llm_schema_lite import Function

extract_email = Function(
    name="ExtractEmail",
    input_schema={"text": str},
    output_schema=str,
    prompt="""
    Extract email from: {{ text }}
    {{ ctx.output_format }}
    """
)
```

**Implementation path:**
- Add `Function` class (optional feature)
- Support Jinja templating with `ctx.output_format`
- Generate prompts with schema injection
- (Significant scope expansion; evaluate if this fits llm-schema-lite's mission)

---

## Summary

**BAML's Core Strengths:**
1. **Comprehensive framework:** DSL → codegen → runtime
2. **Advanced parsing:** Schema-Aligned Parsing with 92-94% accuracy
3. **Dynamic types:** Runtime schema modification
4. **Rich type system:** Literals, unions, type aliases, assertions
5. **Streaming:** Partial parsing with automatic JSON repair
6. **Multimodal:** First-class support for images, audio, video

**llm-schema-lite's Current Strengths:**
1. **Focused scope:** Schema simplification + validation
2. **Pydantic/JSON Schema native:** Works with existing Python ecosystem
3. **Multiple formatters:** JSONish, TypeScript, YAML
4. **Token optimization:** Already aligned with BAML's "jsonish" philosophy
5. **Validation:** Full Draft 2020-12 support

**Top 3 Opportunities:**
1. **Add parsing layer** with JSON repair and type coercion (biggest gap vs. BAML)
2. **Implement TypeBuilder** for dynamic runtime schema modification
3. **Enhanced enum support** with per-value descriptions and aliases

**What Not to Adopt:**
- Full DSL + codegen (out of scope; Pydantic is our DSL)
- Prompt template framework (unless we expand scope significantly)
- LLM client management (out of scope)

---

## References

- [BAML Documentation](https://docs.boundaryml.com/)
- [BAML GitHub](https://github.com/BoundaryML/baml)
- [BoundaryML Blog: Schema-Aligned Parsing](https://www.boundaryml.com/blog/schema-aligned-parsing)
- [BoundaryML Blog: Every Way to Get Structured Output](https://www.boundaryml.com/blog/structured-output-from-llms)
- [BoundaryML Blog: Type Definitions vs JSON Schema](https://www.boundaryml.com/blog/type-definition-prompting-baml)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)

---

*Document compiled from BAML documentation, blog posts, and web research — 2026-02-15*
