# llm-schema-lite

[![PyPI version](https://img.shields.io/pypi/v/llm-schema-lite)](https://pypi.org/project/llm-schema-lite/)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm-schema-lite.svg)](https://pypi.org/project/llm-schema-lite/)
[![CI](https://github.com/rohitgarud/llm-schema-lite/actions/workflows/ci.yaml/badge.svg)](https://github.com/rohitgarud/llm-schema-lite/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/rohitgarud/llm-schema-lite/branch/main/graph/badge.svg)](https://codecov.io/gh/rohitgarud/llm-schema-lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Turn a Pydantic model into a compact schema string an LLM can follow, and turn the reply
back into a validated model. Constraints ride inside the type line instead of a separate
block, which costs **typically 40-70% fewer schema tokens** than raw JSON Schema — measure
your own with `compare_tokens()`. Parsing is built for small models that don't quite follow
instructions: brace-balanced extraction, JSON repair, and opt-in rescue for the mistakes
1B-class models actually make.

Framework agnostic by construction: the package turns a schema into a string and a reply back into a model, so it drops into the OpenAI SDK, any OpenAI-compatible endpoint, or a framework like DSPy without tying you to any of them.

It takes **raw JSON Schema as readily as a Pydantic model**, and it aims at the whole of the
specification rather than a convenient subset. Measured over
[JSONSchemaBench](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench) — **9,542
real-world schemas across 10 configs** — ingestion is **100%**, with **zero** errors and
**zero** timeouts, at a **46.8% median token reduction**.
[What that does and doesn't claim ↓](#-json-schema-coverage)

### Using DSPy?

```python
import dspy

from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter

dspy.configure(adapter=StructuredOutputAdapter(output_mode=OutputMode.JSONISH))
```

That one line swaps DSPy's full JSON Schema prompt for the compact format below and adds
parse-time repair for small-model replies. Measured on the synthetic extraction benchmark
with `qwen3.5:0.8b` (30 cases), enabling the rescue tier against an otherwise
**token-identical prompt** moved field accuracy 0.745 → 0.929 and parsed records 24/30 →
30/30 — paired bootstrap 95% CI `[+0.061, +0.324]`, with invented values unchanged at
`[+0.000, +0.000]`. It is not a uniform win: on the `pii` and `patient-notes` corpora the
same rescue is a statistical tie, and on `insurance-claims` it buys recall by inventing
more. The corpus-by-corpus tables, including where it loses, are in
[the DSPy integration README](src/llm_schema_lite/dspy_integration/README.md).

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

## 📐 JSON Schema Coverage

The goal is the whole specification, not a comfortable subset — so it is measured rather than
asserted, against [JSONSchemaBench](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench)
(`epfl-dlab/JSONSchemaBench`): **10 configs, 9,542 real-world schemas**, from `Github_trivial`
through `Github_ultra`, Kubernetes, Snowplow and the JSON Schema Store.

| Measured over all 9,542 | |
|---|---|
| Ingested without error | **9,542 / 9,542 (100.0%)** |
| Schemas that raise | **0** |
| Too slow to render (10 s/schema budget) | **0** |
| Median token reduction | **46.8%** — per-config medians span 31.4%–57.0% |
| Keywords rendered | JSONish **32/39** · YAML **33/39** · TypeScript **31/39** |
| Validation | full Draft 2020-12 via `jsonschema` |

Reproduce it yourself — the dataset fetch is ~100 MB:

```bash
uv run python -m benchmarking.jsonschemabench.fetch_dataset
uv run python -m benchmarking.jsonschemabench.coverage --all-configs
```

Two qualifications, because the headline invites a stronger reading than it supports.

**Rendering and validation are different surfaces.** Every keyword is *validated* — the full
draft, enforced by `jsonschema`, whether or not it reaches the prompt. The `32/39` counts only
what survives into the schema string. Those gaps are also the corpus's rarest keywords:
`minProperties` 1.6%, `maxProperties` 0.6%, `if`/`then` 0.5%, `else` 0.2%,
`unevaluatedProperties` 0.02% — and `dependentRequired` never appears in the corpus at all.

**Quote the median, never the mean.** `Github_hard` has a median reduction of **40.1%** against
a mean of **13.9%**: the typical schema compacts by half while a handful of pathological ones
drag the average down. A mean over this corpus describes its worst tail, not its behaviour.

### Against constrained-decoding engines

A different mechanism — a grammar constrains the sampler, we write the prompt — but the same
question of what actually reaches the model:

| Feature | LLGuidance | llama.cpp | Outlines | XGrammar | OpenAI | Gemini | **llm-schema-lite** |
|---|---|---|---|---|---|---|---|
| oneOf | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| patternProperties | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| not | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| contains | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| min/max (integer) | ✔ | ✔ | ❌ | ❌ | ❌ | ❌ | **✅** |
| minLength / maxLength | ✔ | ✔ | ✔ | ❌ | ❌ | ❌ | **✅** |
| pattern | ✔ | ✔ | ✔ | ✔ | ❌ | ❌ | **✅** |
| if / then / else | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

The asymmetry is the point: their gaps are **architectural** — an FSM or grammar cannot express
`if/then/else`, `not`, or unbounded recursion. Ours are **implementation debt** — already parsed
and validated, simply not yet rendered into the string.

### The known failure mode: `$ref` expansion

Worth stating plainly rather than burying. A `$ref` is expanded **inline at its use site**, so a
cyclic or heavily-shared definition graph can render *larger* than the raw schema it came from.
A definition rendered once is replaced at later sites by a named back-reference
(`object // defined above: Address`), which is what keeps this rare:

- **56 of 9,542 schemas (0.59%)** still render larger than their input.
- Past the single worst case, the next is **8.67×**, and everything outside the top ten is
  under **3×**.
- The worst, `o13029`, renders at **82×** — seven *mutually* recursive definitions, and the
  accepted limit of repeat-suppression. Closing it needs depth-keyed caching, which is a
  separate decision, not a bug fix.
- Where a `$ref` exhausts the expansion budget the output **says so** —
  `object // budget exhausted: Name` — instead of emitting a bare `object` indistinguishable
  from an untyped one. Fires on **32 schemas (0.34%)**.

Both thresholds are yours to move, on `FormatterConfig`, if your definition graph is
legitimately wide rather than deep:

| Field | Default | Effect |
|---|---|---|
| `max_ref_expansions` | `150` | Total `$ref` expansions allowed across the whole render. Past it, every further `$ref` becomes `object // budget exhausted: Name`. Raise it to render a large graph in full; it also tiers the `anyOf`/`oneOf` member caps. |
| `backreference_min_chars` | `200` | Smallest body worth naming instead of inlining a second copy. `0` names every repeat; a very large value inlines every repeat. The default sits in the gap between the two populations — the largest body two sibling fields share in the test suite is 158 chars, the smallest definition in the 244-reference corpus schema is 409. |

```python
from llm_schema_lite import FormatterConfig, simplify_schema

# 200 distinct definitions, each referenced once: wide, not deep.
wide = {
    "type": "object",
    "properties": {f"f{i}": {"$ref": f"#/$defs/D{i}"} for i in range(200)},
    "$defs": {
        f"D{i}": {"type": "object", "title": f"D{i}", "properties": {"v": {"type": "string"}}}
        for i in range(200)
    },
}

# The default budget stops at 150 and says where it stopped.
assert "budget exhausted" in simplify_schema(wide).to_string()

# Raised past 200, every definition renders.
config = FormatterConfig(max_ref_expansions=500)
assert "budget exhausted" not in simplify_schema(wide, config=config).to_string()
```

Neither default has moved, so raising one is opt-in and costs tokens; leaving them alone
renders exactly as the measured figures above.

For scale: the worst schema in the corpus used to render at **45×** its input and time out.
Scoping one over-broad cache-invalidation rule took it to **0.35×**, removed every timeout, and
is what moved ingestion to 100%. Full methodology, per-config tables, and the profiling behind
that fix are in [the JSONSchemaBench report](docs/JSONSchemaBench_feature_report.md).

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

`StructuredOutputAdapter` wires the same two calls into DSPy's adapter protocol, so a DSPy
program gets the compact schema and the small-model parsing without changing any of its
signatures. Skip this section if you are not using DSPy.

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

### Your existing Pydantic models, unchanged

You do not flatten your domain into scalar `InputField`s, and you do not maintain a second set
of models for DSPy. Annotate with the models you already have — **on inputs as well as
outputs** — and the compact schema is rendered for both:

```python
import dspy
from pydantic import BaseModel, Field

from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter


class Address(BaseModel):
    street: str = Field(min_length=3, description="Street name")
    zipcode: str = Field(pattern=r"^[0-9]{5}$", description="ZIP")


class Company(BaseModel):
    name: str = Field(description="Company name")
    hq: Address
    branches: list[Address] = Field(description="Branch offices")


class Summarise(dspy.Signature):
    """Summarise the company."""

    company: Company = dspy.InputField()
    summary: str = dspy.OutputField()


adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
print(adapter.format_field_structure(Summarise))
```

The **input** block carries the nested schema, its constraints and its descriptions intact:

```
[[ ## company ## ]]
{company}        # note: this value follows the schema:
//Title: Company
{
  name*: string // Company name,
  hq*: {
    street*: string (>= 3 chars) // Street name,
    zipcode*: string (PATTERN: ^[0-9]{5}$) // ZIP
  },
  branches*: [{ ... }]  // Branch offices
}
```

Three consequences worth stating outright:

- **Pydantic stays the single source of truth.** `description`, `ge`/`le`, `min_length` /
  `max_length` and `pattern` all survive into the prompt — inside nested models and inside
  `list[Model]` items alike. Field metadata lives on the field, rather than being restated in a
  signature docstring and drifting from the model it describes.
- **Nothing here is DSPy-specific.** No base class, no mixin, no decorator, no registration
  step: the adapter reads the raw annotation. The same `Company` serves your API layer, your
  database code and your DSPy program.
- **Nesting is followed** — models within models, and lists of models.

Two limits to know. Compact input schemas are a **`JSONISH`/`YAML`** feature: `OutputMode.JSON`
sends the verbose raw JSON Schema instead. And `include_input_schemas=False` (default `True`)
turns input schemas off altogether when you want a shorter prompt.

Every other constructor option — `formatter_config` / `parse_config` forwarding,
`prompt_layout`, the `json_object` response-format flag and its tool-call interaction,
streaming registration and known limits — is documented in
[the DSPy integration README](src/llm_schema_lite/dspy_integration/README.md).

### Typed decisions with Jev

[Jev](https://docs.typesafe.ai/introduction) (TypeSafe AI) is a *System One* model: it takes
a `state` and a set of typed questions and returns calibrated probabilities instead of text,
so there is nothing to parse or repair. `JevAdapter` compiles a signature into that request
and decodes the answers back into typed output fields; `JevLM` sends it (OpenRouter by
default, reading `$OPENROUTER_API_KEY`).

<!-- lsl-docs: skip: issues a live Jev decision request via OpenRouter -->
```python
from enum import Enum
from typing import Annotated, Literal

import dspy
from pydantic import BaseModel, Field

from llm_schema_lite.dspy_integration import JevAdapter, JevLM


class Team(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SALES = "sales"


class Route(BaseModel):
    team: Team = Field(description="Which team should handle this?")
    escalate: bool = Field(description="Should this go to a manager?")


class Triage(dspy.Signature):
    """Triage a support message."""

    message: str = dspy.InputField()
    is_urgent: bool = dspy.OutputField(desc="Does this message convey urgency?")
    frustration: Annotated[
        Literal["calm", "frustrated", "angry"],
        Field(json_schema_extra={"jev": {"type": "score"}}),
    ] = dspy.OutputField(desc="How frustrated is the customer?")
    route: Route = dspy.OutputField()


dspy.configure(lm=JevLM(), adapter=JevAdapter())
pred = dspy.Predict(Triage)(message="Help! My payouts have been failing for 3 days.")

pred.route.team                        # Team.BILLING
pred.jev["route.team"]["confidence"]   # raw answers: probabilities and confidence per question
```

| Output annotation | Jev question | Decoded as |
|---|---|---|
| `bool` | `noul` | `True` when the probability reaches `threshold` (default `0.5`) |
| `Literal[...]` / `Enum` | `choice` | the chosen option, as the `Literal` value or `Enum` member |
| `Literal[...]` / `Enum` marked `{"jev": {"type": "score"}}` | `score` over the options, in order | the most probable level |
| nested `BaseModel` | one question per leaf, keyed by dotted path (`route.team`) | the rebuilt model |

- **Per-field options** go in `json_schema_extra={"jev": {...}}` via `Annotated[T, Field(...)]`
  (`dspy.OutputField` drops unknown kwargs): `type="score"`, `criteria` (your own option or
  level descriptions, or `{"true": ..., "false": ...}` for a `bool`) and `threshold`.
- **Instructions:** the signature docstring and each field's description become the
  question's `instructions`, so DSPy instruction optimizers still apply. Demos are ignored,
  since Jev takes no few-shot examples.
- **Only decisions:** `str`, free-form numbers and lists raise `TypeError` before any
  request, since Jev cannot produce them. That includes `dspy.ChainOfThought`'s
  `reasoning` field, so use `dspy.Predict`.
- **`JevLM`** supports `acall`, DSPy's request cache (`cache=False` to bypass) and saving
  programs. Its saved state keeps `url` but never the API key. Pass
  `url="https://api.typesafe.ai/v1/systemone"` to call TypeSafe directly.

#### Local open models (SemIf)

`SemIfLM` answers the same `JevAdapter` requests with a local model instead, using
[SemIf](https://github.com/TheoLeeCJ/SemIf)'s direct option-logit readout. Each option gets
a letter, and the probability of each option is read from that letter's next-token
logprob, so each question costs one single-token request. Use any OpenAI-compatible server
that returns `top_logprobs` (vLLM, llama.cpp `llama-server`, SGLang, Ollama):

<!-- lsl-docs: skip: needs a local model server -->
```python
from llm_schema_lite.dspy_integration import JevAdapter, SemIfLM

lm = SemIfLM("openai/Qwen/Qwen3.5-4B", api_base="http://localhost:8000/v1", api_key="local")
dspy.configure(lm=lm, adapter=JevAdapter())
```

The probabilities are uncalibrated and only compare the options you gave. Calibrate them on
your own data before you rely on a threshold. `SemIfLM` is an ordinary `dspy.LM`, so
caching, retries, `acall` (which sends the questions concurrently) and saving work as
usual. For thinking models, also pass
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.

On SemIf's own 144-row workload, `SemIfLM` sends byte-identical prompts. It agrees with
SemIf's published decisions on 96–98% of rows, and its accuracy lands within 0.011 of
SemIf's reported numbers. See the [parity benchmark](benchmarking/semif/README.md).

`dspy.Image` inputs reach the model as image content, so the same readout works on a vision
model. The server must return `top_logprobs` for an image request; llama.cpp does. A server
without vision support fails with an error rather than dropping the image.

A question can have up to 256 options. Past 16, options get two-letter labels (`AA` to `PP`)
and are read by the chain rule. When a label is split into two tokens, the server has to
continue a pre-filled assistant turn, which llama.cpp does. Small models pick poorly from
long option lists, though: on Wikipedia pages, asking Qwen3-VL-4B about each link
separately chose better links than one 255-option question.

For fan-out, meaning many requests that differ only in their evidence (candidates, frames,
links), pass `SemIfLM(..., question_first=True)`. The criterion and options then come
before the evidence, so the server can reuse them from its KV cache. With llama.cpp started
with `-np 16 --kv-unified --cache-ram 0`, 16 parallel requests ran 3.2x faster (5.9 against
19 ms per request). It stays off by default. It departs from SemIf's prompt, and on
JevBench it cost Qwen3-0.6B 37 of 231 items (p < 0.0001), while making no measurable
difference on the 2B and 4B models. Measure it on your own model before you turn it on.

##### Demo: SemIf plays Doom from pixels

![A local vision model playing Doom through SemIfLM, with its option probabilities beside the game](https://raw.githubusercontent.com/rohitgarud/llm-schema-lite/main/demos/doom/doom_overlay.webp)

Qwen3-VL-4B (Q4_K_M, llama.cpp, one 8 GB laptop GPU) plays ViZDoom's `defend_the_center`
from raw screenshots. Each frame is one question, "where is the nearest monster?", with four
options: centre, left, right, none. That makes one single-token request, answered in a median
of about 130 ms. The panel shows the probabilities the model returned.

The agent turns toward whichever side has more probability. It fires only when
`P(centre) >= 0.95`, not whenever `centre` is the argmax. That gate is what makes it play:

| policy (20 episodes unless noted) | mean score |
|---|---|
| best policy that ignores the screen (random, biased to turn right) | +1.70 ± 0.28 |
| SemIf, act on the argmax (10 episodes) | +1.90 |
| SemIf, fire only if `P(centre) >= 0.95` | **+6.50 ± 0.97** |

On the argmax the agent fires on about three quarters of frames. It then scores the same as a
policy that never looks. The 0.95 threshold was picked by trying 0.5, 0.8 and 0.95 on this
task, which is the calibration step above. A scripted bot would still play better. The demo
shows what the probabilities are for, not a strong Doom player. For comparison, TypeSafe's
[Jev Doom demo](https://typesafe.ai/blog/introducing-system-one-models-and-jev) drives the
game from structured state given as text; this one reads pixels.

To run it, start `llama-server` with the model and its `--mmproj` on port 8089. Then run:

<!-- lsl-docs: skip: needs vizdoom and a local vision model server -->
```bash
pip install vizdoom pillow imageio
python demos/doom/doomoverlay.py      # plays and records demos/doom/doom_overlay.webp
python demos/doom/doomconfirm.py 20   # the gated policy vs random, with standard errors
python demos/doom/doombase.py 20      # screen-blind baselines, no model needed
```

### Benchmark results

Six sub-1.2B models × eight adapters × five corpora, 30 labeled cases each, run against a
local Ollama. Every cell's raw replies are committed under
`benchmarking/dspy_adapters/results/`, so the numbers below can be re-scored with no model
and no network: `make bench-dspy BENCH_ARGS="--accuracy --corpus pii --replay <csv>"`.

**Read every corpus against its all-null floor** — the score a reply that extracts nothing
would get. Field accuracy counts a correct `null` as a match, so on a sparse corpus it pays
an adapter for extracting nothing; *recall on non-null gold* is the extraction headline and
has a floor of 0 by construction.

| corpus | all-null floor | best adapter (recall) | `JSONAdapter` |
|---|---|---|---|
| synthetic | 0.025 | `sola-jsonish-rescue` **0.936** | 0.889 |
| insurance-claims | 0.018 | `sola-yaml-rescue` **0.749** | 0.119 |
| financial-ner | 0.590 | `sola-yaml-sections` **0.529** | 0.069 |
| pii | **0.949** | `sola-jsonish-rescue` 0.279 | 0.244 |
| patient-notes | **0.356** | `sola-jsonish-sections` 0.214 | 0.000 |

`qwen3.5:0.8b`, the only model that functions across all five. Paired bootstrap, 5000
resamples, 95% CI on the difference:

| comparison | corpus | diff (recall) | 95% CI | verdict |
|---|---|---|---|---|
| `sola-jsonish-rescue` vs identical prompt without rescue | synthetic | +0.193 | `[+0.065, +0.339]` | **higher** |
| — its invented rate | synthetic | +0.000 | `[+0.000, +0.000]` | tie |
| `sola-yaml-rescue` vs `JSONAdapter` | insurance-claims | +0.630 | `[+0.523, +0.722]` | **higher** |
| — its invented rate | insurance-claims | +0.459 | `[+0.339, +0.574]` | **worse** |
| `sola-jsonish-rescue` vs `JSONAdapter` | synthetic | +0.047 | `[-0.019, +0.137]` | tie |
| `sola-jsonish-rescue` vs identical prompt without rescue | pii | +0.012 | `[+0.000, +0.036]` | tie |

Honest reading, because the floors matter more than the wins:

- **The cleanest result is the rescue A/B on synthetic**: same prompt, same tokens, only
  parse-time repair differs — recall +0.193 with the invented rate provably unchanged.
- **Nothing in the table above beats the all-null floor on `pii` or `patient-notes`.** On
  those two corpora every one of these eight adapters, ours included, loses to extracting
  nothing. `patient-notes` is only unbeaten *within this table*: a ninth arm not in it —
  upstream `JSONAdapter` forced to send the signature's JSON Schema as `response_format`,
  which Ollama honours even though litellm reports the capability unsupported — clears that
  0.356 floor with 0.385 on `falcon3:1b`. Where schema-constrained decoding is available it
  is a stronger baseline than anything measured above, and it buys that reach by inventing:
  on the same run it filled every nullable field whose gold was null.
- **Where we win big we also invent more.** The insurance-claims result buys its recall by
  filling null-gold fields, and that cost is significant, not noise.
- Ollama moves a cell by up to 0.02 on identical code, and across dozens of comparisons
  about one in twenty clears zero by chance. Treat single thin wins accordingly.

Full per-corpus tables, the failure-cause analysis, and the reproduction commands are in
[the benchmark README](benchmarking/dspy_adapters/README.md).

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
- The third-party benchmark corpora (`pii`, `financial-ner`, `insurance-claims`, `patient-notes`) were chosen following [thedataquarry/structured-outputs](https://github.com/thedataquarry/structured-outputs)
- `SemIfLM` follows [SemIf](https://github.com/TheoLeeCJ/SemIf)'s direct option-logit readout for local, open-model typed decisions

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/rohitgarud/llm-schema-lite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rohitgarud/llm-schema-lite/discussions)
- **PyPI**: [llm-schema-lite](https://pypi.org/project/llm-schema-lite/)

---

<div align="center">

**[⬆ Back to Top](#llm-schema-lite)**

</div>
