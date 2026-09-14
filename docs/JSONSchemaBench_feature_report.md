# JSONSchemaBench Feature Coverage Report: llm-schema-lite

Every figure below is **measured against the code in this repository**, not asserted. The
feature matrix comes from running one minimal schema per keyword through each formatter and
checking that the keyword's *semantics* survive into the output; the corpus figures come
from running `simplify_schema` over schemas from
[JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench)
([epfl-dlab/JSONSchemaBench](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench)).

Reproduce with:

```bash
uv run python -m benchmarking.jsonschemabench.fetch_dataset   # ~100 MB, 9,542 schemas, 10 configs
uv run python -m benchmarking.jsonschemabench.coverage --limit 2000
```

**Scope.** llm-schema-lite is a **schema simplification and validation** library: it turns
Pydantic models or raw JSON Schema into token-optimized string formats (JSONish, TypeScript,
YAML) for LLM prompts, and validates candidate output with `jsonschema` (Draft 2020-12). It
does **not** perform constrained decoding. So "coverage" here means *which JSON Schema
features survive into the simplified output* — validation separately enforces the full draft.

---

## 1. Executive Summary

| Aspect | Measured |
|--------|----------|
| **Ingestion** | **300/300 (100%)** of the first 300 corpus schemas simplify without raising |
| **Median token reduction** | **46.2%** across those 300 |
| **Schemas that compact** | **294/300**, median reduction **46.6%** |
| **Schemas that still expand** | **4/300**, worst case **−2012%** (see §4) |
| **Feature coverage** | JSONish **30/39**, YAML **33/39**, TypeScript **31/39** |
| **Validation** | Full Draft 2020-12 via `jsonschema` — every keyword enforced regardless of whether it renders |

**Read that first row carefully.** The flat dump's first 300 records are all `Github_trivial`
and `Github_easy`, so those figures describe the *easiest* slice of the corpus and must not
be quoted as a whole-corpus result. Sampled per config instead (45 schemas each, `train`):

| Config | Median reduction | Compact | Expand | Worst |
|--------|------------------|---------|--------|-------|
| Github_hard | **42.8%** | 44/44 | 0 | +2% |
| Github_ultra | **47.9%** | 40/44 | 2 | **−979%** |

The medians survive the difficulty jump — the compaction claim is real on hard schemas. What
does not survive is the tail: `Github_ultra` contains schemas the easy slice never reaches,
including one that renders 45× larger than its input (§4). Any headline number should carry
both halves.

The headline gap is **not** breadth, it is that JSONish — the default mode — renders five
fewer keywords than YAML, despite all three sharing `base.py`. Every one of those five is
already implemented in the base class and simply never reaches JSONish's output.

---

## 2. Feature-by-Feature Status (measured)

✅ = the keyword's semantics appear in that formatter's output. ❌ = they do not (validation
still enforces it).

| Feature | JSONish | YAML | TypeScript |
|---------|---------|------|------------|
| required | ✅ | ✅ | ✅ |
| items | ✅ | ✅ | ✅ |
| additionalProperties: false | ✅ | ✅ | ✅ |
| additionalProperties: schema | ✅ | ✅ | ✅ |
| enum | ✅ | ✅ | ✅ |
| $ref / $defs | ✅ | ✅ | ✅ |
| recursive $ref | ✅ | ✅ | ✅ |
| pattern | ✅ | ✅ | ✅ |
| format | ✅ | ✅ | ✅ |
| oneOf | ✅ | ✅ | ✅ |
| anyOf | ✅ | ✅ | ✅ |
| allOf | ✅ | ✅ | ✅ |
| not | ✅ | ✅ | ✅ |
| **not: true (boolean schema)** | ❌ | ✅ | ✅ |
| const | ✅ | ✅ | ✅ |
| minLength / maxLength | ✅ | ✅ | ✅ |
| minimum / maximum | ✅ | ✅ | ✅ |
| exclusiveMinimum / exclusiveMaximum | ✅ | ✅ | ✅ |
| multipleOf | ✅ | ✅ | ✅ |
| minItems / maxItems | ✅ | ✅ | ✅ |
| uniqueItems | ✅ | ✅ | ✅ |
| prefixItems (tuple) | ✅ | ✅ | ✅ |
| contains | ✅ | ✅ | ✅ |
| boolean schema `true` | ✅ | ✅ | ✅ |
| boolean schema `false` | ✅ | ✅ | ✅ |
| items: true (boolean schema) | ✅ | ✅ | ✅ |
| **patternProperties** | ❌ | ✅ | ✅ |
| **propertyNames** | ❌ | ✅ | ❌ |
| minProperties | ❌ | ❌ | ❌ |
| maxProperties | ❌ | ❌ | ❌ |
| dependencies | ✅ | ✅ | ❌ |
| dependentRequired | ❌ | ❌ | ❌ |
| if / then / else | ❌ | ❌ | ❌ |
| unevaluatedProperties | ❌ | ❌ | ❌ |
| title | ✅ | ✅ | ✅ |
| description | ✅ | ✅ | ✅ |
| default | ✅ | ✅ | ✅ |
| examples (opt-in) | ❌ | ❌ | ❌ |
| nullable (`anyOf` + null) | ✅ | ✅ | ✅ |
| **Total** | **30/39** | **33/39** | **31/39** |

`examples` is excluded by default for token savings and renders when enabled
(`FormatterConfig(metadata_inclusion={"examples": True})`); it is listed ❌ because the
default output omits it.

### Why JSONish trails

JSONish surfaces only keywords that have a **dedicated inline token** (pattern, format,
length/numeric ranges, item counts, uniqueItems, enum, const, default, description, title).
Everything else in `BaseFormatter.METADATA_MAP` is live for YAML and TypeScript and absent
from JSONish.

An earlier revision of this report blamed `JSONishFormatter.add_metadata` being a
pass-through, and called the fix "one contained change". **That was wrong.** `add_metadata`
has exactly one caller — the last line of `BaseFormatter.process_property` — and JSONish
never calls `process_property`: it renders object bodies in its own
`_process_schema_recursive_inner` loop. Instrumenting a full JSONish render confirms it,
with both counters at zero:

```
rendered: // Fields marked with * are required { a*: int, b: string [] }
call counts: {'add_metadata': 0, 'process_property': 0}
```

(`multipleOf: 5` and `contains` are both absent from that output.) JSONish's
`add_metadata` is a dead override on a path JSONish does not use; implementing it would
change nothing observable.

The real cause is structural: that loop emits each property's token directly across ~10
branches, so a keyword with no inline token has nowhere to appear. Closing a gap means
adding a fragment on the path each mode actually uses — which is a larger change than this
report previously claimed, and larger than it looks: `multipleOf` alone needed the token
joined at **three** separate call sites (`jsonish_formatter.py`, `typescript_formatter.py`,
and YAML's `_format_number_range_jsonish`), because none of the three modes reaches
`BaseFormatter.process_type_value` for a scalar number.

**Closed since that finding:** `multipleOf` and `contains` now render in all three modes,
with one shared spelling — `int (multiple of 5)`, `list[string] (contains "z")` — from a
single owner each (`multiple_of_token`, `array_constraint_tokens`). `contains` previously
had three live emitters, printing two or three times on one YAML line, and leaked a raw
Python dict repr (`contains: {'const': 'z'}`) whenever the target carried a `const`.

**Still open in JSONish:** `patternProperties`, `propertyNames`, and `not: true`. All three
are key-position or node-position rather than value-position, so they need
`classify_container` to classify them (it currently returns `kind='object'`,
`key_schema=None` for the first two) rather than a new token.

---

## 3. Keyword frequency in the corpus

From 3,000 corpus schemas (share of schemas containing the keyword at least once):

| Keyword | Schemas | Keyword | Schemas |
|---------|---------|---------|---------|
| type | 97.0% | oneOf | 7.7% |
| properties | 96.0% | **patternProperties** | 6.7% |
| description | 70.7% | minItems | 6.0% |
| required | 70.3% | maximum | 5.7% |
| items | 41.7% | maxItems / anyOf | 4.0% |
| enum | 38.0% | allOf | 2.0% |
| title | 35.0% | additionalItems | 1.7% |
| **additionalProperties** | 32.3% | uniqueItems / const | 1.0% |
| $ref | 27.7% | dependencies | 0.7% |
| pattern | 16.3% | exclusiveMaximum | 0.3% |
| default | 13.7% | propertyNames / not | 0.3% |
| format | 12.0% | minProperties / maxProperties | 0.3% |
| minimum / minLength / maxLength | ~10% | | |

Boolean schemas, separately: `additionalProperties: false` 6,597 sites, `: true` 535,
`additionalItems: false` 82 / `: true` 28, and 28 boolean-valued *properties*. The last of
those used to crash JSONish outright.

This is what makes `patternProperties` (6.7%) the highest-value JSONish gap — it is not an
exotic keyword.

---

## 4. Where output still expands instead of compacting

4 of 300 schemas render **larger** than their raw JSON Schema. The cause is structural and
worth stating plainly: a `$ref` is expanded **inline at its use site**, so the whole
reachable definition graph is materialised, while raw JSON Schema stores each definition
once and points at it.

Repeated references are now handled. A definition rendered once is replaced at later
occurrences by a named back-reference (`object // defined above: Address`) whenever its body
exceeds `BaseFormatter.BACKREFERENCE_MIN_CHARS` (200). Measured effect:

| Schema | refs | distinct defs | before | after |
|--------|------|---------------|--------|-------|
| idx=88 | 244 | 10 | 55,828 tok (−558%) | **4,215 tok (+50.3%)** |
| idx=43 | — | 20 | 24,030 tok (−540%) | **3,074 tok (+18.2%)** |
| idx=106 | 50 | 14 | 329,124 tok | 127,247 tok |
| idx=267 | 247 | 73 | 807,375 tok | 152,115 tok |
| idx=270 | 209 | 70 | 200,335 tok | 86,074 tok |

The threshold exists because duplicating a *small* definition costs almost nothing and reads
better in place — the sibling-inline acceptance criterion (two sibling fields sharing a
`$ref` both render inline) depends on it. The two populations separate cleanly: the largest
body those tests rely on is 158 chars; the smallest definition in the 244-reference schema
is 409.

**The remaining three are not fixable by repeat-suppression.** They carry 70+ *distinct*
definitions nested 7 deep, each referenced only ~3 times, so the cost is transitive
expansion rather than duplication. Closing that requires emitting a leading `$defs` section
and naming every use site — a change to output shape for all ref-bearing schemas, not yet
made.

### The worst case in the corpus

`Github_ultra`'s `o48404` is the same mechanism at full scale, and it is the reason this is
not a four-schema tail:

| | |
|--|--|
| raw | 208,433 chars |
| structure | 174 defs, 534 refs, **152 distinct**, 128 inline object nodes, nesting depth 4 |
| rendered, before back-references | **49,236,120 chars** |
| rendered, after back-references | **9,327,895 chars** (5.3× better, still **45× the input**) |

Back-references did real work here — 49 MB down to 9 MB — and still left 45×, because 152
*distinct* definitions are each materialised transitively. Repeat-suppression cannot reach
this by construction; only rendering each definition once, in a `$defs` section, can.

This case also has a practical cost beyond token count: rendering it took resident memory
from 240 MB to 801 MB, which is enough to make a full-corpus sweep look like it has hung.

---

## 5. Comparison to JSONSchemaBench engines

JSONSchemaBench reports feature coverage for constrained-decoding engines. We are not one,
so this compares *what reaches the model* — their grammar vs our prompt text.

| Feature | LLGuidance | llama.cpp | Outlines | XGrammar | OpenAI | Gemini | llm-schema-lite |
|---------|-----------|-----------|----------|----------|--------|--------|-----------------|
| required | ✔ | ✔ | ✔ | ✔ | ❌ | ✔ | ✅ |
| items | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| additionalProperties | ✔ | ✔ | ✔ | ✔ | ❌ | ❌ | ✅ |
| enum | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| $ref | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| pattern | ✔ | ✔ | ✔ | ✔ | ❌ | ❌ | ✅ |
| format | ✔ | ✔ | ✔ | ❌ | ❌ | ✔ | ✅ |
| oneOf | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| anyOf / allOf | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| minLength/maxLength | ✔ | ✔ | ✔ | ❌ | ❌ | ❌ | ✅ |
| min/max (integer) | ✔ | ✔ | ❌ | ❌ | ❌ | ❌ | ✅ |
| minItems/maxItems | ✔ | ✔ | ✔ | ❌ | ❌ | ❌ | ✅ |
| const | ✔ | ✔ | ✔ | ✔ | ✔ | ❌ | ✅ |
| patternProperties | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ YAML/TS only |
| not | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| if / then / else | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| contains | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ YAML/TS only |

The asymmetry worth noting: their gaps are **architectural** — an FSM/grammar cannot express
`if/then/else`, `not`, or unbounded recursion. Ours are **implementation debt**: the
keywords are already parsed and validated, they just do not reach the rendered string. That
is a materially cheaper problem, and it is the honest form of the claim.

---

## 6. Recommendations, in measured priority order

1. **Emit a `$defs` section.** Render each definition once and name every use site. This is
   the only fix for transitive expansion, and `o48404` moves it from "a four-schema tail" to
   the dominant failure mode at the hard end of the corpus: 45× the input after
   back-references have already done their work, and 801 MB resident to produce. Changes
   output shape for every ref-bearing schema, so it wants its own decision — but the
   evidence for it is now much stronger than the easy-slice sample suggested.
2. **Surface the three remaining keywords in JSONish.** `patternProperties`,
   `propertyNames` and `not: true` — `patternProperties` alone appears in 6.7% of real
   schemas. `multipleOf` and `contains` are done. Note this is *not* a matter of
   implementing `add_metadata`: that method is unreachable from JSONish (see §2). These
   three are key/node-position, so they need `classify_container` to classify them, which
   changes a shared classifier used by all three modes.
3. **`minProperties` / `maxProperties` / `dependentRequired`** — absent from all three
   formatters; `dependentRequired` is not in `METADATA_MAP` at all.
4. **`if/then/else`** — `_format_conditional` exists but never fires from a root-level `if`.

---

## 7. References

- [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench) — benchmark and feature checklist
- [epfl-dlab/JSONSchemaBench](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench) — the dataset (10 configs, 9,542 schemas)
- [JSON Schema Test Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite)
- [Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-core.html) — enforced by `jsonschema` at validation time
- Geng et al., 2025, [Generating Structured Outputs from Language Models](https://arxiv.org/abs/2501.10868)

---

*Measured against `benchmarking/jsonschemabench/` on the first 300 corpus schemas (token
figures, `cl100k_base`) and 3,000 schemas (keyword frequency). Regenerate with the commands
at the top of this file.*
