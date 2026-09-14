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
uv run python -m benchmarking.jsonschemabench.coverage --all-configs
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
| **Ingestion** | **9,532/9,542 (99.9%)** — the *whole* corpus, all 10 configs |
| **Schemas that raise** | **0** — every remaining failure is a timeout, not an exception |
| **Median token reduction** | **47.6%** (median of the ten per-config medians; range 31.4%–58.5%) |
| **Schemas too slow to render** | **10/9,542**, at a 10 s/schema budget (see §4) |
| **Feature coverage** | JSONish **31/39**, YAML **33/39**, TypeScript **31/39** |
| **Validation** | Full Draft 2020-12 via `jsonschema` — every keyword enforced regardless of whether it renders |

Earlier revisions of this report quoted "300/300, 46.2%". That was the flat dump's first 300
records, which are *all* `Github_trivial` and `Github_easy` — the easiest slice of the
corpus. The table below is every schema in every config:

| Config | Schemas | Ingested | Median ↓ | Mean ↓ | Worst | Slow |
|--------|--------:|---------:|---------:|-------:|------:|-----:|
| Github_trivial | 444 | 100% | **58.5%** | 56.7% | −250% | 0 |
| Github_easy | 1938 | 100% | **53.3%** | 53.6% | −34% | 0 |
| Github_medium | 1969 | 100% | **47.2%** | 44.9% | −2884% | 0 |
| Github_hard | 1236 | 99.8% | **40.9%** | 35.0% | −1601% | 3 |
| Github_ultra | 164 | 97.0% | **47.6%** | 38.9% | −354% | 5 |
| Glaiveai2K | 1707 | 100% | **45.6%** | 45.9% | +21% | 0 |
| JsonSchemaStore | 492 | 99.6% | **45.5%** | 18.0% | −6064% | 2 |
| Kubernetes | 1064 | 100% | **31.4%** | 27.4% | −1019% | 0 |
| Snowplow | 403 | 100% | **53.6%** | 53.1% | +24% | 0 |
| WashingtonPost | 125 | 100% | **49.1%** | −51.0% | −1436% | 0 |
| **TOTAL** | **9542** | **99.9%** | — | — | — | **10** |

Three things this table says that the 300-schema sample could not.

**The compaction claim holds across difficulty.** Every config's *median* is a real
reduction, from 31.4% (Kubernetes) to 58.5% (Github_trivial). It erodes with difficulty
(58.5 → 53.3 → 47.2 → 40.9 across the Github ladder) but never inverts.

**Quote the median, never the mean.** `WashingtonPost` has a median of **+49.1%** and a mean
of **−51.0%**: the typical schema compacts by half while a few catastrophic expansions drag
the average below zero. `JsonSchemaStore` shows the same split (45.5% vs 18.0%). A mean over
this corpus is a statement about the worst tail, not about typical behaviour.

**The tail is a real defect, not noise.** Worst cases reach **−6064%** (JsonSchemaStore) and
**−2884%** (Github_medium) — schemas rendering ~29–61× *larger* than their own JSON. Same
root cause as the 10 slow schemas: repeated inline `$ref` expansion (§4).

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
| **propertyNames** (open mapping) | ✅ | ✅ | ❌ |
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
| **Total** | **31/39** | **33/39** | **31/39** |

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

**Corrected:** an earlier revision listed `propertyNames` as a JSONish gap needing a
`classify_container` change. That was wrong on both counts. `classify_container` already
populates `key_schema` from `propertyNames` (rules 6 and 7), `key_token` already renders it,
and JSONish already calls both — so for an **open mapping** JSONish emits the key constraint
today:

```
{"type": "object", "additionalProperties": {"type": "integer"},
 "propertyNames": {"enum": ["alpha", "beta"]}}

jsonish     { <alpha OR beta>: int }
yaml        <alpha OR beta>: int
typescript  interface Schema { [key: string]: number; }   <- drops it
```

TypeScript is the mode that loses the constraint, not JSONish. A `propertyNames` sitting
alongside `properties` still does not surface in any mode, but that is Decision C1 holding
(a schema declaring `properties` is an object, never a mapping), not a defect.

**Still open in JSONish:** `patternProperties` and `not: true`. `patternProperties` is the
substantive one: JSONish renders `{"type":"object","patternProperties":{"^S_":{...}}}` as a
bare `{}`, silently dropping the keys, while YAML and TypeScript surface it through
`BaseFormatter.process_property` — a method JSONish never calls, since it renders object
bodies in its own `_process_schema_recursive_inner` loop. The attach point is therefore that
loop's empty-object base case, not the classifier: Decision C1 excludes `patternProperties`
from `mapping` deliberately, and loosening it would silently reclassify these nodes in YAML
and TypeScript too.

---

## 3. Keyword frequency in the corpus

All **9,542** corpus schemas (share containing the keyword at least once, counted
recursively):

| Keyword | Schemas | Share | Keyword | Schemas | Share |
|---------|--------:|------:|---------|--------:|------:|
| type | 9256 | 97.0% | maximum | 454 | 4.8% |
| properties | 9050 | 94.8% | **patternProperties** | 416 | 4.4% |
| description | 6924 | 72.6% | maxItems | 340 | 3.6% |
| required | 6819 | 71.5% | additionalItems | 196 | 2.1% |
| items | 3900 | 40.9% | allOf | 179 | 1.9% |
| **additionalProperties** | 3483 | 36.5% | minProperties | 117 | 1.2% |
| title | 3403 | 35.7% | uniqueItems | 90 | 0.9% |
| enum | 2826 | 29.6% | const | 81 | 0.8% |
| $ref | 2701 | 28.3% | dependencies | 68 | 0.7% |
| default | 1319 | 13.8% | not | 57 | 0.6% |
| pattern | 1232 | 12.9% | multipleOf | 44 | 0.5% |
| format | 1151 | 12.1% | maxProperties | 43 | 0.5% |
| minimum | 885 | 9.3% | if / then | 27 | 0.3% |
| minLength | 850 | 8.9% | exclusiveMinimum | 21 | 0.2% |
| maxLength | 812 | 8.5% | **propertyNames** | 20 | 0.2% |
| oneOf | 638 | 6.7% | else | 7 | 0.1% |
| minItems | 616 | 6.5% | exclusiveMaximum | 6 | 0.1% |
| anyOf | 480 | 5.0% | contains | 3 | 0.0% |

Only 37 of the 39 tracked keywords appear anywhere in the corpus.

Boolean schemas, separately: `additionalProperties: false` 6,597 sites, `: true` 535,
`additionalItems: false` 82 / `: true` 28, and 28 boolean-valued *properties*. The last of
those used to crash JSONish outright.

This is what makes `patternProperties` (**4.4%**, 416 schemas) the highest-value JSONish gap
— it is not an exotic keyword, and it outweighs the other two open gaps by an order of
magnitude (`propertyNames` 0.2%, `not` 0.6%). An earlier revision quoted 6.7% here from the
3,000-schema sample; 6.7% is `oneOf`'s share, not `patternProperties`'.

---

## 4. Where output still expands instead of compacting

Some schemas render **larger** than their raw JSON Schema. The cause is structural and
worth stating plainly: a `$ref` is expanded **inline at its use site**, so the whole
reachable definition graph is materialised, while raw JSON Schema stores each definition
once and points at it.

Across the full corpus the worst case per config reaches **−6064%** (JsonSchemaStore),
**−2884%** (Github_medium), **−1601%** (Github_hard) and **−1436%** (WashingtonPost) — up to
61× the input. Only `Glaiveai2K` and `Snowplow`, the two configs with *no* cyclic `$ref`
graphs, stay positive throughout. (This run recorded each config's worst case, not a count
of how many schemas expand; that count is not claimed here.)

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

### The cost is CPU, not memory

An earlier revision of this report attributed full-corpus sweep stalls to memory, citing
`o48404` taking resident memory from 240 MB to 801 MB. **That was the wrong diagnosis.**
Instrumenting a full `Github_hard` pass shows RSS pinned flat at **233 MB** from start to
finish, and a bounded render of the worst offender is killed by its 300 s clock
(`timeout -s KILL 300` → exit 137), never by the OOM killer.

These schemas do not exhaust memory, and they do **terminate** — the same `o27039` render
left unbounded exited cleanly after roughly 20–25 minutes of wall time (with CPU contention
from concurrent runs). They are unusably slow, not hung: a 300 s cap kills them, an
unbounded run eventually returns. The practical consequence is the same for a prompt-time
renderer, but the distinction matters for diagnosis — nothing here is deadlocked.

Profiling `o27039` (42 definitions, 109 refs, 3 mutually-recursive `$ref` cycles) names the
hot path exactly:

```
   ncalls  tottime  cumtime  function
   322749   25.874   42.897  jsonish_formatter.py:989(_scan_remove_string_delimiters)
118791893   16.127   16.127  {method 'append' of 'list' objects}
   302699    1.062   59.997  jsonish_formatter.py:207(process_ref)
```

`process_ref` is called 302,699 times even though `_global_expansion_budget` is 150 — the
budget caps *expansions*, not *calls*, so the cheap cache/"return object" paths still run
hundreds of thousands of times. But they are not the cost. The cost is quadratic string
post-processing: a character-at-a-time scanner re-walking a string that inline `$ref`
expansion keeps regrowing, 118 million list appends deep.

**10 of 9,542 schemas (0.1%) exceed a 10 s/schema budget**, all for this reason:

| Config | Slow schemas |
|--------|--------------|
| Github_hard | `o27039`, `o69207`, `o13029` |
| Github_ultra | `o39449`, `o48404`, `o50639`, `o69209`, `o21764` |
| JsonSchemaStore | `meta`, `accelerator` |

Note what is *not* on this list: cyclic schemas as a class. 371 of 9,542 (3.9%) have cyclic
`$ref` graphs — 189 in `Github_hard` alone — and the re-entry guard handles them. Cycles are
survivable; unbounded *width* of distinct-definition expansion is what is not.

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
| contains | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| propertyNames | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ JSONish/YAML only |

The asymmetry worth noting: their gaps are **architectural** — an FSM/grammar cannot express
`if/then/else`, `not`, or unbounded recursion. Ours are **implementation debt**: the
keywords are already parsed and validated, they just do not reach the rendered string. That
is a materially cheaper problem, and it is the honest form of the claim.

---

## 6. Recommendations, in measured priority order

1. **Emit a `$defs` section.** Render each definition once and name every use site. This is
   the only fix for transitive expansion, and the full-corpus run makes it the dominant
   failure mode at the hard end: worst cases of **−6064%** (JsonSchemaStore) and **−2884%**
   (Github_medium), plus the **10 schemas** too slow to render inside 10 s. `o48404` is
   still 45× its input after back-references have done their work. Changes output shape for
   every ref-bearing schema, so it wants its own decision — but the evidence is far stronger
   than the easy-slice sample suggested.
2. **Surface `patternProperties` in JSONish.** It appears in **4.4%** of corpus schemas
   (416 of 9,542) and JSONish drops it silently, rendering the node as a bare `{}`.
   `multipleOf`, `contains` and `propertyNames` are done (see §2). Note this is *not* a
   matter of implementing `add_metadata`: that method is unreachable from JSONish. Nor is
   it a `classify_container` change — Decision C1 excludes `patternProperties` from
   `mapping` deliberately, and loosening it would reclassify these nodes in YAML and
   TypeScript too. The attach point is JSONish's own empty-object base case in
   `_process_schema_recursive_inner`. `not: true` remains open and is rarer (`not` totals
   0.6%).
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

*Measured against `benchmarking/jsonschemabench/` over all **9,542** corpus schemas in all
10 configs — token figures (`cl100k_base`) and keyword frequency alike, at a 10 s/schema
budget. Regenerate with the commands at the top of this file.*
