# JSONSchemaBench Feature Coverage Report: llm-schema-lite

This report compares **llm-schema-lite**’s support for JSON Schema features against the [JSONSchemaBench](https://github.com/epfl-dlab/jsonschemabench) benchmark and the engines/APIs it evaluates (LLGuidance, llama.cpp, Outlines, XGrammar, OpenAI, Gemini).

**Scope:** llm-schema-lite is a **schema simplification and validation** library: it turns Pydantic models or JSON Schema into token-optimized string formats (JSONish, TypeScript, YAML) for LLM prompts and validates candidate outputs with the **jsonschema** library (Draft 2020-12). It does **not** perform constrained decoding; the comparison below is about **which JSON Schema features are reflected in the simplified schema output** and **which are fully supported at validation time**.

---

## 1. Executive Summary

| Aspect | Summary |
|--------|--------|
| **Role** | Schema simplification (reduce tokens) + validation; not a constrained decoding engine. |
| **Validation** | Full Draft 2020-12 via `jsonschema` — all keywords supported for *validation*. |
| **Simplified output** | Many core keywords are represented in the simplified format; several advanced keywords are missing or only partially surfaced. |
| **vs. JSONSchemaBench engines** | We support a broad set of features in our *output format*; gaps are mainly in `prefixItems`, first-class `const`, and consistent `additionalProperties` / `patternProperties` in JSONish. |

**Recommendation:** Prioritize surfacing **additionalProperties**, **prefixItems**, and **const** in all formatters (especially JSONish) to align with high-frequency JSONSchemaBench features and with what constrained decoders expose.

---

## 2. Feature-by-Feature Status

Features are ordered by approximate prevalence in JSONSchemaBench (from the benchmark’s feature frequency chart and checklist).
**Supported** = represented in at least one formatter’s simplified output and/or validation.
**Partial** = only in some formatters or only in metadata/comments.
**Missing** = not reflected in simplified output (validation may still apply).

| Feature | Schemas (approx.) | llm-schema-lite | Notes |
|--------|--------------------|------------------|--------|
| **required** | 7473 | ✅ Supported | Asterisk notation and required-fields comment in all formatters. |
| **items** | 4703 | ✅ Supported | Array item type in all formatters (single schema). |
| **additionalProperties** | 4290 | ⚠️ Partial | Base has `process_additional_properties`; TypeScript/YAML use it only in schema-level (no properties) branch. **JSONish never surfaces it.** |
| **enum** | 3898 | ✅ Supported | Full support in all formatters (OPTIONS / union literals / Literal). |
| **$ref** | 2786 | ✅ Supported | Resolved via `$defs`/definitions in all formatters; depth/expansion limits in base. |
| **pattern** | 1829 | ✅ Supported | Inline (PATTERN) or metadata in all formatters. |
| **format** | 1570 | ✅ Supported | Inline (FORMAT) or metadata; validation uses FormatChecker. |
| **oneOf** | 1437 | ✅ Supported | “ONE OF: …” / “oneOf: …” / unions in all formatters. |
| **@minmaxLength** (minLength/maxLength) | 1388 | ✅ Supported | String length range in process_types / metadata. |
| **@siblingKeys** | 1230 | ❌ Missing | Benchmark-specific / dependency-like; not implemented. |
| **@minmaxInteger** (minimum/maximum for integer) | 1217 | ✅ Supported | Number range in all formatters. |
| **@minmaxItems** (minItems/maxItems) | 1132 | ✅ Supported | Array length range in all formatters. |
| **additionalProperties:object** | 1086 | ⚠️ Partial | Same as additionalProperties — only where formatter calls `process_additional_properties`. |
| **anyOf** | 924 | ✅ Supported | Union handling in all formatters. |
| **allOf** | 756 | ✅ Supported | Intersection / “AND” in all formatters. |
| **patternProperties** | 783 | ⚠️ Partial | Base `process_property` and `process_pattern_properties`; TypeScript/YAML use in schema-level branch. **JSONish has no branch for patternProperties on objects.** |
| **not** | 670 | ⚠️ Partial | Base has `process_not`; used in `process_property`. Not in JSONish’s recursive flow. |
| **dependencies** | 156 | ✅ Supported | List form in JSONish `_get_fields_dependencies`; base process_dependencies; TypeScript/YAML schema-level. |
| **additionalItems** | 221 | ⚠️ Partial | In base METADATA_MAP and format_metadata_parts; no tuple array (prefixItems) support. |
| **@minmaxProperties** | 235 | ⚠️ Partial | In base METADATA_MAP; only as metadata where formatter uses it. |
| **const** | 183 | ⚠️ Partial | Handled inside anyOf/oneOf/ref in base, TypeScript, YAML. **JSONish has no dedicated handling for top-level or property-level `const`.** |
| **@minmaxNumber** | 123 | ✅ Supported | Same as minimum/maximum for number. |
| **uniqueItems** | 117 | ✅ Supported | “UNIQUE” / “unique” in array in all formatters. |
| **propertyNames** | 56 | ⚠️ Partial | Base `process_property_names`; TypeScript/YAML in schema-level branch only. |
| **if / if-then-else** | 63 | ⚠️ Partial | Base `process_conditional` and METADATA_MAP; TypeScript/YAML schema-level. Not in JSONish. |
| **contains** | 23 | ⚠️ Partial | Base `process_contains` and `_format_contains`; TypeScript/YAML in array metadata. JSONish does not surface contains. |
| **unevaluatedProperties** | 2 | ⚠️ Partial | Base `process_unevaluated_properties`; TypeScript/YAML schema-level only. |
| **prefixItems** | — | ❌ Missing | Tuple-style arrays not implemented; only single-schema `items` supported. |
| **multipleOf** | 134 | ⚠️ Partial | In base METADATA_MAP; only as metadata where formatter includes it. |
| **exclusiveMinimum / exclusiveMaximum** | — | ⚠️ Partial | In base METADATA_MAP; tests (e.g. `test_jsonish_formatter_with_exclusive_min_max`) confirm metadata. |
| **default** | — | ✅ Supported | Description/default in comments or metadata in all formatters. |
| **defs / $defs** | — | ✅ Supported | Used for $ref resolution; not rendered as a separate section. |
| **dependentRequired / dependentSchemas** | — | ❌ Missing | Not implemented (dependencies list form is). |
| **dynamicRef / $dynamicRef** | — | ❌ Missing | Not implemented. |
| **anchor / $anchor** | — | ❌ Missing | Not implemented. |
| **minContains / maxContains** | — | ❌ Missing | Not implemented. |
| **@recursiveSchemas** | 0 | ⚠️ Partial | Ref cycles handled by depth/expansion limits; no explicit recursive type output. |

---

## 3. Comparison to JSONSchemaBench Engines

JSONSchemaBench reports **coverage** (support for JSON Schema features) for constrained decoding engines and APIs. The following table summarizes their stated support (✔/❌) vs. llm-schema-lite’s **simplified-output** support (we do not do constrained decoding).

| Feature | LLGuidance | llama.cpp | Outlines | XGrammar | OpenAI | Gemini | llm-schema-lite (output) |
|---------|------------|-----------|----------|----------|--------|--------|---------------------------|
| required | ✔ | ✔ | ✔ | ✔ | ❌ | ✔ | ✅ |
| items | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| additionalProperties | ✔ | ✔ | ✔ | ✔ | ❌ | ❌ | ⚠️ Partial |
| enum | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| $ref | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| pattern | ✔ | ✔ | ✔ | ✔ | ❌ | ❌ | ✅ |
| format | ✔ | ✔ | ✔ | ❌ | ❌ | ✔ | ✅ |
| oneOf | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| anyOf | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| allOf | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| @minmaxLength | ✔ | ✔ | ✔ | ❌ | ❌ | ❌ | ✅ |
| @minmaxInteger | ✔ | ✔ | ❌ | ❌ | ❌ | ❌ | ✅ |
| @minmaxItems | ✔ | ✔ | ✔ | ❌ | ❌ | ❌ | ✅ |
| const | ✔ | ✔ | ✔ | ✔ | ✔ | ❌ | ⚠️ Partial |
| patternProperties | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ Partial |
| not | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ Partial |
| dependencies | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (list form) |
| uniqueItems | — | — | — | — | — | — | ✅ |
| if / conditional | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ Partial |
| contains | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ Partial |
| unevaluatedProperties | — | — | ✔ (best) | ✔ | — | — | ⚠️ Partial |

**Where llm-schema-lite is stronger (in simplified output):**

- **oneOf** — we represent it; all six benchmark engines mark it unsupported.
- **pattern**, **format** — we surface them; OpenAI/Gemini do not support in the benchmark.
- **@minmaxLength**, **@minmaxInteger**, **@minmaxItems** — we represent them; XGrammar/OpenAI/Gemini do not in the benchmark.
- **dependencies** (list form) — we surface it; benchmark engines do not.

**Where we are weaker or aligned:**

- **additionalProperties** — engines that support it surface it; we only do in TypeScript/YAML schema-level branch; JSONish does not.
- **const** — several engines support it; we only handle it inside unions/refs and not as first-class in JSONish.
- **patternProperties** — no engine in the benchmark supports it; we have partial (base/TS/YAML) but not in JSONish.

---

## 4. Test Suite Coverage (JSON Schema Test Suite)

JSONSchemaBench also references the [JSON Schema Test Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite) and reports “fraction of tests passed” per feature for Outlines, Llamacpp, XGrammar, and Guidance. Those numbers measure **conformance of generated output** to the schema.

llm-schema-lite does not generate output; it **validates** with `jsonschema` (Draft 2020-12). So:

- **Validation:** Any keyword supported by Draft 2020-12 is enforced (e.g. type, required, enum, minimum/maximum, minLength/maxLength, pattern, format, items, additionalProperties, oneOf/anyOf/allOf/not, $ref, etc.).
- **Simplified output:** Our “coverage” is what appears in the simplified string (see Section 2). We do not run the JSON Schema Test Suite for generation; adding such a run would require a separate “round-trip” or “generation + validation” benchmark.

---

## 5. Implementation Notes (Where Features Live)

| Area | File(s) | Features |
|------|--------|----------|
| Core entrypoints | `core.py` | `simplify_schema`, `validate` (jsonschema Draft202012Validator). |
| Base processing | `formatters/base.py` | $ref, defs, required, type, enum, anyOf, oneOf, allOf, not, const (in ref/union), additionalProperties, patternProperties, dependencies, if-then-else, contains, propertyNames, unevaluatedProperties, exclusiveMin/Max, multipleOf, min/maxProperties, additionalItems; METADATA_MAP. |
| JSONish | `formatters/jsonish_formatter.py` | required, properties, type, items, enum, $ref, anyOf, oneOf, allOf, min/max length, min/max value, min/max items, uniqueItems, pattern, format, default, dependencies (list). No: additionalProperties, patternProperties, const (first-class), not, if-then-else, contains, propertyNames, unevaluatedProperties. |
| TypeScript | `formatters/typescript_formatter.py` | Uses base; schema-level additionalProperties, patternProperties, dependencies, if-then-else, propertyNames, unevaluatedProperties when no properties. |
| YAML | `formatters/yaml_formatter.py` | Same as TypeScript for schema-level features. |
| Validation | `core.py` | Full Draft 2020-12 via jsonschema; error formatting for required, type, minimum, maximum, minLength, maxLength, minItems, maxItems, pattern, enum. |

---

## 6. Recommendations

1. **additionalProperties (high impact)**
   - Call `process_additional_properties` (or equivalent) in **JSONish** when processing object schemas (e.g. in `_process_schema_recursive` for `type: object` or when leaving an object block) and ensure TypeScript/YAML attach it to object types with properties, not only in the no-properties branch.

2. **const (medium impact)**
   - In **JSONish**, add an explicit branch for `"const" in value` in `_process_schema_recursive` (and for top-level const) and render as a single allowed value (e.g. literal or “CONST: value”). Align with base/TypeScript/YAML where const is already handled inside unions/refs.

3. **prefixItems (medium impact)**
   - Add support for **prefixItems** (and optionally **items** as additionalItems) in all formatters so tuple-style arrays are represented (e.g. “tuple of (string, number, …)” or equivalent in each format).

4. **patternProperties in JSONish**
   - In `_process_schema_recursive`, when handling an object schema, if `patternProperties` is present, either recurse with a synthetic “pattern” property or append a comment/line from a new helper that formats patternProperties (reuse base’s logic).

5. **Optional: schema-level features in JSONish**
   - Consider surfacing **if-then-else**, **propertyNames**, **contains** (for arrays), and **unevaluatedProperties** in JSONish as comments or suffixes when present, for parity with TypeScript/YAML.

6. **Benchmarking**
   - Measure how many JSONSchemaBench schemas are **ingestible** (simplify without error) and how many **use only supported keywords** in the simplified output: `benchmarking/fetch_dataset.py` downloads the dataset and `benchmarking/format_jsonschembench_schema.py` formats one schema from it, so a small script that runs `simplify_schema` over a sample and reports errors or fallbacks is all that is missing.

---

## 7. References

- [JSONSchemaBench](https://github.com/epfl-dlab/jsonschemabench) — benchmark and feature checklist.
- [JSON Schema Test Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite) — official test suite.
- [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-core.html) — validation in llm-schema-lite via `jsonschema`.
- ArXiv: [Generating Structured Outputs from Language Models: Benchmark and Studies](https://arxiv.org/abs/2501.10868) (Geng et al., 2025).

---

*Report generated from codebase analysis of `src/llm_schema_lite/` and `tests/`.*
