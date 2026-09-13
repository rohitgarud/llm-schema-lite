# docs/ — internal design material

This directory is **not** user documentation. User documentation lives in the
[top-level README](../README.md) and, for the DSPy adapter, in
[src/llm_schema_lite/dspy_integration/README.md](../src/llm_schema_lite/dspy_integration/README.md).

What is here is working material kept in the repo so design decisions stay reviewable:

| File | What it is |
|---|---|
| `structured_output_parsing_spec.md` | A clean-room design spec for the parsing pipeline. Describes intended behaviour, including parts that are not implemented. |
| `BAML_capabilities_reference.md` | A capability teardown of BAML, used to decide what this package should and should not copy. |
| `JSONSchemaBench_feature_report.md` | A gap matrix against JSON Schema features. Rows marked ❌ are deliberately unimplemented, not bugs. |

Treat every statement in these files as a design note, not a promise about the
shipped package. Where they disagree with the top-level README, the README wins.
