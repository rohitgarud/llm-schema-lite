```
arm: accuracy
generated: 2026-09-13T16:53:01.625272+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus insurance-claims --cases 30 --adapters json,json-constrained,sola-jsonish-rescue,sola-yaml-rescue --out /tmp/claude-1000/-home-rohitgarud-llm-schema-lite/a9b95a98-8b4c-4773-96d3-50a0b9893834/scratchpad/constrained-llama
git_head: 50c4c8d
dspy_version: 3.3.1
llm_schema_lite_version: 0.7.0
model: ollama_chat/llama3.2:1b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'Cleanlab/insurance-claims-extraction@e86f533a4d9a855d5ae6c8bbeaed4f511f7b49eb', 'cases': 30}
supports_response_schema: False
supports_function_calling: True
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. A correct `None` also counts as a match, so on a sparse corpus field accuracy pays a reply for extracting nothing. **recall (non-null gold)** is the extraction-quality headline: matches over only the fields whose gold value is not `None`, a cell that raised scoring 0 against them, so extracting nothing scores 0. **invented (null gold)** is the other half: of the fields whose gold is `None`, how many the reply filled anyway. Ground truth is the label shipped with the third-party `insurance-claims` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format | recall (non-null gold) | invented (null gold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| json | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 6.192 | 2854.0 | json_object | 0.000 (0/506) | 0.000 (0/61) |
| json-constrained | 30 | 0.691 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 2.654 | 2459.5 | json_schema | 0.749 (379/506) | 0.787 (48/61) |
| sola-jsonish-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 9 | 21 | 0 | 0 | 0 | 2.316 | 1666.0 | json_object | 0.000 (0/506) | 0.000 (0/61) |
| sola-yaml-rescue | 30 | 0.019 | 0/30 (0.00) | 1 | 25 | 4 | 0 | 0 | 0 | 4.879 | 1856.5 | none | 0.022 (11/506) | 0.000 (0/61) |

**All-null floor: 0.018** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

**json**

| field | wrong | missing |
|---|---|---|
| `header.channel` | 0 | 30 |
| `header.claim_id` | 0 | 30 |
| `header.incident_date` | 0 | 30 |
| `header.report_date` | 0 | 30 |
| `header.reported_by` | 0 | 30 |
| `incident_description.estimated_damage_amount` | 0 | 30 |
| `incident_description.incident_type` | 0 | 30 |
| `incident_description.location_type` | 0 | 30 |

**json-constrained**

| field | wrong | missing |
|---|---|---|
| `insured_objects[].make_model` | 23 | 0 |
| `incident_description.location_type` | 22 | 0 |
| `insured_objects[].object_id` | 22 | 0 |
| `insured_objects[].location_address` | 15 | 0 |
| `incident_description.incident_type` | 13 | 0 |
| `policy_details.effective_date` | 12 | 0 |
| `incident_description.estimated_damage_amount` | 11 | 0 |
| `policy_details.expiration_date` | 10 | 0 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `header.channel` | 0 | 30 |
| `header.claim_id` | 0 | 30 |
| `header.incident_date` | 0 | 30 |
| `header.report_date` | 0 | 30 |
| `header.reported_by` | 0 | 30 |
| `incident_description.estimated_damage_amount` | 0 | 30 |
| `incident_description.incident_type` | 0 | 30 |
| `incident_description.location_type` | 0 | 30 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `header.incident_date` | 1 | 29 |
| `incident_description.location_type` | 1 | 29 |
| `header.channel` | 0 | 29 |
| `header.claim_id` | 0 | 29 |
| `header.report_date` | 0 | 29 |
| `header.reported_by` | 0 | 29 |
| `incident_description.estimated_damage_amount` | 0 | 29 |
| `incident_description.incident_type` | 0 | 29 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
