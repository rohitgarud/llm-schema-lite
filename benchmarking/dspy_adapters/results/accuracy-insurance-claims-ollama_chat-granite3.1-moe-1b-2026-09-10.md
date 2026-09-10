```
arm: accuracy
generated: 2026-09-10T08:13:12.972357+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus insurance-claims --cases 30
git_head: 867819b
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/granite3.1-moe:1b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'Cleanlab/insurance-claims-extraction@e86f533a4d9a855d5ae6c8bbeaed4f511f7b49eb', 'cases': 30}
supports_response_schema: False
supports_function_calling: True
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. Ground truth is the label shipped with the third-party `insurance-claims` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 4.014 | 2959.5 | none |
| json | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 2.225 | 2898.0 | json_object |
| baml | 30 | 0.152 | 0/30 (0.00) | 7 | 0 | 23 | 0 | 0 | 0 | 1.604 | 1774.0 | json_object |
| sola-json-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 5 | 25 | 0 | 0 | 0 | 2.748 | 2868.5 | json_object |
| sola-jsonish-sections | 30 | 0.030 | 0/30 (0.00) | 1 | 0 | 29 | 0 | 0 | 0 | 1.722 | 1905.0 | json_object |
| sola-yaml-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 1 | 29 | 0 | 0 | 0 | 1.382 | 1785.0 | none |
| sola-jsonish-rescue | 30 | 0.030 | 0/30 (0.00) | 1 | 0 | 29 | 0 | 0 | 0 | 1.745 | 1875.5 | json_object |
| sola-yaml-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 1.318 | 1785.0 | none |

**All-null floor: 0.018** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so read every score above as a distance from this.

## Most-missed fields

**chat**

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

**baml**

| field | wrong | missing |
|---|---|---|
| `incident_description.location_type` | 4 | 23 |
| `insured_objects[].location_address` | 5 | 22 |
| `insured_objects[].make_model` | 5 | 22 |
| `header.incident_date` | 3 | 23 |
| `incident_description.incident_type` | 3 | 23 |
| `insured_objects[].object_id` | 3 | 22 |
| `header.channel` | 1 | 23 |
| `incident_description.estimated_damage_amount` | 1 | 23 |

**sola-json-sections**

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

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `header.channel` | 0 | 29 |
| `header.claim_id` | 0 | 29 |
| `header.incident_date` | 0 | 29 |
| `header.report_date` | 0 | 29 |
| `header.reported_by` | 0 | 29 |
| `incident_description.estimated_damage_amount` | 0 | 29 |
| `incident_description.incident_type` | 0 | 29 |
| `incident_description.location_type` | 0 | 29 |

**sola-yaml-sections**

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

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `header.channel` | 0 | 29 |
| `header.claim_id` | 0 | 29 |
| `header.incident_date` | 0 | 29 |
| `header.report_date` | 0 | 29 |
| `header.reported_by` | 0 | 29 |
| `incident_description.estimated_damage_amount` | 0 | 29 |
| `incident_description.incident_type` | 0 | 29 |
| `incident_description.location_type` | 0 | 29 |

**sola-yaml-rescue**

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

Per-case detail is in the companion `.csv`; it is not duplicated here.
