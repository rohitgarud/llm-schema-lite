```
arm: accuracy
generated: 2026-09-10T07:53:09.545610+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus insurance-claims --cases 30
git_head: 867819b
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/qwen3.5:0.8b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'Cleanlab/insurance-claims-extraction@e86f533a4d9a855d5ae6c8bbeaed4f511f7b49eb', 'cases': 30}
supports_response_schema: False
supports_function_calling: False
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. Ground truth is the label shipped with the third-party `insurance-claims` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 3.459 | 2578.0 | none |
| json | 30 | 0.111 | 0/30 (0.00) | 4 | 0 | 26 | 0 | 0 | 0 | 2.476 | 2568.5 | json_object |
| baml | 30 | 0.026 | 0/30 (0.00) | 1 | 0 | 29 | 0 | 0 | 0 | 2.473 | 1636.5 | json_object |
| sola-json-sections | 30 | 0.686 | 0/30 (0.00) | 29 | 0 | 1 | 0 | 0 | 0 | 2.740 | 2402.5 | json_object |
| sola-jsonish-sections | 30 | 0.219 | 0/30 (0.00) | 8 | 0 | 22 | 0 | 0 | 0 | 2.612 | 1681.0 | json_object |
| sola-yaml-sections | 30 | 0.628 | 0/30 (0.00) | 27 | 0 | 3 | 0 | 0 | 0 | 2.417 | 1684.0 | none |
| sola-jsonish-rescue | 30 | 0.233 | 0/30 (0.00) | 9 | 0 | 21 | 0 | 0 | 0 | 2.434 | 1681.0 | json_object |
| sola-yaml-rescue | 30 | 0.670 | 0/30 (0.00) | 28 | 0 | 2 | 0 | 0 | 0 | 2.396 | 1684.0 | none |

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
| `incident_description.incident_type` | 4 | 26 |
| `incident_description.location_type` | 1 | 26 |
| `incident_description.police_report_number` | 1 | 26 |
| `header.channel` | 0 | 26 |
| `header.claim_id` | 0 | 26 |
| `header.incident_date` | 0 | 26 |
| `header.report_date` | 0 | 26 |
| `header.reported_by` | 0 | 26 |

**baml**

| field | wrong | missing |
|---|---|---|
| `incident_description.incident_type` | 1 | 29 |
| `incident_description.location_type` | 1 | 29 |
| `header.channel` | 0 | 29 |
| `header.claim_id` | 0 | 29 |
| `header.incident_date` | 0 | 29 |
| `header.report_date` | 0 | 29 |
| `header.reported_by` | 0 | 29 |
| `incident_description.estimated_damage_amount` | 0 | 29 |

**sola-json-sections**

| field | wrong | missing |
|---|---|---|
| `insured_objects[].make_model` | 21 | 1 |
| `incident_description.incident_type` | 19 | 2 |
| `incident_description.location_type` | 19 | 2 |
| `insured_objects[].location_address` | 14 | 1 |
| `insured_objects[].object_id` | 12 | 1 |
| `insured_objects[].year` | 12 | 1 |
| `policy_details.effective_date` | 9 | 1 |
| `policy_details.expiration_date` | 9 | 1 |

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `incident_description.incident_type` | 5 | 22 |
| `incident_description.location_type` | 5 | 22 |
| `incident_description.estimated_damage_amount` | 3 | 22 |
| `header.incident_date` | 2 | 22 |
| `insured_objects[].make_model` | 6 | 18 |
| `header.channel` | 1 | 22 |
| `insured_objects[].year` | 5 | 18 |
| `header.claim_id` | 0 | 22 |

**sola-yaml-sections**

| field | wrong | missing |
|---|---|---|
| `incident_description.location_type` | 21 | 3 |
| `incident_description.incident_type` | 20 | 3 |
| `insured_objects[].make_model` | 19 | 4 |
| `insured_objects[].year` | 16 | 4 |
| `insured_objects[].object_id` | 13 | 4 |
| `insured_objects[].location_address` | 9 | 4 |
| `insured_objects[].estimated_value` | 8 | 4 |
| `policy_details.effective_date` | 8 | 3 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `incident_description.location_type` | 6 | 21 |
| `incident_description.incident_type` | 5 | 21 |
| `incident_description.estimated_damage_amount` | 3 | 21 |
| `insured_objects[].make_model` | 6 | 18 |
| `header.incident_date` | 2 | 21 |
| `insured_objects[].year` | 5 | 18 |
| `header.channel` | 1 | 21 |
| `header.claim_id` | 0 | 21 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `incident_description.location_type` | 21 | 2 |
| `insured_objects[].make_model` | 21 | 2 |
| `incident_description.incident_type` | 20 | 2 |
| `insured_objects[].year` | 16 | 2 |
| `insured_objects[].object_id` | 13 | 2 |
| `insured_objects[].location_address` | 9 | 2 |
| `insured_objects[].estimated_value` | 8 | 2 |
| `policy_details.effective_date` | 8 | 2 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
