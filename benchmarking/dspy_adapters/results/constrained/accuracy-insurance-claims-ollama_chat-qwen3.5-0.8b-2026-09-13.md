```
arm: accuracy
generated: 2026-09-13T15:56:18.129120+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus insurance-claims --cases 30 --adapters json,json-constrained,sola-jsonish-rescue,sola-yaml-rescue --out /tmp/claude-1000/-home-rohitgarud-llm-schema-lite/a9b95a98-8b4c-4773-96d3-50a0b9893834/scratchpad/constrained-qwen
git_head: 50c4c8d
dspy_version: 3.3.1
llm_schema_lite_version: 0.7.0
model: ollama_chat/qwen3.5:0.8b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'Cleanlab/insurance-claims-extraction@e86f533a4d9a855d5ae6c8bbeaed4f511f7b49eb', 'cases': 30}
supports_response_schema: False
supports_function_calling: False
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. A correct `None` also counts as a match, so on a sparse corpus field accuracy pays a reply for extracting nothing. **recall (non-null gold)** is the extraction-quality headline: matches over only the fields whose gold value is not `None`, a cell that raised scoring 0 against them, so extracting nothing scores 0. **invented (null gold)** is the other half: of the fields whose gold is `None`, how many the reply filled anyway. Ground truth is the label shipped with the third-party `insurance-claims` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format | recall (non-null gold) | invented (null gold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| json | 30 | 0.111 | 0/30 (0.00) | 4 | 0 | 26 | 0 | 0 | 0 | 2.880 | 2574.0 | json_object | 0.119 (60/506) | 0.049 (3/61) |
| json-constrained | 30 | 0.688 | 0/30 (0.00) | 29 | 0 | 1 | 0 | 0 | 0 | 3.423 | 2546.0 | mixed | 0.708 (358/506) | 0.443 (27/61) |
| sola-jsonish-rescue | 30 | 0.720 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 2.846 | 1681.0 | json_object | 0.731 (370/506) | 0.344 (21/61) |
| sola-yaml-rescue | 30 | 0.721 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 2.812 | 1684.0 | none | 0.749 (379/506) | 0.508 (31/61) |

**All-null floor: 0.018** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

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

**json-constrained**

| field | wrong | missing |
|---|---|---|
| `incident_description.location_type` | 21 | 1 |
| `incident_description.incident_type` | 20 | 1 |
| `insured_objects[].make_model` | 19 | 2 |
| `insured_objects[].object_id` | 12 | 2 |
| `insured_objects[].year` | 10 | 2 |
| `insured_objects[].estimated_value` | 9 | 2 |
| `policy_details.effective_date` | 10 | 1 |
| `policy_details.expiration_date` | 10 | 1 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `incident_description.location_type` | 22 | 0 |
| `incident_description.incident_type` | 21 | 0 |
| `insured_objects[].make_model` | 21 | 0 |
| `insured_objects[].year` | 17 | 0 |
| `insured_objects[].location_address` | 12 | 0 |
| `policy_details.effective_date` | 10 | 0 |
| `insured_objects[].estimated_value` | 9 | 0 |
| `policy_details.expiration_date` | 9 | 0 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `incident_description.location_type` | 23 | 0 |
| `insured_objects[].make_model` | 22 | 0 |
| `incident_description.incident_type` | 20 | 0 |
| `insured_objects[].year` | 16 | 0 |
| `insured_objects[].object_id` | 14 | 0 |
| `insured_objects[].location_address` | 11 | 0 |
| `policy_details.effective_date` | 10 | 0 |
| `policy_details.expiration_date` | 9 | 0 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
