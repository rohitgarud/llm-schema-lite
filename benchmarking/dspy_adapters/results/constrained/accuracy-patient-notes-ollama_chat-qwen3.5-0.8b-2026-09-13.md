```
arm: accuracy
generated: 2026-09-13T15:49:49.822396+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus patient-notes --cases 30 --adapters json,json-constrained,sola-jsonish-rescue,sola-yaml-rescue --out /tmp/claude-1000/-home-rohitgarud-llm-schema-lite/a9b95a98-8b4c-4773-96d3-50a0b9893834/scratchpad/constrained-qwen
git_head: 50c4c8d
dspy_version: 3.3.1
llm_schema_lite_version: 0.7.0
model: ollama_chat/qwen3.5:0.8b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'thedataquarry/structured-outputs@47924a59d8b5b4eb19426c3e0defe4e33ff4f861', 'cases': 30}
supports_response_schema: False
supports_function_calling: False
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. A correct `None` also counts as a match, so on a sparse corpus field accuracy pays a reply for extracting nothing. **recall (non-null gold)** is the extraction-quality headline: matches over only the fields whose gold value is not `None`, a cell that raised scoring 0 against them, so extracting nothing scores 0. **invented (null gold)** is the other half: of the fields whose gold is `None`, how many the reply filled anyway. Ground truth is the label shipped with the third-party `patient-notes` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format | recall (non-null gold) | invented (null gold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| json | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 2.820 | 2400.5 | json_object | 0.000 (0/341) | 0.000 (0/277) |
| json-constrained | 30 | 0.561 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 3.324 | 2419.5 | json_schema | 0.701 (239/341) | 0.596 (165/277) |
| sola-jsonish-rescue | 30 | 0.162 | 0/30 (0.00) | 8 | 0 | 22 | 0 | 0 | 0 | 3.334 | 1409.0 | json_object | 0.214 (73/341) | 0.152 (42/277) |
| sola-yaml-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 3.232 | 1442.5 | none | 0.000 (0/341) | 0.000 (0/277) |

**All-null floor: 0.356** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

**json**

| field | wrong | missing |
|---|---|---|
| `name.given[]` | 0 | 55 |
| `age` | 0 | 30 |
| `birthDate` | 0 | 30 |
| `email` | 0 | 30 |
| `gender` | 0 | 30 |
| `maritalStatus` | 0 | 30 |
| `name.family` | 0 | 30 |
| `name.prefix` | 0 | 30 |

**json-constrained**

| field | wrong | missing |
|---|---|---|
| `age` | 29 | 0 |
| `name.given[]` | 15 | 14 |
| `name.prefix` | 24 | 0 |
| `primaryLanguage` | 17 | 0 |
| `maritalStatus` | 16 | 0 |
| `immunization` | 0 | 15 |
| `address` | 0 | 13 |
| `gender` | 13 | 0 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `name.given[]` | 3 | 46 |
| `age` | 8 | 22 |
| `name.prefix` | 7 | 22 |
| `allergy` | 0 | 28 |
| `birthDate` | 3 | 22 |
| `gender` | 3 | 22 |
| `maritalStatus` | 3 | 22 |
| `primaryLanguage` | 3 | 22 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `name.given[]` | 0 | 55 |
| `age` | 0 | 30 |
| `birthDate` | 0 | 30 |
| `email` | 0 | 30 |
| `gender` | 0 | 30 |
| `maritalStatus` | 0 | 30 |
| `name.family` | 0 | 30 |
| `name.prefix` | 0 | 30 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
