```
arm: accuracy
generated: 2026-09-12T08:34:55.721351+00:00
command: python -m benchmarking.dspy_adapters --accuracy --cases 30
git_head: b1016fb
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/llama3.2:1b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'cases': 30, 'cases_seed': 0}
supports_response_schema: False
supports_function_calling: True
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. A correct `None` also counts as a match, so on a sparse corpus field accuracy pays a reply for extracting nothing. **recall (non-null gold)** is the extraction-quality headline: matches over only the fields whose gold value is not `None`, a cell that raised scoring 0 against them, so extracting nothing scores 0. **invented (null gold)** is the other half: of the fields whose gold is `None`, how many the reply filled anyway. Ground truth is the generated record the prose was rendered from, so the corpus measures schema-following under paraphrase — not real-world extraction.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format | recall (non-null gold) | invented (null gold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.022 | 0/30 (0.00) | 1 | 29 | 0 | 0 | 0 | 0 | 5.868 | 1554.0 | none | 0.024 (7/296) | 0.034 (1/29) |
| json | 30 | 0.203 | 0/30 (0.00) | 9 | 0 | 21 | 0 | 0 | 0 | 1.183 | 981.5 | json_object | 0.216 (64/296) | 0.034 (1/29) |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 0.988 | 511.0 | json_object | 0.000 (0/296) | 0.000 (0/29) |
| sola-json-sections | 30 | 0.363 | 1/30 (0.03) | 14 | 14 | 2 | 0 | 0 | 0 | 1.222 | 892.5 | json_object | 0.395 (117/296) | 0.448 (13/29) |
| sola-jsonish-sections | 30 | 0.215 | 2/30 (0.07) | 9 | 10 | 11 | 0 | 0 | 0 | 1.083 | 545.0 | json_object | 0.233 (69/296) | 0.207 (6/29) |
| sola-yaml-sections | 30 | 0.255 | 4/30 (0.13) | 8 | 4 | 18 | 0 | 0 | 0 | 1.175 | 540.5 | none | 0.260 (77/296) | 0.103 (3/29) |
| sola-jsonish-rescue | 30 | 0.271 | 2/30 (0.07) | 13 | 6 | 11 | 0 | 0 | 0 | 0.953 | 545.0 | json_object | 0.291 (86/296) | 0.310 (9/29) |
| sola-yaml-rescue | 30 | 0.357 | 6/30 (0.20) | 12 | 4 | 14 | 0 | 0 | 0 | 1.127 | 540.0 | none | 0.355 (105/296) | 0.103 (3/29) |

**All-null floor: 0.025** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

**chat**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 45 |
| `address.city` | 0 | 29 |
| `address.postcode` | 0 | 29 |
| `address.street` | 0 | 29 |
| `age` | 0 | 29 |
| `name` | 0 | 29 |
| `contacts[].email` | 0 | 27 |
| `contacts[].phone` | 0 | 27 |

**json**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 45 |
| `contacts[].email` | 0 | 26 |
| `contacts[].phone` | 0 | 26 |
| `address.street` | 2 | 21 |
| `address.postcode` | 1 | 21 |
| `address.city` | 0 | 21 |
| `age` | 0 | 21 |
| `name` | 0 | 21 |

**baml**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 45 |
| `address.city` | 0 | 30 |
| `address.postcode` | 0 | 30 |
| `address.street` | 0 | 30 |
| `age` | 0 | 30 |
| `name` | 0 | 30 |
| `contacts[].email` | 0 | 28 |
| `contacts[].phone` | 0 | 28 |

**sola-json-sections**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 42 |
| `contacts[].phone` | 11 | 14 |
| `address.postcode` | 4 | 16 |
| `contacts[].email` | 4 | 14 |
| `address.city` | 0 | 16 |
| `address.street` | 0 | 16 |
| `age` | 0 | 16 |
| `name` | 0 | 16 |

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 38 |
| `address.postcode` | 4 | 21 |
| `contacts[].email` | 0 | 23 |
| `contacts[].phone` | 0 | 23 |
| `address.city` | 0 | 21 |
| `address.street` | 0 | 21 |
| `age` | 0 | 21 |
| `name` | 0 | 21 |

**sola-yaml-sections**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 32 |
| `contacts[].phone` | 3 | 21 |
| `contacts[].email` | 2 | 21 |
| `address.city` | 0 | 22 |
| `address.postcode` | 0 | 22 |
| `address.street` | 0 | 22 |
| `age` | 0 | 22 |
| `name` | 0 | 22 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 38 |
| `address.postcode` | 7 | 17 |
| `contacts[].email` | 0 | 23 |
| `contacts[].phone` | 0 | 23 |
| `employment.company` | 0 | 18 |
| `employment.role` | 0 | 18 |
| `employment.years` | 0 | 18 |
| `address.city` | 0 | 17 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 27 |
| `contacts[].phone` | 3 | 21 |
| `contacts[].email` | 2 | 21 |
| `address.city` | 0 | 18 |
| `address.postcode` | 0 | 18 |
| `address.street` | 0 | 18 |
| `age` | 0 | 18 |
| `name` | 0 | 18 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
