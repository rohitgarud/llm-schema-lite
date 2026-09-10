```
arm: accuracy
generated: 2026-09-10T07:08:59.838655+00:00
command: python -m benchmarking.dspy_adapters --accuracy --cases 30
git_head: daf210b
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/falcon3:1b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'cases': 30, 'cases_seed': 0}
supports_response_schema: False
supports_function_calling: False
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. Ground truth is the generated record the prose was rendered from, so the corpus measures schema-following under paraphrase — not real-world extraction.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 7.838 | 1838.0 | none |
| json | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 7.398 | 1644.0 | json_object |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 1.460 | 614.0 | json_object |
| sola-json-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 29 | 1 | 0 | 0 | 0 | 2.144 | 1170.5 | json_object |
| sola-jsonish-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 1.331 | 671.5 | json_object |
| sola-yaml-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 2 | 28 | 0 | 0 | 0 | 1.137 | 586.5 | none |
| sola-jsonish-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 1.377 | 671.5 | json_object |
| sola-yaml-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 2 | 28 | 0 | 0 | 0 | 1.124 | 586.5 | none |

## Most-missed fields

**chat**

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

**json**

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
| `tags[]` | 0 | 45 |
| `address.city` | 0 | 30 |
| `address.postcode` | 0 | 30 |
| `address.street` | 0 | 30 |
| `age` | 0 | 30 |
| `name` | 0 | 30 |
| `contacts[].email` | 0 | 28 |
| `contacts[].phone` | 0 | 28 |

**sola-jsonish-sections**

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

**sola-yaml-sections**

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

**sola-jsonish-rescue**

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

**sola-yaml-rescue**

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

Per-case detail is in the companion `.csv`; it is not duplicated here.
