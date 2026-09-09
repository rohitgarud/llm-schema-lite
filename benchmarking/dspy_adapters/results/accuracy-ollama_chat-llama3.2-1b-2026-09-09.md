```
arm: accuracy
generated: 2026-09-09T14:01:50.694001+00:00
command: python -m benchmarking.dspy_adapters --accuracy --cases 30
git_head: 054da0a
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/llama3.2:1b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'cases': 30, 'cases_seed': 0}
supports_response_schema: False
supports_function_calling: True
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. Ground truth is the generated record the prose was rendered from, so the corpus measures schema-following under paraphrase — not real-world extraction.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.022 | 0/30 (0.00) | 1 | 29 | 0 | 0 | 0 | 0 | 5.758 | 1554.0 | none |
| json | 30 | 0.203 | 0/30 (0.00) | 9 | 0 | 21 | 0 | 0 | 0 | 1.184 | 981.5 | json_object |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 0.991 | 511.0 | json_object |
| sola-json-sections | 30 | 0.363 | 1/30 (0.03) | 14 | 14 | 2 | 0 | 0 | 0 | 1.189 | 892.5 | json_object |
| sola-jsonish-sections | 30 | 0.215 | 2/30 (0.07) | 9 | 10 | 11 | 0 | 0 | 0 | 0.946 | 545.0 | json_object |
| sola-yaml-sections | 30 | 0.206 | 3/30 (0.10) | 7 | 5 | 18 | 0 | 0 | 0 | 1.044 | 533.0 | none |
| sola-jsonish-rescue | 30 | 0.215 | 2/30 (0.07) | 9 | 10 | 11 | 0 | 0 | 0 | 0.953 | 545.0 | json_object |

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
| `tags[]` | 0 | 39 |
| `address.city` | 0 | 23 |
| `address.postcode` | 0 | 23 |
| `address.street` | 0 | 23 |
| `age` | 0 | 23 |
| `contacts[].phone` | 1 | 22 |
| `name` | 0 | 23 |
| `contacts[].email` | 0 | 22 |

**sola-jsonish-rescue**

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

Per-case detail is in the companion `.csv`; it is not duplicated here.
