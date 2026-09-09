```
arm: accuracy
generated: 2026-09-09T14:04:21.181561+00:00
command: python -m benchmarking.dspy_adapters --accuracy --cases 30
git_head: 054da0a
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/granite3.1-moe:1b
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
| chat | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 0.749 | 964.5 | none |
| json | 30 | 0.797 | 6/30 (0.20) | 30 | 0 | 0 | 0 | 0 | 0 | 0.557 | 982.5 | json_object |
| baml | 30 | 0.760 | 8/30 (0.27) | 27 | 3 | 0 | 0 | 0 | 0 | 0.668 | 563.0 | json_object |
| sola-json-sections | 30 | 0.157 | 0/30 (0.00) | 6 | 0 | 24 | 0 | 0 | 0 | 0.685 | 933.5 | json_object |
| sola-jsonish-sections | 30 | 0.858 | 9/30 (0.30) | 29 | 0 | 1 | 0 | 0 | 0 | 0.707 | 626.5 | json_object |
| sola-yaml-sections | 30 | 0.363 | 2/30 (0.07) | 14 | 0 | 16 | 0 | 0 | 0 | 0.499 | 539.0 | none |
| sola-jsonish-rescue | 30 | 0.858 | 9/30 (0.30) | 29 | 0 | 1 | 0 | 0 | 0 | 0.709 | 626.5 | json_object |

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
| `contacts[].email` | 0 | 28 |
| `contacts[].phone` | 0 | 28 |
| `employment` | 0 | 6 |
| `tags[]` | 0 | 4 |

**baml**

| field | wrong | missing |
|---|---|---|
| `address.postcode` | 10 | 3 |
| `contacts[].phone` | 8 | 4 |
| `tags[]` | 0 | 11 |
| `contacts[].email` | 5 | 4 |
| `employment` | 0 | 8 |
| `address.street` | 4 | 3 |
| `address.city` | 0 | 3 |
| `age` | 0 | 3 |

**sola-json-sections**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 36 |
| `contacts[].phone` | 2 | 24 |
| `address.city` | 0 | 24 |
| `address.postcode` | 0 | 24 |
| `address.street` | 0 | 24 |
| `age` | 0 | 24 |
| `contacts[].email` | 0 | 24 |
| `name` | 0 | 24 |

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `contacts[].phone` | 10 | 2 |
| `contacts[].email` | 7 | 2 |
| `employment` | 0 | 8 |
| `address.postcode` | 6 | 1 |
| `address.street` | 4 | 1 |
| `tags[]` | 0 | 2 |
| `address.city` | 0 | 1 |
| `age` | 0 | 1 |

**sola-yaml-sections**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 37 |
| `contacts[].phone` | 3 | 19 |
| `contacts[].email` | 2 | 19 |
| `address.street` | 4 | 16 |
| `address.city` | 0 | 16 |
| `address.postcode` | 0 | 16 |
| `age` | 0 | 16 |
| `name` | 0 | 16 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `contacts[].phone` | 10 | 2 |
| `contacts[].email` | 7 | 2 |
| `employment` | 0 | 8 |
| `address.postcode` | 6 | 1 |
| `address.street` | 4 | 1 |
| `tags[]` | 0 | 2 |
| `address.city` | 0 | 1 |
| `age` | 0 | 1 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
