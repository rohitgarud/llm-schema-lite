```
arm: accuracy
generated: 2026-09-10T06:20:18.287813+00:00
command: python -m benchmarking.dspy_adapters --accuracy --cases 30
git_head: daf210b
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/qwen3.5:0.8b
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
| chat | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 1.864 | 968.0 | none |
| json | 30 | 0.889 | 20/30 (0.67) | 29 | 0 | 1 | 0 | 0 | 0 | 1.297 | 995.0 | json_object |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 1.264 | 571.5 | json_object |
| sola-json-sections | 30 | 0.886 | 14/30 (0.47) | 30 | 0 | 0 | 0 | 0 | 0 | 1.228 | 905.0 | json_object |
| sola-jsonish-sections | 30 | 0.745 | 13/30 (0.43) | 24 | 0 | 6 | 0 | 0 | 0 | 1.357 | 622.0 | json_object |
| sola-yaml-sections | 30 | 0.711 | 11/30 (0.37) | 22 | 0 | 8 | 0 | 0 | 0 | 1.368 | 579.5 | none |
| sola-jsonish-rescue | 30 | 0.929 | 19/30 (0.63) | 30 | 0 | 0 | 0 | 0 | 0 | 1.274 | 622.0 | json_object |
| sola-yaml-rescue | 30 | 0.831 | 15/30 (0.50) | 26 | 0 | 4 | 0 | 0 | 0 | 1.274 | 579.5 | none |

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
| `tags[]` | 0 | 10 |
| `contacts[].email` | 8 | 0 |
| `contacts[].phone` | 8 | 0 |
| `address.street` | 2 | 1 |
| `address.city` | 0 | 1 |
| `address.postcode` | 0 | 1 |
| `age` | 0 | 1 |
| `employment.company` | 0 | 1 |

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
| `tags[]` | 3 | 19 |
| `contacts[].phone` | 8 | 0 |
| `contacts[].email` | 6 | 0 |
| `address.street` | 1 | 0 |

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 4 | 14 |
| `contacts[].phone` | 7 | 2 |
| `address.street` | 2 | 6 |
| `contacts[].email` | 6 | 2 |
| `address.city` | 0 | 6 |
| `address.postcode` | 0 | 6 |
| `age` | 0 | 6 |
| `name` | 0 | 6 |

**sola-yaml-sections**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 14 |
| `contacts[].email` | 6 | 7 |
| `contacts[].phone` | 6 | 7 |
| `address.city` | 0 | 8 |
| `address.postcode` | 0 | 8 |
| `address.street` | 0 | 8 |
| `age` | 0 | 8 |
| `name` | 0 | 8 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `contacts[].phone` | 7 | 2 |
| `contacts[].email` | 6 | 2 |
| `tags[]` | 4 | 0 |
| `address.street` | 2 | 0 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `contacts[].email` | 6 | 7 |
| `contacts[].phone` | 6 | 7 |
| `tags[]` | 0 | 5 |
| `address.city` | 0 | 4 |
| `address.postcode` | 0 | 4 |
| `address.street` | 0 | 4 |
| `age` | 0 | 4 |
| `employment` | 0 | 4 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
