```
arm: accuracy
generated: 2026-09-12T08:28:07.492952+00:00
command: python -m benchmarking.dspy_adapters --accuracy --cases 30
git_head: b1016fb
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/granite3.1-moe:1b
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
| chat | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 0.755 | 964.5 | none | 0.000 (0/296) | 0.000 (0/29) |
| json | 30 | 0.797 | 6/30 (0.20) | 30 | 0 | 0 | 0 | 0 | 0 | 0.553 | 982.5 | json_object | 0.834 (247/296) | 0.207 (6/29) |
| baml | 30 | 0.760 | 8/30 (0.27) | 27 | 3 | 0 | 0 | 0 | 0 | 0.661 | 563.0 | json_object | 0.791 (234/296) | 0.414 (12/29) |
| sola-json-sections | 30 | 0.157 | 0/30 (0.00) | 6 | 0 | 24 | 0 | 0 | 0 | 0.673 | 933.5 | json_object | 0.169 (50/296) | 0.207 (6/29) |
| sola-jsonish-sections | 30 | 0.858 | 9/30 (0.30) | 29 | 0 | 1 | 0 | 0 | 0 | 0.695 | 626.5 | json_object | 0.899 (266/296) | 0.414 (12/29) |
| sola-yaml-sections | 30 | 0.422 | 5/30 (0.17) | 14 | 0 | 16 | 0 | 0 | 0 | 0.508 | 534.5 | none | 0.419 (124/296) | 0.172 (5/29) |
| sola-jsonish-rescue | 30 | 0.858 | 9/30 (0.30) | 29 | 0 | 1 | 0 | 0 | 0 | 0.769 | 626.5 | json_object | 0.899 (266/296) | 0.414 (12/29) |
| sola-yaml-rescue | 30 | 0.446 | 6/30 (0.20) | 15 | 0 | 15 | 0 | 0 | 0 | 0.513 | 534.5 | none | 0.439 (130/296) | 0.172 (5/29) |

**All-null floor: 0.025** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

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
| `tags[]` | 0 | 23 |
| `contacts[].phone` | 3 | 18 |
| `contacts[].email` | 2 | 18 |
| `address.street` | 2 | 16 |
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

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `contacts[].phone` | 3 | 18 |
| `tags[]` | 0 | 21 |
| `contacts[].email` | 2 | 18 |
| `address.street` | 2 | 15 |
| `address.city` | 0 | 15 |
| `address.postcode` | 0 | 15 |
| `age` | 0 | 15 |
| `name` | 0 | 15 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
