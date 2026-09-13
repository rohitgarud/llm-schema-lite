```
arm: accuracy
generated: 2026-09-13T16:28:17.198555+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus synthetic --cases 30 --adapters json,json-constrained,sola-jsonish-rescue,sola-yaml-rescue --out /tmp/claude-1000/-home-rohitgarud-llm-schema-lite/a9b95a98-8b4c-4773-96d3-50a0b9893834/scratchpad/constrained-llama
git_head: 50c4c8d
dspy_version: 3.3.1
llm_schema_lite_version: 0.7.0
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
| json | 30 | 0.200 | 0/30 (0.00) | 9 | 0 | 21 | 0 | 0 | 0 | 1.329 | 982.0 | json_object | 0.213 (63/296) | 0.034 (1/29) |
| json-constrained | 30 | 0.794 | 4/30 (0.13) | 30 | 0 | 0 | 0 | 0 | 0 | 1.276 | 962.0 | json_schema | 0.861 (255/296) | 0.862 (25/29) |
| sola-jsonish-rescue | 30 | 0.271 | 2/30 (0.07) | 13 | 6 | 11 | 0 | 0 | 0 | 1.111 | 545.0 | json_object | 0.291 (86/296) | 0.310 (9/29) |
| sola-yaml-rescue | 30 | 0.357 | 6/30 (0.20) | 12 | 4 | 14 | 0 | 0 | 0 | 1.144 | 540.5 | none | 0.355 (105/296) | 0.103 (3/29) |

**All-null floor: 0.025** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

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

**json-constrained**

| field | wrong | missing |
|---|---|---|
| `tags[]` | 0 | 27 |
| `contacts[].phone` | 12 | 1 |
| `address.postcode` | 10 | 0 |
| `contacts[].email` | 7 | 1 |
| `employment` | 0 | 8 |
| `address.street` | 1 | 0 |

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
