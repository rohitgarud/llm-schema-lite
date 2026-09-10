```
arm: accuracy
generated: 2026-09-10T09:48:06.350744+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus financial-ner --cases 30
git_head: 867819b
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/falcon3:1b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'Cleanlab/fire-financial-ner-extraction@d354ba26e96b07d216db1c22888a7f7e8be52fed', 'cases': 30}
supports_response_schema: False
supports_function_calling: False
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. Ground truth is the label shipped with the third-party `financial-ner` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.028 | 0/30 (0.00) | 2 | 28 | 0 | 0 | 0 | 0 | 5.419 | 1601.5 | none |
| json | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 1.496 | 1156.0 | json_object |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 1.854 | 792.0 | json_object |
| sola-json-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 0.605 | 988.5 | json_object |
| sola-jsonish-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 6 | 24 | 0 | 0 | 0 | 1.522 | 809.0 | json_object |
| sola-yaml-sections | 30 | 0.112 | 0/30 (0.00) | 6 | 8 | 16 | 0 | 0 | 0 | 0.698 | 722.5 | none |
| sola-jsonish-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 6 | 24 | 0 | 0 | 0 | 1.550 | 807.5 | json_object |
| sola-yaml-rescue | 30 | 0.112 | 0/30 (0.00) | 6 | 9 | 15 | 0 | 0 | 0 | 0.795 | 750.5 | none |

**All-null floor: 0.590** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so read every score above as a distance from this.

## Most-missed fields

**chat**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 5 | 56 |
| `Person` | 0 | 27 |
| `Product` | 0 | 27 |
| `Quantity` | 0 | 26 |
| `Date` | 0 | 25 |
| `Money[]` | 2 | 20 |
| `Location` | 0 | 20 |
| `Money` | 0 | 13 |

**json**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 0 | 61 |
| `Product` | 0 | 29 |
| `Person` | 0 | 28 |
| `Quantity` | 0 | 28 |
| `Date` | 0 | 26 |
| `Money[]` | 0 | 22 |
| `Location` | 0 | 21 |
| `Money` | 0 | 13 |

**baml**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 0 | 61 |
| `Product` | 0 | 29 |
| `Person` | 0 | 28 |
| `Quantity` | 0 | 28 |
| `Date` | 0 | 26 |
| `Money[]` | 0 | 22 |
| `Location` | 0 | 21 |
| `Money` | 0 | 13 |

**sola-json-sections**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 0 | 61 |
| `Product` | 0 | 29 |
| `Person` | 0 | 28 |
| `Quantity` | 0 | 28 |
| `Date` | 0 | 26 |
| `Money[]` | 0 | 22 |
| `Location` | 0 | 21 |
| `Money` | 0 | 13 |

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 0 | 61 |
| `Product` | 0 | 29 |
| `Person` | 0 | 28 |
| `Quantity` | 0 | 28 |
| `Date` | 0 | 26 |
| `Money[]` | 0 | 22 |
| `Location` | 0 | 21 |
| `Money` | 0 | 13 |

**sola-yaml-sections**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 16 | 45 |
| `Person` | 0 | 23 |
| `Product` | 0 | 23 |
| `Money[]` | 5 | 17 |
| `Quantity` | 0 | 22 |
| `Date` | 0 | 21 |
| `Location` | 0 | 17 |
| `Money` | 0 | 11 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 0 | 61 |
| `Product` | 0 | 29 |
| `Person` | 0 | 28 |
| `Quantity` | 0 | 28 |
| `Date` | 0 | 26 |
| `Money[]` | 0 | 22 |
| `Location` | 0 | 21 |
| `Money` | 0 | 13 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 16 | 45 |
| `Person` | 0 | 23 |
| `Product` | 0 | 23 |
| `Money[]` | 5 | 17 |
| `Quantity` | 0 | 22 |
| `Date` | 0 | 21 |
| `Location` | 0 | 17 |
| `Money` | 0 | 11 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
