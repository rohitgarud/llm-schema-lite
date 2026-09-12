```
arm: accuracy
generated: 2026-09-12T11:29:01.919417+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus financial-ner --cases 30
git_head: 18b38b4
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/llama3.2:1b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'Cleanlab/fire-financial-ner-extraction@d354ba26e96b07d216db1c22888a7f7e8be52fed', 'cases': 30}
supports_response_schema: False
supports_function_calling: True
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. A correct `None` also counts as a match, so on a sparse corpus field accuracy pays a reply for extracting nothing. **recall (non-null gold)** is the extraction-quality headline: matches over only the fields whose gold value is not `None`, a cell that raised scoring 0 against them, so extracting nothing scores 0. **invented (null gold)** is the other half: of the fields whose gold is `None`, how many the reply filled anyway. Ground truth is the label shipped with the third-party `financial-ner` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format | recall (non-null gold) | invented (null gold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.189 | 0/30 (0.00) | 11 | 19 | 0 | 0 | 0 | 0 | 2.106 | 1068.0 | none | 0.049 (5/102) | 0.116 (17/147) |
| json | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 0.557 | 892.5 | json_object | 0.000 (0/102) | 0.000 (0/147) |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 1.823 | 718.5 | json_object | 0.000 (0/102) | 0.000 (0/147) |
| sola-json-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 1.739 | 986.0 | json_object | 0.000 (0/102) | 0.000 (0/147) |
| sola-jsonish-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 2 | 28 | 0 | 0 | 0 | 1.216 | 690.5 | json_object | 0.000 (0/102) | 0.000 (0/147) |
| sola-yaml-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 1 | 29 | 0 | 0 | 0 | 2.227 | 832.0 | none | 0.000 (0/102) | 0.000 (0/147) |
| sola-jsonish-rescue | 30 | 0.281 | 0/30 (0.00) | 29 | 0 | 1 | 0 | 0 | 0 | 1.112 | 690.5 | json_object | 0.608 (62/102) | 0.918 (135/147) |
| sola-yaml-rescue | 30 | 0.080 | 0/30 (0.00) | 4 | 0 | 26 | 0 | 0 | 0 | 1.989 | 832.0 | none | 0.098 (10/102) | 0.048 (7/147) |

**All-null floor: 0.590** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

**chat**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 20 | 36 |
| `Money[]` | 5 | 17 |
| `Product` | 0 | 21 |
| `Person` | 0 | 20 |
| `Quantity` | 0 | 20 |
| `Date` | 0 | 18 |
| `Location` | 0 | 16 |
| `Location[]` | 3 | 7 |

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
| `Company[]` | 0 | 61 |
| `Product` | 0 | 29 |
| `Person` | 0 | 28 |
| `Quantity` | 0 | 28 |
| `Date` | 0 | 26 |
| `Money[]` | 0 | 22 |
| `Location` | 0 | 21 |
| `Money` | 0 | 13 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `Person` | 0 | 27 |
| `Product` | 0 | 27 |
| `Quantity` | 0 | 26 |
| `Date` | 0 | 25 |
| `Location` | 0 | 20 |
| `Money[]` | 13 | 1 |
| `Company[]` | 11 | 2 |
| `Money` | 0 | 12 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 4 | 52 |
| `Product` | 0 | 27 |
| `Person` | 0 | 26 |
| `Quantity` | 0 | 26 |
| `Date` | 0 | 24 |
| `Money[]` | 1 | 20 |
| `Location` | 0 | 20 |
| `Money` | 0 | 12 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
