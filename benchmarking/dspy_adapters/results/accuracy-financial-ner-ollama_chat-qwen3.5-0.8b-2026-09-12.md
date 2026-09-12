```
arm: accuracy
generated: 2026-09-12T11:12:48.089593+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus financial-ner --cases 30
git_head: 18b38b4
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/qwen3.5:0.8b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'Cleanlab/fire-financial-ner-extraction@d354ba26e96b07d216db1c22888a7f7e8be52fed', 'cases': 30}
supports_response_schema: False
supports_function_calling: False
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. A correct `None` also counts as a match, so on a sparse corpus field accuracy pays a reply for extracting nothing. **recall (non-null gold)** is the extraction-quality headline: matches over only the fields whose gold value is not `None`, a cell that raised scoring 0 against them, so extracting nothing scores 0. **invented (null gold)** is the other half: of the fields whose gold is `None`, how many the reply filled anyway. Ground truth is the label shipped with the third-party `financial-ner` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format | recall (non-null gold) | invented (null gold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 0.967 | 905.0 | none | 0.000 (0/102) | 0.000 (0/147) |
| json | 30 | 0.044 | 0/30 (0.00) | 1 | 0 | 29 | 0 | 0 | 0 | 0.800 | 920.5 | json_object | 0.069 (7/102) | 0.007 (1/147) |
| baml | 30 | 0.570 | 0/30 (0.00) | 29 | 0 | 1 | 0 | 0 | 0 | 1.423 | 681.5 | json_object | 0.000 (0/102) | 0.000 (0/147) |
| sola-json-sections | 30 | 0.213 | 1/30 (0.03) | 9 | 0 | 21 | 0 | 0 | 0 | 0.769 | 861.0 | json_object | 0.235 (24/102) | 0.082 (12/147) |
| sola-jsonish-sections | 30 | 0.639 | 6/30 (0.20) | 26 | 1 | 3 | 0 | 0 | 0 | 0.669 | 627.5 | json_object | 0.422 (43/102) | 0.075 (11/147) |
| sola-yaml-sections | 30 | 0.699 | 1/30 (0.03) | 27 | 0 | 3 | 0 | 0 | 0 | 0.712 | 637.0 | none | 0.529 (54/102) | 0.082 (12/147) |
| sola-jsonish-rescue | 30 | 0.639 | 6/30 (0.20) | 26 | 1 | 3 | 0 | 0 | 0 | 0.671 | 627.5 | json_object | 0.422 (43/102) | 0.075 (11/147) |
| sola-yaml-rescue | 30 | 0.695 | 1/30 (0.03) | 27 | 0 | 3 | 0 | 0 | 0 | 0.698 | 637.0 | none | 0.529 (54/102) | 0.088 (13/147) |

**All-null floor: 0.590** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

**chat**

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

**json**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 0 | 58 |
| `Product` | 0 | 29 |
| `Person` | 0 | 27 |
| `Quantity` | 0 | 27 |
| `Date` | 0 | 25 |
| `Location` | 0 | 20 |
| `Money[]` | 1 | 17 |
| `Money` | 0 | 13 |

**baml**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 59 | 2 |
| `Money[]` | 21 | 1 |
| `Location[]` | 10 | 0 |
| `Date[]` | 4 | 0 |
| `Person[]` | 2 | 0 |
| `Quantity[]` | 2 | 0 |
| `Date` | 0 | 1 |
| `Location` | 0 | 1 |

**sola-json-sections**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 4 | 44 |
| `Product` | 0 | 24 |
| `Quantity` | 0 | 22 |
| `Date` | 0 | 20 |
| `Person` | 0 | 20 |
| `Location` | 0 | 18 |
| `Money[]` | 3 | 10 |
| `Money` | 0 | 12 |

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 24 | 6 |
| `Money[]` | 8 | 3 |
| `Location[]` | 7 | 3 |
| `Date` | 0 | 7 |
| `Location` | 0 | 6 |
| `Product` | 0 | 5 |
| `Date[]` | 4 | 0 |
| `Person` | 0 | 4 |

**sola-yaml-sections**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 11 | 6 |
| `Money[]` | 11 | 3 |
| `Product` | 0 | 13 |
| `Location[]` | 10 | 0 |
| `Date` | 0 | 3 |
| `Date[]` | 3 | 0 |
| `Location` | 0 | 3 |
| `Person` | 0 | 3 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 24 | 6 |
| `Money[]` | 8 | 3 |
| `Location[]` | 7 | 3 |
| `Date` | 0 | 7 |
| `Location` | 0 | 6 |
| `Product` | 0 | 5 |
| `Date[]` | 4 | 0 |
| `Person` | 0 | 4 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `Company[]` | 11 | 6 |
| `Money[]` | 11 | 3 |
| `Product` | 0 | 14 |
| `Location[]` | 10 | 0 |
| `Date` | 0 | 3 |
| `Date[]` | 3 | 0 |
| `Location` | 0 | 3 |
| `Person` | 0 | 3 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
