```
arm: accuracy
generated: 2026-09-13T15:34:04.077150+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus financial-ner --cases 30 --adapters json,json-constrained,sola-jsonish-rescue,sola-yaml-rescue --out /tmp/claude-1000/-home-rohitgarud-llm-schema-lite/a9b95a98-8b4c-4773-96d3-50a0b9893834/scratchpad/constrained-qwen
git_head: 50c4c8d
dspy_version: 3.3.1
llm_schema_lite_version: 0.7.0
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
| json | 30 | 0.044 | 0/30 (0.00) | 1 | 0 | 29 | 0 | 0 | 0 | 0.713 | 920.5 | json_object | 0.069 (7/102) | 0.007 (1/147) |
| json-constrained | 30 | 0.691 | 3/30 (0.10) | 30 | 0 | 0 | 0 | 0 | 0 | 0.938 | 944.0 | json_schema | 0.686 (70/102) | 0.306 (45/147) |
| sola-jsonish-rescue | 30 | 0.639 | 6/30 (0.20) | 26 | 1 | 3 | 0 | 0 | 0 | 0.576 | 627.5 | json_object | 0.422 (43/102) | 0.075 (11/147) |
| sola-yaml-rescue | 30 | 0.763 | 3/30 (0.10) | 29 | 0 | 1 | 0 | 0 | 0 | 0.614 | 635.5 | none | 0.588 (60/102) | 0.082 (12/147) |

**All-null floor: 0.590** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

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

**json-constrained**

| field | wrong | missing |
|---|---|---|
| `Date` | 0 | 18 |
| `Company[]` | 12 | 0 |
| `Location` | 0 | 12 |
| `Location[]` | 9 | 0 |
| `Money[]` | 5 | 0 |
| `Product` | 0 | 5 |
| `Money` | 0 | 4 |
| `Company` | 0 | 2 |

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
| `Money[]` | 12 | 1 |
| `Company[]` | 10 | 2 |
| `Product` | 0 | 11 |
| `Location[]` | 10 | 0 |
| `Date[]` | 3 | 0 |
| `Company` | 0 | 2 |
| `Quantity[]` | 2 | 0 |
| `Date` | 0 | 1 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
