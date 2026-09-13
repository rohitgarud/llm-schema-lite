```
arm: accuracy
generated: 2026-09-13T14:46:44.349364+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus pii --cases 30 --adapters json,json-constrained,sola-jsonish-rescue,sola-yaml-rescue --out /tmp/claude-1000/-home-rohitgarud-llm-schema-lite/a9b95a98-8b4c-4773-96d3-50a0b9893834/scratchpad/constrained
git_head: 3d012b8-dirty
dspy_version: 3.3.1
llm_schema_lite_version: 0.7.0
model: ollama_chat/falcon3:1b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'Cleanlab/pii-extraction@ca07a3a51edbc5ca99d13e452c6c1c49fd83ff9f', 'cases': 30}
supports_response_schema: False
supports_function_calling: False
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. A correct `None` also counts as a match, so on a sparse corpus field accuracy pays a reply for extracting nothing. **recall (non-null gold)** is the extraction-quality headline: matches over only the fields whose gold value is not `None`, a cell that raised scoring 0 against them, so extracting nothing scores 0. **invented (null gold)** is the other half: of the fields whose gold is `None`, how many the reply filled anyway. Ground truth is the label shipped with the third-party `pii` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format | recall (non-null gold) | invented (null gold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| json | 30 | 0.095 | 0/30 (0.00) | 3 | 0 | 27 | 0 | 0 | 0 | 1.326 | 2933.0 | json_object | 0.000 (0/86) | 0.000 (0/1594) |
| json-constrained | 30 | 0.263 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 7.736 | 3575.0 | json_schema | 0.291 (25/86) | 0.738 (1177/1594) |
| sola-jsonish-rescue | 30 | 0.464 | 0/30 (0.00) | 26 | 0 | 4 | 0 | 0 | 0 | 5.554 | 1637.5 | json_object | 0.221 (19/86) | 0.390 (622/1594) |
| sola-yaml-rescue | 30 | 0.374 | 0/30 (0.00) | 22 | 1 | 7 | 0 | 0 | 0 | 3.472 | 1268.0 | none | 0.116 (10/86) | 0.346 (552/1594) |

**All-null floor: 0.949** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

**json**

| field | wrong | missing |
|---|---|---|
| `AGE` | 2 | 27 |
| `IPV6` | 2 | 27 |
| `FIRSTNAME` | 1 | 27 |
| `GENDER` | 1 | 27 |
| `MAC` | 1 | 27 |
| `TIME` | 1 | 27 |
| `USERAGENT` | 1 | 27 |
| `ACCOUNTNAME` | 0 | 27 |

**json-constrained**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 30 | 0 |
| `AGE` | 30 | 0 |
| `BIC` | 30 | 0 |
| `BITCOINADDRESS` | 30 | 0 |
| `ACCOUNTNAME` | 29 | 0 |
| `AMOUNT` | 25 | 0 |
| `CREDITCARDNUMBER` | 25 | 0 |
| `CITY` | 24 | 0 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 26 | 4 |
| `AGE` | 26 | 4 |
| `ACCOUNTNAME` | 24 | 4 |
| `AMOUNT` | 24 | 4 |
| `BIC` | 17 | 4 |
| `CREDITCARDISSUER` | 14 | 4 |
| `CREDITCARDNUMBER` | 14 | 4 |
| `CURRENCYCODE` | 14 | 4 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 22 | 8 |
| `AGE` | 22 | 8 |
| `BIC` | 22 | 8 |
| `ACCOUNTNAME` | 21 | 8 |
| `AMOUNT` | 19 | 8 |
| `BITCOINADDRESS` | 19 | 8 |
| `CURRENCYCODE` | 14 | 8 |
| `COMPANYNAME` | 13 | 8 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
