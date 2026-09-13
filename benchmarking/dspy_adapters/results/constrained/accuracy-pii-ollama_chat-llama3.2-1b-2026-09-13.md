```
arm: accuracy
generated: 2026-09-13T16:37:38.316044+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus pii --cases 30 --adapters json,json-constrained,sola-jsonish-rescue,sola-yaml-rescue --out /tmp/claude-1000/-home-rohitgarud-llm-schema-lite/a9b95a98-8b4c-4773-96d3-50a0b9893834/scratchpad/constrained-llama
git_head: 50c4c8d
dspy_version: 3.3.1
llm_schema_lite_version: 0.7.0
model: ollama_chat/llama3.2:1b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'Cleanlab/pii-extraction@ca07a3a51edbc5ca99d13e452c6c1c49fd83ff9f', 'cases': 30}
supports_response_schema: False
supports_function_calling: True
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. A correct `None` also counts as a match, so on a sparse corpus field accuracy pays a reply for extracting nothing. **recall (non-null gold)** is the extraction-quality headline: matches over only the fields whose gold value is not `None`, a cell that raised scoring 0 against them, so extracting nothing scores 0. **invented (null gold)** is the other half: of the fields whose gold is `None`, how many the reply filled anyway. Ground truth is the label shipped with the third-party `pii` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format | recall (non-null gold) | invented (null gold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| json | 30 | 0.537 | 0/30 (0.00) | 18 | 0 | 12 | 0 | 0 | 0 | 0.378 | 2440.0 | json_object | 0.012 (1/86) | 0.034 (54/1594) |
| json-constrained | 30 | 0.676 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 3.955 | 2870.0 | json_schema | 0.256 (22/86) | 0.302 (481/1594) |
| sola-jsonish-rescue | 30 | 0.320 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 4.480 | 1440.0 | json_object | 0.372 (32/86) | 0.683 (1088/1594) |
| sola-yaml-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 0.208 | 997.5 | none | 0.000 (0/86) | 0.000 (0/1594) |

**All-null floor: 0.949** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

**json**

| field | wrong | missing |
|---|---|---|
| `FIRSTNAME` | 7 | 12 |
| `PHONEIMEI` | 4 | 12 |
| `VEHICLEVIN` | 4 | 12 |
| `ACCOUNTNAME` | 3 | 12 |
| `AGE` | 3 | 12 |
| `AMOUNT` | 3 | 12 |
| `CREDITCARDISSUER` | 3 | 12 |
| `CURRENCY` | 3 | 12 |

**json-constrained**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNAME` | 29 | 0 |
| `ACCOUNTNUMBER` | 27 | 0 |
| `CREDITCARDNUMBER` | 18 | 0 |
| `COMPANYNAME` | 14 | 0 |
| `CURRENCY` | 13 | 0 |
| `FIRSTNAME` | 12 | 0 |
| `USERNAME` | 12 | 0 |
| `CREDITCARDISSUER` | 11 | 0 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 28 | 0 |
| `ACCOUNTNAME` | 27 | 0 |
| `CREDITCARDNUMBER` | 24 | 0 |
| `CURRENCY` | 23 | 0 |
| `AMOUNT` | 22 | 0 |
| `IBAN` | 22 | 0 |
| `IP` | 22 | 0 |
| `PHONEIMEI` | 22 | 0 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNAME` | 0 | 30 |
| `ACCOUNTNUMBER` | 0 | 30 |
| `AGE` | 0 | 30 |
| `AMOUNT` | 0 | 30 |
| `BIC` | 0 | 30 |
| `BITCOINADDRESS` | 0 | 30 |
| `BUILDINGNUMBER` | 0 | 30 |
| `CITY` | 0 | 30 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
