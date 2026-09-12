```
arm: accuracy
generated: 2026-09-12T11:08:52.977255+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus pii --cases 30
git_head: 18b38b4
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
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
| chat | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 1.221 | 2787.0 | none | 0.000 (0/86) | 0.000 (0/1594) |
| json | 30 | 0.095 | 0/30 (0.00) | 3 | 0 | 27 | 0 | 0 | 0 | 1.225 | 2932.0 | json_object | 0.000 (0/86) | 0.000 (0/1594) |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 1 | 29 | 0 | 0 | 0 | 0.355 | 799.5 | json_object | 0.000 (0/86) | 0.000 (0/1594) |
| sola-json-sections | 30 | 0.346 | 0/30 (0.00) | 15 | 1 | 14 | 0 | 0 | 0 | 9.153 | 3524.5 | json_object | 0.105 (9/86) | 0.143 (228/1594) |
| sola-jsonish-sections | 30 | 0.321 | 0/30 (0.00) | 23 | 5 | 2 | 0 | 0 | 0 | 7.627 | 1949.0 | json_object | 0.267 (23/86) | 0.442 (704/1594) |
| sola-yaml-sections | 30 | 0.383 | 0/30 (0.00) | 24 | 1 | 5 | 0 | 0 | 0 | 4.396 | 1284.0 | none | 0.198 (17/86) | 0.408 (650/1594) |
| sola-jsonish-rescue | 30 | 0.370 | 0/30 (0.00) | 28 | 0 | 2 | 0 | 0 | 0 | 7.529 | 1949.0 | json_object | 0.291 (25/86) | 0.560 (892/1594) |
| sola-yaml-rescue | 30 | 0.396 | 0/30 (0.00) | 24 | 1 | 5 | 0 | 0 | 0 | 4.183 | 1267.0 | none | 0.186 (16/86) | 0.393 (627/1594) |

**All-null floor: 0.949** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

**chat**

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

**baml**

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

**sola-json-sections**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 15 | 15 |
| `AMOUNT` | 15 | 15 |
| `BIC` | 14 | 15 |
| `BITCOINADDRESS` | 13 | 15 |
| `AGE` | 12 | 15 |
| `ACCOUNTNAME` | 6 | 15 |
| `CREDITCARDCVV` | 6 | 15 |
| `CREDITCARDNUMBER` | 6 | 15 |

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 23 | 7 |
| `AGE` | 23 | 7 |
| `BIC` | 23 | 7 |
| `ACCOUNTNAME` | 22 | 7 |
| `AMOUNT` | 21 | 7 |
| `BITCOINADDRESS` | 16 | 7 |
| `CREDITCARDNUMBER` | 16 | 7 |
| `CURRENCYCODE` | 16 | 7 |

**sola-yaml-sections**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 24 | 6 |
| `AGE` | 24 | 6 |
| `BIC` | 24 | 6 |
| `ACCOUNTNAME` | 23 | 6 |
| `AMOUNT` | 21 | 6 |
| `BITCOINADDRESS` | 20 | 6 |
| `CREDITCARDCVV` | 16 | 6 |
| `CREDITCARDISSUER` | 16 | 6 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 28 | 2 |
| `AGE` | 28 | 2 |
| `ACCOUNTNAME` | 26 | 2 |
| `AMOUNT` | 26 | 2 |
| `BIC` | 21 | 2 |
| `CREDITCARDNUMBER` | 20 | 2 |
| `CURRENCYCODE` | 20 | 2 |
| `COMPANYNAME` | 19 | 2 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 24 | 6 |
| `AGE` | 24 | 6 |
| `BIC` | 24 | 6 |
| `ACCOUNTNAME` | 23 | 6 |
| `AMOUNT` | 21 | 6 |
| `BITCOINADDRESS` | 20 | 6 |
| `CREDITCARDCVV` | 16 | 6 |
| `CREDITCARDISSUER` | 16 | 6 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
