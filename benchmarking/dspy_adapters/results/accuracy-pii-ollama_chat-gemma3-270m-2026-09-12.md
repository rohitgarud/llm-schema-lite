```
arm: accuracy
generated: 2026-09-12T10:20:09.946268+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus pii --cases 30
git_head: 18b38b4
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/gemma3:270m
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
| chat | 30 | 0.632 | 0/30 (0.00) | 20 | 10 | 0 | 0 | 0 | 0 | 3.184 | 3001.0 | none | 0.000 (0/86) | 0.000 (0/1594) |
| json | 30 | 0.918 | 0/30 (0.00) | 29 | 1 | 0 | 0 | 0 | 0 | 3.354 | 3303.5 | json_object | 0.000 (0/86) | 0.000 (0/1594) |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 3.215 | 1616.0 | json_object | 0.000 (0/86) | 0.000 (0/1594) |
| sola-json-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 13 | 17 | 0 | 0 | 0 | 1.218 | 2389.0 | json_object | 0.000 (0/86) | 0.000 (0/1594) |
| sola-jsonish-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 0.227 | 1101.0 | json_object | 0.000 (0/86) | 0.000 (0/1594) |
| sola-yaml-sections | 30 | 0.000 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 3.060 | 1995.5 | none | 0.000 (0/86) | 1.000 (1594/1594) |
| sola-jsonish-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 0.269 | 1093.5 | json_object | 0.000 (0/86) | 0.000 (0/1594) |
| sola-yaml-rescue | 30 | 0.000 | 0/30 (0.00) | 29 | 1 | 0 | 0 | 0 | 0 | 3.537 | 1995.5 | none | 0.000 (0/86) | 0.967 (1542/1594) |

**All-null floor: 0.949** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

**chat**

| field | wrong | missing |
|---|---|---|
| `FIRSTNAME` | 7 | 10 |
| `AMOUNT` | 4 | 10 |
| `PHONEIMEI` | 3 | 10 |
| `CURRENCY` | 2 | 10 |
| `DATE` | 2 | 10 |
| `GENDER` | 2 | 10 |
| `IBAN` | 2 | 10 |
| `IPV6` | 2 | 10 |

**json**

| field | wrong | missing |
|---|---|---|
| `FIRSTNAME` | 9 | 1 |
| `AMOUNT` | 5 | 1 |
| `IPV4` | 4 | 1 |
| `CURRENCY` | 3 | 1 |
| `IPV6` | 3 | 1 |
| `LASTNAME` | 3 | 1 |
| `MIDDLENAME` | 3 | 1 |
| `PHONEIMEI` | 3 | 1 |

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
| `ACCOUNTNAME` | 0 | 30 |
| `ACCOUNTNUMBER` | 0 | 30 |
| `AGE` | 0 | 30 |
| `AMOUNT` | 0 | 30 |
| `BIC` | 0 | 30 |
| `BITCOINADDRESS` | 0 | 30 |
| `BUILDINGNUMBER` | 0 | 30 |
| `CITY` | 0 | 30 |

**sola-jsonish-sections**

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

**sola-yaml-sections**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNAME` | 30 | 0 |
| `ACCOUNTNUMBER` | 30 | 0 |
| `AGE` | 30 | 0 |
| `AMOUNT` | 30 | 0 |
| `BIC` | 30 | 0 |
| `BITCOINADDRESS` | 30 | 0 |
| `BUILDINGNUMBER` | 30 | 0 |
| `CITY` | 30 | 0 |

**sola-jsonish-rescue**

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

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNAME` | 29 | 1 |
| `ACCOUNTNUMBER` | 29 | 1 |
| `AGE` | 29 | 1 |
| `AMOUNT` | 29 | 1 |
| `BIC` | 29 | 1 |
| `BITCOINADDRESS` | 29 | 1 |
| `BUILDINGNUMBER` | 29 | 1 |
| `CITY` | 29 | 1 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
