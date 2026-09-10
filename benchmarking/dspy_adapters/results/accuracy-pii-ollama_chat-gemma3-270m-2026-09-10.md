```
arm: accuracy
generated: 2026-09-10T10:23:20.541442+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus pii --cases 30
git_head: 867819b
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

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. Ground truth is the label shipped with the third-party `pii` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.632 | 0/30 (0.00) | 20 | 10 | 0 | 0 | 0 | 0 | 3.332 | 3001.0 | none |
| json | 30 | 0.918 | 0/30 (0.00) | 29 | 1 | 0 | 0 | 0 | 0 | 3.473 | 3303.5 | json_object |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 3.255 | 1616.0 | json_object |
| sola-json-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 13 | 17 | 0 | 0 | 0 | 1.197 | 2389.0 | json_object |
| sola-jsonish-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 0.239 | 1101.0 | json_object |
| sola-yaml-sections | 30 | 0.000 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 3.323 | 1995.5 | none |
| sola-jsonish-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 0.245 | 1093.5 | json_object |
| sola-yaml-rescue | 30 | 0.000 | 0/30 (0.00) | 29 | 1 | 0 | 0 | 0 | 0 | 3.276 | 1995.5 | none |

**All-null floor: 0.949** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so read every score above as a distance from this.

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
