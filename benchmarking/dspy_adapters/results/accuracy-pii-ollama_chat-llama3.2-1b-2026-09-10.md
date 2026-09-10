```
arm: accuracy
generated: 2026-09-10T08:23:05.801665+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus pii --cases 30
git_head: 867819b
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/llama3.2:1b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'Cleanlab/pii-extraction@ca07a3a51edbc5ca99d13e452c6c1c49fd83ff9f', 'cases': 30}
supports_response_schema: False
supports_function_calling: True
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. Ground truth is the label shipped with the third-party `pii` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.159 | 0/30 (0.00) | 5 | 25 | 0 | 0 | 0 | 0 | 0.174 | 2145.5 | none |
| json | 30 | 0.507 | 0/30 (0.00) | 17 | 0 | 13 | 0 | 0 | 0 | 0.358 | 2432.0 | json_object |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 3.564 | 1174.0 | json_object |
| sola-json-sections | 30 | 0.402 | 0/30 (0.00) | 27 | 2 | 1 | 0 | 0 | 0 | 3.438 | 2552.0 | json_object |
| sola-jsonish-sections | 30 | 0.082 | 0/30 (0.00) | 7 | 23 | 0 | 0 | 0 | 0 | 4.057 | 1446.0 | json_object |
| sola-yaml-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 0.193 | 997.5 | none |
| sola-jsonish-rescue | 30 | 0.082 | 0/30 (0.00) | 7 | 23 | 0 | 0 | 0 | 0 | 4.135 | 1446.0 | json_object |
| sola-yaml-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 0.199 | 997.5 | none |

**All-null floor: 0.949** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so read every score above as a distance from this.

## Most-missed fields

**chat**

| field | wrong | missing |
|---|---|---|
| `FIRSTNAME` | 2 | 25 |
| `PHONEIMEI` | 2 | 25 |
| `ACCOUNTNAME` | 1 | 25 |
| `EYECOLOR` | 1 | 25 |
| `IP` | 1 | 25 |
| `IPV4` | 1 | 25 |
| `IPV6` | 1 | 25 |
| `LASTNAME` | 1 | 25 |

**json**

| field | wrong | missing |
|---|---|---|
| `FIRSTNAME` | 6 | 13 |
| `PHONEIMEI` | 4 | 13 |
| `AGE` | 3 | 13 |
| `AMOUNT` | 3 | 13 |
| `CURRENCY` | 3 | 13 |
| `GENDER` | 3 | 13 |
| `IPV6` | 3 | 13 |
| `MASKEDNUMBER` | 3 | 13 |

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
| `ACCOUNTNUMBER` | 24 | 3 |
| `ACCOUNTNAME` | 22 | 3 |
| `CREDITCARDNUMBER` | 18 | 3 |
| `AMOUNT` | 17 | 3 |
| `COMPANYNAME` | 17 | 3 |
| `CREDITCARDISSUER` | 17 | 3 |
| `CURRENCY` | 17 | 3 |
| `FIRSTNAME` | 17 | 3 |

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 7 | 23 |
| `ACCOUNTNAME` | 6 | 23 |
| `AGE` | 6 | 23 |
| `AMOUNT` | 6 | 23 |
| `BIC` | 6 | 23 |
| `JOBTYPE` | 6 | 23 |
| `BITCOINADDRESS` | 5 | 23 |
| `CREDITCARDNUMBER` | 5 | 23 |

**sola-yaml-sections**

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

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 7 | 23 |
| `ACCOUNTNAME` | 6 | 23 |
| `AGE` | 6 | 23 |
| `AMOUNT` | 6 | 23 |
| `BIC` | 6 | 23 |
| `JOBTYPE` | 6 | 23 |
| `BITCOINADDRESS` | 5 | 23 |
| `CREDITCARDNUMBER` | 5 | 23 |

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
