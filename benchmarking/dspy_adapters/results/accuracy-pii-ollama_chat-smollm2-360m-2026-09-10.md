```
arm: accuracy
generated: 2026-09-10T08:57:22.824162+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus pii --cases 30
git_head: 867819b
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/smollm2:360m
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
| chat | 30 | 0.095 | 0/30 (0.00) | 3 | 27 | 0 | 0 | 0 | 0 | 1.899 | 2586.5 | none |
| json | 30 | 0.190 | 0/30 (0.00) | 6 | 0 | 24 | 0 | 0 | 0 | 0.223 | 2932.0 | json_object |
| baml | 30 | 0.142 | 0/30 (0.00) | 11 | 0 | 19 | 0 | 0 | 0 | 0.468 | 817.5 | json_object |
| sola-json-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 2 | 28 | 0 | 0 | 0 | 0.210 | 2284.0 | json_object |
| sola-jsonish-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 0.202 | 1111.5 | json_object |
| sola-yaml-sections | 30 | 0.443 | 0/30 (0.00) | 27 | 0 | 3 | 0 | 0 | 0 | 4.559 | 2002.0 | none |
| sola-jsonish-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 0.212 | 1111.5 | json_object |
| sola-yaml-rescue | 30 | 0.443 | 0/30 (0.00) | 27 | 0 | 3 | 0 | 0 | 0 | 4.570 | 2002.0 | none |

**All-null floor: 0.949** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so read every score above as a distance from this.

## Most-missed fields

**chat**

| field | wrong | missing |
|---|---|---|
| `AMOUNT` | 1 | 27 |
| `CURRENCY` | 1 | 27 |
| `DATE` | 1 | 27 |
| `EMAIL` | 1 | 27 |
| `FIRSTNAME` | 1 | 27 |
| `IBAN` | 1 | 27 |
| `IPV4` | 1 | 27 |
| `USERNAME` | 1 | 27 |

**json**

| field | wrong | missing |
|---|---|---|
| `AGE` | 2 | 24 |
| `IPV6` | 2 | 24 |
| `CREDITCARDISSUER` | 1 | 24 |
| `CREDITCARDNUMBER` | 1 | 24 |
| `DATE` | 1 | 24 |
| `EMAIL` | 1 | 24 |
| `ETHEREUMADDRESS` | 1 | 24 |
| `FIRSTNAME` | 1 | 24 |

**baml**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 11 | 19 |
| `AGE` | 11 | 19 |
| `ACCOUNTNAME` | 10 | 19 |
| `AMOUNT` | 10 | 19 |
| `BITCOINADDRESS` | 10 | 19 |
| `GENDER` | 10 | 19 |
| `BUILDINGNUMBER` | 9 | 19 |
| `CITY` | 9 | 19 |

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
| `FIRSTNAME` | 18 | 3 |
| `IPV4` | 16 | 3 |
| `ACCOUNTNAME` | 15 | 3 |
| `GENDER` | 15 | 3 |
| `IPV6` | 15 | 3 |
| `JOBTITLE` | 15 | 3 |
| `JOBTYPE` | 15 | 3 |
| `LASTNAME` | 15 | 3 |

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
| `FIRSTNAME` | 18 | 3 |
| `IPV4` | 16 | 3 |
| `ACCOUNTNAME` | 15 | 3 |
| `GENDER` | 15 | 3 |
| `IPV6` | 15 | 3 |
| `JOBTITLE` | 15 | 3 |
| `JOBTYPE` | 15 | 3 |
| `LASTNAME` | 15 | 3 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
