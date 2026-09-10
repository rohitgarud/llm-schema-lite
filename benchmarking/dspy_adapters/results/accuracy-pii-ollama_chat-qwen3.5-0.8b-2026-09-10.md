```
arm: accuracy
generated: 2026-09-10T07:38:48.307170+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus pii --cases 30
git_head: 867819b
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/qwen3.5:0.8b
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
| chat | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 3.708 | 2642.5 | none |
| json | 30 | 0.907 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 3.945 | 2927.0 | json_object |
| baml | 30 | 0.833 | 0/30 (0.00) | 27 | 0 | 3 | 0 | 0 | 0 | 0.642 | 852.5 | json_object |
| sola-json-sections | 30 | 0.496 | 0/30 (0.00) | 16 | 7 | 7 | 0 | 0 | 0 | 3.937 | 2637.0 | json_object |
| sola-jsonish-sections | 30 | 0.864 | 0/30 (0.00) | 28 | 0 | 2 | 0 | 0 | 0 | 3.789 | 1489.5 | json_object |
| sola-yaml-sections | 30 | 0.928 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 3.772 | 1547.5 | none |
| sola-jsonish-rescue | 30 | 0.862 | 0/30 (0.00) | 28 | 0 | 2 | 0 | 0 | 0 | 3.804 | 1490.0 | json_object |
| sola-yaml-rescue | 30 | 0.929 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 3.902 | 1545.0 | none |

**All-null floor: 0.949** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so read every score above as a distance from this.

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
| `FIRSTNAME` | 11 | 0 |
| `LASTNAME` | 9 | 0 |
| `CITY` | 5 | 0 |
| `CREDITCARDNUMBER` | 5 | 0 |
| `CURRENCYCODE` | 5 | 0 |
| `ACCOUNTNUMBER` | 4 | 0 |
| `AMOUNT` | 4 | 0 |
| `BITCOINADDRESS` | 4 | 0 |

**baml**

| field | wrong | missing |
|---|---|---|
| `LASTNAME` | 10 | 3 |
| `FIRSTNAME` | 8 | 3 |
| `CITY` | 5 | 3 |
| `CURRENCYCODE` | 5 | 3 |
| `PHONEIMEI` | 5 | 3 |
| `EMAIL` | 4 | 3 |
| `IP` | 4 | 3 |
| `PHONENUMBER` | 4 | 3 |

**sola-json-sections**

| field | wrong | missing |
|---|---|---|
| `LASTNAME` | 7 | 14 |
| `FIRSTNAME` | 6 | 14 |
| `ACCOUNTNUMBER` | 3 | 14 |
| `CREDITCARDNUMBER` | 3 | 14 |
| `CURRENCY` | 3 | 14 |
| `AMOUNT` | 2 | 14 |
| `BITCOINADDRESS` | 2 | 14 |
| `CITY` | 2 | 14 |

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `LASTNAME` | 9 | 2 |
| `CURRENCYCODE` | 7 | 2 |
| `FIRSTNAME` | 7 | 2 |
| `CITY` | 5 | 2 |
| `CURRENCYNAME` | 5 | 2 |
| `JOBTITLE` | 5 | 2 |
| `CURRENCY` | 4 | 2 |
| `IPV4` | 4 | 2 |

**sola-yaml-sections**

| field | wrong | missing |
|---|---|---|
| `LASTNAME` | 10 | 0 |
| `FIRSTNAME` | 8 | 0 |
| `CURRENCYCODE` | 6 | 0 |
| `CITY` | 5 | 0 |
| `JOBTITLE` | 5 | 0 |
| `ACCOUNTNAME` | 4 | 0 |
| `CREDITCARDNUMBER` | 4 | 0 |
| `CURRENCY` | 4 | 0 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `LASTNAME` | 9 | 2 |
| `FIRSTNAME` | 7 | 2 |
| `CURRENCYCODE` | 6 | 2 |
| `CURRENCYNAME` | 6 | 2 |
| `JOBTITLE` | 6 | 2 |
| `CITY` | 5 | 2 |
| `CURRENCY` | 5 | 2 |
| `CREDITCARDNUMBER` | 4 | 2 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `LASTNAME` | 10 | 0 |
| `FIRSTNAME` | 8 | 0 |
| `CURRENCYCODE` | 6 | 0 |
| `JOBTITLE` | 5 | 0 |
| `ACCOUNTNAME` | 4 | 0 |
| `CITY` | 4 | 0 |
| `CREDITCARDNUMBER` | 4 | 0 |
| `CURRENCY` | 4 | 0 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
