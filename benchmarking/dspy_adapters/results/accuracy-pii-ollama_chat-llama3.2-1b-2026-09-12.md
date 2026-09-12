```
arm: accuracy
generated: 2026-09-12T10:41:15.130835+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus pii --cases 30
git_head: 18b38b4
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

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. A correct `None` also counts as a match, so on a sparse corpus field accuracy pays a reply for extracting nothing. **recall (non-null gold)** is the extraction-quality headline: matches over only the fields whose gold value is not `None`, a cell that raised scoring 0 against them, so extracting nothing scores 0. **invented (null gold)** is the other half: of the fields whose gold is `None`, how many the reply filled anyway. Ground truth is the label shipped with the third-party `pii` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format | recall (non-null gold) | invented (null gold) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.159 | 0/30 (0.00) | 5 | 25 | 0 | 0 | 0 | 0 | 0.210 | 2145.5 | none | 0.000 (0/86) | 0.001 (1/1594) |
| json | 30 | 0.507 | 0/30 (0.00) | 17 | 0 | 13 | 0 | 0 | 0 | 0.399 | 2432.0 | json_object | 0.012 (1/86) | 0.033 (52/1594) |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 4.222 | 1174.0 | json_object | 0.000 (0/86) | 0.000 (0/1594) |
| sola-json-sections | 30 | 0.402 | 0/30 (0.00) | 27 | 2 | 1 | 0 | 0 | 0 | 3.828 | 2552.0 | json_object | 0.198 (17/86) | 0.487 (776/1594) |
| sola-jsonish-sections | 30 | 0.082 | 0/30 (0.00) | 7 | 23 | 0 | 0 | 0 | 0 | 4.586 | 1446.0 | json_object | 0.070 (6/86) | 0.152 (242/1594) |
| sola-yaml-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 0.217 | 997.5 | none | 0.000 (0/86) | 0.000 (0/1594) |
| sola-jsonish-rescue | 30 | 0.340 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 4.615 | 1446.0 | json_object | 0.372 (32/86) | 0.662 (1055/1594) |
| sola-yaml-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 0.221 | 997.5 | none | 0.000 (0/86) | 0.000 (0/1594) |

**All-null floor: 0.949** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

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
| `ACCOUNTNUMBER` | 29 | 0 |
| `ACCOUNTNAME` | 28 | 0 |
| `CREDITCARDNUMBER` | 23 | 0 |
| `AMOUNT` | 22 | 0 |
| `CURRENCY` | 22 | 0 |
| `IBAN` | 22 | 0 |
| `AGE` | 21 | 0 |
| `BIC` | 21 | 0 |

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
