```
arm: accuracy
generated: 2026-09-12T10:30:00.490144+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus pii --cases 30
git_head: 18b38b4
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/granite3.1-moe:1b
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
| chat | 30 | 0.256 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 2.951 | 2678.0 | none | 0.233 (20/86) | 0.743 (1184/1594) |
| json | 30 | 0.767 | 0/30 (0.00) | 29 | 0 | 1 | 0 | 0 | 0 | 2.338 | 2889.5 | json_object | 0.128 (11/86) | 0.164 (262/1594) |
| baml | 30 | 0.696 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 2.458 | 1243.0 | json_object | 0.163 (14/86) | 0.275 (439/1594) |
| sola-json-sections | 30 | 0.575 | 0/30 (0.00) | 21 | 9 | 0 | 0 | 0 | 0 | 2.392 | 2537.5 | json_object | 0.070 (6/86) | 0.098 (157/1594) |
| sola-jsonish-sections | 30 | 0.710 | 0/30 (0.00) | 26 | 0 | 4 | 0 | 0 | 0 | 2.350 | 1484.5 | json_object | 0.151 (13/86) | 0.124 (198/1594) |
| sola-yaml-sections | 30 | 0.289 | 0/30 (0.00) | 12 | 0 | 18 | 0 | 0 | 0 | 1.784 | 1431.5 | none | 0.081 (7/86) | 0.099 (158/1594) |
| sola-jsonish-rescue | 30 | 0.710 | 0/30 (0.00) | 26 | 0 | 4 | 0 | 0 | 0 | 2.308 | 1484.5 | json_object | 0.151 (13/86) | 0.124 (198/1594) |
| sola-yaml-rescue | 30 | 0.289 | 0/30 (0.00) | 12 | 0 | 18 | 0 | 0 | 0 | 1.755 | 1431.5 | none | 0.081 (7/86) | 0.099 (158/1594) |

**All-null floor: 0.949** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

**chat**

| field | wrong | missing |
|---|---|---|
| `AGE` | 27 | 0 |
| `CITY` | 26 | 0 |
| `DOB` | 26 | 0 |
| `EYECOLOR` | 25 | 0 |
| `ACCOUNTNAME` | 24 | 0 |
| `CURRENCY` | 24 | 0 |
| `CURRENCYNAME` | 24 | 0 |
| `DATE` | 24 | 0 |

**json**

| field | wrong | missing |
|---|---|---|
| `FIRSTNAME` | 14 | 1 |
| `ACCOUNTNUMBER` | 11 | 1 |
| `AGE` | 11 | 1 |
| `CURRENCY` | 10 | 1 |
| `CURRENCYNAME` | 10 | 1 |
| `CITY` | 9 | 1 |
| `DOB` | 9 | 1 |
| `LASTNAME` | 9 | 1 |

**baml**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 20 | 0 |
| `CITY` | 19 | 0 |
| `CURRENCY` | 19 | 0 |
| `DATE` | 19 | 0 |
| `FIRSTNAME` | 19 | 0 |
| `CURRENCYNAME` | 18 | 0 |
| `CURRENCYSYMBOL` | 18 | 0 |
| `DOB` | 17 | 0 |

**sola-json-sections**

| field | wrong | missing |
|---|---|---|
| `FIRSTNAME` | 8 | 9 |
| `CREDITCARDISSUER` | 6 | 9 |
| `ACCOUNTNUMBER` | 5 | 9 |
| `DATE` | 5 | 9 |
| `EMAIL` | 5 | 9 |
| `IPV4` | 5 | 9 |
| `IPV6` | 5 | 9 |
| `LASTNAME` | 5 | 9 |

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 19 | 4 |
| `FIRSTNAME` | 12 | 4 |
| `CITY` | 11 | 4 |
| `DATE` | 10 | 4 |
| `CURRENCY` | 9 | 4 |
| `CURRENCYNAME` | 9 | 4 |
| `CURRENCYSYMBOL` | 9 | 4 |
| `AMOUNT` | 8 | 4 |

**sola-yaml-sections**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNAME` | 10 | 18 |
| `IPV6` | 5 | 18 |
| `JOBTITLE` | 5 | 18 |
| `MIDDLENAME` | 5 | 18 |
| `PHONEIMEI` | 5 | 18 |
| `VEHICLEVIN` | 5 | 18 |
| `ACCOUNTNUMBER` | 4 | 18 |
| `COMPANYNAME` | 4 | 18 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNUMBER` | 19 | 4 |
| `FIRSTNAME` | 12 | 4 |
| `CITY` | 11 | 4 |
| `DATE` | 10 | 4 |
| `CURRENCY` | 9 | 4 |
| `CURRENCYNAME` | 9 | 4 |
| `CURRENCYSYMBOL` | 9 | 4 |
| `AMOUNT` | 8 | 4 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `ACCOUNTNAME` | 10 | 18 |
| `IPV6` | 5 | 18 |
| `JOBTITLE` | 5 | 18 |
| `MIDDLENAME` | 5 | 18 |
| `PHONEIMEI` | 5 | 18 |
| `VEHICLEVIN` | 5 | 18 |
| `ACCOUNTNUMBER` | 4 | 18 |
| `COMPANYNAME` | 4 | 18 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
