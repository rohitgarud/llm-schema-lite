```
arm: accuracy
generated: 2026-09-13T15:42:26.128403+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus pii --cases 30 --adapters json,json-constrained,sola-jsonish-rescue,sola-yaml-rescue --out /tmp/claude-1000/-home-rohitgarud-llm-schema-lite/a9b95a98-8b4c-4773-96d3-50a0b9893834/scratchpad/constrained-qwen
git_head: 50c4c8d
dspy_version: 3.3.1
llm_schema_lite_version: 0.7.0
model: ollama_chat/qwen3.5:0.8b
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
| json | 30 | 0.904 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 3.944 | 2929.0 | json_object | 0.244 (21/86) | 0.061 (97/1594) |
| json-constrained | 30 | 0.935 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 4.563 | 2927.0 | json_schema | 0.256 (22/86) | 0.028 (45/1594) |
| sola-jsonish-rescue | 30 | 0.864 | 0/30 (0.00) | 28 | 0 | 2 | 0 | 0 | 0 | 4.130 | 1489.5 | json_object | 0.267 (23/86) | 0.036 (58/1594) |
| sola-yaml-rescue | 30 | 0.929 | 0/30 (0.00) | 30 | 0 | 0 | 0 | 0 | 0 | 4.220 | 1547.5 | none | 0.244 (21/86) | 0.035 (55/1594) |

**All-null floor: 0.949** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so field accuracy is only a distance from this floor, and an adapter near it may simply have extracted nothing. Recall (non-null gold) has a floor of 0 by construction: compare adapters on that.

## Most-missed fields

**json**

| field | wrong | missing |
|---|---|---|
| `FIRSTNAME` | 11 | 0 |
| `LASTNAME` | 10 | 0 |
| `CREDITCARDNUMBER` | 6 | 0 |
| `ACCOUNTNUMBER` | 5 | 0 |
| `CITY` | 5 | 0 |
| `CURRENCYCODE` | 5 | 0 |
| `PHONENUMBER` | 5 | 0 |
| `AMOUNT` | 4 | 0 |

**json-constrained**

| field | wrong | missing |
|---|---|---|
| `FIRSTNAME` | 10 | 0 |
| `LASTNAME` | 9 | 0 |
| `ACCOUNTNUMBER` | 4 | 0 |
| `CITY` | 4 | 0 |
| `CREDITCARDNUMBER` | 4 | 0 |
| `CURRENCYCODE` | 4 | 0 |
| `AMOUNT` | 3 | 0 |
| `BITCOINADDRESS` | 3 | 0 |

**sola-jsonish-rescue**

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
