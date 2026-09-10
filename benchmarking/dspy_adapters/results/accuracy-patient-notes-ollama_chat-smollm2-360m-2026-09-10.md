```
arm: accuracy
generated: 2026-09-10T11:47:32.001972+00:00
command: python -m benchmarking.dspy_adapters --accuracy --corpus patient-notes --cases 30
git_head: 867819b
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/smollm2:360m
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False, 'corpus': 'thedataquarry/structured-outputs@47924a59d8b5b4eb19426c3e0defe4e33ff4f861', 'cases': 30}
supports_response_schema: False
supports_function_calling: False
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

field accuracy is micro-averaged over the **expected** fields of every case: a cell that raised scores 0 against its full denominator rather than being excluded, and emitting fewer fields can never raise the score. Fields the model invents are reported under `spurious` and break `exact`, but do not enter the denominator. Ground truth is the label shipped with the third-party `patient-notes` corpus (Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own signature, not one written by this package's authors.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Extraction accuracy — aggregate

| adapter | cases | field accuracy | exact records | ok | parse | validation | empty | transport | format | median wall_s | median total_tokens | response_format |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 0.000 | 0/30 (0.00) | 0 | 30 | 0 | 0 | 0 | 0 | 4.933 | 2851.0 | none |
| json | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 30 | 0 | 0 | 0 | 5.167 | 3325.0 | json_object |
| baml | 30 | 0.000 | 0/30 (0.00) | 0 | 0 | 0 | 0 | 0 | 30 | 0.012 | — | none |
| sola-json-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 5 | 25 | 0 | 0 | 0 | 5.349 | 2784.0 | json_object |
| sola-jsonish-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 2 | 28 | 0 | 0 | 0 | 2.226 | 1400.5 | json_object |
| sola-yaml-sections | 30 | 0.000 | 0/30 (0.00) | 0 | 27 | 3 | 0 | 0 | 0 | 5.139 | 1894.0 | none |
| sola-jsonish-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 2 | 28 | 0 | 0 | 0 | 2.251 | 1400.5 | json_object |
| sola-yaml-rescue | 30 | 0.000 | 0/30 (0.00) | 0 | 27 | 3 | 0 | 0 | 0 | 5.062 | 1894.0 | none |

**All-null floor: 0.356** — the field accuracy of a reply that extracts nothing (every output field `None`) on these same cases. A correct `None` counts as a match, so read every score above as a distance from this.

## Most-missed fields

**chat**

| field | wrong | missing |
|---|---|---|
| `name.given[]` | 0 | 55 |
| `age` | 0 | 30 |
| `birthDate` | 0 | 30 |
| `email` | 0 | 30 |
| `gender` | 0 | 30 |
| `maritalStatus` | 0 | 30 |
| `name.family` | 0 | 30 |
| `name.prefix` | 0 | 30 |

**json**

| field | wrong | missing |
|---|---|---|
| `name.given[]` | 0 | 55 |
| `age` | 0 | 30 |
| `birthDate` | 0 | 30 |
| `email` | 0 | 30 |
| `gender` | 0 | 30 |
| `maritalStatus` | 0 | 30 |
| `name.family` | 0 | 30 |
| `name.prefix` | 0 | 30 |

**baml**

| field | wrong | missing |
|---|---|---|
| `name.given[]` | 0 | 55 |
| `age` | 0 | 30 |
| `birthDate` | 0 | 30 |
| `email` | 0 | 30 |
| `gender` | 0 | 30 |
| `maritalStatus` | 0 | 30 |
| `name.family` | 0 | 30 |
| `name.prefix` | 0 | 30 |

**sola-json-sections**

| field | wrong | missing |
|---|---|---|
| `name.given[]` | 0 | 55 |
| `age` | 0 | 30 |
| `birthDate` | 0 | 30 |
| `email` | 0 | 30 |
| `gender` | 0 | 30 |
| `maritalStatus` | 0 | 30 |
| `name.family` | 0 | 30 |
| `name.prefix` | 0 | 30 |

**sola-jsonish-sections**

| field | wrong | missing |
|---|---|---|
| `name.given[]` | 0 | 55 |
| `age` | 0 | 30 |
| `birthDate` | 0 | 30 |
| `email` | 0 | 30 |
| `gender` | 0 | 30 |
| `maritalStatus` | 0 | 30 |
| `name.family` | 0 | 30 |
| `name.prefix` | 0 | 30 |

**sola-yaml-sections**

| field | wrong | missing |
|---|---|---|
| `name.given[]` | 0 | 55 |
| `age` | 0 | 30 |
| `birthDate` | 0 | 30 |
| `email` | 0 | 30 |
| `gender` | 0 | 30 |
| `maritalStatus` | 0 | 30 |
| `name.family` | 0 | 30 |
| `name.prefix` | 0 | 30 |

**sola-jsonish-rescue**

| field | wrong | missing |
|---|---|---|
| `name.given[]` | 0 | 55 |
| `age` | 0 | 30 |
| `birthDate` | 0 | 30 |
| `email` | 0 | 30 |
| `gender` | 0 | 30 |
| `maritalStatus` | 0 | 30 |
| `name.family` | 0 | 30 |
| `name.prefix` | 0 | 30 |

**sola-yaml-rescue**

| field | wrong | missing |
|---|---|---|
| `name.given[]` | 0 | 55 |
| `age` | 0 | 30 |
| `birthDate` | 0 | 30 |
| `email` | 0 | 30 |
| `gender` | 0 | 30 |
| `maritalStatus` | 0 | 30 |
| `name.family` | 0 | 30 |
| `name.prefix` | 0 | 30 |

Per-case detail is in the companion `.csv`; it is not duplicated here.
