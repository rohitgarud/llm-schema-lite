```
arm: live
generated: 2026-09-10T06:14:32.776767+00:00
command: python -m benchmarking.dspy_adapters --live --trials 3
git_head: daf210b
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: ollama_chat/qwen3:8b
api_base: http://localhost:11434
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7, 'think': False}
supports_response_schema: False
supports_function_calling: True
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Live outcomes — aggregate

| adapter | signature | trials | ok | parse | validation | empty | transport | format | parse rate | validation rate | median wall_s | stddev wall_s | median total_tokens | response_format |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.198 | 4.788 | 225 | none |
| chat | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.210 | 0.175 | 372 | none |
| chat | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.134 | 0.111 | 395 | none |
| chat | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.485 | 0.065 | 259 | none |
| chat | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.470 | 0.055 | 246 | none |
| chat | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.369 | 0.054 | 312 | none |
| json | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.566 | 0.063 | 211 | json_object |
| json | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.209 | 0.079 | 396 | json_object |
| json | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.111 | 0.095 | 415 | json_object |
| json | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.404 | 0.070 | 248 | json_object |
| json | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.257 | 0.082 | 275 | json_object |
| json | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.316 | 0.062 | 320 | json_object |
| baml | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.632 | 0.047 | 204 | json_object |
| baml | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.804 | 0.155 | 239 | json_object |
| baml | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.564 | 0.172 | 248 | json_object |
| baml | enum | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 1.00 | 0.00 | 0.559 | 0.086 | 217 | json_object |
| baml | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.543 | 0.087 | 208 | json_object |
| baml | recursive | 3 | 0 | 0 | 0 | 0 | 0 | 3 | — | — | 0.022 | 0.004 | — | none |
| sola-json-sections | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.659 | 0.103 | 203 | json_object |
| sola-json-sections | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.350 | 0.171 | 367 | json_object |
| sola-json-sections | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.410 | 0.103 | 394 | json_object |
| sola-json-sections | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.412 | 0.058 | 241 | json_object |
| sola-json-sections | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.410 | 0.051 | 227 | json_object |
| sola-json-sections | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.297 | 0.046 | 298 | json_object |
| sola-jsonish-sections | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.541 | 0.027 | 203 | json_object |
| sola-jsonish-sections | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.214 | 0.044 | 276 | json_object |
| sola-jsonish-sections | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.117 | 0.051 | 264 | json_object |
| sola-jsonish-sections | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.406 | 0.030 | 241 | json_object |
| sola-jsonish-sections | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.392 | 0.044 | 219 | json_object |
| sola-jsonish-sections | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.301 | 0.051 | 263 | json_object |
| sola-yaml-sections | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.295 | 0.043 | 187 | none |
| sola-yaml-sections | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.793 | 0.048 | 247 | none |
| sola-yaml-sections | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.658 | 0.058 | 233 | none |
| sola-yaml-sections | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.329 | 0.042 | 227 | none |
| sola-yaml-sections | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.313 | 0.093 | 202 | none |
| sola-yaml-sections | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.109 | 0.159 | 230 | none |
| sola-jsonish-rescue | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.563 | 0.073 | 200 | json_object |
| sola-jsonish-rescue | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.407 | 0.037 | 276 | json_object |
| sola-jsonish-rescue | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.279 | 0.035 | 264 | json_object |
| sola-jsonish-rescue | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.445 | 0.101 | 241 | json_object |
| sola-jsonish-rescue | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.450 | 0.115 | 219 | json_object |
| sola-jsonish-rescue | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.291 | 0.085 | 263 | json_object |
| sola-yaml-rescue | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.297 | 0.031 | 187 | none |
| sola-yaml-rescue | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.804 | 0.015 | 247 | none |
| sola-yaml-rescue | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.647 | 0.023 | 233 | none |
| sola-yaml-rescue | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.198 | 0.050 | 227 | none |
| sola-yaml-rescue | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.230 | 0.034 | 202 | none |
| sola-yaml-rescue | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.726 | 0.009 | 230 | none |

## Live outcomes — detail

| adapter | signature | trial | outcome | error class | wall_s | lm_calls | response_format |
|---|---|---|---|---|---|---|---|
| chat | flat | 1 | ok |  | 9.353 | 1 | none |
| chat | flat | 2 | ok |  | 1.198 | 1 | none |
| chat | flat | 3 | ok |  | 0.929 | 1 | none |
| chat | nested | 1 | ok |  | 1.505 | 1 | none |
| chat | nested | 2 | ok |  | 1.193 | 1 | none |
| chat | nested | 3 | ok |  | 1.210 | 1 | none |
| chat | list_of_model | 1 | ok |  | 1.324 | 1 | none |
| chat | list_of_model | 2 | ok |  | 1.134 | 1 | none |
| chat | list_of_model | 3 | ok |  | 1.130 | 1 | none |
| chat | enum | 1 | ok |  | 0.596 | 1 | none |
| chat | enum | 2 | ok |  | 0.482 | 1 | none |
| chat | enum | 3 | ok |  | 0.485 | 1 | none |
| chat | optional | 1 | ok |  | 0.564 | 1 | none |
| chat | optional | 2 | ok |  | 0.470 | 1 | none |
| chat | optional | 3 | ok |  | 0.467 | 1 | none |
| chat | recursive | 1 | ok |  | 1.463 | 1 | none |
| chat | recursive | 2 | ok |  | 1.369 | 1 | none |
| chat | recursive | 3 | ok |  | 1.369 | 1 | none |
| json | flat | 1 | ok |  | 0.672 | 1 | json_object |
| json | flat | 2 | ok |  | 0.562 | 1 | json_object |
| json | flat | 3 | ok |  | 0.566 | 1 | json_object |
| json | nested | 1 | ok |  | 1.343 | 1 | json_object |
| json | nested | 2 | ok |  | 1.203 | 1 | json_object |
| json | nested | 3 | ok |  | 1.209 | 1 | json_object |
| json | list_of_model | 1 | ok |  | 1.275 | 1 | json_object |
| json | list_of_model | 2 | ok |  | 1.109 | 1 | json_object |
| json | list_of_model | 3 | ok |  | 1.111 | 1 | json_object |
| json | enum | 1 | ok |  | 0.524 | 1 | json_object |
| json | enum | 2 | ok |  | 0.401 | 1 | json_object |
| json | enum | 3 | ok |  | 0.404 | 1 | json_object |
| json | optional | 1 | ok |  | 1.395 | 1 | json_object |
| json | optional | 2 | ok |  | 1.257 | 1 | json_object |
| json | optional | 3 | ok |  | 1.250 | 1 | json_object |
| json | recursive | 1 | ok |  | 1.413 | 1 | json_object |
| json | recursive | 2 | ok |  | 1.316 | 1 | json_object |
| json | recursive | 3 | ok |  | 1.297 | 1 | json_object |
| baml | flat | 1 | ok |  | 0.712 | 1 | json_object |
| baml | flat | 2 | ok |  | 0.627 | 1 | json_object |
| baml | flat | 3 | ok |  | 0.632 | 1 | json_object |
| baml | nested | 1 | ok |  | 1.804 | 1 | json_object |
| baml | nested | 2 | ok |  | 1.831 | 1 | json_object |
| baml | nested | 3 | ok |  | 1.550 | 1 | json_object |
| baml | list_of_model | 1 | ok |  | 1.858 | 1 | json_object |
| baml | list_of_model | 2 | ok |  | 1.564 | 1 | json_object |
| baml | list_of_model | 3 | ok |  | 1.556 | 1 | json_object |
| baml | enum | 1 | validation_error | ValueError | 0.658 | 1 | json_object |
| baml | enum | 2 | validation_error | ValueError | 0.559 | 1 | json_object |
| baml | enum | 3 | validation_error | ValueError | 0.486 | 1 | json_object |
| baml | optional | 1 | ok |  | 0.679 | 1 | json_object |
| baml | optional | 2 | ok |  | 0.543 | 1 | json_object |
| baml | optional | 3 | ok |  | 0.518 | 1 | json_object |
| baml | recursive | 1 | format_error | ValueError | 0.027 | 0 | none |
| baml | recursive | 2 | format_error | ValueError | 0.022 | 0 | none |
| baml | recursive | 3 | format_error | ValueError | 0.020 | 0 | none |
| sola-json-sections | flat | 1 | ok |  | 0.802 | 1 | json_object |
| sola-json-sections | flat | 2 | ok |  | 0.659 | 1 | json_object |
| sola-json-sections | flat | 3 | ok |  | 0.602 | 1 | json_object |
| sola-json-sections | nested | 1 | ok |  | 1.626 | 1 | json_object |
| sola-json-sections | nested | 2 | ok |  | 1.350 | 1 | json_object |
| sola-json-sections | nested | 3 | ok |  | 1.313 | 1 | json_object |
| sola-json-sections | list_of_model | 1 | ok |  | 1.582 | 1 | json_object |
| sola-json-sections | list_of_model | 2 | ok |  | 1.397 | 1 | json_object |
| sola-json-sections | list_of_model | 3 | ok |  | 1.410 | 1 | json_object |
| sola-json-sections | enum | 1 | ok |  | 0.509 | 1 | json_object |
| sola-json-sections | enum | 2 | ok |  | 0.405 | 1 | json_object |
| sola-json-sections | enum | 3 | ok |  | 0.412 | 1 | json_object |
| sola-json-sections | optional | 1 | ok |  | 0.494 | 1 | json_object |
| sola-json-sections | optional | 2 | ok |  | 0.410 | 1 | json_object |
| sola-json-sections | optional | 3 | ok |  | 0.403 | 1 | json_object |
| sola-json-sections | recursive | 1 | ok |  | 1.376 | 1 | json_object |
| sola-json-sections | recursive | 2 | ok |  | 1.297 | 1 | json_object |
| sola-json-sections | recursive | 3 | ok |  | 1.296 | 1 | json_object |
| sola-jsonish-sections | flat | 1 | ok |  | 0.586 | 1 | json_object |
| sola-jsonish-sections | flat | 2 | ok |  | 0.539 | 1 | json_object |
| sola-jsonish-sections | flat | 3 | ok |  | 0.541 | 1 | json_object |
| sola-jsonish-sections | nested | 1 | ok |  | 1.288 | 1 | json_object |
| sola-jsonish-sections | nested | 2 | ok |  | 1.214 | 1 | json_object |
| sola-jsonish-sections | nested | 3 | ok |  | 1.210 | 1 | json_object |
| sola-jsonish-sections | list_of_model | 1 | ok |  | 1.199 | 1 | json_object |
| sola-jsonish-sections | list_of_model | 2 | ok |  | 1.117 | 1 | json_object |
| sola-jsonish-sections | list_of_model | 3 | ok |  | 1.106 | 1 | json_object |
| sola-jsonish-sections | enum | 1 | ok |  | 0.456 | 1 | json_object |
| sola-jsonish-sections | enum | 2 | ok |  | 0.406 | 1 | json_object |
| sola-jsonish-sections | enum | 3 | ok |  | 0.403 | 1 | json_object |
| sola-jsonish-sections | optional | 1 | ok |  | 0.465 | 1 | json_object |
| sola-jsonish-sections | optional | 2 | ok |  | 0.385 | 1 | json_object |
| sola-jsonish-sections | optional | 3 | ok |  | 0.392 | 1 | json_object |
| sola-jsonish-sections | recursive | 1 | ok |  | 1.386 | 1 | json_object |
| sola-jsonish-sections | recursive | 2 | ok |  | 1.295 | 1 | json_object |
| sola-jsonish-sections | recursive | 3 | ok |  | 1.301 | 1 | json_object |
| sola-yaml-sections | flat | 1 | ok |  | 0.367 | 1 | none |
| sola-yaml-sections | flat | 2 | ok |  | 0.291 | 1 | none |
| sola-yaml-sections | flat | 3 | ok |  | 0.295 | 1 | none |
| sola-yaml-sections | nested | 1 | ok |  | 0.874 | 1 | none |
| sola-yaml-sections | nested | 2 | ok |  | 0.789 | 1 | none |
| sola-yaml-sections | nested | 3 | ok |  | 0.793 | 1 | none |
| sola-yaml-sections | list_of_model | 1 | ok |  | 0.755 | 1 | none |
| sola-yaml-sections | list_of_model | 2 | ok |  | 0.653 | 1 | none |
| sola-yaml-sections | list_of_model | 3 | ok |  | 0.658 | 1 | none |
| sola-yaml-sections | enum | 1 | ok |  | 0.338 | 1 | none |
| sola-yaml-sections | enum | 2 | ok |  | 0.329 | 1 | none |
| sola-yaml-sections | enum | 3 | ok |  | 0.262 | 1 | none |
| sola-yaml-sections | optional | 1 | ok |  | 0.466 | 1 | none |
| sola-yaml-sections | optional | 2 | ok |  | 0.297 | 1 | none |
| sola-yaml-sections | optional | 3 | ok |  | 0.313 | 1 | none |
| sola-yaml-sections | recursive | 1 | ok |  | 1.290 | 1 | none |
| sola-yaml-sections | recursive | 2 | ok |  | 1.109 | 1 | none |
| sola-yaml-sections | recursive | 3 | ok |  | 0.974 | 1 | none |
| sola-jsonish-rescue | flat | 1 | ok |  | 0.656 | 1 | json_object |
| sola-jsonish-rescue | flat | 2 | ok |  | 0.511 | 1 | json_object |
| sola-jsonish-rescue | flat | 3 | ok |  | 0.563 | 1 | json_object |
| sola-jsonish-rescue | nested | 1 | ok |  | 1.450 | 1 | json_object |
| sola-jsonish-rescue | nested | 2 | ok |  | 1.407 | 1 | json_object |
| sola-jsonish-rescue | nested | 3 | ok |  | 1.376 | 1 | json_object |
| sola-jsonish-rescue | list_of_model | 1 | ok |  | 1.279 | 1 | json_object |
| sola-jsonish-rescue | list_of_model | 2 | ok |  | 1.268 | 1 | json_object |
| sola-jsonish-rescue | list_of_model | 3 | ok |  | 1.334 | 1 | json_object |
| sola-jsonish-rescue | enum | 1 | ok |  | 0.616 | 1 | json_object |
| sola-jsonish-rescue | enum | 2 | ok |  | 0.437 | 1 | json_object |
| sola-jsonish-rescue | enum | 3 | ok |  | 0.445 | 1 | json_object |
| sola-jsonish-rescue | optional | 1 | ok |  | 0.636 | 1 | json_object |
| sola-jsonish-rescue | optional | 2 | ok |  | 0.425 | 1 | json_object |
| sola-jsonish-rescue | optional | 3 | ok |  | 0.450 | 1 | json_object |
| sola-jsonish-rescue | recursive | 1 | ok |  | 1.435 | 1 | json_object |
| sola-jsonish-rescue | recursive | 2 | ok |  | 1.291 | 1 | json_object |
| sola-jsonish-rescue | recursive | 3 | ok |  | 1.283 | 1 | json_object |
| sola-yaml-rescue | flat | 1 | ok |  | 0.346 | 1 | none |
| sola-yaml-rescue | flat | 2 | ok |  | 0.289 | 1 | none |
| sola-yaml-rescue | flat | 3 | ok |  | 0.297 | 1 | none |
| sola-yaml-rescue | nested | 1 | ok |  | 0.815 | 1 | none |
| sola-yaml-rescue | nested | 2 | ok |  | 0.804 | 1 | none |
| sola-yaml-rescue | nested | 3 | ok |  | 0.786 | 1 | none |
| sola-yaml-rescue | list_of_model | 1 | ok |  | 0.686 | 1 | none |
| sola-yaml-rescue | list_of_model | 2 | ok |  | 0.647 | 1 | none |
| sola-yaml-rescue | list_of_model | 3 | ok |  | 0.647 | 1 | none |
| sola-yaml-rescue | enum | 1 | ok |  | 0.284 | 1 | none |
| sola-yaml-rescue | enum | 2 | ok |  | 0.198 | 1 | none |
| sola-yaml-rescue | enum | 3 | ok |  | 0.198 | 1 | none |
| sola-yaml-rescue | optional | 1 | ok |  | 0.285 | 1 | none |
| sola-yaml-rescue | optional | 2 | ok |  | 0.224 | 1 | none |
| sola-yaml-rescue | optional | 3 | ok |  | 0.230 | 1 | none |
| sola-yaml-rescue | recursive | 1 | ok |  | 0.739 | 1 | none |
| sola-yaml-rescue | recursive | 2 | ok |  | 0.723 | 1 | none |
| sola-yaml-rescue | recursive | 3 | ok |  | 0.726 | 1 | none |

### Provider accounting — NOT comparable across the `response_format` groups

| adapter | signature | response_format | prompt_tokens_reported | completion_tokens_reported | total_tokens |
|---|---|---|---|---|---|
| chat | flat | none | 195 | 30 | 225 |
| chat | flat | none | 195 | 30 | 225 |
| chat | flat | none | 195 | 30 | 225 |
| chat | nested | none | 326 | 46 | 372 |
| chat | nested | none | 326 | 46 | 372 |
| chat | nested | none | 326 | 46 | 372 |
| chat | list_of_model | none | 346 | 49 | 395 |
| chat | list_of_model | none | 346 | 49 | 395 |
| chat | list_of_model | none | 346 | 49 | 395 |
| chat | enum | none | 239 | 20 | 259 |
| chat | enum | none | 239 | 20 | 259 |
| chat | enum | none | 239 | 20 | 259 |
| chat | optional | none | 226 | 20 | 246 |
| chat | optional | none | 226 | 20 | 246 |
| chat | optional | none | 226 | 20 | 246 |
| chat | recursive | none | 253 | 59 | 312 |
| chat | recursive | none | 253 | 59 | 312 |
| chat | recursive | none | 253 | 59 | 312 |
| json | flat | json_object | 187 | 24 | 211 |
| json | flat | json_object | 187 | 24 | 211 |
| json | flat | json_object | 187 | 24 | 211 |
| json | nested | json_object | 344 | 52 | 396 |
| json | nested | json_object | 344 | 52 | 396 |
| json | nested | json_object | 344 | 52 | 396 |
| json | list_of_model | json_object | 367 | 48 | 415 |
| json | list_of_model | json_object | 367 | 48 | 415 |
| json | list_of_model | json_object | 367 | 48 | 415 |
| json | enum | json_object | 231 | 17 | 248 |
| json | enum | json_object | 231 | 17 | 248 |
| json | enum | json_object | 231 | 17 | 248 |
| json | optional | json_object | 222 | 53 | 275 |
| json | optional | json_object | 222 | 53 | 275 |
| json | optional | json_object | 222 | 53 | 275 |
| json | recursive | json_object | 264 | 56 | 320 |
| json | recursive | json_object | 264 | 56 | 320 |
| json | recursive | json_object | 264 | 56 | 320 |
| baml | flat | json_object | 177 | 27 | 204 |
| baml | flat | json_object | 177 | 27 | 204 |
| baml | flat | json_object | 177 | 27 | 204 |
| baml | nested | json_object | 187 | 52 | 239 |
| baml | nested | json_object | 187 | 52 | 239 |
| baml | nested | json_object | 187 | 52 | 239 |
| baml | list_of_model | json_object | 188 | 60 | 248 |
| baml | list_of_model | json_object | 188 | 60 | 248 |
| baml | list_of_model | json_object | 188 | 60 | 248 |
| baml | enum | json_object | 199 | 18 | 217 |
| baml | enum | json_object | 199 | 18 | 217 |
| baml | enum | json_object | 199 | 18 | 217 |
| baml | optional | json_object | 192 | 16 | 208 |
| baml | optional | json_object | 192 | 16 | 208 |
| baml | optional | json_object | 192 | 16 | 208 |
| sola-json-sections | flat | json_object | 180 | 23 | 203 |
| sola-json-sections | flat | json_object | 180 | 23 | 203 |
| sola-json-sections | flat | json_object | 180 | 23 | 203 |
| sola-json-sections | nested | json_object | 315 | 52 | 367 |
| sola-json-sections | nested | json_object | 315 | 52 | 367 |
| sola-json-sections | nested | json_object | 315 | 52 | 367 |
| sola-json-sections | list_of_model | json_object | 334 | 60 | 394 |
| sola-json-sections | list_of_model | json_object | 334 | 60 | 394 |
| sola-json-sections | list_of_model | json_object | 334 | 60 | 394 |
| sola-json-sections | enum | json_object | 224 | 17 | 241 |
| sola-json-sections | enum | json_object | 224 | 17 | 241 |
| sola-json-sections | enum | json_object | 224 | 17 | 241 |
| sola-json-sections | optional | json_object | 210 | 17 | 227 |
| sola-json-sections | optional | json_object | 210 | 17 | 227 |
| sola-json-sections | optional | json_object | 210 | 17 | 227 |
| sola-json-sections | recursive | json_object | 242 | 56 | 298 |
| sola-json-sections | recursive | json_object | 242 | 56 | 298 |
| sola-json-sections | recursive | json_object | 242 | 56 | 298 |
| sola-jsonish-sections | flat | json_object | 180 | 23 | 203 |
| sola-jsonish-sections | flat | json_object | 180 | 23 | 203 |
| sola-jsonish-sections | flat | json_object | 180 | 23 | 203 |
| sola-jsonish-sections | nested | json_object | 224 | 52 | 276 |
| sola-jsonish-sections | nested | json_object | 224 | 52 | 276 |
| sola-jsonish-sections | nested | json_object | 224 | 52 | 276 |
| sola-jsonish-sections | list_of_model | json_object | 216 | 48 | 264 |
| sola-jsonish-sections | list_of_model | json_object | 216 | 48 | 264 |
| sola-jsonish-sections | list_of_model | json_object | 216 | 48 | 264 |
| sola-jsonish-sections | enum | json_object | 224 | 17 | 241 |
| sola-jsonish-sections | enum | json_object | 224 | 17 | 241 |
| sola-jsonish-sections | enum | json_object | 224 | 17 | 241 |
| sola-jsonish-sections | optional | json_object | 203 | 16 | 219 |
| sola-jsonish-sections | optional | json_object | 203 | 16 | 219 |
| sola-jsonish-sections | optional | json_object | 203 | 16 | 219 |
| sola-jsonish-sections | recursive | json_object | 207 | 56 | 263 |
| sola-jsonish-sections | recursive | json_object | 207 | 56 | 263 |
| sola-jsonish-sections | recursive | json_object | 207 | 56 | 263 |
| sola-yaml-sections | flat | none | 175 | 12 | 187 |
| sola-yaml-sections | flat | none | 175 | 12 | 187 |
| sola-yaml-sections | flat | none | 175 | 12 | 187 |
| sola-yaml-sections | nested | none | 213 | 34 | 247 |
| sola-yaml-sections | nested | none | 213 | 34 | 247 |
| sola-yaml-sections | nested | none | 213 | 34 | 247 |
| sola-yaml-sections | list_of_model | none | 205 | 28 | 233 |
| sola-yaml-sections | list_of_model | none | 205 | 28 | 233 |
| sola-yaml-sections | list_of_model | none | 205 | 28 | 233 |
| sola-yaml-sections | enum | none | 219 | 8 | 227 |
| sola-yaml-sections | enum | none | 219 | 8 | 227 |
| sola-yaml-sections | enum | none | 219 | 8 | 227 |
| sola-yaml-sections | optional | none | 193 | 9 | 202 |
| sola-yaml-sections | optional | none | 193 | 9 | 202 |
| sola-yaml-sections | optional | none | 193 | 9 | 202 |
| sola-yaml-sections | recursive | none | 199 | 31 | 230 |
| sola-yaml-sections | recursive | none | 199 | 31 | 230 |
| sola-yaml-sections | recursive | none | 199 | 31 | 230 |
| sola-jsonish-rescue | flat | json_object | 180 | 20 | 200 |
| sola-jsonish-rescue | flat | json_object | 180 | 20 | 200 |
| sola-jsonish-rescue | flat | json_object | 180 | 20 | 200 |
| sola-jsonish-rescue | nested | json_object | 224 | 52 | 276 |
| sola-jsonish-rescue | nested | json_object | 224 | 52 | 276 |
| sola-jsonish-rescue | nested | json_object | 224 | 52 | 276 |
| sola-jsonish-rescue | list_of_model | json_object | 216 | 48 | 264 |
| sola-jsonish-rescue | list_of_model | json_object | 216 | 48 | 264 |
| sola-jsonish-rescue | list_of_model | json_object | 216 | 48 | 264 |
| sola-jsonish-rescue | enum | json_object | 224 | 17 | 241 |
| sola-jsonish-rescue | enum | json_object | 224 | 17 | 241 |
| sola-jsonish-rescue | enum | json_object | 224 | 17 | 241 |
| sola-jsonish-rescue | optional | json_object | 203 | 16 | 219 |
| sola-jsonish-rescue | optional | json_object | 203 | 16 | 219 |
| sola-jsonish-rescue | optional | json_object | 203 | 16 | 219 |
| sola-jsonish-rescue | recursive | json_object | 207 | 56 | 263 |
| sola-jsonish-rescue | recursive | json_object | 207 | 56 | 263 |
| sola-jsonish-rescue | recursive | json_object | 207 | 56 | 263 |
| sola-yaml-rescue | flat | none | 175 | 12 | 187 |
| sola-yaml-rescue | flat | none | 175 | 12 | 187 |
| sola-yaml-rescue | flat | none | 175 | 12 | 187 |
| sola-yaml-rescue | nested | none | 213 | 34 | 247 |
| sola-yaml-rescue | nested | none | 213 | 34 | 247 |
| sola-yaml-rescue | nested | none | 213 | 34 | 247 |
| sola-yaml-rescue | list_of_model | none | 205 | 28 | 233 |
| sola-yaml-rescue | list_of_model | none | 205 | 28 | 233 |
| sola-yaml-rescue | list_of_model | none | 205 | 28 | 233 |
| sola-yaml-rescue | enum | none | 219 | 8 | 227 |
| sola-yaml-rescue | enum | none | 219 | 8 | 227 |
| sola-yaml-rescue | enum | none | 219 | 8 | 227 |
| sola-yaml-rescue | optional | none | 193 | 9 | 202 |
| sola-yaml-rescue | optional | none | 193 | 9 | 202 |
| sola-yaml-rescue | optional | none | 193 | 9 | 202 |
| sola-yaml-rescue | recursive | none | 199 | 31 | 230 |
| sola-yaml-rescue | recursive | none | 199 | 31 | 230 |
| sola-yaml-rescue | recursive | none | 199 | 31 | 230 |
