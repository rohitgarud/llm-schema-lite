```
arm: live
generated: 2026-09-09T13:51:03.438177+00:00
command: python -m benchmarking.dspy_adapters --live --trials 3
git_head: 054da0a
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
| chat | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.750 | 6.591 | 225 | none |
| chat | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.080 | 0.085 | 372 | none |
| chat | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.145 | 0.084 | 395 | none |
| chat | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.480 | 0.064 | 259 | none |
| chat | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.488 | 0.046 | 246 | none |
| chat | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.383 | 0.060 | 312 | none |
| json | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.575 | 0.051 | 211 | json_object |
| json | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.219 | 0.078 | 396 | json_object |
| json | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.201 | 0.072 | 415 | json_object |
| json | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.417 | 0.067 | 248 | json_object |
| json | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.273 | 0.066 | 275 | json_object |
| json | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.342 | 0.046 | 320 | json_object |
| baml | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.629 | 0.051 | 204 | json_object |
| baml | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.194 | 0.047 | 239 | json_object |
| baml | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.376 | 0.057 | 248 | json_object |
| baml | enum | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 1.00 | 0.00 | 0.426 | 0.045 | 217 | json_object |
| baml | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.417 | 0.039 | 208 | json_object |
| baml | recursive | 3 | 0 | 0 | 0 | 0 | 0 | 3 | — | — | 0.008 | 0.002 | — | none |
| sola-json-sections | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.546 | 0.037 | 203 | json_object |
| sola-json-sections | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.201 | 0.075 | 367 | json_object |
| sola-json-sections | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.416 | 0.086 | 394 | json_object |
| sola-json-sections | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.404 | 0.079 | 241 | json_object |
| sola-json-sections | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.409 | 0.034 | 227 | json_object |
| sola-json-sections | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.351 | 0.043 | 298 | json_object |
| sola-jsonish-sections | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.535 | 0.043 | 203 | json_object |
| sola-jsonish-sections | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.205 | 0.033 | 276 | json_object |
| sola-jsonish-sections | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.113 | 0.075 | 264 | json_object |
| sola-jsonish-sections | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.405 | 0.032 | 241 | json_object |
| sola-jsonish-sections | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.387 | 0.027 | 219 | json_object |
| sola-jsonish-sections | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.295 | 0.041 | 263 | json_object |
| sola-yaml-sections | flat | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.497 | 0.054 | 202 | none |
| sola-yaml-sections | nested | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.818 | 0.038 | 249 | none |
| sola-yaml-sections | list_of_model | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.681 | 0.052 | 239 | none |
| sola-yaml-sections | enum | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.345 | 0.060 | 239 | none |
| sola-yaml-sections | optional | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.337 | 0.042 | 215 | none |
| sola-yaml-sections | recursive | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.694 | 0.041 | 231 | none |
| sola-jsonish-rescue | flat | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.470 | 0.036 | 200 | json_object |
| sola-jsonish-rescue | nested | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.198 | 0.013 | 276 | json_object |
| sola-jsonish-rescue | list_of_model | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.101 | 0.024 | 264 | json_object |
| sola-jsonish-rescue | enum | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.404 | 0.050 | 241 | json_object |
| sola-jsonish-rescue | optional | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 0.383 | 0.032 | 219 | json_object |
| sola-jsonish-rescue | recursive | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.282 | 0.010 | 263 | json_object |

## Live outcomes — detail

| adapter | signature | trial | outcome | error class | wall_s | lm_calls | response_format |
|---|---|---|---|---|---|---|---|
| chat | flat | 1 | ok |  | 12.144 | 1 | none |
| chat | flat | 2 | ok |  | 0.750 | 1 | none |
| chat | flat | 3 | ok |  | 0.704 | 1 | none |
| chat | nested | 1 | ok |  | 1.223 | 1 | none |
| chat | nested | 2 | ok |  | 1.080 | 1 | none |
| chat | nested | 3 | ok |  | 1.072 | 1 | none |
| chat | list_of_model | 1 | ok |  | 1.289 | 1 | none |
| chat | list_of_model | 2 | ok |  | 1.145 | 1 | none |
| chat | list_of_model | 3 | ok |  | 1.144 | 1 | none |
| chat | enum | 1 | ok |  | 0.587 | 1 | none |
| chat | enum | 2 | ok |  | 0.473 | 1 | none |
| chat | enum | 3 | ok |  | 0.480 | 1 | none |
| chat | optional | 1 | ok |  | 0.564 | 1 | none |
| chat | optional | 2 | ok |  | 0.488 | 1 | none |
| chat | optional | 3 | ok |  | 0.479 | 1 | none |
| chat | recursive | 1 | ok |  | 1.477 | 1 | none |
| chat | recursive | 2 | ok |  | 1.366 | 1 | none |
| chat | recursive | 3 | ok |  | 1.383 | 1 | none |
| json | flat | 1 | ok |  | 0.660 | 1 | json_object |
| json | flat | 2 | ok |  | 0.570 | 1 | json_object |
| json | flat | 3 | ok |  | 0.575 | 1 | json_object |
| json | nested | 1 | ok |  | 1.352 | 1 | json_object |
| json | nested | 2 | ok |  | 1.216 | 1 | json_object |
| json | nested | 3 | ok |  | 1.219 | 1 | json_object |
| json | list_of_model | 1 | ok |  | 1.299 | 1 | json_object |
| json | list_of_model | 2 | ok |  | 1.158 | 1 | json_object |
| json | list_of_model | 3 | ok |  | 1.201 | 1 | json_object |
| json | enum | 1 | ok |  | 0.527 | 1 | json_object |
| json | enum | 2 | ok |  | 0.407 | 1 | json_object |
| json | enum | 3 | ok |  | 0.417 | 1 | json_object |
| json | optional | 1 | ok |  | 1.383 | 1 | json_object |
| json | optional | 2 | ok |  | 1.264 | 1 | json_object |
| json | optional | 3 | ok |  | 1.273 | 1 | json_object |
| json | recursive | 1 | ok |  | 1.414 | 1 | json_object |
| json | recursive | 2 | ok |  | 1.329 | 1 | json_object |
| json | recursive | 3 | ok |  | 1.342 | 1 | json_object |
| baml | flat | 1 | ok |  | 0.716 | 1 | json_object |
| baml | flat | 2 | ok |  | 0.628 | 1 | json_object |
| baml | flat | 3 | ok |  | 0.629 | 1 | json_object |
| baml | nested | 1 | ok |  | 1.269 | 1 | json_object |
| baml | nested | 2 | ok |  | 1.184 | 1 | json_object |
| baml | nested | 3 | ok |  | 1.194 | 1 | json_object |
| baml | list_of_model | 1 | ok |  | 1.476 | 1 | json_object |
| baml | list_of_model | 2 | ok |  | 1.376 | 1 | json_object |
| baml | list_of_model | 3 | ok |  | 1.376 | 1 | json_object |
| baml | enum | 1 | validation_error | ValueError | 0.503 | 1 | json_object |
| baml | enum | 2 | validation_error | ValueError | 0.426 | 1 | json_object |
| baml | enum | 3 | validation_error | ValueError | 0.426 | 1 | json_object |
| baml | optional | 1 | ok |  | 0.485 | 1 | json_object |
| baml | optional | 2 | ok |  | 0.417 | 1 | json_object |
| baml | optional | 3 | ok |  | 0.416 | 1 | json_object |
| baml | recursive | 1 | format_error | ValueError | 0.012 | 0 | none |
| baml | recursive | 2 | format_error | ValueError | 0.008 | 0 | none |
| baml | recursive | 3 | format_error | ValueError | 0.008 | 0 | none |
| sola-json-sections | flat | 1 | ok |  | 0.609 | 1 | json_object |
| sola-json-sections | flat | 2 | ok |  | 0.545 | 1 | json_object |
| sola-json-sections | flat | 3 | ok |  | 0.546 | 1 | json_object |
| sola-json-sections | nested | 1 | ok |  | 1.331 | 1 | json_object |
| sola-json-sections | nested | 2 | ok |  | 1.201 | 1 | json_object |
| sola-json-sections | nested | 3 | ok |  | 1.200 | 1 | json_object |
| sola-json-sections | list_of_model | 1 | ok |  | 1.556 | 1 | json_object |
| sola-json-sections | list_of_model | 2 | ok |  | 1.399 | 1 | json_object |
| sola-json-sections | list_of_model | 3 | ok |  | 1.416 | 1 | json_object |
| sola-json-sections | enum | 1 | ok |  | 0.540 | 1 | json_object |
| sola-json-sections | enum | 2 | ok |  | 0.404 | 1 | json_object |
| sola-json-sections | enum | 3 | ok |  | 0.401 | 1 | json_object |
| sola-json-sections | optional | 1 | ok |  | 0.466 | 1 | json_object |
| sola-json-sections | optional | 2 | ok |  | 0.409 | 1 | json_object |
| sola-json-sections | optional | 3 | ok |  | 0.406 | 1 | json_object |
| sola-json-sections | recursive | 1 | ok |  | 1.380 | 1 | json_object |
| sola-json-sections | recursive | 2 | ok |  | 1.296 | 1 | json_object |
| sola-json-sections | recursive | 3 | ok |  | 1.351 | 1 | json_object |
| sola-jsonish-sections | flat | 1 | ok |  | 0.609 | 1 | json_object |
| sola-jsonish-sections | flat | 2 | ok |  | 0.535 | 1 | json_object |
| sola-jsonish-sections | flat | 3 | ok |  | 0.534 | 1 | json_object |
| sola-jsonish-sections | nested | 1 | ok |  | 1.254 | 1 | json_object |
| sola-jsonish-sections | nested | 2 | ok |  | 1.192 | 1 | json_object |
| sola-jsonish-sections | nested | 3 | ok |  | 1.205 | 1 | json_object |
| sola-jsonish-sections | list_of_model | 1 | ok |  | 1.242 | 1 | json_object |
| sola-jsonish-sections | list_of_model | 2 | ok |  | 1.113 | 1 | json_object |
| sola-jsonish-sections | list_of_model | 3 | ok |  | 1.112 | 1 | json_object |
| sola-jsonish-sections | enum | 1 | ok |  | 0.460 | 1 | json_object |
| sola-jsonish-sections | enum | 2 | ok |  | 0.405 | 1 | json_object |
| sola-jsonish-sections | enum | 3 | ok |  | 0.404 | 1 | json_object |
| sola-jsonish-sections | optional | 1 | ok |  | 0.432 | 1 | json_object |
| sola-jsonish-sections | optional | 2 | ok |  | 0.384 | 1 | json_object |
| sola-jsonish-sections | optional | 3 | ok |  | 0.387 | 1 | json_object |
| sola-jsonish-sections | recursive | 1 | ok |  | 1.364 | 1 | json_object |
| sola-jsonish-sections | recursive | 2 | ok |  | 1.292 | 1 | json_object |
| sola-jsonish-sections | recursive | 3 | ok |  | 1.295 | 1 | json_object |
| sola-yaml-sections | flat | 1 | parse_error | AdapterParseError | 0.590 | 1 | none |
| sola-yaml-sections | flat | 2 | parse_error | AdapterParseError | 0.497 | 1 | none |
| sola-yaml-sections | flat | 3 | parse_error | AdapterParseError | 0.495 | 1 | none |
| sola-yaml-sections | nested | 1 | parse_error | AdapterParseError | 0.863 | 1 | none |
| sola-yaml-sections | nested | 2 | parse_error | AdapterParseError | 0.818 | 1 | none |
| sola-yaml-sections | nested | 3 | parse_error | AdapterParseError | 0.788 | 1 | none |
| sola-yaml-sections | list_of_model | 1 | parse_error | AdapterParseError | 0.769 | 1 | none |
| sola-yaml-sections | list_of_model | 2 | parse_error | AdapterParseError | 0.678 | 1 | none |
| sola-yaml-sections | list_of_model | 3 | parse_error | AdapterParseError | 0.681 | 1 | none |
| sola-yaml-sections | enum | 1 | parse_error | AdapterParseError | 0.444 | 1 | none |
| sola-yaml-sections | enum | 2 | parse_error | AdapterParseError | 0.345 | 1 | none |
| sola-yaml-sections | enum | 3 | parse_error | AdapterParseError | 0.337 | 1 | none |
| sola-yaml-sections | optional | 1 | parse_error | AdapterParseError | 0.408 | 1 | none |
| sola-yaml-sections | optional | 2 | parse_error | AdapterParseError | 0.335 | 1 | none |
| sola-yaml-sections | optional | 3 | parse_error | AdapterParseError | 0.337 | 1 | none |
| sola-yaml-sections | recursive | 1 | parse_error | AdapterParseError | 0.764 | 1 | none |
| sola-yaml-sections | recursive | 2 | parse_error | AdapterParseError | 0.691 | 1 | none |
| sola-yaml-sections | recursive | 3 | parse_error | AdapterParseError | 0.694 | 1 | none |
| sola-jsonish-rescue | flat | 1 | ok |  | 0.531 | 1 | json_object |
| sola-jsonish-rescue | flat | 2 | ok |  | 0.470 | 1 | json_object |
| sola-jsonish-rescue | flat | 3 | ok |  | 0.468 | 1 | json_object |
| sola-jsonish-rescue | nested | 1 | ok |  | 1.219 | 1 | json_object |
| sola-jsonish-rescue | nested | 2 | ok |  | 1.196 | 1 | json_object |
| sola-jsonish-rescue | nested | 3 | ok |  | 1.198 | 1 | json_object |
| sola-jsonish-rescue | list_of_model | 1 | ok |  | 1.141 | 1 | json_object |
| sola-jsonish-rescue | list_of_model | 2 | ok |  | 1.100 | 1 | json_object |
| sola-jsonish-rescue | list_of_model | 3 | ok |  | 1.101 | 1 | json_object |
| sola-jsonish-rescue | enum | 1 | ok |  | 0.490 | 1 | json_object |
| sola-jsonish-rescue | enum | 2 | ok |  | 0.404 | 1 | json_object |
| sola-jsonish-rescue | enum | 3 | ok |  | 0.402 | 1 | json_object |
| sola-jsonish-rescue | optional | 1 | ok |  | 0.439 | 1 | json_object |
| sola-jsonish-rescue | optional | 2 | ok |  | 0.383 | 1 | json_object |
| sola-jsonish-rescue | optional | 3 | ok |  | 0.382 | 1 | json_object |
| sola-jsonish-rescue | recursive | 1 | ok |  | 1.298 | 1 | json_object |
| sola-jsonish-rescue | recursive | 2 | ok |  | 1.279 | 1 | json_object |
| sola-jsonish-rescue | recursive | 3 | ok |  | 1.282 | 1 | json_object |

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
| sola-yaml-sections | flat | none | 181 | 21 | 202 |
| sola-yaml-sections | flat | none | 181 | 21 | 202 |
| sola-yaml-sections | flat | none | 181 | 21 | 202 |
| sola-yaml-sections | nested | none | 215 | 34 | 249 |
| sola-yaml-sections | nested | none | 215 | 34 | 249 |
| sola-yaml-sections | nested | none | 215 | 34 | 249 |
| sola-yaml-sections | list_of_model | none | 210 | 29 | 239 |
| sola-yaml-sections | list_of_model | none | 210 | 29 | 239 |
| sola-yaml-sections | list_of_model | none | 210 | 29 | 239 |
| sola-yaml-sections | enum | none | 225 | 14 | 239 |
| sola-yaml-sections | enum | none | 225 | 14 | 239 |
| sola-yaml-sections | enum | none | 225 | 14 | 239 |
| sola-yaml-sections | optional | none | 201 | 14 | 215 |
| sola-yaml-sections | optional | none | 201 | 14 | 215 |
| sola-yaml-sections | optional | none | 201 | 14 | 215 |
| sola-yaml-sections | recursive | none | 201 | 30 | 231 |
| sola-yaml-sections | recursive | none | 201 | 30 | 231 |
| sola-yaml-sections | recursive | none | 201 | 30 | 231 |
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
