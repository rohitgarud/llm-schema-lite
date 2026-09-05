```
arm: live
generated: 2026-09-05T06:59:20.788791+00:00
command: python -m benchmarking.dspy_adapters --live
git_head: 41810a5
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
model: openai/qwen3:8b
api_base: http://localhost:11434/v1
lm_kwargs: {'temperature': 0.0, 'max_tokens': 900, 'cache': False, 'num_retries': 0, 'seed': 7}
supports_response_schema: False
supports_function_calling: False
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Live outcomes — aggregate

| adapter | signature | trials | ok | parse | validation | empty | transport | format | median wall_s | stddev wall_s | median total_tokens | response_format |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | flat | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 23.092 | 0.000 | 488 | none |
| chat | nested | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 8.530 | 0.000 | 678 | none |
| chat | list_of_model | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 8.906 | 0.000 | 719 | none |
| chat | enum | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 7.680 | 0.000 | 562 | none |
| chat | optional | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 3.764 | 0.000 | 378 | none |
| chat | recursive | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 15.650 | 0.000 | 809 | none |
| json | flat | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 11.160 | 0.000 | 587 | json_object |
| json | nested | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 7.889 | 0.000 | 652 | json_object |
| json | list_of_model | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 11.361 | 0.000 | 829 | json_object |
| json | enum | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 14.753 | 0.000 | 735 | json_object |
| json | optional | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 23.626 | 0.000 | 1085 | json_object |
| json | recursive | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 21.398 | 0.000 | 1158 | json_object |
| baml | flat | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 14.554 | 0.000 | 614 | json_object |
| baml | nested | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 10.672 | 0.000 | 562 | json_object |
| baml | list_of_model | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 7.594 | 0.000 | 493 | json_object |
| baml | enum | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 5.696 | 0.000 | 424 | json_object |
| baml | optional | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 5.512 | 0.000 | 411 | json_object |
| baml | recursive | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.007 | 0.000 | — | none |
| sola-json-sections | flat | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 9.185 | 0.000 | 541 | json_object |
| sola-json-sections | nested | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 8.186 | 0.000 | 566 | json_object |
| sola-json-sections | list_of_model | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 9.330 | 0.000 | 677 | json_object |
| sola-json-sections | enum | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 16.544 | 0.000 | 642 | json_object |
| sola-json-sections | optional | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 23.162 | 0.000 | 331 | json_object |
| sola-json-sections | recursive | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 36.282 | 0.000 | 782 | json_object |
| sola-jsonish-sections | flat | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 28.342 | 0.000 | 624 | json_object |
| sola-jsonish-sections | nested | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 15.150 | 0.000 | 520 | json_object |
| sola-jsonish-sections | list_of_model | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 18.228 | 0.000 | 575 | json_object |
| sola-jsonish-sections | enum | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 15.033 | 0.000 | 429 | json_object |
| sola-jsonish-sections | optional | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 23.088 | 0.000 | 340 | json_object |
| sola-jsonish-sections | recursive | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 25.532 | 0.000 | 633 | json_object |
| sola-yaml-sections | flat | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 11.675 | 0.000 | 604 | none |
| sola-yaml-sections | nested | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 10.785 | 0.000 | 450 | none |
| sola-yaml-sections | list_of_model | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 23.230 | 0.000 | 1163 | none |
| sola-yaml-sections | enum | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 14.013 | 0.000 | 473 | none |
| sola-yaml-sections | optional | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 22.466 | 0.000 | 459 | none |
| sola-yaml-sections | recursive | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 13.685 | 0.000 | 713 | none |

## Live outcomes — detail

| adapter | signature | trial | outcome | error class | wall_s | lm_calls | response_format |
|---|---|---|---|---|---|---|---|
| chat | flat | 1 | ok |  | 23.092 | 1 | none |
| chat | nested | 1 | ok |  | 8.530 | 1 | none |
| chat | list_of_model | 1 | ok |  | 8.906 | 1 | none |
| chat | enum | 1 | parse_error | AdapterParseError | 7.680 | 1 | none |
| chat | optional | 1 | ok |  | 3.764 | 1 | none |
| chat | recursive | 1 | ok |  | 15.650 | 1 | none |
| json | flat | 1 | ok |  | 11.160 | 1 | json_object |
| json | nested | 1 | ok |  | 7.889 | 1 | json_object |
| json | list_of_model | 1 | ok |  | 11.361 | 1 | json_object |
| json | enum | 1 | ok |  | 14.753 | 1 | json_object |
| json | optional | 1 | ok |  | 23.626 | 1 | json_object |
| json | recursive | 1 | parse_error | AdapterParseError | 21.398 | 1 | json_object |
| baml | flat | 1 | ok |  | 14.554 | 1 | json_object |
| baml | nested | 1 | ok |  | 10.672 | 1 | json_object |
| baml | list_of_model | 1 | ok |  | 7.594 | 1 | json_object |
| baml | enum | 1 | ok |  | 5.696 | 1 | json_object |
| baml | optional | 1 | ok |  | 5.512 | 1 | json_object |
| baml | recursive | 1 | format_error | ValueError | 0.007 | 0 | none |
| sola-json-sections | flat | 1 | ok |  | 9.185 | 1 | json_object |
| sola-json-sections | nested | 1 | ok |  | 8.186 | 1 | json_object |
| sola-json-sections | list_of_model | 1 | ok |  | 9.330 | 1 | json_object |
| sola-json-sections | enum | 1 | ok |  | 16.544 | 1 | json_object |
| sola-json-sections | optional | 1 | ok |  | 23.162 | 1 | json_object |
| sola-json-sections | recursive | 1 | ok |  | 36.282 | 1 | json_object |
| sola-jsonish-sections | flat | 1 | ok |  | 28.342 | 1 | json_object |
| sola-jsonish-sections | nested | 1 | ok |  | 15.150 | 1 | json_object |
| sola-jsonish-sections | list_of_model | 1 | ok |  | 18.228 | 1 | json_object |
| sola-jsonish-sections | enum | 1 | ok |  | 15.033 | 1 | json_object |
| sola-jsonish-sections | optional | 1 | ok |  | 23.088 | 1 | json_object |
| sola-jsonish-sections | recursive | 1 | parse_error | AdapterParseError | 25.532 | 1 | json_object |
| sola-yaml-sections | flat | 1 | parse_error | AdapterParseError | 11.675 | 1 | none |
| sola-yaml-sections | nested | 1 | parse_error | AdapterParseError | 10.785 | 1 | none |
| sola-yaml-sections | list_of_model | 1 | parse_error | AdapterParseError | 23.230 | 1 | none |
| sola-yaml-sections | enum | 1 | parse_error | AdapterParseError | 14.013 | 1 | none |
| sola-yaml-sections | optional | 1 | parse_error | AdapterParseError | 22.466 | 1 | none |
| sola-yaml-sections | recursive | 1 | ok |  | 13.685 | 1 | none |

### Provider accounting — NOT comparable across the `response_format` groups

| adapter | signature | response_format | prompt_tokens_reported | completion_tokens_reported | total_tokens |
|---|---|---|---|---|---|
| chat | flat | none | 189 | 299 | 488 |
| chat | nested | none | 320 | 358 | 678 |
| chat | list_of_model | none | 340 | 379 | 719 |
| chat | enum | none | 233 | 329 | 562 |
| chat | optional | none | 220 | 158 | 378 |
| chat | recursive | none | 247 | 562 | 809 |
| json | flat | json_object | 545 | 42 | 587 |
| json | nested | json_object | 600 | 52 | 652 |
| json | list_of_model | json_object | 769 | 60 | 829 |
| json | enum | json_object | 718 | 17 | 735 |
| json | optional | json_object | 1069 | 16 | 1085 |
| json | recursive | json_object | 258 | 900 | 1158 |
| baml | flat | json_object | 567 | 47 | 614 |
| baml | nested | json_object | 510 | 52 | 562 |
| baml | list_of_model | json_object | 433 | 60 | 493 |
| baml | enum | json_object | 411 | 13 | 424 |
| baml | optional | json_object | 395 | 16 | 411 |
| sola-json-sections | flat | json_object | 492 | 49 | 541 |
| sola-json-sections | nested | json_object | 514 | 52 | 566 |
| sola-json-sections | list_of_model | json_object | 617 | 60 | 677 |
| sola-json-sections | enum | json_object | 629 | 13 | 642 |
| sola-json-sections | optional | json_object | 315 | 16 | 331 |
| sola-json-sections | recursive | json_object | 726 | 56 | 782 |
| sola-jsonish-sections | flat | json_object | 585 | 39 | 624 |
| sola-jsonish-sections | nested | json_object | 468 | 52 | 520 |
| sola-jsonish-sections | list_of_model | json_object | 515 | 60 | 575 |
| sola-jsonish-sections | enum | json_object | 416 | 13 | 429 |
| sola-jsonish-sections | optional | json_object | 324 | 16 | 340 |
| sola-jsonish-sections | recursive | json_object | 584 | 49 | 633 |
| sola-yaml-sections | flat | none | 182 | 422 | 604 |
| sola-yaml-sections | nested | none | 227 | 223 | 450 |
| sola-yaml-sections | list_of_model | none | 263 | 900 | 1163 |
| sola-yaml-sections | enum | none | 226 | 247 | 473 |
| sola-yaml-sections | optional | none | 202 | 257 | 459 |
| sola-yaml-sections | recursive | none | 228 | 485 | 713 |
