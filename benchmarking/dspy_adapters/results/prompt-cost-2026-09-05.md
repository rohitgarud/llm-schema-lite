```
arm: prompt-cost
generated: 2026-09-05T12:07:39.445038+00:00
command: python -m benchmarking.dspy_adapters --offline
git_head: e9e8204
dspy_version: 3.3.1
llm_schema_lite_version: 0.6.1
encoding: cl100k_base
```

## Metric integrity

Same prompt, same model, greedy: without `response_format` → `{'prompt_tokens': 21, 'completion_tokens': 160}`; with `response_format={"type": "json_object"}` → `{'prompt_tokens': 173, 'completion_tokens': 7}` (identical 649-char reasoning trace in both). Ollama re-prefills the thinking trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode and `ChatAdapter` send **no** `response_format` while JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are **not** on the same accounting basis. Total tokens stay comparable (181 vs 180).

prompt/completion token counts are NOT comparable across the `response_format` groups

the offline count measures the textual prompt only — it excludes `_call_preprocess` tool/native-type handling and the `response_format` request kwarg, which is a request field and not a message. For this matrix that is correct: the thing under test is the text.

The offline prompt-cost table and the live outcomes table are never joined into one table or one derived score.

## Prompt cost — pivot (prompt tokens per adapter x signature)

| adapter | flat | nested | list_of_model | enum | optional | recursive |
|---|---|---|---|---|---|---|
| chat | 174 | 303 | 323 | 218 | 205 | 232 |
| json | 166 | 321 | 344 | 210 | 201 | 243 |
| baml | 156 | 164 | 165 | 178 | 171 | — |
| sola-json-sections | 166 | 299 | 318 | 210 | 196 | 228 |
| sola-jsonish-sections | 166 | 205 | 199 | 210 | 186 | 185 |
| sola-yaml-sections | 167 | 199 | 194 | 211 | 187 | 182 |
| sola-json-block | 166 | 301 | 320 | 210 | 196 | 230 |
| sola-jsonish-block | 166 | 206 | 200 | 210 | 186 | 186 |
| sola-yaml-block | 167 | 200 | 195 | 211 | 187 | 183 |

## Prompt cost — detail

| adapter | adapter config | signature | messages | chars | prompt tokens | outcome |
|---|---|---|---|---|---|---|
| chat | ChatAdapter(use_json_adapter_fallback=False) | flat | 2 | 750 | 174 | ok |
| chat | ChatAdapter(use_json_adapter_fallback=False) | nested | 2 | 1124 | 303 | ok |
| chat | ChatAdapter(use_json_adapter_fallback=False) | list_of_model | 2 | 1180 | 323 | ok |
| chat | ChatAdapter(use_json_adapter_fallback=False) | enum | 2 | 858 | 218 | ok |
| chat | ChatAdapter(use_json_adapter_fallback=False) | optional | 2 | 815 | 205 | ok |
| chat | ChatAdapter(use_json_adapter_fallback=False) | recursive | 2 | 887 | 232 | ok |
| json | JSONAdapter() | flat | 2 | 728 | 166 | ok |
| json | JSONAdapter() | nested | 2 | 1209 | 321 | ok |
| json | JSONAdapter() | list_of_model | 2 | 1277 | 344 | ok |
| json | JSONAdapter() | enum | 2 | 836 | 210 | ok |
| json | JSONAdapter() | optional | 2 | 803 | 201 | ok |
| json | JSONAdapter() | recursive | 2 | 930 | 243 | ok |
| baml | BAMLAdapter() | flat | 2 | 667 | 156 | ok |
| baml | BAMLAdapter() | nested | 2 | 639 | 164 | ok |
| baml | BAMLAdapter() | list_of_model | 2 | 651 | 165 | ok |
| baml | BAMLAdapter() | enum | 2 | 685 | 178 | ok |
| baml | BAMLAdapter() | optional | 2 | 688 | 171 | ok |
| baml | BAMLAdapter() | recursive | — | — | — | format_error |
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | flat | 2 | 734 | 166 | ok |
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | nested | 2 | 1120 | 299 | ok |
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | list_of_model | 2 | 1176 | 318 | ok |
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | enum | 2 | 842 | 210 | ok |
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | optional | 2 | 799 | 196 | ok |
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | recursive | 2 | 883 | 228 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | flat | 2 | 734 | 166 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | nested | 2 | 848 | 205 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | list_of_model | 2 | 826 | 199 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | enum | 2 | 842 | 210 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | optional | 2 | 785 | 186 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | recursive | 2 | 793 | 185 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | flat | 2 | 741 | 167 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | nested | 2 | 833 | 199 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | list_of_model | 2 | 815 | 194 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | enum | 2 | 849 | 211 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | optional | 2 | 792 | 187 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | recursive | 2 | 780 | 182 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | flat | 2 | 728 | 166 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | nested | 2 | 1119 | 301 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | list_of_model | 2 | 1175 | 320 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | enum | 2 | 836 | 210 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | optional | 2 | 793 | 196 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | recursive | 2 | 882 | 230 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | flat | 2 | 728 | 166 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | nested | 2 | 845 | 206 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | list_of_model | 2 | 823 | 200 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | enum | 2 | 836 | 210 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | optional | 2 | 777 | 186 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | recursive | 2 | 790 | 186 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | flat | 2 | 735 | 167 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | nested | 2 | 830 | 200 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | list_of_model | 2 | 812 | 195 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | enum | 2 | 843 | 211 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | optional | 2 | 784 | 187 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | recursive | 2 | 777 | 183 | ok |
