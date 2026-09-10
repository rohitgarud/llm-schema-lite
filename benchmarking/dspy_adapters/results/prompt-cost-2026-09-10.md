```
arm: prompt-cost
generated: 2026-09-10T12:54:51.106511+00:00
command: python -m benchmarking.dspy_adapters --offline
git_head: 867819b
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
| sola-json-sections | 159 | 292 | 311 | 203 | 189 | 221 |
| sola-jsonish-sections | 159 | 201 | 193 | 203 | 182 | 186 |
| sola-yaml-sections | 154 | 190 | 182 | 198 | 172 | 178 |
| sola-json-block | 159 | 294 | 313 | 203 | 189 | 223 |
| sola-jsonish-block | 159 | 199 | 192 | 203 | 179 | 184 |
| sola-json-rescue | 159 | 292 | 311 | 203 | 189 | 221 |
| sola-jsonish-rescue | 159 | 201 | 193 | 203 | 182 | 186 |
| sola-yaml-rescue | 154 | 190 | 182 | 198 | 172 | 178 |
| sola-yaml-block | 160 | 193 | 188 | 204 | 180 | 181 |

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
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | flat | 2 | 691 | 159 | ok |
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | nested | 2 | 1077 | 292 | ok |
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | list_of_model | 2 | 1133 | 311 | ok |
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | enum | 2 | 799 | 203 | ok |
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | optional | 2 | 756 | 189 | ok |
| sola-json-sections | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS) | recursive | 2 | 840 | 221 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | flat | 2 | 691 | 159 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | nested | 2 | 815 | 201 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | list_of_model | 2 | 791 | 193 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | enum | 2 | 799 | 203 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | optional | 2 | 750 | 182 | ok |
| sola-jsonish-sections | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS) | recursive | 2 | 772 | 186 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | flat | 2 | 675 | 154 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | nested | 2 | 782 | 190 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | list_of_model | 2 | 762 | 182 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | enum | 2 | 783 | 198 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | optional | 2 | 721 | 172 | ok |
| sola-yaml-sections | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS) | recursive | 2 | 745 | 178 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | flat | 2 | 685 | 159 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | nested | 2 | 1076 | 294 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | list_of_model | 2 | 1132 | 313 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | enum | 2 | 793 | 203 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | optional | 2 | 750 | 189 | ok |
| sola-json-block | StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK) | recursive | 2 | 839 | 223 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | flat | 2 | 685 | 159 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | nested | 2 | 802 | 199 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | list_of_model | 2 | 778 | 192 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | enum | 2 | 793 | 203 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | optional | 2 | 734 | 179 | ok |
| sola-jsonish-block | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK) | recursive | 2 | 761 | 184 | ok |
| sola-json-rescue | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS, parse_config=ParseConfig()) | flat | 2 | 691 | 159 | ok |
| sola-json-rescue | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS, parse_config=ParseConfig()) | nested | 2 | 1077 | 292 | ok |
| sola-json-rescue | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS, parse_config=ParseConfig()) | list_of_model | 2 | 1133 | 311 | ok |
| sola-json-rescue | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS, parse_config=ParseConfig()) | enum | 2 | 799 | 203 | ok |
| sola-json-rescue | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS, parse_config=ParseConfig()) | optional | 2 | 756 | 189 | ok |
| sola-json-rescue | StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS, parse_config=ParseConfig()) | recursive | 2 | 840 | 221 | ok |
| sola-jsonish-rescue | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS, parse_config=ParseConfig()) | flat | 2 | 691 | 159 | ok |
| sola-jsonish-rescue | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS, parse_config=ParseConfig()) | nested | 2 | 815 | 201 | ok |
| sola-jsonish-rescue | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS, parse_config=ParseConfig()) | list_of_model | 2 | 791 | 193 | ok |
| sola-jsonish-rescue | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS, parse_config=ParseConfig()) | enum | 2 | 799 | 203 | ok |
| sola-jsonish-rescue | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS, parse_config=ParseConfig()) | optional | 2 | 750 | 182 | ok |
| sola-jsonish-rescue | StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS, parse_config=ParseConfig()) | recursive | 2 | 772 | 186 | ok |
| sola-yaml-rescue | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS, parse_config=ParseConfig()) | flat | 2 | 675 | 154 | ok |
| sola-yaml-rescue | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS, parse_config=ParseConfig()) | nested | 2 | 782 | 190 | ok |
| sola-yaml-rescue | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS, parse_config=ParseConfig()) | list_of_model | 2 | 762 | 182 | ok |
| sola-yaml-rescue | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS, parse_config=ParseConfig()) | enum | 2 | 783 | 198 | ok |
| sola-yaml-rescue | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS, parse_config=ParseConfig()) | optional | 2 | 721 | 172 | ok |
| sola-yaml-rescue | StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS, parse_config=ParseConfig()) | recursive | 2 | 745 | 178 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | flat | 2 | 692 | 160 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | nested | 2 | 787 | 193 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | list_of_model | 2 | 769 | 188 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | enum | 2 | 800 | 204 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | optional | 2 | 741 | 180 | ok |
| sola-yaml-block | StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK) | recursive | 2 | 750 | 181 | ok |
