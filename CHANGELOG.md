# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- **Private parser helpers are no longer re-exported, and `BaseFormatter` loses two
  uncalled methods.** `_get_json_schema`, `_validate_field` and `_build_result` leave
  `llm_schema_lite.__all__` and `llm_schema_lite.parsers`; import them from
  `llm_schema_lite.parsers.schema_parser`. `BaseFormatter.process_schema` and
  `process_unique_items` are gone (nothing called them), and `process_anyof` is now
  abstract; the three shipped formatters all implement it, so only a third-party subclass
  that relied on the base version is affected. The `regex` runtime dependency is dropped:
  nothing imported it.
- **Constraint spelling is ASCII everywhere.** TypeScript's scalar length/range fragments
  and JSONish's nested-position (tuple element, scalar `$ref`) fragments drop `≥`/`≤` for
  `>=`/`<=`; JSONish and YAML property-position output is byte-identical to before. Numeric
  two-sided ranges now read `(n to m)` (e.g. `(1 to 10)`, `(-5 to 10)`); string length and
  item-count ranges keep `(n-m unit)`. (`lsl-2026-09-05-009`)
- **An enum's allowed-value set is structural and survives `include_metadata=False` /
  `include_constraints=False` in all three formatters.** `const` and `Literal` values ride
  the same route. `metadata_inclusion={"enum": False}` no longer suppresses the value list
  (it is no longer a recognised override point for this keyword). (`lsl-2026-09-05-009`)
- **DSPy prompts no longer embed the schema inside a JSON string.** The adapter renders
  the field structure as plain text, and the layout is selectable through the new
  `PromptLayout` enum. Any snapshot asserting on the previous prompt text will change.
  (`c1fb265`)
- **`StructuredOutputAdapter.__call__`/`acall` now dispatch past `JSONAdapter` to
  `ChatAdapter`**, ending the MRO re-entry that overwrote `response_format` in every mode.
  JSON mode now matches upstream DSPy 3.3.1 gate for gate and re-raises `LMError`.
  (`447b7eb`)
- **DSPy prompts now carry descriptions and constraints by default.** The adapter's
  hardcoded `include_metadata=False` is gone; schema rendering is governed by
  `FormatterConfig` like everything else. (`99b6aa7`)
- **Enums and `Literal`s render as `string // one of: "a", "b"`** instead of
  `OPTIONS: a| b`. (`a47ad7c`)
- **`dict`, `tuple`, `set` and typeless fields render as real container shapes** —
  `{<string>: int}`, `[int, string]`, `string [] (unique)`, `any` — instead of leaking
  `string` or an `additional:` key. (`91daa6b`)
- **The `dspy` extra and dependency-group floor rises from `>=3.0.3` to `>=3.3.1`.** The
  integration adapter targets DSPy 3.3.1 APIs and no 3.0.x compatibility path is
  maintained. (`f31ac3e`)
- **TypeScript no longer emits a per-`$defs` `interface` block.** Every nested model's body
  is rendered inline at each use site and nowhere else; the only `interface` in a render is
  `interface Schema`. The removed blocks were unreachable output — no render anywhere
  referenced a non-`Schema` interface by name, because `process_ref` always inlines — so
  they were pure token cost. Snapshot-visible for any TypeScript output over a model with
  `$defs`: 35 blocks across the fixture corpus drop to 0, the corpus token total falls
  27.8%, and `Order` goes from 727 tokens to 340, below JSONish's 414. A schema with more
  than ten distinct `$defs` may also render *wider* unions than before, because the dead
  interface pass no longer consumes the recursion-expansion budget that was collapsing them
  into `anyOf: N options`. Nothing hoists `$defs` into named interfaces, and no such mode is
  planned — see the `hoist_enums`/`hoist_classes` removal below. (`lsl-2026-09-05-011`)
- **A TypeScript interface member terminates before its comment, not after it.**
  `name*: string  // Full name;` now reads `name*: string;  // Full name`, and a multi-line
  description's continuation lines stay pure comments instead of collecting the `;`. The
  same applies to `[key: string]` index-signature members. Snapshot-visible for any
  TypeScript output carrying a description or a metadata comment: 170 offending lines across
  the fixture corpus drop to 0. Per-model token counts can rise slightly for models with no
  `$defs` (`Product`: 105 to 109), because the `;` re-tokenises away from the type.
  (`lsl-2026-09-05-011`)
- **TypeScript string literals and `const` values are JSON-quoted and escaped.**
  `api_version*: v1` now reads `api_version*: "v1"`, and enum unions escape embedded quotes:
  `q*: "say "hi"" | "plain"` becomes `q*: "say \"hi\"" | "plain"`. Numbers, booleans and
  `null` stay bare (`version*: 1`, `true | false`), so the rendering *decision* is unchanged
  — only the escaping and the quoting of strings. Enum mapping keys are unaffected and
  remain bare (`Record<red | green, number>`, `lsl-2026-09-05-010`). Snapshot-visible for any
  TypeScript output containing a `Literal` or `const` string. (`lsl-2026-09-05-011`)
- **YAML output no longer hoists nested models into a `Class.field` `$defs` section.** A
  nested `$ref`, a `Model | None` and a `list[Model]` now render as real YAML mappings and
  sequences inline on the property that references them; both hard-coded `$defs` loops that
  emitted `Class.field` keys are deleted, along with the duplication they caused and the
  non-idempotent cached-branch rendering path. `OR null` and closed-world ("no additional
  properties") notes now attach to the block's key through the deferred-comment slot so
  PyYAML never re-quotes the line; defaults are emitted once instead of twice; nested
  required-field markers are resolved against the nested `$def` rather than the root; and
  `YAMLFormatter.__init__` no longer mutates the caller's `FormatterConfig`. Every
  continuation line of a multi-line description is commented, so all 148 fixture renders
  round-trip through `yaml.safe_load`. Breaking for any consumer parsing the old hoisted
  shape. (`ba5d7af`)
- **`FormatterConfig.hoist_enums` and `hoist_classes` are removed.** No formatter ever read
  either field, so no rendered output changes — the options were accepted and silently
  ignored. Code that passed `FormatterConfig(hoist_enums=...)` or
  `FormatterConfig(hoist_classes=...)` now raises `TypeError` instead of quietly doing
  nothing; delete the argument. Hoisting is not a supported rendering mode and none is
  planned. (`lsl-2026-04-29-002`)

### Added

- **Four more rescues in the DSPy adapter, for reply shapes reported on DSPy's tracker.**
  With `parse_config` set, an `Optional[X]` field holding a scalar is rescued against
  `X`, so `14` for `Optional[str]` becomes `"14"` and an enum member's name works under
  `Optional` as it does bare (stanfordnlp/dspy#1962, #9994). A numeric string such as
  `"2"` now matches an integer or float enum member. With `allow_coercion`, a reply
  missing output fields is retried once more in two cases:
  - fields sent under one extra key, as a dict or as JSON text (`{"json_input": {...}}`,
    #8539), are unwrapped;
  - a key that unambiguously near-misses an absent field (`tool_args` for
    `next_tool_args`, #8377) is renamed. `name` against `first_name` and `last_name` is
    left alone.

  Null, dicts and lists are never stringified, and a retry that fails raises the
  original error. Not yet measured on the benchmark.
- **Constraints passed as `InputField`/`OutputField` kwargs reach the prompt.**
  `OutputField(ge=0.0, le=1.0)` now renders as `float (0.0 to 1.0)` in JSONish and YAML
  and as `minimum`/`maximum` in JSON mode's schema text. They were dropped because they
  live in the field's metadata, not its annotation (stanfordnlp/dspy#10195). The
  benchmark's prompts are byte-identical.
- **With `parse_config` set, the DSPy adapter checks more and rescues more.**
  - Constraints passed as `OutputField` kwargs are enforced at parse time, so `7.0` for
    `OutputField(le=1.0)` fails like any invalid value (#10195). Upstream checks only the
    annotation, so `parse_config=None` still accepts it.
  - A JSON reply cut off mid-string, as at `max_tokens`, loses the value it was writing:
    the repair closed the string, so `"answer": "bl` passed as a complete answer
    (#1727). The field then fails as missing. A cut mid-number is not detected.
  - The coercion rescue tries a scalar against each member of any union, not only
    `Optional[X]`, and matches an enum member named or valued in another case (`"red"`
    for `RED = "crimson"`).

  Not yet measured on the benchmark.
- **The accuracy benchmark records each raw reply, and `--replay CSV` re-scores them.**
  Every accuracy CSV row gains a `replies` column, a JSON list with one reply text per LM
  call. `--replay` feeds them back through the current adapter code with no model and no
  environment, so a parser change can be checked against recorded replies in seconds.
  It is valid while the adapters' prompts are unchanged.
- **Held-out cases and paired intervals for the benchmark.** `--cases-offset N` skips
  the first N cases of any accuracy corpus, so the `ParseConfig()` repairs, all found by
  reading failures in the first 30 cases, can be scored on cases they were not tuned on.
  `python -m benchmarking.dspy_adapters.paired` reads per-case accuracy CSVs and gives
  each adapter's difference from a baseline (`--baseline json`) a paired bootstrap 95%
  interval on recall, field accuracy or invented values.
- **List-of-records merge in the DSPy adapter.** With `parse_config` set
  (`allow_coercion`), an output value sent as a list of two or more records, where the
  schema wants one object whose every field is a list, is merged into that object: each
  field becomes the concatenation across the records, a lone value appended whole, a null
  adding nothing. A slot that also admits a list is never touched. Parsing 1,350 captured
  completions (3 models x 5 corpora x 3 modes) with and without it: financial-NER went
  from 126 to 216 of 270 parsed and recall on non-null gold from 0.232 to 0.407
  (`llama3.2:1b` JSON 0.016 -> 0.422, 2/30 -> 29/30); no other corpus changed and no case
  got worse. The rescued records keep what the model invented: categories filled where
  the gold is empty rose 217 -> 543 of 1,323, half of the new ones placeholder strings
  ("not explicitly mentioned") that unrepaired replies carry too.
- **The accuracy arm reports recall on non-null gold and invented values on null gold.**
  Field accuracy counts a correct `None` as a match, so a reply that extracts nothing
  already scores 0.949 on `pii` and 0.590 on `financial-ner`. `score()` now also counts, in
  the same pass and under the same matching rules, the matches among fields whose gold is
  not `None` (recall, whose all-null floor is 0) and the gold-`None` fields the reply
  filled, with a scalar or a whole list or object. Both are new aggregate-table and
  per-case CSV columns (`recall_matched`, `recall_total`, `recall`, `invented`), appended
  after the existing ones; the report names recall as the extraction-quality headline.
- **Extraction-accuracy benchmark arm.** `benchmarking/dspy_adapters/{cases,accuracy}.py`
  generate labeled extraction cases and score a reply field-by-field against ground truth,
  with `--accuracy --cases N --cases-seed S` on the benchmark CLI and a third results file
  stem (`accuracy-<model>-<date>.{md,csv}`). This measures *correctness*, which the existing
  outcomes arm cannot: a reply can parse, satisfy the schema, and be entirely wrong. Ground
  truth is the record the prose was rendered from, so labels are exact and need no dataset.
  A failed cell scores 0 against its full denominator rather than being excluded, and the
  denominator is the expected fields, never the produced ones. The arm is opt-in and never
  runs by default — it costs `adapters x cases` live calls.
- **List-unwrap rescue in the DSPy adapter.** With `parse_config` set (`allow_coercion`),
  an output value sent as a one-item list where the schema wants an object is unwrapped
  before validation, ahead of the all-null-item prune. It is schema-guided: a list-typed
  field is never touched, and a list of two or more objects is left for validation to
  reject. Measured by parsing the same live completions with and without it: in JSON mode
  `qwen3.5:0.8b` wraps the whole financial-NER record so on 29 of 30 cases, and the repair
  takes it from 9/30 to 26/30 parsed (field accuracy 0.209 -> 0.606); JSONISH and YAML
  replies were unchanged on every corpus measured. The benchmark gains an opt-in
  `sola-json-rescue` cell to reproduce it.
- **Five more structural repairs in the DSPy adapter**, behind the same `parse_config`
  gate. A field that failed validation is also offered these repairs:
  - fields the model hoisted out of a nested object move back under it, including from the
    reply's top level into the output field;
  - the prompt's `*` marker is stripped from nested keys;
  - an optional object whose every field is null becomes null;
  - a lone value is wrapped in a list where the schema wants a list of values.

  And before the missing-field error, an output field's contents sent without its key are
  moved under it. Each repair only reshapes what the model sent, never adds a value.
  Measured by parsing the same captured completions with and without them (30 cases per
  corpus x mode):
  - `qwen3.5:0.8b`:
    - insurance-claims JSONISH 0.233 -> 0.720 (9/30 -> 30/30 parsed);
    - insurance-claims YAML 0.670 -> 0.721;
    - pii JSON 0.713 -> 0.926 (23 -> 30);
    - synthetic YAML 0.831 -> 0.926.
  - `granite3.1-moe:1b`:
    - financial-NER parsed 0 -> 16 (JSON), 1 -> 13 (YAML) and 0 -> 8 (JSONISH);
    - pii JSON 0.607 -> 0.738.
  - `llama3.2:1b`:
    - pii JSONISH 0.082 -> 0.339 (7/30 -> 30/30 parsed);
    - synthetic JSONISH 0.215 -> 0.271 and YAML 0.311 -> 0.357.

  No case scored lower on any model, across 1,350 captured completions.
- **Third-party corpora for the accuracy arm**
  (`--corpus pii|financial-ner|insurance-claims|patient-notes`).
  `benchmarking/dspy_adapters/external.py` runs the four structured-output tasks of
  thedataquarry/structured-outputs — three Cleanlab benchmarks and clinical patient notes —
  with that repo's own signature instructions and field descriptions, so the arm is no
  longer scored only on a corpus this package wrote. Rows are fetched at pinned revisions
  (Hugging Face via the `benchmark` extra, or thedataquarry's GitHub for patient notes) and
  never vendored — no source states a licence. The patient-notes schema is reshaped to
  match its own gold, which upstream's does not; each deviation is marked at its field. Reports name the corpus in the file stem
  (`accuracy-<corpus>-<model>-<date>`) and print an all-null floor under the aggregate: a
  correct `None` is a match, so a reply that extracts nothing already scores 0.949 on the
  first 30 PII cases. `flatten` now dumps models in JSON mode so a `date` compares equal
  to its label; synthetic-corpus scores are unchanged.
- **`sola-jsonish-rescue` benchmark cell** — `sola-jsonish-sections` with
  `parse_config=ParseConfig()` and nothing else changed, so the pair isolates what
  parse-time repair is worth. A test asserts the two render byte-identical prompts for all
  six signatures; the offline arm makes the same claim checkable in the token counts.

- `benchmarking/fetch_dataset.py` — downloads JSONSchemaBench into the repo-root
  `jsonschembench_dataset.json` (~100 MB, now gitignored) in the shape
  `format_jsonschembench_schema.py` reads. Idempotent: skips when the file already
  exists. Requires the `benchmark` extra. (`lsl-2026-09-04-013`)
- `StructuredOutputAdapter(prompt_layout=...)` and the `PromptLayout` enum
  (`PromptLayout.SECTIONS`, `PromptLayout.JSON_BLOCK`). (`c1fb265`)
- `StructuredOutputAdapter(formatter_config=...)` — the single passthrough for schema
  rendering options. When given it wins entirely and `max_recursion_depth` is ignored.
  (`99b6aa7`)
- `StructuredOutputAdapter(max_recursion_depth=...)`, default `2`. (`669e129`)
- `StructuredOutputAdapter(use_json_object_response_format=...)`, default `True`, for
  OpenAI-compatible servers that advertise `response_format` but reject the `json_object`
  type; and `StructuredOutputAdapter(parallel_tool_calls=...)`, default `None`.
  (`447b7eb`)
- `StructuredOutputAdapter(parse_config=...)` and `ParseConfig.strip_required_marker`
  (default `"*"`), the reply-side counterpart of `FormatterConfig.required_marker`. The same
  marker value later grew to strip at every nested object level — see the `Fixed` entries for
  `lsl-2026-09-05-003`. (`7a29f2d`)
- DSPy per-field streaming support for `StructuredOutputAdapter` in JSON and JSONish
  modes: `register_streaming_support()` runs on `llm_schema_lite.dspy_integration` import,
  and the new public `StreamingNotSupportedError` is raised — before any LM request — when
  YAML mode is combined with stream listeners. (`c88a0ce`)
- `StructuredOutputAdapter.format_finetune_data()` is implemented and returns an OpenAI
  chat-format record instead of raising `NotImplementedError`. (`8bcf58b`)
- `FormatterConfig.max_recursion_depth`, default `2`. (`2a68581`)
- A DSPy-latest CI canary job. (`f31ac3e`)
- **A DSPy adapter benchmark harness**, `benchmarking/dspy_adapters/`, run with
  `make bench-dspy` (`BENCH_ARGS=--offline` for the no-network arm, `BENCH_ARGS=--live`
  against a real endpoint, no `BENCH_ARGS` for both). It measures prompt cost and parse /
  validation outcomes per adapter x signature and writes markdown and CSV reports.
  Development-only: `pyproject.toml` ships `src/llm_schema_lite` alone, so the harness is in
  the repository but not in the installed wheel. (`2db8c3d`)
- **Newly public helpers on already-exported classes and modules.**
  `FormatterConfig.includes(key)`; `BaseFormatter.effective_root_schema()` and
  `BaseFormatter.root_decorations()`; `BaseFormatter.sanitize_comment_text()` (overridden in
  `JSONishFormatter`); the previously undocumented `exclude` tuple of
  `BaseFormatter.format_metadata_parts(value, exclude=())`;
  `formatters.config.with_format_default_separator()`; `parsers.normalize_marker_keys()`; and
  `formatters.base.IDENTITY_TAG`. The module `llm_schema_lite.schema_normalization` is also
  new — reachable by full path (`normalize_schema_titles()`), deliberately not re-exported
  from the package `__all__`; every `BaseFormatter.__init__` routes its incoming schema
  through it, so a schema dict passed to two formatters is deep-copied and never mutated by
  the first. (`lsl-2026-09-05-001`, `lsl-2026-09-05-002`, `lsl-2026-09-05-003`,
  `lsl-2026-09-05-004`, `lsl-2026-09-05-005`)
- The DSPy adapter benchmark's live aggregate table now reports `parse rate` and `validation
  rate` per adapter x signature. Both are computed over *attempted* cells only — a cell that
  failed in `format()` or whose request was rejected is excluded from the denominator rather
  than scored — and an undefined rate renders as `—`, like every other undefined numeric in
  the report. The CSV files are unchanged. (`e9e8204`)

### Changed

- **`ParseConfig(partial=True)` now keeps the valid part of a field in the DSPy adapter.**
  A field still rejected after the rescues used to be dropped whole, which recovers nothing
  for a single-output signature. Now its invalid values are nulled (an invalid list item is
  dropped, and a value that may not be null takes its parent with it) and the rest is
  kept; only a field with nothing valid left is dropped as before. The steps are logged at
  DEBUG when `log_coercions` is on. On the same replay, recall on non-null gold rose:
  insurance-claims 0.259 -> 0.455, patient-notes 0.038 -> 0.323, synthetic 0.603 -> 0.705,
  no case worse. Values where the gold is null rose with it (patient-notes 0.022 -> 0.369
  of null-gold fields), which is why it stays opt-in. `partial=False`, `parse_config=None`
  and `loads(schema=..., partial=True)` are unchanged.
- Documentation: the top-level README gains Installation, Quick Start and DSPy Integration
  sections; the DSPy integration README is corrected against the shipped adapter; and
  every runnable code block in both, plus `examples/basic_usage.py`, is now executed by
  `tests/test_docs_examples.py`. The DSPy integration README's parsing pipeline and
  `parse_config` bullet now describe shipped behaviour. Both READMEs drop the YAML
  "experimental" label — the two formatter defects it named, hoisted `Class.field` keys and
  quoted multi-line strings, are fixed — in favour of a note that the YAML output is a
  YAML-flavoured schema sketch optimised for prompts rather than a serialization format. The
  README's make-target list gains `bench-dspy`. (`lsl-2026-09-05-003`, `lsl-2026-09-05-012`)
- **An enum-typed mapping key now renders its allowed values instead of just its type.**
  `dict[Color, int]` with `Color` an enum of `red`/`green` renders `<red OR green>: int`
  (JSONish), `dict[red OR green, int]` (YAML) and `Record<red | green, number>` (TypeScript)
  — joined with each formatter's own `union_separator`, so YAML/JSONish read `OR` and
  TypeScript reads `|`. The same applies through an `Optional[...]` wrapper and inside a
  `list[...]`, and to non-string enums (`dict[IntEnum, int]` → `<1 OR 2>`). Values are
  rendered bare, not quoted, matching the existing bare `<string>` / `<int>` key convention;
  note this differs from the quoted spelling used in the value-position `one of: "red",
  "green"` comment. The key set is **structural**: it survives `include_metadata=False`,
  `include_constraints=False` and `metadata_inclusion`, exactly like an enum's value set in
  the value position. Snapshot-visible for any schema with an enum-keyed mapping.
  (`lsl-2026-09-05-010`)

- **Five behaviours copied per formatter now have one owner each; no output changes.** The
  recursion-truncation predicate (4 copies) becomes `BaseFormatter._reentry_truncated`; the
  tuple length-suffix suppression (3 copies) becomes `BaseFormatter.render_tuple_token`;
  `BaseFormatter.format_field_name` finally reads the `_nested_required_stack` it owns, so the
  YAML and TypeScript overrides that re-added that read are deleted; `YAMLFormatter`'s
  `add_metadata` and `_metadata_parts` share one part-collector behind a `deferred` flag, and
  the `get_available_metadata` pre-check that was dead in both (`format_metadata_parts` already
  applies a strictly stronger filter) is gone, matching TypeScript; and
  `YAMLFormatter._mapping_value_pairs` routes its required/optional marker through
  `format_field_name` like `_properties_block` already did. `JSONishFormatter`'s
  `deferred_comment_gap` override, an exact copy of the base property, is deleted. Net -144
  lines in `src/llm_schema_lite/formatters/`, with a byte-identical 39,133-line corpus render
  across 14 configs x 103 schemas x 3 formatters. Deliberately **not** unified, each because it
  would change first-render output or cost more than it saves: `$ref`→`$defs` resolution (8
  sites, three return shapes, five guards); merging `_mapping_value_pairs` into
  `_properties_block` (would make mapping values render as nested blocks); giving
  `_mapping_value_pairs` the global expansion budget (would make mapping expansions visible to
  the anyOf/oneOf member-cap tiers); JSONish's two parallel dispatch chains; the adapter's
  `__call__`/`acall` pair (the async exception surfaces at `await`, not at coroutine creation);
  and YAML's `str(item["const"])`, which is not a duplicate of `process_const` at all —
  routing through it turns `x: US OR null` into `x: string  # one of: "US" OR null`.
  (`lsl-2026-09-05-013`)
- The non-partial `loads(text, schema=Model)` route now builds its result with
  `model_validate` instead of `model_construct`, so nested data becomes real sub-model
  instances, enums become enum members and lax numerics are narrowed. Pydantic-only
  validation failures are raised as `ConversionError("Validation failed: ...")`, never as a
  leaked `pydantic.ValidationError`. The partial route is unchanged and still uses
  `model_construct`. (`944b193`)
- **The benchmark harness degrades instead of raising when `tiktoken` has no encoding.**
  `runner._get_encoding()` now returns `tiktoken.Encoding | None` and catches `OSError` (the
  base class of `requests.exceptions.RequestException`) rather than propagating; both
  outcomes are memoised and `runner.encoding_available()` exposes the result. The new
  stdlib-only `benchmarking/dspy_adapters/encoding.py` locates a bundled `cl100k_base` blob
  inside an installed dependency via `importlib.util.find_spec` (never importing it) and
  seeds `TIKTOKEN_CACHE_DIR` / `CUSTOM_TIKTOKEN_CACHE_DIR`, never overriding a cache
  directory that already works. `--offline` prints a one-line stderr warning when the
  encoding is still unavailable and records `encoding: cl100k_base (unavailable)` in the
  report provenance; the exit code is unchanged (`0`). Seven over-strong "no network"
  statements across the benchmark package were corrected. (`f1191eb`)
- Bare `pytest`/`make test` no longer force `--cov-report=html` and `--cov-report=xml` by
  default; `make test-cov`/`make test-cov-full` and CI already request those reports
  explicitly and are unaffected. (`lsl-2026-09-04-013`)

### Removed

- Unreferenced, empty `src/llm_schema_lite/parsers/typescript_parser.py` and
  `src/llm_schema_lite/validators/typescript_validators.py` (no imports, no tests).
  (`lsl-2026-09-04-013`)
- The unreachable `BaseModel = None` `ImportError` fallback in `core.py` — pydantic is a
  hard dependency, so the except branch never ran. (`lsl-2026-09-04-013`)
- The dead, already-commented-out `_warm_cache` method and its orphaned call-site comment
  in `formatters/base.py`. (`lsl-2026-09-04-013`)
- **About 5,200 lines of dead or duplicated code, with no output change.** Every
  formatter output over 297 schemas, every DSPy prompt, and 1,350 captured small-model
  completions replayed through every parse path compare byte-identical before and after
  (36,223 probes), as do the offline prompt-cost report and a live accuracy rerun.
  - Legacy JSONSchemaBench feature-coverage tooling (`benchmarking/base.py`,
    `benchmarking/jsonschemabench/`) and its 54 tests; 89 unused `conftest.py` fixtures,
    `TestDataFactory`, and the 16 raw schemas only they used.
  - Uncalled helpers (`_is_problematic_schema`, `build_alias_map`,
    `BaseParser._extract_content`, the `_parse_json`/`_parse_yaml` adapter wrappers),
    unreachable YAML-extraction branches, and import guards for required dependencies.
  - Duplicates folded into one copy: the JSONish and YAML `anyOf`/`oneOf`/`allOf`
    renderers, three `process_additional_properties`, the two validators' `validate`,
    `SchemaLite`'s and `JSONishFormatter`'s token counting, the rescue repairs' tree walk,
    and the benchmark's report writers, markdown tables and adapter registry.
  - `StructuredOutputAdapter` code that repeated DSPy's `JSONAdapter`:
    `format_assistant_message_content` is inherited, and `user_message_output_requirements`
    and the user branch of `format_field_with_value` call `super()`.
  - Tooling nobody used: the `psutil`, `types-requests`, `coverage`, `gitlint` and
    `commitlint` dev packages (the last two come from their pre-commit repos),
    `[dependency-groups]`, `MANIFEST.in`, the unused pytest markers,
    `make test-parallel`/`test-fast`/`test-slow`, and ruff settings that restated defaults.

### Fixed

- **A leading `<think>`/`<thinking>` block is no longer parsed as the answer.** JSON
  extraction took the first `{`, so a reasoning model that restated the format
  (`{"answer": ...}`) inside its think block had that placeholder returned as the answer,
  with no error. JSON and YAML parsing now drop such a block at the very start of the
  reply; the tag inside a value is untouched. When the chat template supplied the opening
  tag, so the reply holds only `</think>`, everything up to that tag is dropped if it
  starts a line.
- **A `{` or `[` inside a JSON string no longer breaks extraction.** Brace counting
  ignored strings, so a code snippet such as `"if (user) {"` in a prose-wrapped reply
  never balanced, and the reply failed where DSPy's `JSONAdapter` parsed it
  (stanfordnlp/dspy#8759).
- **JSON mode no longer sends Pydantic `x-*` keys to the provider.**
  `json_schema_extra={"x-...": ...}` reached `response_format`, which strict-schema
  providers such as Bedrock reject with a 400 (stanfordnlp/dspy#9686). Field names that
  start with `x-` are kept.
- `SchemaLite.compare_tokens()` for TypeScript and YAML cached its first counts and
  returned them for every later call, even with a different `original_schema`,
  `simplified_schema` or `encoding`. Each call now counts afresh.
- **JSONish no longer turns a described union's members into comments.** For
  `u: A | B = Field(description="either")` where `A` has a field with a default, the
  union's comment was split at the first `" // "` in the whole rendering, which was
  `A`'s `(default=1)`. The rest of `A` and all of `B` came out as `//` comment lines, so
  the model never saw those fields, and internal `⟪lslid…⟫` tokens leaked into the
  prompt. The split now happens only inside the node's own comment, which also stops a
  `pattern` containing `" // "` from being cut. Output that was already correct is
  unchanged.
- **Output-field envelope: a schema is now bound to the key it must be emitted under.**
  In SECTIONS layout an output schema opened a brace at column 0 on its own line, so a
  small model read it as the response *envelope* and emitted the record bare —
  `{"name": "Ada", ...}` instead of `{"record": {"name": "Ada", ...}}`. Valid JSON, every
  value correct, no output field for DSPy to find: `AdapterParseError`, and a perfect
  extraction scored zero. The schema is now prefixed `"<field>": `. Measured on
  `qwen3.5:0.8b` over 30 labeled extraction cases, JSONISH went 0/30 parsed to 30/30
  (0.000 -> 0.745 field accuracy); JSON mode was unchanged. Costs 2-3 prompt tokens on the
  8 of 53 matrix cells that carry an output schema, and nothing elsewhere. YAML gets the
  same fix in its own syntax (below) rather than this JSON prefix, which measured as a
  *regression* there (0.849 -> 0.760) because it is not valid YAML.
- **YAML mode's prompt no longer demonstrates a shape it is not asking for.** The output
  section rendered `[[ ## record ## ]]` markers under a "respond with YAML" instruction —
  two incompatible requests. JSON and JSONish hide the same contradiction because they
  send `response_format={"type": "json_object"}`, which forces the reply back into shape;
  YAML sends no response format, so the demonstration *is* the specification and models
  follow it. `qwen3:8b` returned 0 parseable replies out of 18 in the outcomes arm. The
  block is now real YAML (`record:` with the schema nested under it), which also removes
  the envelope ambiguity: 18/18 `ok`, and 2-8 fewer prompt tokens per signature. YAML
  mode wants `parse_config=ParseConfig()`: its replies hit the same all-null list item as
  JSONish (see the rescue below), and pruning it moved `qwen3.5:0.8b` from 0.714 to 0.848
  field accuracy over 40 cases, 16/40 -> 22/40 exact. YAML's typed plain scalars
  (`postcode: 95014` loads as an int) are not repaired yet: the coercion rescue covers
  scalar output fields only, not fields nested inside a model.
- **A markdown code fence no longer has to carry a language tag to be extracted.**
  `_extract_from_markdown` matched only ` ```json ` / ` ```yaml `, so a reply fenced with a
  bare ` ``` ` — or mislabelled ` ```python ` — was parsed whole, and a single line of
  prose after the closing fence was enough to fail it. Tagged patterns are still tried
  first. Measured on `llama3.2:1b` in YAML mode: 15/30 parse errors -> 3/30, field
  accuracy 0.080 -> 0.225.
- **Parse-time rescue for all-null list items** (`ParseConfig` only, off when
  `parse_config is None`). Small models answer an empty list with a placeholder rather than
  `[]` — `{"contacts": [{"email": null, "phone": null}]}` — which fails a required
  `email: str` and costs the whole record. Coercion cannot repair it without inventing a
  value; dropping an item whose every field is `null` removes nothing the model extracted.
  Rescue-only, so a reply that validates is never touched and a legitimately all-optional
  item survives. Measured on `qwen3.5:0.8b`: 0.745 -> 0.929 field accuracy, 24/30 -> 30/30
  parsed; inert on `granite3.1-moe:1b` and `llama3.2:1b`, which do not make the mistake.
- **Benchmark trial isolation.** `run_live_arm` shared one `dspy.LM` across every cell, so
  a fixed `seed` made all N trials of a cell one deterministic request replayed N times —
  N trials carried one trial's information and every `stddev` column read `0.000` as if it
  were confidence. The seed is now offset per trial.
- **Input-structure header dropped for all-bare signatures.** When every input is a plain
  `str` the "Inputs will have the following structure:" header introduced nothing but field
  markers. Exactly -7 tokens on all 36 `sola-*` matrix cells, zero on `chat`/`json`/`baml`;
  byte-identical whenever any input carries a note or a schema.

- **YAML no longer appends a document-level `# any properties allowed` for a nested mapping.**
  A `dict[str, Model]` property used to increment a placeholder counter whose only effect was
  a trailer comment after the last root field, which read as "the root object accepts
  arbitrary keys" even when the root schema had no `additionalProperties` at all. YAML now
  matches JSONish and TypeScript, which have been silent for mappings since
  `lsl-2026-09-04-015`. This also removes a double emission: a root with both a nested
  structural mapping and a complex root `additionalProperties` used to print the comment
  twice. The root `additionalProperties` channel itself is unchanged and still describes an
  open or typed root exactly once. (`lsl-2026-09-05-010`)
- **A `propertyNames` schema no longer leaks a raw Python dict repr into a comment.** An
  object-shaped property carrying `propertyNames` used to render
  `# propertyNames: {'pattern': '^[a-z]+$'}` in YAML and TypeScript; `propertyNames` is now
  suppressed from the metadata channel for every node kind, not just mappings, where the key
  token already states it. (`lsl-2026-09-05-010`)
- **Constraint metadata at nested positions (tuple element, mapping value, array item) now
  honours `include_constraints` and `metadata_inclusion` in every formatter.** Only the
  JSONish tuple element leaked at HEAD; the fix is in the shared base method every nested
  position funnels through, so array item and mapping value positions (already correct)
  are unaffected. Snapshot-visible: a JSONish tuple element's constraint spelling now
  matches its sibling scalar property exactly. (`lsl-2026-09-05-009`)
- **TypeScript's `minLength`/`maxLength` and `minimum`/`maximum` gates no longer `or` over
  the pair.** Disabling one bound via `metadata_inclusion` now renders the surviving bound
  in its one-sided form (`>= n` / `<= n`) instead of the full two-sided range. Snapshot-visible
  for any TypeScript output using `metadata_inclusion` to disable exactly one bound of a pair.
  (`lsl-2026-09-05-009`)
- **JSONish comment text now reproduces the authored description verbatim.** Quotes,
  backslashes and `//` in a `description` no longer leak `\"` or `\\` into the rendered
  comment, and a description containing a real newline collapses to a single `//` line
  instead of escaping to a literal `\n` or breaking out of the comment as a bare
  document line. Nothing is decoded: an authored two-character `\n` still renders as two
  characters. YAML and TypeScript output is unchanged. (`lsl-2026-09-05-005`)
- **A property-level `description` on a `$ref` field is no longer dropped in JSONish.**
  An enum `$ref` renders `one of: "US", "CA"; Country code`; an object `$ref` renders the
  property's description as the block's trailing comment while the referenced
  definition's docstring keeps the opening line. The two are de-duplicated when their
  text is equal, and both honour `include_metadata` / `include_descriptions`.
  (`lsl-2026-09-05-005`)
- **No JSONish line ends in whitespace**, across every model and every
  `include_metadata` / `include_descriptions` / `include_constraints` combination, and
  `transform_schema()` now returns the identical string on repeat calls instead of
  serving a whitespace-collapsed copy from its cache. (`lsl-2026-09-05-005`)
- **Root-level `list[Model]`, `Model | None` and `list[Model] | None` schemas now render
  with the same block conventions as nested ones.** JSONish no longer leaks a Python dict
  repr at the root, no longer drops `OR null` for a root `anyOf`, and no longer prints a
  stray `// Array of (items):` header; YAML renders root arrays and root optionals as
  block form with per-field metadata on its own field and nullability as a leading
  `# OR null`. (`lsl-2026-09-05-002`)
- **A recursive root model keeps its title/docstring header and required-marker legend**
  in all three formats, and TypeScript no longer emits a duplicated `interface <Def>`
  beside `interface Schema`. (`lsl-2026-09-05-002`)
- **The DSPy adapter's required-marker legend now appears for YAML block sequences**
  (`- name*: …`), closing a latent gap that also affected nested list blocks. The adapter's
  temporary tier-2 root-array unwrap is deleted — the formatters own root-array rendering
  now, and the adapter's rendered block equals `simplify_schema(...).to_string()` for every
  probed shape in both modes. (`lsl-2026-09-05-002`)
- Stray trailing quote after schema comments in JSONish output. (`32145bb`)
- Pydantic auto-generated titles no longer leak into comments: property titles are dropped
  at every depth when the title merely restates the field name, and the root title is
  dropped when it is identifier-shaped and the model has a docstring. User-set titles
  survive. (`15b937e`)
- Nested `$ref` models render as inline schema blocks instead of Python dict reprs,
  including `Model | None`, `list[Model]` and a model referenced from two sibling fields.
  (`0047c95`)
- Self-referential models no longer recurse without bound: rendering stops at
  `FormatterConfig.max_recursion_depth` and leaves a `recursive: <TypeName>` marker, in
  all three formatters. (`d35090b`, `7430cb2`, `caa8122`, `3d6e8b2`, `4548d98`)
- `include_metadata`, `include_descriptions` and `include_constraints` are now live rather
  than dead: every keyword passes one category gate. Structural output (required markers,
  container tokens) is correctly classified as non-metadata and always emitted. (`99b6aa7`)
- A DSPy reply that omits an optional output field now yields that field's default via
  `apply_output_field_defaults`, instead of failing the completeness check with
  `AdapterParseError`. The YAML→JSON rescue is narrowed to a `ConversionError` around
  extraction only. (`7a29f2d`)
- `make test-dspy` matched no files and exited 4; it now runs
  `pytest tests -k dspy -v --no-cov`. (`e3f2274`)
- `JSONishFormatter` no longer rewrites `union_separator` on the caller's
  `FormatterConfig`. Both JSONish and YAML now apply their `" OR "` default through
  one helper in `formatters/config.py` that copies rather than writes through, so a
  single config can be reused across formatters -- and by `StructuredOutputAdapter` --
  without a later TypeScript render emitting `number OR string`. (`bfa544e`)
- JSONish deferred postfix/prefix/recursion comments are now keyed by a per-occurrence
  identity instead of by bare property name: a nested field no longer inherits an outer
  same-named field's description or constraint, and `recursive: <Type>` now marks only
  the truncated level of a recursive model instead of every expanded level above it.
  A property literally named `__additional_properties__` also renders correctly instead
  of being swallowed by the sentinel machinery. Snapshot-visible: `recursive:` marker
  placement and same-name-collision comment text change on the next render; output for
  every schema without a same-named property at two depths is byte-identical. (`5cbfc23`)
- YAML no longer splits a field's value at the first `#`. The deferred-comment hoist used
  to treat any `comment_prefix` inside a rendered line as a pre-existing comment marker, so
  a regex pattern containing `#` (`^#[0-9a-f]{6}$`) was cut in half and the constraint the
  LM had to satisfy was destroyed. The hoist now owns only the text it minted and emits the
  value intact; a line that already carries a literal comment ends with two comments, which
  YAML and JavaScript both accept. (`4da3d1b`)
- Multi-line per-field descriptions no longer emit bare document lines. Every continuation
  line becomes a comment at the field's own indent -- including inside nested blocks and
  `- ` sequence items in YAML, and at the fixed member indent in TypeScript -- so
  `yaml.safe_load` accepts the output for every model, and a TypeScript interface no longer
  gains an orphan `line two;`. An inline object literal folds the continuation lines back
  into its `/* ... */` comment instead of leaking a `//`. (`4da3d1b`)
- YAML states `pattern` and `format` once per field instead of twice. The type token
  already renders `(PATTERN: ...)` / `(FORMAT: ...)` for a `"type": "string"` node, and the
  trailing metadata comment restated both. The comment now omits them for exactly that node
  shape; a `pattern` on a node that is not a typed string is still reported, and TypeScript
  and JSONish are unaffected. (`4da3d1b`)
- **A TypeScript `Literal` containing a quote, a backslash or a newline no longer emits
  unparseable output.** `Literal["line1\nline2"]` previously wrote a raw newline that split
  the interface across three physical lines and left a bare, uncommented document line;
  `Literal['say "hi"']` previously emitted unbalanced quotes. Both now route through the
  same `format_literal_value` helper JSONish and YAML already used.
  (`lsl-2026-09-05-011`)
- **A `const` field no longer restates its own value as a comment.** `version: v1  //
  (defaults to v1), const: v1` now reads `version: "v1";  // (defaults to v1)` — the default
  survives, the tautology is gone. Separately, a `$ref` to a const-only `$def` and a `const`
  member of an `anyOf`/`oneOf` now render through each formatter's own const renderer
  instead of a bare `str()`, so YAML gets its `# one of: ...` idiom where it previously
  leaked the raw value; JSONish and YAML output is byte-identical for every fixture model.
  (`lsl-2026-09-05-011`)

- **Rendering the same schema twice on one formatter instance now returns the same string.**
  A second `YAMLFormatter.transform_schema()` on an instance built with
  `FormatterConfig(prefix=...)` silently dropped the prefix — `"# P\n# Title: Outer\n..."` on
  the first call, `"# Title: Outer\n..."` on the second — because the cached `_processed_data`
  branch returned before the prefix line the main flow ends on. Measured at 80/80 conftest
  models under every prefix-carrying config. Both formatters' cached branches are gone, so
  there is exactly one render path and the second call reproduces the first by construction;
  a second, latent asymmetry went with them (the cached YAML branch inferred "did we inject a
  placeholder key?" by scanning rendered keys for `<...>`, which a property literally named
  `<x>` would have fooled). **First-render output is unchanged** for all 4,326 corpus
  renderings — this is a repeat-render fix only. Reachable only by calling `transform_schema()`
  twice on one instance; `simplify_schema(...).to_string()` memoizes the string and was never
  affected. Covered by the new `tests/test_repeat_render.py`. (`lsl-2026-09-05-013`)
- `StructuredOutputAdapter.parse` now strips the required marker its own prompt renders
  (`FormatterConfig.required_marker`) from reply keys, in JSON, JSONish and YAML modes, with
  and without a `ParseConfig`. A key matching an output field verbatim always wins.
  (`944b193`)
- `StructuredOutputAdapter.parse` now accepts a top-level JSON array wrapping the reply
  object, matching upstream `dspy.JSONAdapter` (including nested arrays, fenced and indented
  forms, and arrays whose leading elements are not objects). (`944b193`)
- `loads(text, schema=Model)` now strips required markers at every nested object level,
  resolving `$ref`, `anyOf`/`oneOf`/`allOf`, `items`, `prefixItems` and
  `additionalProperties` values, depth-capped at 8. Keys of an open-ended `dict[str, X]`
  field are never rewritten. (`944b193`)
- **`dspy.ToolCalls` output fields now render schema guidance in the system prompt** instead
  of a bare placeholder. In `OutputMode.JSON` the note stem and the JSON schema object match
  upstream `dspy.JSONAdapter`; in JSONish and YAML the equivalent simplified schema is
  emitted by the project's own formatters. (`lsl-2026-09-05-007`)
- `user_message_output_requirements` now emits upstream's concrete hint for a `ToolCalls`
  output — `(must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]})` —
  in all three output modes, replacing the unhelpful "must be formatted as a valid Python
  ToolCalls". (`lsl-2026-09-05-007`)
- The `dspy.Type` / `dspy.History` carve-out is now element-aware: parameterised annotations
  such as `list[dspy.Tool]`, `list[dspy.Image]`, `Optional[dspy.Image]` and
  `dict[str, dspy.Image]` no longer receive a misleading schema note, where previously only
  bare class annotations were carved out. A `list[dspy.ToolCalls]` output is deliberately
  silent — DSPy only recognises an exact `ToolCalls` annotation. (`lsl-2026-09-05-007`)
- `FormatterConfig(required_marker="")` no longer emits the nonsense legend line
  `Fields marked with  are required`. (`lsl-2026-09-05-007`)
- **The offline benchmark arm, the doc-example tests and `examples/basic_usage.py` no longer
  require network access.** `tiktoken` downloads the `cl100k_base` BPE table on a cold cache;
  every one of these surfaces crashed on a machine with no cache and no network, and a
  full-suite run passed only because importing `litellm` silently repointed
  `TIKTOKEN_CACHE_DIR`. The benchmark runner now reports `prompt_tokens` as unavailable (`—`
  in markdown, an empty CSV cell) instead of raising, `pytest` seeds the cache deliberately
  before collection, and the example script prints an honest "unavailable" line. (`f1191eb`)
- **The benchmark's `response_format_sent` column no longer reports `none` for requests the
  endpoint rejected.** The value is now observed from the LM call itself through a scoped
  `dspy.BaseCallback`, so a call that raises before DSPy records its history still reports
  what was sent. The DSPy issue #1871 reproduction table now correctly shows `json_object`
  for the four adapters whose `json_object` request is refused; `none` now means "no LM call
  was attempted". (`e9e8204`)
- **Benchmark results can no longer record an absolute machine path or a credential.** The
  provenance `command:` line is reconstructed as the documented
  `python -m benchmarking.dspy_adapters …` form with shell-quoted arguments instead of
  `" ".join(sys.argv)`, and credential-shaped keys in `LSL_BENCH_LM_KWARGS` (plus
  `user:pass@` in `LSL_BENCH_API_BASE`) are redacted to `<redacted>`, including inside nested
  objects. Matching is by whole key segment, so `max_tokens` and the other measurement kwargs
  are untouched. (`00d07ad`)
- The committed offline results (`results/prompt-cost-2026-09-05.*`) were regenerated. Eight
  rows change: six YAML-mode rows from `ba5d7af` and two JSONish `recursive` rows from
  `52d59c9`, not from these fixes. (`00d07ad`)
- **Pre-commit runs the test suite once per commit, under the project venv.** The `pytest`
  hook is now staged at `pre-commit` only (it also ran at `commit-msg`), runs with `--no-cov`
  so `htmlcov/` and `coverage.xml` are no longer written into the tree, and is invoked through
  `uv run --no-sync`, so the DSPy integration tests are no longer silently skipped by a
  pyenv interpreter without dspy. ruff is pinned to the same 0.14.x in pre-commit and the dev
  extra, so `make format` and the hook agree on formatting, and
  `benchmarking/dspy_adapters` is now type-checked by `make lint` and CI.
  (`lsl-2026-09-05-014`, `560cf64`)

<!--
  Everything above this marker is hand-written. `make changelog` (git-changelog, in-place via
  [tool.git-changelog] in pyproject.toml) preserves it byte-for-byte and replaces only the
  marker line below: it appends an auto-generated section for a `v0.7.0` tag that does not
  exist, with links built from a local SSH host alias. Treat the target as manual-only —
  review and discard that generated section before committing.
-->
<!-- insertion marker -->
## [v0.6.1](https://github.com/rohitgarud/llm-schema-lite/releases/tag/v0.6.1) - 2025-10-27

<small>[Compare with v0.6.0](https://github.com/rohitgarud/llm-schema-lite/compare/v0.6.0...v0.6.1)</small>

### Bug Fixes

- issue due to boolean ref definition ([fd00aba](https://github.com/rohitgarud/llm-schema-lite/commit/fd00aba1b9a0522cc8cf4285251416d6fbad9874) by Rohit Garud).

### Chore

- Handle edge cases ([aea4ff1](https://github.com/rohitgarud/llm-schema-lite/commit/aea4ff16a3e8af6e25924bd9debd4ff530183010) by Rohit Garud).

### Tests

- Improve test coverage ([a58a5c0](https://github.com/rohitgarud/llm-schema-lite/commit/a58a5c047622072a248563dd31f092287e14fdef) by Rohit Garud).

## [v0.6.0](https://github.com/rohitgarud/llm-schema-lite/releases/tag/v0.6.0) - 2025-10-25

<small>[Compare with v0.5.0](https://github.com/rohitgarud/llm-schema-lite/compare/v0.5.0...v0.6.0)</small>

### Features

- enhance constraint integration across all formatters ([66fc219](https://github.com/rohitgarud/llm-schema-lite/commit/66fc219d3567b314b5e0ee4c2df931b6b80e9876) by Rohit Garud).
- Add required field highlighting with asterisk notation ([84b9e3e](https://github.com/rohitgarud/llm-schema-lite/commit/84b9e3edb304bbd2aad3db2fa0636728c3b88658) by Rohit Garud).
- add comprehensive schema validation with multiple error collection ([9c311c7](https://github.com/rohitgarud/llm-schema-lite/commit/9c311c7e5bd28cd93c81a67929abd87a7a39d115) by Rohit Garud).

## [v0.5.0](https://github.com/rohitgarud/llm-schema-lite/releases/tag/v0.5.0) - 2025-10-24

<small>[Compare with v0.4.0](https://github.com/rohitgarud/llm-schema-lite/compare/v0.4.0...v0.5.0)</small>

### Features

- enhance core functionality with advanced JSON Schema features ([0e88e4b](https://github.com/rohitgarud/llm-schema-lite/commit/0e88e4b149787e7121ae0b076d3c7a1ac5ced58a) by Rohit Garud).

### Bug Fixes

- resolve mypy type errors in formatters ([657468f](https://github.com/rohitgarud/llm-schema-lite/commit/657468f177a72645f676759cbfb906ba6ea06408) by Rohit Garud).

### Tests

- Improve test coverage ([37b6776](https://github.com/rohitgarud/llm-schema-lite/commit/37b67763888990cf52995a265057940d831490bb) by Rohit Garud).
- comprehensive test refactoring and consolidation ([46b395d](https://github.com/rohitgarud/llm-schema-lite/commit/46b395d46aee8175299eb114b8d6ddc4993a9e34) by Rohit Garud).

## [v0.4.0](https://github.com/rohitgarud/llm-schema-lite/releases/tag/v0.4.0) - 2025-10-19

<small>[Compare with v0.3.0](https://github.com/rohitgarud/llm-schema-lite/compare/v0.3.0...v0.4.0)</small>

### Features

- add robust loads() function for unified JSON/YAML parsing ([0cb5f2b](https://github.com/rohitgarud/llm-schema-lite/commit/0cb5f2b8285e59a6697e5488e3fc24919210419e) by Rohit Garud).

### Bug Fixes

- Mypy issues ([b907784](https://github.com/rohitgarud/llm-schema-lite/commit/b907784b62c7289d57487312e7baf68c9b6f22de) by Rohit Garud).

## [v0.3.0](https://github.com/rohitgarud/llm-schema-lite/releases/tag/v0.3.0) - 2025-10-18

<small>[Compare with v0.2.2](https://github.com/rohitgarud/llm-schema-lite/compare/v0.2.2...v0.3.0)</small>

### Features

- add DSPy integration with StructuredOutputAdapter ([0577d79](https://github.com/rohitgarud/llm-schema-lite/commit/0577d790392fd0c48756a837a61655ad7757fb3f) by Rohit Garud). BREAKING CHANGES: None

### Bug Fixes

- CI issues ([ca15865](https://github.com/rohitgarud/llm-schema-lite/commit/ca15865dbff247bd62ddc3fb748ad23d78654af0) by Rohit Garud).
- CI issue ([1079832](https://github.com/rohitgarud/llm-schema-lite/commit/107983288cddb34b5509f043d386597057164089) by Rohit Garud).

### Tests

- Add tests for dspy integration ([5b44d11](https://github.com/rohitgarud/llm-schema-lite/commit/5b44d1115a9c909c5390c703babe2a54bf42e3b7) by Rohit Garud).

## [v0.2.2](https://github.com/rohitgarud/llm-schema-lite/releases/tag/v0.2.2) - 2025-10-15

<small>[Compare with v0.2.1](https://github.com/rohitgarud/llm-schema-lite/compare/v0.2.1...v0.2.2)</small>

### Bug Fixes

- update CI workflow for codecov ([ae06ec4](https://github.com/rohitgarud/llm-schema-lite/commit/ae06ec48b112d574c0181b2ee9cfb5b63d572f7d) by Rohit Garud).

### Chore

- update ci pipeline for codecov ([74d3abd](https://github.com/rohitgarud/llm-schema-lite/commit/74d3abd745b6744fec21aec0a8c38fba42feaaa9) by Rohit Garud).

### Performance Improvements

- apply optimizations formatters ([736f221](https://github.com/rohitgarud/llm-schema-lite/commit/736f221c55e998c5ec96602bd4bbe838d36a1cff) by Rohit Garud).

## [v0.2.1](https://github.com/rohitgarud/llm-schema-lite/releases/tag/v0.2.1) - 2025-10-14

<small>[Compare with v0.2.0](https://github.com/rohitgarud/llm-schema-lite/compare/v0.2.0...v0.2.1)</small>

### Docs

- update CHANGELOG for v0.2.0 ([c504369](https://github.com/rohitgarud/llm-schema-lite/commit/c50436906db08107c3a26e0dafea1e83af62fd8d) by github-actions[bot]).

## [v0.2.0](https://github.com/rohitgarud/llm-schema-lite/releases/tag/v0.2.0) - 2025-10-14

<small>[Compare with v0.1.0](https://github.com/rohitgarud/llm-schema-lite/compare/v0.1.0...v0.2.0)</small>

### Features

- implement core llm-schema-lite with multi-format support ([849e87f](https://github.com/rohitgarud/llm-schema-lite/commit/849e87f33d39f941579a550eab963ac00cf02aaf) by Rohit Garud).

### Bug Fixes

- remove manual v0.2.0 entry to allow auto-generation ([172a2cc](https://github.com/rohitgarud/llm-schema-lite/commit/172a2ccc32d027ae6629d34f904f72f59c3511ee) by Rohit Garud).
- fix email-validator dependecy issue ([99251db](https://github.com/rohitgarud/llm-schema-lite/commit/99251db68c4af4217b8e3ed0010ff890e8fbfc72) by Rohit Garud).
- dynamic version issue in pyproject.toml ([40afd5e](https://github.com/rohitgarud/llm-schema-lite/commit/40afd5ec03eb16c8e95131e75b5887396864344b) by Rohit Garud).

### Docs

- Update readme and changelog ([a4696cb](https://github.com/rohitgarud/llm-schema-lite/commit/a4696cb58c99c8888cd0cdaadec5102869119332) by Rohit Garud).

### Chore

- add examples and update dependencies ([ffde05b](https://github.com/rohitgarud/llm-schema-lite/commit/ffde05b22c8b7a786a98c0dcb01125039ce80df1) by Rohit Garud).
- remove static versioning from pyproject.toml ([6fced87](https://github.com/rohitgarud/llm-schema-lite/commit/6fced872ffc1126527ff99e847cd582a490d95d9) by Rohit Garud).
- rename to llm-schema-lite ([914da1f](https://github.com/rohitgarud/llm-schema-lite/commit/914da1fca8873731302453b5eeaafcacc7b23749) by Rohit Garud).

### Tests

- add comprehensive test suite with 112 tests ([ea4f608](https://github.com/rohitgarud/llm-schema-lite/commit/ea4f60852bbe9bd917d68bb8bf5f3f3e9ce9fbad) by Rohit Garud).

## [v0.1.0](https://github.com/rohitgarud/llm-schema-lite/releases/tag/v0.1.0) - 2025-10-12

<small>[Compare with first commit](https://github.com/rohitgarud/llm-schema-lite/compare/51766385b3c7e7172bfae5e8e8c8b1b431ad0a24...v0.1.0)</small>

### Chore

- initial package structure ([17462ec](https://github.com/rohitgarud/llm-schema-lite/commit/17462ec0c8e135e202d58cf808ac732396ed8d58) by Rohit Garud).
