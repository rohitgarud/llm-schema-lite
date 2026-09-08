# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

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

### Added

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
  (default `"*"`), the reply-side counterpart of `FormatterConfig.required_marker`.
  (`7a29f2d`)
- DSPy per-field streaming support for `StructuredOutputAdapter` in JSON and JSONish
  modes: `register_streaming_support()` runs on `llm_schema_lite.dspy_integration` import,
  and the new public `StreamingNotSupportedError` is raised — before any LM request — when
  YAML mode is combined with stream listeners. (`c88a0ce`)
- `StructuredOutputAdapter.format_finetune_data()` is implemented and returns an OpenAI
  chat-format record instead of raising `NotImplementedError`. (`8bcf58b`)
- `FormatterConfig.max_recursion_depth`, default `2`. (`2a68581`)
- A DSPy-latest CI canary job. (`f31ac3e`)

### Changed

- Documentation: the top-level README gains Installation, Quick Start and DSPy Integration
  sections; the DSPy integration README is corrected against the shipped adapter; and
  every runnable code block in both, plus `examples/basic_usage.py`, is now executed by
  `tests/test_docs_examples.py`.
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

### Fixed

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
  than dead: every keyword passes one category gate, and `FormatterConfig` is copied rather
  than mutated in place. Structural output (required markers, container tokens) is
  correctly classified as non-metadata and always emitted. (`99b6aa7`)
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
