# DSPy adapter benchmark

## 1. What this is

This package benchmarks `llm_schema_lite.dspy_integration.StructuredOutputAdapter`
against the upstream DSPy adapters (`ChatAdapter`, `JSONAdapter`, `BAMLAdapter`) across
a small matrix of signatures.

It has **two arms**, and they share one cell registry (the same adapter ids, the same
signature ids) but they are **never joined into one table**, anywhere, in any file:

- **`prompt-cost`** — offline, exact, deterministic, model-free. It calls
  `adapter.format()` directly and counts the resulting prompt with `tiktoken`
  (`cl100k_base`). No `dspy.LM`, no network, no environment variables.
- **`outcomes`** — live, against a real `dspy.LM` endpoint (verified against Ollama
  serving `qwen3:8b`). Token counts here are whatever the provider reports back, and a
  live run is not deterministic run to run.

These are fundamentally different kinds of numbers — one is a property of the text, the
other is a property of a specific model's specific response on a specific day — so
`report.py` writes them to separate `PromptRow` / `TrialRow` dataclasses, separate CSV
files, and separate markdown files. There is no code path that merges a `prompt-cost`
row with a `live` row.

**Live-run status:** the committed live artefacts in `results/` were produced by a real
run against Ollama (`qwen3:8b`), see `results/live-openai-qwen3-8b-2026-09-05.md` for the exact
command, git head, and effective `lm_kwargs`. If no `results/live-*.md` file is present
for the current date, the live pass was not run and no live numbers exist for that date
— this file does not claim otherwise.

## 2. Quick start

Offline arm — no endpoint needed:

```
make bench-dspy BENCH_ARGS=--offline
```

Live arm — needs a running OpenAI-compatible endpoint (verified against Ollama):

```
export LSL_BENCH_MODEL=openai/qwen3:8b
export LSL_BENCH_API_BASE=http://localhost:11434/v1
export LSL_BENCH_API_KEY=not-needed                # optional, default "not-needed"
export LSL_BENCH_LM_KWARGS='{"max_tokens": 900}'   # optional, JSON object

make bench-dspy BENCH_ARGS=--live
```

`make bench-dspy` with no `BENCH_ARGS` runs **both** arms (the live env must be set).

## 3. `LSL_BENCH_*` environment variables

| var | required | default | semantics |
|---|---|---|---|
| `LSL_BENCH_MODEL` | **yes** (live only) | — | litellm model string, `dspy.LM`'s first positional arg. Verified: `openai/qwen3:8b`. |
| `LSL_BENCH_API_BASE` | **yes** (live only) | — | OpenAI-compatible base URL. Verified: `http://localhost:11434/v1`. |
| `LSL_BENCH_API_KEY` | no | `"not-needed"` | Must be non-empty for litellm; Ollama/LM Studio ignore the value. |
| `LSL_BENCH_LM_KWARGS` | no | `{}` | JSON object, merged **last** into `dspy.LM(**kwargs)`. Sole tuning escape hatch. |

Only `config.py` reads these. `--offline`, `--list`, and `--repro-1871` (with no live
probe configured) never touch `os.environ` at all.

## 4. CLI flags and exit codes

`python -m benchmarking.dspy_adapters` (equivalently `make bench-dspy BENCH_ARGS=...`)
exposes eight flags:

| flag | `argparse` type | default | meaning |
|---|---|---|---|
| `--offline` | `store_true` | `False` | Offline prompt-cost arm **plus** the synthetic #1871 reproduction. No env, no network, no `dspy.LM`. Sub-second. |
| `--live` | `store_true` | `False` | Live outcomes arm only. Requires the two env vars. |
| `--adapters` | `str` (comma-separated) | offline: all 9 · live: the 6 `*-sections` ids | Filter by adapter id. Unknown id → stderr listing valid ids, exit 2. |
| `--signatures` | `str` (comma-separated) | all 6 | Filter by signature id. Unknown id → same treatment. |
| `--trials` | `int` | `1` | Live-arm repetitions per cell. Ignored by the offline arm (deterministic). |
| `--out` | `Path` | `<package dir>/results` | Output directory; created if absent. |
| `--repro-1871` | `store_true` | `False` | Run **only** the #1871 work: the offline synthetic reproduction always, plus `probe_1871_live()` iff the live env is configured. |
| `--list` | `store_true` | `False` | Print every adapter id and signature id with its `config_repr`, then exit 0. |

Dispatch: neither `--offline` nor `--live` given ⇒ **both arms** run (live env
required). `--offline --live` together ⇒ same as neither (both arms run). `--list` and
`--repro-1871` short-circuit everything else and never write a file.

Exit codes:

| code | meaning |
|---|---|
| `0` | The requested arms ran and every requested results file was written. **Recorded cell failures do not affect this** — a `format_error` row is data, not a harness fault. |
| `1` | The harness itself failed: output dir unwritable, or the live arm ran ≥ 1 cell **and** every row has `error_class == "LMTransportError"` (the exact leaf class, not merely `outcome == transport_error`) — reported as `error: no live cell reached the endpoint; is <api_base> up?` |
| `2` | Misconfiguration: missing env, bad `LSL_BENCH_LM_KWARGS` JSON, unknown `--adapters` / `--signatures` id. |

Use `--adapters`/`--signatures`/`--list` to do a tiny pass rather than the full matrix
— see §11 below on live-call latency.

## 5. The two arms, explained

**`prompt-cost` (offline).** Each cell in the matrix calls the adapter's `format()`
method directly on a signature and its inputs, with no `dspy.LM` involved. The
resulting list of chat messages is concatenated and tokenized with `tiktoken`'s
`cl100k_base` encoding. This is exact and deterministic: the same adapter, the same
signature, and the same inputs always produce the same `prompt_tokens` count, on any
machine, with no model required.

**`outcomes` (live).** Each cell runs the signature through a real `dspy.Predict` call
against the configured `dspy.LM`. Whether the call succeeds, how it fails if it
doesn't, wall-clock time, and token counts are all **provider-reported** — litellm
relays whatever the endpoint's response says. These numbers can vary run to run (model
sampling, provider-side batching, load) even with `temperature=0.0` and a fixed `seed`,
so the live arm supports `--trials` to run each cell more than once.

## 6. The C5 caveat — response_format re-bills the thinking trace

On Ollama, sending `response_format={"type": "json_object"}` changes how the *same*
underlying model output is billed. Measured directly: for one adapter/signature pair,
the same 649-character reasoning ("thinking") trace was reported as **21 prompt tokens
/ 160 completion tokens** without `response_format`, and as **173 prompt tokens / 7
completion tokens** with `response_format` — the totals are close (181 vs 180) but the
split is not.

This matters because not all adapters send `response_format` the same way: `chat` and
the three `sola-yaml-*` adapters send **no** `response_format` at all, while the other
six adapters (`json`, `baml`, `sola-json-sections`, `sola-jsonish-sections`,
`sola-json-block`, `sola-jsonish-block`) send `{"type": "json_object"}`. That means
`prompt_tokens_reported` and `completion_tokens_reported` are **not on the same
accounting basis** across those two groups, and comparing them across groups would be
comparing two different billing conventions, not two different prompts.

Consequently: the live report's provider-accounting block is headed
"Provider accounting — NOT comparable across the `response_format` groups", and
`report.py` only ever aggregates **`total_tokens`** across adapters. Every live CSV and
markdown row also carries `response_format_sent` so a reader can group correctly before
comparing anything.

## 7. Known limitations of the offline count

The offline count measures the textual prompt only — it excludes `_call_preprocess`
tool/native-type handling and the `response_format` request kwarg, which is a request
field and not a message. For this matrix that is correct: the thing under test is the
text.

Concretely: `_call_preprocess`'s tool-definition injection and native-type coercion
happen outside `format()`, and `response_format` is never part of the message list that
`tiktoken` counts. Neither omission is a bug in the offline arm — it is scoped
deliberately to the prompt text, which is what `StructuredOutputAdapter` differs on.

## 8. CSV provenance rule

CSV files (`prompt-cost-*.csv`, `live-*.csv`) carry **no** header/provenance block —
they must stay plainly machine-parseable (a `csv.DictReader` should not have to skip
anything). Provenance for a CSV (generated timestamp, command line, git head, dspy
version, `llm_schema_lite` version, and — for the live CSV — the model, api base,
effective `lm_kwargs`, and capability flags) lives entirely in the **same-named** `.md`
sibling, at the top of that file, before any table.

## 9. The #1871 status

DSPy issue #1871 reports that some adapters break when `response_format` uses a JSON
Schema (`{"type": "json_schema", ...}`) rather than a bare JSON object, against certain
OpenAI-compatible endpoints.

This is **not reproducible against Ollama**: Ollama accepts both `json_object` and
`json_schema` request shapes and returns HTTP 200 for both. AC-4 is answered honestly,
not papered over, by two things together:

1. An **offline synthetic reproduction** (`run_repro_1871_offline`, always run under
   `--offline` and `--repro-1871`) that fakes the failure mode directly, independent of
   any real endpoint, so the classification logic (`classify()`) is exercised and
   locked in as a regression test.
2. A **live probe row** (`probe_1871_live`, run only when the live environment is
   configured) that hits the real endpoint and records its outcome as
   `not_reproducible` — because, on this endpoint, it genuinely does not reproduce.

The issue **does reproduce for real against LM Studio, with no code change** — only a
different `LSL_BENCH_API_BASE`. That was not run as part of this matrix (only
`qwen3:8b` via Ollama was available), so no LM Studio results file is committed here;
this is stated as a fact about the bug, not a claim of a run that didn't happen.

## 10. Adapter and signature ids

**Adapters** (nine live in `ADAPTERS`, plus one repro-only tenth in
`REPRO_1871_ADAPTERS` used exclusively by the #1871 cell set):

| id | `config_repr` |
|---|---|
| `chat` | `ChatAdapter(use_json_adapter_fallback=False)` |
| `json` | `JSONAdapter()` |
| `baml` | `BAMLAdapter()` |
| `sola-json-sections` | `StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS)` |
| `sola-jsonish-sections` | `StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS)` |
| `sola-yaml-sections` | `StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS)` |
| `sola-json-block` | `StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK)` |
| `sola-jsonish-block` | `StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK)` |
| `sola-yaml-block` | `StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK)` |
| `sola-jsonish-nojsonobject` (repro-only) | `StructuredOutputAdapter(output_mode=JSONISH, use_json_object_response_format=False)` |

`chat` is always constructed with `use_json_adapter_fallback=False` — the default
`True` turns a real `AdapterParseError` into a fake `ok` result at `lm_calls=2`, which
would silently mask parsing failures (Δ1/D3.1 in the design notes).

The live arm defaults to the six `*-sections` ids (`chat`, `json`, `baml`,
`sola-json-sections`, `sola-jsonish-sections`, `sola-yaml-sections`) — the offline arm
still covers all nine.

**Signatures** (six, `SIGNATURE_IDS` order):

| id | signature | inputs |
|---|---|---|
| `flat` | `Flat(question -> answer, confidence)` | `question` |
| `nested` | `Nested(text -> person: Person)` | `text` |
| `list_of_model` | `ListOfModel(text -> people: list[Person])` | `text` |
| `enum` | `EnumSig(text -> colour: Colour, tier: Literal["a","b"])` | `text` |
| `optional` | `OptionalSig(question -> answer, note: str \| None)` | `question` |
| `recursive` | `Recursive(text -> tree: Node)` (self-referential Pydantic model) | `text` |

**`baml` cannot run `recursive` at all.** `BAMLAdapter` raises `ValueError` at
`format()` time — before any LM call is even attempted — for the recursive `Node`
model. Both arms record this cell as `outcome == "format_error"`,
`error_class == "ValueError"`, with no `prompt_tokens` and no live call ever made for
that cell.

## 11. Practical note on live-run time

`qwen3:8b` via Ollama takes roughly 4-12 seconds per call. A full live pass (six
adapters × six signatures × `--trials`) is therefore several minutes, not seconds. Use
`--adapters`, `--signatures`, and `--list` to scope a quick check instead of running the
whole matrix, e.g.:

```
make bench-dspy BENCH_ARGS="--live --adapters json --signatures flat"
```
