# DSPy adapter benchmark

## 1. What this is

This package benchmarks `llm_schema_lite.dspy_integration.StructuredOutputAdapter`
against the upstream DSPy adapters (`ChatAdapter`, `JSONAdapter`, `BAMLAdapter`) across
a small matrix of signatures.

It has **three arms**. The first two share one cell registry (the same adapter ids, the
same signature ids); the third has its own corpus. None of them is **ever joined into one
table**, anywhere, in any file:

- **`prompt-cost`** — offline, exact, deterministic, model-free. It calls
  `adapter.format()` directly and counts the resulting prompt with `tiktoken`
  (`cl100k_base`). No `dspy.LM`, no environment variables. The arm itself makes no
  network call; `tiktoken` fetches the `cl100k_base` table once if no cache is
  reachable, and when that fetch is impossible the arm still completes and reports
  `prompt_tokens` as `—` (unavailable) rather than failing.
- **`outcomes`** — live, against a real `dspy.LM` endpoint (verified against Ollama
  serving `qwen3:8b`). Token counts here are whatever the provider reports back, and a
  live run is not deterministic run to run. It scores **validity**: did the reply parse
  and satisfy the schema.
- **`accuracy`** — live, over a synthetic corpus of labeled extraction cases, scored
  field-by-field against ground truth. It scores **correctness**, which validity cannot:
  a reply can be perfectly shaped, pass every schema check, and be entirely wrong. This
  is the arm that speaks to the package's actual claim — that a compact, explicit schema
  helps *small* models follow it — because token cost is not the thing being sold.
  See §12.

These are fundamentally different kinds of numbers — one is a property of the text, one
is a property of a specific model's specific response on a specific day, one is a
property of that response measured against a known answer — so `report.py` writes them
to separate `PromptRow` / `TrialRow` / `AccuracyRow` dataclasses, separate CSV files, and
separate markdown files. There is no code path that merges rows from two arms.

**Live-run status:** the committed live artefacts in `results/` were produced by real
runs against Ollama on the **`ollama_chat/` provider with `{"think": false}`**:

| file | arm | model | scope |
|---|---|---|---|
| `prompt-cost-2026-09-09.md` | offline | — | 10 adapters x 6 signatures |
| `live-ollama_chat-qwen3-8b-2026-09-09.md` | outcomes | `qwen3:8b` | 7 adapters x 6 signatures x 3 trials |
| `accuracy-ollama_chat-<model>-2026-09-09.md` | accuracy | six sub-1.2B models | 7 adapters x 30 cases each |

Each file's provenance block carries the exact command, git head and effective
`lm_kwargs`. If no results file is present for a given arm and date, that pass was not
run and no numbers exist for it — this file does not claim otherwise.

Two sets of superseded artefacts were deleted rather than kept:

- The `2026-09-05` live pair was generated at `41810a5`, before the
  `response_format_sent` fix and the `parse rate` / `validation rate` columns, so its
  `response_format` column was wrong on exactly the rows that made no LM call.
- The `2026-09-09` **`openai/`** live pair was generated before the trial-isolation fix
  in `run_live_arm`. A fixed `seed` made all N trials of a cell one deterministic request
  replayed N times, so every `stddev wall_s` in that file is an artefact reading `0.000`
  as if it were confidence. It also ran on a provider where `think` is silently ignored,
  so most of its latency is a thinking trace. Superseded on both counts by the
  `ollama_chat` pair.

## 2. Quick start

Offline arm — no endpoint needed:

```
make bench-dspy BENCH_ARGS=--offline
```

Live arms — need a running endpoint (verified against Ollama):

```
export LSL_BENCH_MODEL=ollama_chat/qwen3:8b
export LSL_BENCH_API_BASE=http://localhost:11434
export LSL_BENCH_API_KEY=not-needed                # optional, default "not-needed"
export LSL_BENCH_LM_KWARGS='{"think": false}'      # optional, JSON object

make bench-dspy BENCH_ARGS=--live
make bench-dspy BENCH_ARGS="--accuracy --cases 30"
```

**Use the `ollama_chat/` provider, not `openai/`, for any reasoning model.** Ollama
honours `think` only on its native `/api/chat`; litellm's `openai/` provider posts to
`/v1/chat/completions`, where the flag is silently ignored and the model spends its
whole `max_tokens` budget on a thinking trace it never gets to finish — which the
outcomes arm then records as `empty_response`. `ollama_chat/` posts to `/api/chat`, so
`{"think": false}` takes effect. The two providers advertise identical
`response_format` capability (`supports_response_schema=False`, `json_object`
accepted), so switching changes what the model *does*, not what any adapter *sends*.
Note the api_base has **no** `/v1` suffix for `ollama_chat/`.

`make bench-dspy` with no `BENCH_ARGS` runs the offline and outcomes arms (the live env
must be set). The accuracy arm is **never** part of that default — it costs
`adapters × cases` live calls and must be asked for explicitly.

## 3. `LSL_BENCH_*` environment variables

| var | required | default | semantics |
|---|---|---|---|
| `LSL_BENCH_MODEL` | **yes** (live only) | — | litellm model string, `dspy.LM`'s first positional arg. Verified: `ollama_chat/qwen3:8b`, `ollama_chat/qwen3.5:0.8b`, `openai/qwen3:8b`. |
| `LSL_BENCH_API_BASE` | **yes** (live only) | — | Base URL. `http://localhost:11434` for `ollama_chat/`; `http://localhost:11434/v1` for `openai/`. |
| `LSL_BENCH_API_KEY` | no | `"not-needed"` | Must be non-empty for litellm; Ollama/LM Studio ignore the value. |
| `LSL_BENCH_LM_KWARGS` | no | `{}` | JSON object, merged **last** into `dspy.LM(**kwargs)`. Sole tuning escape hatch. |

Only `config.py` reads these. `--offline`, `--list`, and `--repro-1871` (with no live
probe configured) never touch `os.environ` at all.

## 4. CLI flags and exit codes

`python -m benchmarking.dspy_adapters` (equivalently `make bench-dspy BENCH_ARGS=...`)
exposes eleven flags:

| flag | `argparse` type | default | meaning |
|---|---|---|---|
| `--offline` | `store_true` | `False` | Offline prompt-cost arm **plus** the synthetic #1871 reproduction. No `dspy.LM`; no network required — token counts degrade to `—` if `cl100k_base` cannot be loaded. Sets `TIKTOKEN_CACHE_DIR` only when doing so is what makes an offline count possible. Sub-second. |
| `--live` | `store_true` | `False` | Live outcomes arm only. Requires the two env vars. |
| `--accuracy` | `store_true` | `False` | Live accuracy arm **only**, and never run by default. Requires the same two env vars. |
| `--cases` | `int` | `30` | Accuracy-arm case count. Ignored by every other arm. |
| `--cases-seed` | `int` | `0` | Accuracy-arm corpus seed. Recorded in the report's provenance block, because two accuracy runs are comparable only if they scored the same cases. |
| `--adapters` | `str` (comma-separated) | offline: all 10 · live: the 6 `*-sections` ids + `sola-jsonish-rescue` | Filter by adapter id. Unknown id → stderr listing valid ids, exit 2. |
| `--signatures` | `str` (comma-separated) | all 6 | Filter by signature id. Unknown id → same treatment. |
| `--trials` | `int` | `1` | Live-arm repetitions per cell. Ignored by the offline arm (deterministic). |
| `--out` | `Path` | `<package dir>/results` | Output directory; created if absent. |
| `--repro-1871` | `store_true` | `False` | Run **only** the #1871 work: the offline synthetic reproduction always, plus `probe_1871_live()` iff the live env is configured. |
| `--list` | `store_true` | `False` | Print every adapter id and signature id with its `config_repr`, then exit 0. |

Dispatch: neither `--offline` nor `--live` given ⇒ **both** of those arms run (live env
required). `--offline --live` together ⇒ same as neither (both arms run). `--list`,
`--repro-1871` and `--accuracy` short-circuit everything else; the first two never write
a file.

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
machine, with no model required. When no `cl100k_base` cache is reachable and no
network is available, the count is reported as `—` rather than estimated.

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
The live aggregate's `parse rate` and `validation rate` columns are markdown-only for the
same reason in reverse: they are per-`(adapter, signature)` derived values, and the CSV is
strictly per-trial, so `LIVE_CSV_HEADER` is unchanged.

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

**Adapters** (ten in `ADAPTERS`, plus one repro-only extra in `REPRO_1871_ADAPTERS`
used exclusively by the #1871 cell set):

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
| `sola-jsonish-rescue` | `StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS, parse_config=ParseConfig())` |
| `sola-yaml-block` | `StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK)` |
| `sola-jsonish-nojsonobject` (repro-only) | `StructuredOutputAdapter(output_mode=JSONISH, use_json_object_response_format=False)` |

`chat` is always constructed with `use_json_adapter_fallback=False` — the default
`True` turns a real `AdapterParseError` into a fake `ok` result at `lm_calls=2`, which
would silently mask parsing failures (Δ1/D3.1 in the design notes).

The live arms default to the six `*-sections` ids (`chat`, `json`, `baml`,
`sola-json-sections`, `sola-jsonish-sections`, `sola-yaml-sections`) **plus**
`sola-jsonish-rescue`; the offline arm covers all ten.

`sola-jsonish-rescue` is `sola-jsonish-sections` with one thing changed —
`parse_config=ParseConfig()`, which arms the parse-time rescues — so the pair isolates
what parse-time repair is worth. Its *prompt* is byte-identical to its twin, which the
offline arm makes checkable: the two rows must agree on every token count, and a test
asserts it for all six signatures. The rescue that matters here drops list items whose
every field is `null`, the shape a small model produces instead of `[]`.

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

## 12. The `accuracy` arm

### What it measures

Field-level exact-match accuracy against ground truth — the metric the two published
comparisons this arm is modelled on report ([DSPy PR #8614][pr] and
[thedataquarry/structured-outputs][dq]). Every case is an extraction task: prose in,
one `PersonRecord` out. The record is flattened to `dotted.path -> scalar` (lists
indexed, `contacts[0].email`), and the reply is scored path by path.

[pr]: https://github.com/stanfordnlp/dspy/pull/8614
[dq]: https://github.com/thedataquarry/structured-outputs

### Where ground truth comes from

`cases.py` generates a `PersonRecord` first and renders it to prose second, so the label
*is* the source instance — exact, free, no dataset download, no annotation, no licence
question. `generate(n, seed=S)` is a pure function of the seed and a strict
prefix-extension as `n` grows, so a 50-case run stays comparable to a 10-case run on
their common prefix.

**The honest weakness, stated rather than hidden:** the renderer writes the text the
extractor reads, so difficulty is bounded by how adversarial the rendering is. `render`
varies phrasing per case, shuffles the order-independent trailer, and omits optional
fields outright — and when it omits one, the expected value is `None`, so the case
scores a *correct absence* rather than a guess. It is still synthetic prose. This arm
measures schema-following under paraphrase; it does not measure real-world extraction,
and no number it produces should be quoted as if it did.

A test enforces the fairness invariant directly: for 30 generated cases, every
non-`None` scored value must appear verbatim in the case's prose
(`test_every_scored_value_appears_in_the_text`). A field the prose never states cannot
be scored.

### The two denominator decisions

Both follow the precedent set when the parse/validation rates were fixed:

1. **A cell that raised scores 0.0 — it is not excluded.** Dropping failures would let
   an adapter that answers half its cases outrank one that answers every case
   imperfectly.
2. **The denominator is the EXPECTED fields, not the produced ones.** Otherwise a reply
   that emitted nothing but `name` would score 1.00.

Fields the model invents land in `spurious`: they are reported and they break `exact`,
but they do not enter the denominator — a field that should not exist has no expected
value to match, and folding it in would double-penalise a reply that also got a real
field wrong.

Aggregate `field accuracy` is **micro**-averaged (pooled matches over pooled expected
fields), not the mean of per-case ratios, so a sparse case does not carry the same
weight as a fully-populated one.

### Reading the report

`accuracy-<model-slug>-<date>.md` carries two tables. The aggregate answers *who won*;
**Most-missed fields** answers *where they lost*, per adapter, with list indices
collapsed (`contacts[].email`) so a path counts once instead of fragmenting across
positions. Per-case detail — including the exact `wrong` / `missing` / `spurious` paths
— is in the companion CSV only, never duplicated into the markdown.

### What the sweep found

Six sub-1.2B models across five families, 30 labeled cases each, field accuracy:

| adapter | qwen3.5:0.8b | granite3.1-moe:1b | llama3.2:1b | smollm2:360m | falcon3:1b | gemma3:270m |
|---|---|---|---|---|---|---|
| `sola-jsonish-rescue` | **0.929** | 0.858 | 0.215 | 0.095 | 0.000 | 0.000 |
| `json` (`JSONAdapter`) | 0.889 | 0.797 | 0.203 | 0.000 | 0.000 | 0.000 |
| `sola-json-sections` | 0.886 | 0.157 | **0.363** | 0.000 | 0.000 | 0.000 |
| `sola-jsonish-sections` | 0.745 | **0.858** | 0.215 | **0.095** | 0.000 | 0.000 |
| `sola-yaml-sections` | 0.849 | 0.363 | 0.206 | 0.000 | 0.000 | 0.000 |
| `baml` (`BAMLAdapter`) | 0.000 | 0.760 | 0.000 | 0.000 | 0.000 | 0.000 |
| `chat` (`ChatAdapter`) | 0.000 | 0.000 | 0.022 | 0.000 | 0.000 | 0.000 |

**The dominant failure mode below 1B is the envelope, not the schema.** A zero in this
table is almost never a failed extraction — it is a correct extraction that failed to be
wrapped in its output field. Inspected directly, `baml` and (before the fix)
`sola-jsonish-sections` both emitted `{"name": ..., "age": ...}` instead of
`{"record": {...}}`; `BAMLAdapter.parse` filters to `signature.output_fields` and raises
when the key set does not match, so every value can be right and the cell still scores
0/22. `ChatAdapter` fails differently and worse — it returns well-formed JSON while its
own protocol expects `[[ ## field ## ]]` markers, giving 1 usable cell out of 180.

This is not a defect unique to this package. `BAMLAdapter` renders
``Output field `record` should be of type:`` followed by a schema whose brace opens at
column 0 — the same ambiguity, and it still has it.

**Three models cannot do the task at all.** `falcon3:1b`, `gemma3:270m` and
`smollm2:360m` score ~0 for every adapter. That is a property of the models, not the
harness: `falcon3:1b` answers with a JSON *schema* rather than an instance. Nested
extraction with optionality needs roughly ≥0.8B.

**Adapter ranking is model-dependent, and mode matters more than adapter.** JSON beats
JSONISH on `qwen3.5:0.8b` (0.886 vs 0.745) and loses badly on `granite3.1-moe:1b` (0.157
vs 0.858, where JSON mode's verbose schema draws 24/30 validation errors). No adapter wins
everywhere. The two live arms also disagree with each other — `sola-jsonish-sections` is
18/18 `ok` on `qwen3:8b` in the outcomes arm while `sola-yaml-sections` is 0/18, and on
`qwen3.5:0.8b` they swap — which is exactly why no number from one arm is carried into
the other's table.

**What parse-time repair is worth.** `sola-jsonish-rescue` differs from
`sola-jsonish-sections` only by `parse_config=ParseConfig()`, and the offline arm confirms
the two prompts are token-identical. It converts 6 validation errors into 6 correct
records on `qwen3.5:0.8b` (0.745 -> 0.929, 24/30 -> 30/30) and is inert everywhere else —
all six were the same bug, an empty list answered with `[{"email": null, "phone": null}]`.
