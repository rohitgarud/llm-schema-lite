# Schema-constrained decoding comparison

Artefacts for the `json-constrained` arm, kept in their own directory because they are
**not** interchangeable with their siblings in `../`: same stem convention, different arm
set. Two of these filenames collide exactly with committed files in `../` —
`accuracy-{insurance-claims,patient-notes}-ollama_chat-falcon3-1b-2026-09-13.{csv,md}` —
which are a different run (eight arms, `git_head 9e2d0f3`, llm-schema-lite 0.6.1). Neither
supersedes the other, so they are not merged. Read a stem here as "the constrained-arm
study", never as a newer version of the file above it.

## What was run

Four arms, five corpora, 30 cases each, against a local Ollama:

```
--adapters json,json-constrained,sola-jsonish-rescue,sola-yaml-rescue
```

`json-constrained` is `ConstrainedJSONAdapter`: upstream `JSONAdapter` with the
structured-output path forced on. litellm reports `supports_response_schema=False` for
every Ollama prefix, so the stock `json` arm downgrades to `{"type": "json_object"}` --
valid JSON of any shape, not the signature's schema. Ollama's `/v1` honours a `json_schema`
`response_format` anyway; verified with a `Literal["ZZQX_PURPLE_ONLY"]` field, which forced
that value as the colour of a banana while the unconstrained call answered "yellow".

Models: `falcon3:1b`, `qwen3.5:0.8b`, `llama3.2:1b`. All on llm-schema-lite 0.7.0.

## Reading these

Recall (non-null gold) is the headline; field accuracy is only a distance from each
corpus's all-null floor, printed in every `.md`. `invented (null gold)` matters as much as
recall here: the constrained arm buys reach by filling null-gold fields, so the two must be
read together or the comparison is meaningless.

Two of the fifteen cells are impure: some cases fell back to `json_object` and are *not*
grammar-constrained even though they sit in a `json-constrained` row. A report whose
`response_format` column reads `mixed` rather than `json_schema` is flagging exactly this:

| cell | constrained | downgraded | cases |
|---|---|---|---|
| `insurance-claims` / `qwen3.5:0.8b` | 29/30 | 1 | `insurance-claims-0021` |
| `patient-notes` / `llama3.2:1b` | 28/30 | 2 | `patient-notes-0001`, `patient-notes-0006` |

All three downgraded cases ended in `validation_error`. The other thirteen cells sent
`json_schema` on all 30.

**What triggers the fallback, instrumented.** An earlier version of this note claimed the
downgrade happens "when the structured-output build fails". That was wrong.
`super().__call__` -- the LM call *and* the parse -- sits inside the same `try` as the
schema build (`dspy/adapters/json_adapter.py:80-92`, mirrored at
`structured_output_adapter.py:771-784`), so any exception raised there triggers the
`json_object` retry, not just a build failure.

Measured by attaching a logging handler to the two fallback loggers and reading
`sys.exc_info()` from inside their `except` block -- the warning is logged there, so the
live exception is still on the stack and no library patch is needed. Across three
signatures and both code paths, 13 of 13 fallbacks were a pydantic `ValidationError` on the
model's *reply*. None was a schema-build failure:

| model / corpus | arm | logger | fallbacks | exception |
|---|---|---|---|---|
| `llama3.2:1b` / insurance-claims | `sola-jsonish-constrained` | ours | 9/30 | `ValidationError` (InsuranceClaim) |
| `llama3.2:1b` / synthetic | `sola-jsonish-constrained` | ours | 1/30 | `ValidationError` (PersonRecord) |
| `llama3.2:1b` / patient-notes | `json-constrained` | **dspy** | 2/30 | `ValidationError` (PatientRecord) |
| `llama3.2:1b` / patient-notes | `sola-jsonish-constrained` | ours | 1/30 | `ValidationError` (PatientRecord) |

The stock `dspy.adapters.json_adapter` row is the one that generalises: this is upstream
behaviour, not something our subclass introduces.

The failing replies are **truncated**, not misshapen -- every one is an unterminated string
around 2,800-3,000 characters. Pooled over all 15 cells of the composed sweep, downgraded
cases sit at the token cap and clean ones do not:

| | n | p50 tokens | >=850 tokens |
|---|---|---|---|
| downgraded | 19 | 883 | 15/19 |
| clean (`json_schema`) | 881 | 254 | 8/881 |

(cl100k_base used as a proxy; llama3.2's own tokenizer differs.) The cap is this harness's
own `BASELINE_LM_KWARGS`, measured as a FLOOR (below it qwen3's thinking trace exhausts the
budget and yields `empty_response`) and never validated as a sufficient ceiling for
constrained decoding.

Two distinct things drive a reply into that cap, and an earlier draft of this note named
only the first:

1. **Long output.** `enforce_required` marks every property required, so a
   grammar-constrained reply carries every key in the schema whether or not the input
   supports one.
2. **Degenerate repetition.** At least 7 of the 20 downgraded replies end in a repeating
   unit -- `' Cedar Lane,'` x33, `' Mr.'` x100, `'. Mr'` x100 -- emitted *inside* a single
   JSON string. A grammar does not prevent this: an arbitrarily long string is legal JSON,
   so the sampler stays schema-compliant right until the budget cuts it off mid-string.
   Read 7 as a floor, not a count -- the detector only looks for a repeating unit of 3-60
   characters at the tail, so longer-period loops are missed.

Raising the cap does not fix this. The whole sweep was re-run at `max_tokens=2400` to check,
and it cleared **nothing**: the same seven cells, the same counts, the same case ids. Every
downgraded reply simply grew to the new ceiling and failed the same way:

| case | tokens @900 | tokens @2400 |
|---|---|---|
| `insurance-claims-0000` | 900 | 2400 |
| `insurance-claims-0012` | 882 | 2353 |
| `case-0017` | 867 | 2309 |
| `case-0018` | 830 | 2205 |

Several are looping a whole repeated *unit*, not a single string -- a duplicated array
element such as `'"make_model": "Honda CR-V", "year": ...'` or
`'": "", "phone": null }, { "email'`. The tail detector described above missed these
because it only considers repeating units of 3-60 characters; the real period is longer.

**An earlier version of this note called a `mixed` row "confounded with the token budget"
and told the reader to discard it. That warning is withdrawn.** Recall at 900 and at 2400
agrees to three decimals in nearly every cell, so the budget was never what these cells
were measuring. What a `mixed` row actually records is a grammar-constrained generation
that never terminates: it fills whatever budget it is given, then dies mid-structure.

The wider point is not specific to this harness: **schema-constrained decoding does not
guarantee parseable output.** A grammar constrains structure, not termination.

The downgrade count is not stable run to run either: re-running `synthetic` /
`sola-jsonish-constrained` produced 1/30 where the sweep recorded 4/30, at the same
`temperature=0` and `seed=7`.

Audit any of this directly, rather than trusting this table:

```bash
uv run python -c "
import csv, collections
csv.field_size_limit(10**7)
rows = [r for r in csv.DictReader(open('<csv>')) if r['adapter'] == 'json-constrained']
print(collections.Counter(r['response_format_sent'] for r in rows))
"
```

## Reproducing

Re-score any committed CSV with no model and no network -- the `--corpus`/`--cases` must
match the run that produced it:

```bash
make bench-dspy BENCH_ARGS="--accuracy --corpus pii --replay <csv>"
```

Paired-bootstrap CIs against the constrained arm:

```bash
uv run python benchmarking/dspy_adapters/paired.py <csv> \
  --baseline json-constrained --metric recall --resamples 5000
```
