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

Two of the fifteen cells are impure. DSPy downgrades to `json_object` when the
structured-output build fails, and those cases are *not* grammar-constrained even though
they sit in a `json-constrained` row; a report whose `response_format` column reads `mixed`
rather than `json_schema` is flagging exactly this:

| cell | constrained | downgraded | cases |
|---|---|---|---|
| `insurance-claims` / `qwen3.5:0.8b` | 29/30 | 1 | `insurance-claims-0021` |
| `patient-notes` / `llama3.2:1b` | 28/30 | 2 | `patient-notes-0001`, `patient-notes-0006` |

All three downgraded cases ended in `validation_error`. The other thirteen cells sent
`json_schema` on all 30. Audit any of this directly, rather than trusting this table:

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
