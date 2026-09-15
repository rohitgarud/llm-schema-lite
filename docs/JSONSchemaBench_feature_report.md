# JSONSchemaBench Feature Coverage Report: llm-schema-lite

Every figure below is **measured against the code in this repository**, not asserted. The
feature matrix comes from running one minimal schema per keyword through each formatter and
checking that the keyword's *semantics* survive into the output; the corpus figures come
from running `simplify_schema` over schemas from
[JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench)
([epfl-dlab/JSONSchemaBench](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench)).

Reproduce with:

```bash
uv run python -m benchmarking.jsonschemabench.fetch_dataset   # ~100 MB, 9,542 schemas, 10 configs
uv run python -m benchmarking.jsonschemabench.coverage --all-configs
```

**Scope.** llm-schema-lite is a **schema simplification and validation** library: it turns
Pydantic models or raw JSON Schema into token-optimized string formats (JSONish, TypeScript,
YAML) for LLM prompts, and validates candidate output with `jsonschema` (Draft 2020-12). It
does **not** perform constrained decoding. So "coverage" here means *which JSON Schema
features survive into the simplified output* — validation separately enforces the full draft.

---

## 1. Executive Summary

| Aspect | Measured |
|--------|----------|
| **Ingestion** | **9,542/9,542 (100.0%)** — the *whole* corpus, all 10 configs |
| **Schemas that raise** | **0** — and since the taint fix (§4) nothing times out either |
| **Median token reduction** | **46.8%** (median of the ten per-config medians; range 31.4%–57.0%) |
| **Schemas too slow to render** | **0/9,542** at a 10 s/schema budget, down from 9–10 (§4) |
| **Feature coverage** | JSONish **32/39**, YAML **33/39**, TypeScript **31/39** |
| **Validation** | Full Draft 2020-12 via `jsonschema` — every keyword enforced regardless of whether it renders |

Earlier revisions of this report quoted "300/300, 46.2%". That was the flat dump's first 300
records, which are *all* `Github_trivial` and `Github_easy` — the easiest slice of the
corpus. The table below is every schema in every config:

| Config | Schemas | Ingested | Median ↓ | Mean ↓ | Worst | Slow |
|--------|--------:|---------:|---------:|-------:|------:|-----:|
| Github_trivial | 444 | 100% | **57.0%** | 55.0% | −250% | 0 |
| Github_easy | 1938 | 100% | **52.9%** | 53.1% | −34% | 0 |
| Github_medium | 1969 | 100% | **46.7%** | 45.2% | −1031% | 0 |
| Github_hard | 1236 | 100% | **40.1%** | 13.9% | −23389% | 0 |
| Github_ultra | 164 | 100% | **47.0%** | 36.4% | −354% | 0 |
| Glaiveai2K | 1707 | 100% | **45.6%** | 45.9% | +21% | 0 |
| JsonSchemaStore | 492 | 100% | **43.7%** | 39.5% | −1051% | 0 |
| Kubernetes | 1064 | 100% | **31.4%** | 30.8% | −407% | 0 |
| Snowplow | 403 | 100% | **53.5%** | 52.6% | +24% | 0 |
| WashingtonPost | 125 | 100% | **49.1%** | 36.0% | −144% | 0 |
| **TOTAL** | **9542** | **100.0%** | — | — | — | **0** |

Two things in that table need saying, because one of them looks like a regression and is not.

**The means recovered, and the timeouts are gone.** `WashingtonPost` went from a mean of
**−51.0%** to **+36.0%**, `JsonSchemaStore` from 18.0% to **39.5%**, `Github_ultra` from
−38.2% to **+36.4%**. Every schema in the corpus now renders inside the 10 s budget, so
ingestion is 100.0% and the slow column is empty for the first time. All of that is the
taint-scoping fix in §4.

**`Github_hard`'s worst case reads −23389%, far worse than the −4770% it showed before, and
that is an artefact of the fix rather than damage from it.** Schemas that used to exceed the
budget were *excluded* from these statistics entirely; they now complete, so their expansion
is counted for the first time. The honest way to read the row is that a number which was
previously hidden has become visible. `o13029` alone renders at 82× its input — see §4 for
why it is the one schema repeat-suppression cannot reach.

The medians are unchanged by the fix and remain 0.4–1.9 pp below the pre-`patternProperties`
baseline. That residue is the structural cost of rendering pattern keys, not ref expansion.

Three things this table says that the 300-schema sample could not.

**The compaction claim holds across difficulty.** Every config's *median* is a real
reduction, from 31.4% (Kubernetes) to 57.0% (Github_trivial). It erodes with difficulty
(57.0 → 52.9 → 46.7 → 40.1 across the Github ladder) but never inverts.

**Quote the median, never the mean.** `WashingtonPost` has a median of **+49.1%** and a mean
of **−51.4%**: the typical schema compacts by half while a few catastrophic expansions drag
the average below zero. `Github_hard` now shows the same split even more starkly (40.1% vs
−51.3%). A mean over this corpus is a statement about the worst tail, not about typical
behaviour.

**The tail was one bug, not two, and it is mostly closed.** Both the expansion tail and the
`patternProperties` regression traced to a single taint-scoping defect that switched
back-references off on cyclic schemas. Fixing it took the worst schema from 45× its input to
**0.35×**, eliminated every timeout, and recovered ~97% of the pattern regression. What
survives is **56 schemas (0.59%)** that still render larger than their input — one outlier at
82× (`o13029`, mutual recursion) and a tail where the next worst is 8.67×. See §4.

The headline gap is **not** breadth, it is that JSONish — the default mode — renders five
fewer keywords than YAML, despite all three sharing `base.py`. Every one of those five is
already implemented in the base class and simply never reaches JSONish's output.

---

## 2. Feature-by-Feature Status (measured)

✅ = the keyword's semantics appear in that formatter's output. ❌ = they do not (validation
still enforces it).

| Feature | JSONish | YAML | TypeScript |
|---------|---------|------|------------|
| required | ✅ | ✅ | ✅ |
| items | ✅ | ✅ | ✅ |
| additionalProperties: false | ✅ | ✅ | ✅ |
| additionalProperties: schema | ✅ | ✅ | ✅ |
| enum | ✅ | ✅ | ✅ |
| $ref / $defs | ✅ | ✅ | ✅ |
| recursive $ref | ✅ | ✅ | ✅ |
| pattern | ✅ | ✅ | ✅ |
| format | ✅ | ✅ | ✅ |
| oneOf | ✅ | ✅ | ✅ |
| anyOf | ✅ | ✅ | ✅ |
| allOf | ✅ | ✅ | ✅ |
| not | ✅ | ✅ | ✅ |
| **not: true (boolean schema)** | ❌ | ✅ | ✅ |
| const | ✅ | ✅ | ✅ |
| minLength / maxLength | ✅ | ✅ | ✅ |
| minimum / maximum | ✅ | ✅ | ✅ |
| exclusiveMinimum / exclusiveMaximum | ✅ | ✅ | ✅ |
| multipleOf | ✅ | ✅ | ✅ |
| minItems / maxItems | ✅ | ✅ | ✅ |
| uniqueItems | ✅ | ✅ | ✅ |
| prefixItems (tuple) | ✅ | ✅ | ✅ |
| contains | ✅ | ✅ | ✅ |
| boolean schema `true` | ✅ | ✅ | ✅ |
| boolean schema `false` | ✅ | ✅ | ✅ |
| items: true (boolean schema) | ✅ | ✅ | ✅ |
| **patternProperties** | ✅ | ✅ | ✅ |
| **propertyNames** (open mapping) | ✅ | ✅ | ❌ |
| minProperties | ❌ | ❌ | ❌ |
| maxProperties | ❌ | ❌ | ❌ |
| dependencies | ✅ | ✅ | ❌ |
| dependentRequired | ❌ | ❌ | ❌ |
| if / then / else | ❌ | ❌ | ❌ |
| unevaluatedProperties | ❌ | ❌ | ❌ |
| title | ✅ | ✅ | ✅ |
| description | ✅ | ✅ | ✅ |
| default | ✅ | ✅ | ✅ |
| examples (opt-in) | ❌ | ❌ | ❌ |
| nullable (`anyOf` + null) | ✅ | ✅ | ✅ |
| **Total** | **32/39** | **33/39** | **31/39** |

`examples` is excluded by default for token savings and renders when enabled
(`FormatterConfig(metadata_inclusion={"examples": True})`); it is listed ❌ because the
default output omits it.

### Why JSONish trails

JSONish surfaces only keywords that have a **dedicated inline token** (pattern, format,
length/numeric ranges, item counts, uniqueItems, enum, const, default, description, title).
Everything else in `BaseFormatter.METADATA_MAP` is live for YAML and TypeScript and absent
from JSONish.

An earlier revision of this report blamed `JSONishFormatter.add_metadata` being a
pass-through, and called the fix "one contained change". **That was wrong.** `add_metadata`
has exactly one caller — the last line of `BaseFormatter.process_property` — and JSONish
never calls `process_property`: it renders object bodies in its own
`_process_schema_recursive_inner` loop. Instrumenting a full JSONish render confirms it,
with both counters at zero:

```
rendered: // Fields marked with * are required { a*: int, b: string [] }
call counts: {'add_metadata': 0, 'process_property': 0}
```

(`multipleOf: 5` and `contains` are both absent from that output.) JSONish's
`add_metadata` is a dead override on a path JSONish does not use; implementing it would
change nothing observable.

The real cause is structural: that loop emits each property's token directly across ~10
branches, so a keyword with no inline token has nowhere to appear. Closing a gap means
adding a fragment on the path each mode actually uses — which is a larger change than this
report previously claimed, and larger than it looks: `multipleOf` alone needed the token
joined at **three** separate call sites (`jsonish_formatter.py`, `typescript_formatter.py`,
and YAML's `_format_number_range_jsonish`), because none of the three modes reaches
`BaseFormatter.process_type_value` for a scalar number.

**Closed since that finding:** `multipleOf` and `contains` now render in all three modes,
with one shared spelling — `int (multiple of 5)`, `list[string] (contains "z")` — from a
single owner each (`multiple_of_token`, `array_constraint_tokens`). `contains` previously
had three live emitters, printing two or three times on one YAML line, and leaked a raw
Python dict repr (`contains: {'const': 'z'}`) whenever the target carried a `const`.

**Corrected:** an earlier revision listed `propertyNames` as a JSONish gap needing a
`classify_container` change. That was wrong on both counts. `classify_container` already
populates `key_schema` from `propertyNames` (rules 6 and 7), `key_token` already renders it,
and JSONish already calls both — so for an **open mapping** JSONish emits the key constraint
today:

```
{"type": "object", "additionalProperties": {"type": "integer"},
 "propertyNames": {"enum": ["alpha", "beta"]}}

jsonish     { <alpha OR beta>: int }
yaml        <alpha OR beta>: int
typescript  interface Schema { [key: string]: number; }   <- drops it
```

TypeScript is the mode that loses the constraint, not JSONish. A `propertyNames` sitting
alongside `properties` still does not surface in any mode, but that is Decision C1 holding
(a schema declaring `properties` is an object, never a mapping), not a defect.

**Corrected again — `patternProperties` (now fixed).** The paragraph that stood here called
this a JSONish-only gap whose attach point was "that loop's empty-object base case, **not**
the classifier". Measuring all three modes showed that was wrong twice over. JSONish did
render a bare `{}`, but YAML and TypeScript were not fine either — they emitted `object`
plus a comment that leaked JSONish's `//` marker into a `#` context and, on the property
path, a raw Python dict repr:

```
env: 'object  //pattern: [^[A-Z]+$]: string'  # patternProperties: {'^[A-Z]+$': {'type': 'string'}}
```

So all three modes lost it, and a JSONish-local patch would have left the other two leaking.
The fix is a shared classifier change after all: `classify_container` gained a
`pattern_mapping` kind (rule 7b) holding one `(regex, value schema)` pair per pattern. It is
**not** a `mapping` — a mapping carries one value schema for all keys, so reusing it would
have kept only the first pattern. Decision C1 is untouched: `properties` still wins, so only
a `patternProperties`-only node reaches rule 7b.

```
{"type": "object", "patternProperties": {"^S_": {"type": "string"},
                                         "^N_": {"type": "integer"}}}

jsonish     { <^S_>: string, <^N_>: int }
yaml        <^S_>: string
            <^N_>: int
typescript  type Schema = Record<string, string | number /* keys: ^S_, ^N_ */>;
```

TypeScript names the regexes in a **block** comment rather than rendering them structurally:
an index signature takes a key *type*, not a regex, and a `//` comment would swallow the `;`
that `type Schema = ...;` appends. Measured over a 400-schema pattern-bearing sample, **78.7%**
now expose a pattern key; the remaining 21.3% is Decision C1 holding, not a gap — corpus-wide,
**513 of the 2,401** pattern nodes co-declare `properties` (21.4%).

**Still open in JSONish:** `not: true` only, and it is rare (`not` totals 1.6%).

---

## 3. Keyword frequency in the corpus

All **9,542** corpus schemas (share containing the keyword at least once, counted
recursively through every subschema slot — `definitions` and `$defs` included):

| Keyword | Schemas | Share | Keyword | Schemas | Share |
|---------|--------:|------:|---------|--------:|------:|
| type | 9500 | 99.6% | maximum | 601 | 6.3% |
| properties | 9335 | 97.8% | maxItems | 421 | 4.4% |
| required | 7437 | 77.9% | allOf | 325 | 3.4% |
| description | 7136 | 74.8% | additionalItems | 216 | 2.3% |
| items | 4630 | 48.5% | not | 155 | 1.6% |
| **additionalProperties** | 4179 | 43.8% | minProperties | 153 | 1.6% |
| title | 3481 | 36.5% | dependencies | 134 | 1.4% |
| enum | 3436 | 36.0% | const | 128 | 1.3% |
| $ref | 2759 | 28.9% | uniqueItems | 117 | 1.2% |
| pattern | 1768 | 18.5% | multipleOf | 61 | 0.6% |
| format | 1549 | 16.2% | maxProperties | 58 | 0.6% |
| default | 1516 | 15.9% | if | 52 | 0.5% |
| oneOf | 1375 | 14.4% | then | 52 | 0.5% |
| minimum | 1156 | 12.1% | **propertyNames** | 30 | 0.3% |
| minLength | 1025 | 10.7% | exclusiveMinimum | 26 | 0.3% |
| minItems | 934 | 9.8% | else | 15 | 0.2% |
| maxLength | 922 | 9.7% | exclusiveMaximum | 8 | 0.1% |
| anyOf | 839 | 8.8% | contains | 4 | 0.0% |
| **patternProperties** | 687 | 7.2% | unevaluatedProperties | 2 | 0.0% |

Only 38 of the 39 tracked keywords appear anywhere in the corpus.

Boolean schemas, separately: `additionalProperties: false` 6,597 sites, `: true` 535,
`additionalItems: false` 82 / `: true` 28, and 28 boolean-valued *properties*. The last of
those used to crash JSONish outright.

**These counts replace an earlier table that undercounted every row.**
`analyze_schema_features` used to recurse into a hand-written subset of the subschema slots
that omitted `definitions` and `$defs`, so any keyword living inside a definition was
invisible to it. The counter now walks every slot that can hold a subschema, and the
difference is large and uneven — `oneOf` +737, `items` +730, `required` +618, `enum` +610,
`pattern` +536, `not` nearly tripled (57 → 155). `then` and `unevaluatedProperties` did not
appear in the old table **at all**. Anything read from the previous revision is a lower
bound, not a measurement.

⚠️ **One residual gap, stated rather than hidden.** A handful of schemas nest subschemas
under non-standard container keys (`openscad`, `vega`, `bolts`, `defs`) that no allowlist can
anticipate. A schema-aware walk that descends into unknown keys too finds **691** schemas
carrying `patternProperties` against the counter's 687 — about 5 schemas, ~0.05%. The table
is accurate to roughly that margin.

**Correction — two of my own earlier figures were wrong here.** A previous revision quoted
`patternProperties` at 6.7% (that is `oneOf`'s share), then at 4.4% (the undercount above),
then at **7.5% / 716 schemas**. That last one was also wrong, and the cause was my own
measuring script rather than the counter: it treated `name → schema` maps as schema nodes, so
a **JSON Schema meta-schema** — one whose root `properties` are literally `$schema`,
`additionalItems`, `allOf`, `patternProperties`, … — was counted as *using* the keyword when
it only *documents* it. The true figure is **691 schemas (7.2%)**.

The ordering conclusion survives all of it: `patternProperties` was the highest-value JSONish
gap by an order of magnitude over the alternatives (`propertyNames` 0.3%, `not` 1.6%), and it
is **now closed** (see §2).

---

## 4. Where output still expands instead of compacting

Some schemas render **larger** than their raw JSON Schema. The cause is structural and
worth stating plainly: a `$ref` is expanded **inline at its use site**, so the whole
reachable definition graph is materialised, while raw JSON Schema stores each definition
once and points at it.

Across the full corpus the worst case per config now reaches **−23389%** (Github_hard),
**−1051%** (JsonSchemaStore), **−1031%** (Github_medium) and **−407%** (Kubernetes). Only
`Glaiveai2K` and `Snowplow`, the two configs with *no* cyclic `$ref` graphs, stay positive
throughout. The count of affected schemas is now measured rather than left open: **56 of
9,542 (0.59%)** render larger than their input, and past `o13029`'s 82× the next worst is
8.67×, with everything outside the top ten under 3×.

`Github_hard`'s −23389% is worse than the −4770% earlier revisions quoted, and that is an
artefact of the fix rather than damage from it: the schemas driving it used to exceed the
render budget and be excluded from the statistics entirely. They now complete, so they are
counted for the first time.

Repeated references are now handled. A definition rendered once is replaced at later
occurrences by a named back-reference (`object // defined above: Address`) whenever its body
exceeds `BaseFormatter.BACKREFERENCE_MIN_CHARS` (200). Measured effect:

| Schema | refs | distinct defs | before | after |
|--------|------|---------------|--------|-------|
| idx=88 | 244 | 10 | 55,828 tok (−558%) | **4,215 tok (+50.3%)** |
| idx=43 | — | 20 | 24,030 tok (−540%) | **3,074 tok (+18.2%)** |
| idx=106 | 50 | 14 | 329,124 tok | 127,247 tok |
| idx=267 | 247 | 73 | 807,375 tok | 152,115 tok |
| idx=270 | 209 | 70 | 200,335 tok | 86,074 tok |

The threshold exists because duplicating a *small* definition costs almost nothing and reads
better in place — the sibling-inline acceptance criterion (two sibling fields sharing a
`$ref` both render inline) depends on it. The two populations separate cleanly: the largest
body those tests rely on is 158 chars; the smallest definition in the 244-reference schema
is 409.

**⚠️ The paragraph that stood here was wrong — see the correction below.** It claimed the
remaining three "are not fixable by repeat-suppression" because they carry 70+ *distinct*
definitions each referenced only ~3 times, and that closing the gap required emitting a
leading `$defs` section. Profiling the renderer, rather than reasoning about it, shows
repeat-suppression is exactly the right mechanism and is already implemented — it is
switched off by a taint bug on the very schemas that need it most.

### The worst case in the corpus

`Github_ultra`'s `o48404` is the same mechanism at full scale, and it is the reason this is
not a four-schema tail:

| | |
|--|--|
| raw | 208,433 chars |
| structure | 174 defs, 534 refs, **152 distinct**, 128 inline object nodes, nesting depth 4 |
| rendered, before back-references | **49,236,120 chars** |
| rendered, after back-references | **9,327,895 chars** (5.3× better, still **45× the input**) |

Back-references did real work here — 49 MB down to 9 MB — and still left 45×. The reason is
**not** that 152 distinct definitions are irreducibly transitive, as this paragraph
previously asserted. It is that back-references are suppressed almost everywhere on this
schema by the bug documented next: only **103** of its 51,808 expansions were ever cached.

### Correction (2026-09-15): the tail is a taint-scoping bug, not an architectural limit

`_truncation_epoch` (`base.py:682`) is **global and monotonic**. It is bumped whenever any
`$ref` re-entry is truncated, and both the body cache and `_emitted_refs` are written only
if the epoch is unchanged since that frame was entered (`base.py:805-808`,
`jsonish_formatter.py:261, 268-270`). So a truncation *anywhere* in a subtree poisons
**every ancestor on the stack**, including unrelated siblings. On a cyclic schema
truncations are constant, so essentially no definition ever reaches `_emitted_refs`, no
back-reference can fire, and every use site re-inlines the full body.

Instrumented counts — the instrumented render reproduces this section's own 9,327,895-char
figure byte for byte, so it is measuring the real thing:

| | `o48404` | `o27039` | `o13029` |
|--|--:|--:|--:|
| distinct ref keys | 149 | 40 | **6** |
| full expansions | **51,808** | 186,102 | **172,617** |
| expansions cached | 103 | 19 | 10 |
| expansions tainted, not cached | 51,705 | 186,083 | 172,607 |
| truncations | 25,960 | 100,744 | 5,005,663 |

`JsonRenderer` alone is expanded **70,207 times** in `o13029` — a schema with six distinct
definitions.

Neutralising *only* the taint guard, changing nothing else:

| | production | taint neutralised |
|--|--:|--:|
| `o48404` | 9,327,895 chars, 27.2 s | **130,673 chars, 0.0 s** |
| `o27039` | unfinished at 60 s | **9,310 chars, 0.0 s** |
| `o13029` | unfinished at 120 s | **69,331 chars, 0.0 s** |

`o48404` goes from **45× its input to 0.63×** — a 37% *reduction* — and the full transitive
closure of its 152 definitions costs ~131 KB, not 9.3 MB. The slowness goes with it.

The guard cannot simply be deleted. It exists because a truncated body is **path-dependent**:
one containing `recursive: Node` must not be replayed to a clean sibling entitled to a fuller
expansion, which `tests/test_recursive_models.py:611` pins across all three formatters. The
defect is the *scope* — the taint is global where it should name only the refs actually
responsible. `tests/test_recursive_models.py:245` already pins the other half of the correct
behaviour: a clean sibling must still be cached after an unrelated truncation.

#### The shipped fix, and the one schema it does not reach

`_body_is_replayable` replaces the epoch comparison. A body is refused caching only when a
ref that truncated *inside* it was already on the expansion path at entry — i.e. when an
ancestor spent the depth budget, so the same body would render differently elsewhere. A
truncation a body inflicts on itself leaves it position-independent, and cacheable.

| Schema | before | after | |
|--------|-------:|------:|--|
| `o48404` | 9,327,895 chars (45×) | **115,337 (0.35×)** | fixed |
| `o27039` | unfinished at 60 s | 42,832 chars (2.04×) | terminates |
| `o69207` | 6,086,552 chars (98×) | 149,514 (2.41×) | much improved |
| `o13029` | unfinished at 120 s | 3,539,199 (**82×**) | **not fixed** |

Corpus-wide the timeouts disappear entirely — 9–10 slow schemas become **0**, ingestion
**100.0%** — and only **56 of 9,542 schemas (0.59%)** still render larger than their input.
Past `o13029` the next worst expander is 8.67×, and everything outside the top ten is under
3×, so the residue is one outlier and a mild tail, not a class.

**`o13029` is the accepted limit of repeat-suppression.** Its seven definitions are *mutually*
recursive, so nearly every frame is entered with a ref already on the path that will later
truncate: replay is accepted twice and refused 148 times, and the refusals are correct — those
bodies genuinely are position-dependent. Reaching it would require keying the cache by
remaining depth rather than by ref name, which is a larger change and is not made here. Note
that an earlier experiment reached 69,331 chars on this schema only by force-caching
unconditionally, which replays exactly the bodies `test_recursive_models.py:611` forbids —
smaller output, but not correct output.

A second guard now applies where it never did: JSONish counts its `$ref` expansions against
`_global_expansion_budget` for the first time, and an exhausted budget renders
`object // budget exhausted: Name` rather than a bare `object` indistinguishable from an
untyped one. It fires on **32 of 9,542 schemas (0.34%)**, mostly `WashingtonPost`, and costs
0.2 pp of mean reduction on that config and nothing measurable anywhere else.

### The `patternProperties` fix cost tokens, and the bill is measured

`efb7b16` made `patternProperties` render structurally rather than being dropped or reduced
to a one-line comment (§2). That is a coverage win and a token loss, and the loss is large
enough to state rather than bury. A/B against the immediately preceding commit (`afa03f4`),
same harness, same 10 s budget, same machine, the two runs serialised so neither contends
for CPU:

| Config | pattern-bearing schemas | median ↓ before → after |
|--------|------------------------:|------------------------:|
| Github_trivial | 25 | 79.5% → **47.7%** |
| JsonSchemaStore | 119 | 57.0% → **47.8%** |
| Github_hard | 291 | 37.6% → **32.2%** |
| Github_ultra | 30 | 17.2% → **−441.5%** |

The attribution is isolated, not inferred. Schemas *without* the keyword are byte-identical
across the two commits — same medians, same means, same timeout sets — and `Glaiveai2K` and
`Kubernetes`, the two configs carrying **no** `patternProperties` at all, come through
unchanged to the decimal across 2,771 schemas. Of 141 pattern-bearing schemas measured
individually under both commits, **114 regressed** and 27 did not. Worst named cases:

| Schema | before | after |
|--------|-------:|------:|
| `JsonSchemaStore/tmlanguage` | −195.3% | **−1822.9%** |
| `Github_ultra/o21307` | +40.6% | **−780.8%** |
| `Github_ultra/o21193` | +17.8% | **−674.0%** |

The mechanism is the one described above, multiplied: a pattern's *value schema* is now
materialised once per regex, so when that value schema carries `$ref`s the expansion is
taken once per pattern. The `o21xxx` cluster are variants of a single pattern-heavy shape,
which is why they move together.

Whether the trade is right is a product decision, not a measurement one — the stated goal is
small-model reliability rather than token count, and a pattern key that reaches the model is
worth more than one that silently does not. But turning 17% compaction into a 5.4×
expansion on `Github_ultra`'s pattern tail is not free, and the obvious mitigation — fall
back to the comment form when the structural render exceeds some multiple of its input — is
**not implemented**.

(Group sizes above come from a structural walk that descends every dict value; the stricter
keyword counter of §3 gives 25 / 118 / 285 / 30 for the same four configs. The medians are
computed over the groups as listed.)

### The cost is CPU, not memory

An earlier revision of this report attributed full-corpus sweep stalls to memory, citing
`o48404` taking resident memory from 240 MB to 801 MB. **That was the wrong diagnosis.**
Instrumenting a full `Github_hard` pass shows RSS pinned flat at **233 MB** from start to
finish, and a bounded render of the worst offender is killed by its 300 s clock
(`timeout -s KILL 300` → exit 137), never by the OOM killer.

These schemas do not exhaust memory, and they do **terminate** — the same `o27039` render
left unbounded exited cleanly after roughly 20–25 minutes of wall time (with CPU contention
from concurrent runs). They are unusably slow, not hung: a 300 s cap kills them, an
unbounded run eventually returns. The practical consequence is the same for a prompt-time
renderer, but the distinction matters for diagnosis — nothing here is deadlocked.

Profiling `o27039` (42 definitions, 109 refs, 3 mutually-recursive `$ref` cycles) names the
hot path exactly:

```
   ncalls  tottime  cumtime  function
   322749   25.874   42.897  jsonish_formatter.py:989(_scan_remove_string_delimiters)
118791893   16.127   16.127  {method 'append' of 'list' objects}
   302699    1.062   59.997  jsonish_formatter.py:207(process_ref)
```

`process_ref` is called 302,699 times even though `_global_expansion_budget` is 150. The
explanation offered here — that the budget caps *expansions* rather than *calls* — was too
generous. JSONish's `process_ref` did not increment `_global_expansion_count` at all
(contrast `base.py:727`), so in the **default mode the budget was not loose, it was absent**
— fixed since, along with the taint; see the shipped-fix note above.
The proximate cost is quadratic string post-processing: a character-at-a-time scanner
re-walking a string that inline `$ref` expansion keeps regrowing, 118 million list appends
deep. But that string only regrows because the taint bug above forces the re-expansion —
fix the taint and the scanner has little left to re-walk.

**Historic — no schema in the corpus exceeds the 10 s budget any more.** Before the taint
fix, 9 or 10 of 9,542 (0.1%) did, all for this reason:

| Config | Slow schemas (before the fix) |
|--------|--------------|
| Github_hard | `o27039`, `o13029`, and `o69207` *only sometimes* |
| Github_ultra | `o39449`, `o48404`, `o50639`, `o69209`, `o21764` |
| JsonSchemaStore | `meta`, `accelerator` |

The count was 9 **or** 10, and the difference was one schema sitting on the fence. Timed
alone, `o69207` rendered in **7.8 s** before the `patternProperties` change and **8.9–9.4 s**
after — both under the budget — but it crossed 10 s inside a full sweep, under the memory
pressure of nine other configs. So it was excluded from some runs and included in others, and
because it expanded to **6,086,552 characters from a 62 KB input (98×)**, its presence or
absence swung that config's mean and worst case violently: `Github_hard`'s worst case read
**−30089%** on runs where it completed and **−4770%** on runs where it timed out.

This is recorded rather than deleted because it explains why three separate runs of this
report disagreed about `Github_hard`, and because "the measurement is unstable" is itself a
finding. It no longer applies: `o69207` now renders 149,514 chars and every config's slow
column is empty, so the means and worst cases in §1 are reproducible.

Note what is *not* on this list: cyclic schemas as a class. 371 of 9,542 (3.9%) have cyclic
`$ref` graphs — 189 in `Github_hard` alone — and the re-entry guard keeps them terminating.
But the claim that closed this paragraph, that "unbounded *width* of distinct-definition
expansion" is the unsurvivable case, had it backwards. A cycle is precisely what triggers
the truncation that poisons the cache, so cyclicity is the *cause* of the width, not an
alternative to it: `o13029` has **six** distinct definitions and five million truncations.

---

## 5. Comparison to JSONSchemaBench engines

JSONSchemaBench reports feature coverage for constrained-decoding engines. We are not one,
so this compares *what reaches the model* — their grammar vs our prompt text.

| Feature | LLGuidance | llama.cpp | Outlines | XGrammar | OpenAI | Gemini | llm-schema-lite |
|---------|-----------|-----------|----------|----------|--------|--------|-----------------|
| required | ✔ | ✔ | ✔ | ✔ | ❌ | ✔ | ✅ |
| items | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| additionalProperties | ✔ | ✔ | ✔ | ✔ | ❌ | ❌ | ✅ |
| enum | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| $ref | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| pattern | ✔ | ✔ | ✔ | ✔ | ❌ | ❌ | ✅ |
| format | ✔ | ✔ | ✔ | ❌ | ❌ | ✔ | ✅ |
| oneOf | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| anyOf / allOf | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✅ |
| minLength/maxLength | ✔ | ✔ | ✔ | ❌ | ❌ | ❌ | ✅ |
| min/max (integer) | ✔ | ✔ | ❌ | ❌ | ❌ | ❌ | ✅ |
| minItems/maxItems | ✔ | ✔ | ✔ | ❌ | ❌ | ❌ | ✅ |
| const | ✔ | ✔ | ✔ | ✔ | ✔ | ❌ | ✅ |
| patternProperties | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| not | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| if / then / else | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| contains | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| propertyNames | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ JSONish/YAML only |

The asymmetry worth noting: their gaps are **architectural** — an FSM/grammar cannot express
`if/then/else`, `not`, or unbounded recursion. Ours are **implementation debt**: the
keywords are already parsed and validated, they just do not reach the rendered string. That
is a materially cheaper problem, and it is the honest form of the claim.

---

## 6. Recommendations, in measured priority order

1. ~~**Scope the `_truncation_epoch` taint.**~~ **Done.** This item previously read "emit a
   `$defs` section", on the strength of a claim §4 now retracts; the dominant expansion
   failure mode was a **bug**, not an architectural limit. `_body_is_replayable` now refuses
   a body only when an ancestor caused its truncation. Corpus effect: timeouts **9–10 → 0**,
   ingestion **100.0%**, `o48404` **45× → 0.35×**, and only 0.59% of schemas still expand. No
   output-shape change and no indirection added, which is why it replaced the `$defs` section
   rather than preceding it. JSONish also gained the `_global_expansion_count` accounting it
   never had, and budget exhaustion is now marked rather than silent. **`o13029` (82×)
   remains**, as the accepted limit of repeat-suppression under mutual recursion — closing it
   needs depth-keyed caching, which is a separate decision.

2. **Cap the `patternProperties` structural render — probably unnecessary, re-measure
   first.** The regression logged in §4 is ~97% a symptom of item 1: with the taint
   neutralised, the worst pattern-bearing schemas move from a median of **−645.5% to −0.7%**.
   What remains is ~20 pp on ref-bearing pattern schemas, three ref-free schemas losing
   50–62 pp, and an 81-schema tail averaging ~5 pp — and none of it *expands*: those three
   ref-free cases still compact by 36–44%. Build a cap only if a post-fix corpus run finds a
   schema whose render genuinely exceeds its input. Less compaction is not a bug.
3. ~~**Surface `patternProperties` in JSONish.**~~ **Done**, and the reasoning that stood
   here was wrong. This item asserted it was *not* a `classify_container` change, because
   Decision C1 excludes `patternProperties` from `mapping` deliberately. That conclusion
   does not follow: C1 forbids calling these nodes a **mapping**, not giving them a kind of
   their own. The fix is a new `pattern_mapping` kind (rule 7b) that leaves C1 exactly as
   written, and it had to be shared rather than JSONish-local because YAML and TypeScript
   were leaking a raw dict repr on the same nodes — a JSONish-only patch would have left
   that standing. Measured at **7.2%** of corpus schemas (691 of 9,542), not the 4.4% quoted
   here; see §3 on why the old counter undercounted. `not: true` remains open, and is rarer
   (`not` totals 1.6%).
4. **`minProperties` / `maxProperties` / `dependentRequired`** — absent from all three
   formatters; `dependentRequired` is not in `METADATA_MAP` at all.
5. **`if/then/else`** — `_format_conditional` exists but never fires from a root-level `if`.

---

## 7. References

- [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench) — benchmark and feature checklist
- [epfl-dlab/JSONSchemaBench](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench) — the dataset (10 configs, 9,542 schemas)
- [JSON Schema Test Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite)
- [Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-core.html) — enforced by `jsonschema` at validation time
- Geng et al., 2025, [Generating Structured Outputs from Language Models](https://arxiv.org/abs/2501.10868)

---

*Measured against `benchmarking/jsonschemabench/` over all **9,542** corpus schemas in all
10 configs — token figures (`cl100k_base`) and keyword frequency alike, at a 10 s/schema
budget (`--timeout`, now enforced by the sweep rather than only described here). The
before/after figures in §4 come from a matched run of the same harness against `afa03f4`,
the commit preceding the `patternProperties` change, serialised so the two runs never
contend for CPU. Regenerate with the commands at the top of this file.*
