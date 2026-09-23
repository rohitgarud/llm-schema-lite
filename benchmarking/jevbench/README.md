# JevBench through `SemIfLM`

Runs the public tiers of [JevBench](https://github.com/fstandhartinger/jevbench) (MIT),
the benchmark that TypeSafe's Jev and SemIf are ranked on, through `SemIfLM`. The results
sit next to JevBench's published runs of SemIf and Jev. JevBench's items are already in
TypeSafe's wire format (a `state`, then a `noul`, `choice` or `score` question), so each
one goes to `SemIfLM` exactly as `JevAdapter` would send it. The task files and
JevBench's per-task outcomes are fetched at a pinned commit (`2fa63fa3`) and checked
against their SHA-256 before use.

```bash
# llama.cpp serving any GGUF; the hard tier needs about 4k tokens of context per request
llama-server -m Qwen_Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -c 16384 --port 8089
uv run python -m benchmarking.jevbench.run --api-base http://localhost:8089/v1 \
    --our-model "Qwen3.5-4B Q4_K_M"          # add --question-first for that layout
```

Each run writes `results/jevbench-<model>-<date>.json`, with a summary and every item's
probabilities and latency.

**Not the leaderboard number.** Only three tiers are public: easy (48 of 72), standard
(72 of 96) and hard (111 of 220). The judge tier and the held-out halves are not, so the
runner reports accuracy per tier on the 231 public items and re-scores the references on
the same items. Calibration follows JevBench v1.2's formula on the hard tier: the mean of
`100 x (1 - ECE / 0.5)` and `100 x (1 - mean TVD)` to the gold distributions. Latency is
serial p50/p95, as JevBench measures it. Cost is left out, because it is a hosting price.

## Results (2026-09-24)

These are SemIf's three pinned GGUFs, the same files [`benchmarking/semif`](../semif/README.md)
checks parity against. They ran on a llama.cpp CUDA server (`ghcr.io/ggml-org/llama.cpp:server-cuda`,
`-np 16 --kv-unified`) on an RTX 4070 Laptop, one request at a time. The first two rows
are JevBench's own runs.

| System | Easy (48) | Standard (72) | Hard (111) | Hard calibration | p50 / p95 |
|---|---:|---:|---:|---:|---:|
| Jev 1.13.0 (TypeSafe API) | 100% | 98.6% | 73.0% | | |
| SemIf, Qwen3.5-4B BF16 (SemIf's own code) | 100% | 98.6% | 61.3% | | |
| `SemIfLM`, Qwen3.5-4B Q4_K_M | 100% | 94.4% | 55.9% | 70.9 | 0.23 s / 1.37 s |
| `SemIfLM`, MiniCPM5-2B Q4_K_M | 100% | 69.4% | 43.2% | 41.7 | 0.09 s / 0.88 s |
| `SemIfLM`, Qwen3-0.6B Q8_0 | 87.5% | 48.6% | 36.0% | 28.1 | 0.03 s / 0.75 s |
| `SemIfLM`, Qwen3-VL-4B-Instruct Q4_K_M (not a SemIf model) | 100% | 86.1% | 45.0% | 30.9 | 0.10 s / 2.19 s |

- **Same model, 4-bit, lands a few items short of SemIf's BF16 run.** Of the 21 items only
  one of the two got right, SemIf's run got 15 and `SemIfLM` got 6 (McNemar p = 0.078).
  Two things differ besides the weights. JevBench's SemIf adapter prefixes each option's
  description with its id (`"true: ..."`), while `SemIfLM` sends what `JevAdapter` would.
  And the reference ran in BF16: on authored144 the same Q4_K_M agrees with BF16 on 95.8% of
  rows.
- **No item lost its option letters** from the top 20 logprobs, on any model.
- **p95 is the hard tier's long states,** which run to about 4k tokens of prompt each.

### `question_first=True`

The same runs with the criterion and options sent before the evidence. Each row counts the
items only one layout got right, out of 231.

| Model | Only default right | Only question-first right | McNemar p |
|---|---:|---:|---:|
| Qwen3.5-4B Q4_K_M | 18 | 16 | 0.86 |
| MiniCPM5-2B Q4_K_M | 20 | 14 | 0.39 |
| Qwen3-0.6B Q8_0 | 56 | 19 | < 0.0001 |
| Qwen3-VL-4B-Instruct Q4_K_M | 15 | 18 | 0.73 |

On the 2B and 4B models the layout makes no measurable difference. On Qwen3-0.6B,
question-first costs 37 items, and easy accuracy falls from 87.5% to 31.2%. Likely a small model
leans on the options sitting right before its answer, which is why the flag stays off by
default.
