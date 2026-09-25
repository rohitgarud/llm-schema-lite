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
# or any /v1/systemone endpoint through JevLM, e.g. a local Kev server
uv run python -m benchmarking.jevbench.run --jev-url http://127.0.0.1:8009/v1/systemone \
    --model kev-latest --our-model "Kev-0.8B bf16"
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
| `JevLM` to [Kev](https://github.com/jaredpalmer/kev)-0.8B bf16 (`kev.serve`) | 100% | 75.0% | 29.7% | 55.1 | 0.11 s / 0.24 s |
| `JevLM` to [Decider](https://github.com/Mapika/decider)-2B v10 bf16 (`decider.serve`, eager) | 100% | 86.1% | 45.0% | 44.8 | not serial |
| `JevLM` to [Laya](https://github.com/NandhaKishorM/laya) (`laya-serve`, its router) | 95.8% | 68.1% | 33.3% | 62.4 | 0.05 s / 6.8 s |
| [OpenSourceJev](https://github.com/sabeel111/OpenSourceJev), Qwen3.5-4B Q4_K_M (unsloth) | 100% | 93.1% | 59.5% | 65.1 | 0.82 s / 17.1 s |

- **Same model, 4-bit, lands a few items short of SemIf's BF16 run.** Of the 21 items only
  one of the two got right, SemIf's run got 15 and `SemIfLM` got 6 (McNemar p = 0.078).
  Two things differ besides the weights. JevBench's SemIf adapter prefixes each option's
  description with its id (`"true: ..."`), while `SemIfLM` sends what `JevAdapter` would.
  And the reference ran in BF16: on authored144 the same Q4_K_M agrees with BF16 on 95.8% of
  rows.
- **No item lost its option letters** from the top 20 logprobs, on any model.
- **Kev-0.8B is the only Kev that fits this 8 GB GPU in bf16.** Kev's own README puts the
  4B and 9B well above it, and it has no GGUF. On the hard tier it is the least accurate
  run here, but its calibration (55.1) beats the 2B and 0.6B `SemIfLM` runs. Qwen3.5-4B
  beats it on both.
- **Other open decision models.** Decider and Laya take `JevAdapter`'s requests unchanged,
  so they ran through `JevLM`. Decider ran without its CUDA graphs, whose capture stalled
  on the 8 GB GPU, and its latency is left out because a second client overlapped the run.
  Laya's router picks its checkpoint per request, and its English checkpoint reads 512
  tokens, which may explain its hard-tier score on these long states.
- **OpenSourceJev is not a drop-in backend.** It rejects the `instructions` object
  `JevAdapter` sends (`{"task": ..., "question": ...}`) with an HTTP 500, so it was sent
  JevBench's plain string, with its context raised from 4k to 8k tokens. It reads option
  logits from llama.cpp, as `SemIfLM` does, on the GGUF its code pins (unsloth's, not the
  bartowski file in the `SemIfLM` row). Its 93.1% on the standard tier matches the 93% it
  reports for JevBench's original set.
- **p95 is the hard tier's long states,** which run to about 4k tokens of prompt each.
- **The `SemIfLM` latencies use llama.cpp's default cache settings.** Every JevBench item
  is one question on a new state, which is the case where `--cache-ram 0
  --ctx-checkpoints 0` pays. On an RTX 2000 Ada it cut Qwen3.5-4B Q8_0 from 209 to 85 ms
  per short question and from 719 to 543 ms on the hard tier. These rows were not rerun,
  because the table above comes from a different GPU. See the
  [serving comparison](../semif/README.md#serving-backends-vs-in-process-2026-09-25).

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
