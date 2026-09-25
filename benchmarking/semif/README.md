# SemIf parity benchmark

Checks that `SemIfLM` reproduces [SemIf](https://github.com/TheoLeeCJ/SemIf)'s published
decisions. It scores SemIf's 144-row authored workload (three families: evidence
interpretation, rule application, candidate selection) through `SemIfLM` and compares the
result with SemIf's own row-level predictions for the same model. Both files are fetched
at a pinned commit (`1f2dea3e`) and checked against their SHA-256 before use.

```bash
# llama.cpp serving one of SemIf's browser-demo GGUFs
llama-server -m Qwen3-0.6B-Q8_0.gguf --jinja -ngl 99 --port 8089
uv run python -m benchmarking.semif.parity --reference qwen3-0.6b \
    --api-base http://localhost:8089/v1 --template-url http://localhost:8089
```

`--reference` is `qwen3-0.6b`, `minicpm5-2b` or `qwen3.5-4b`. `--template-url` (llama.cpp
only) renders each prompt through the server's chat template and compares its hash with
SemIf's recorded `prompt_sha256`. Pass `--our-model` when the server runs a different GGUF
from the default. Each run writes `results/parity-<gguf>-<date>.json`, with a summary and
every row's probabilities.

## Results (2026-09-22)

llama.cpp CUDA server (`ghcr.io/ggml-org/llama.cpp:server-cuda`), RTX 4070 Laptop. The
references are SemIf's native BF16 runs. Accuracy is SemIf's headline metric, mean family
balanced accuracy.

| Model (GGUF) | SemIf reported | `SemIfLM` | Argmax agreement | Mean max \|Δp\| | Prompt hash matches |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 0.440 | 0.443 | 97.2% | 0.034 | 144/144 |
| MiniCPM5-2B Q8_0 | 0.686 | 0.681 | 97.9% | 0.023 | 144/144 |
| MiniCPM5-2B Q4_K_M | 0.686 | 0.633 | 82.6% | 0.162 | 144/144 |
| Qwen3.5-4B Q4_K_M | 0.813 | 0.802 | 95.8% | 0.049 | 144/144 |

- **The prompts are identical.** All 576 rendered prompts hash to SemIf's recorded
  `prompt_sha256`, so the input matches byte for byte and only weights and runtime differ.
- **The readout lost no probability mass.** No run had an option letter fall outside the top
  20 logprobs. Recomputing the metric from SemIf's own predictions reproduces its reported
  numbers exactly, so the metric port is correct.
- **The remaining drift is quantization.** At Q8_0 both models agree with BF16 on about 97–98%
  of rows. MiniCPM5-2B at Q4_K_M, SemIf's browser build, drops to 82.6%. Running the same
  model at Q8_0 recovers 97.9%, which puts the gap on the 4-bit weights, not the
  implementation.

## Runtime ablation (2026-09-25)

`runtime.py` scores the same rows in-process with PyTorch, once per runtime variant, to
see which systems optimizations make SemIf's readout faster and what each costs in parity
and calibration. The baseline is SemIf's own `direct.py` path: BF16, SDPA,
`logits_to_keep=1`, full-vocabulary `lm_head`, and for Qwen3.5 Transformers' PyTorch Gated
DeltaNet, as SemIf's pinned dependencies run it. Each later variant adds one change to the
one before it; quantized variants add theirs on top of the fused one.

```bash
uv run --extra dspy --with torch==2.10.0 --with transformers==5.17.0 --with accelerate \
    --with torchao==0.17.0 --with flash-linear-attention==0.5.2 \
    python -m benchmarking.semif.runtime --reference qwen3-0.6b \
    --variants semif,allpos,selected,graph,compiled,int8,fp8   # --workload hard for 4k states
# Qwen3.5-4B: --variants semif,allpos,selected,graph,fla,fla-compiled,fla-int8
```

RTX 2000 Ada (16 GB, sm_89), torch 2.10, transformers 5.17, torchao 0.17, fla 0.5.2, idle
GPU. Serial forward latency, mean over 3 interleaved passes. authored144 prompts run
127–171 tokens; JevBench's public hard tier (`--workload hard`, 111 items) runs 211–3,997.
Agreement and KL are against the SemIf path in the same run. ECE is top-choice, 10 bins,
on gold labels.

| Variant | Qwen3-0.6B<br>authored144 | Qwen3-0.6B<br>hard | MiniCPM5-2B<br>authored144 | MiniCPM5-2B<br>hard | Qwen3.5-4B<br>authored144 | Qwen3.5-4B<br>hard |
|---|---:|---:|---:|---:|---:|---:|
| `semif`: SemIf's path | 15.6 ms | 80.3 ms | 38.6 ms | 208 ms | 98.4 ms | 673 ms |
| `allpos`: logits for every position (naive HF) | +0% | +12% | +1% | +7% | −1% | +5% |
| `selected`: `lm_head` cut to the 16 letter rows | −12% | 0% | −6% | −1% | −8% | −1% |
| `graph`: + CUDA graphs, length-bucketed static buffers | −20% | 0% | −7% | +1% | −11% | +1% |
| `fla`: + fused Gated DeltaNet kernel (Qwen3.5 only) | | | | | −38% | −31% |
| `compiled` / `fla-compiled`: + `torch.compile` fused kernels | **−38%** | **−30%** | **−19%** | **−14%** | **−44%** | **−43%** |
| `int8`: + INT8 weights and activations | −61% | −41% | −49% | −41% | −61% | −63%¹ |
| `fp8`: + FP8 weights and activations, no graph | −33% | | −33% | | −56%² | |

| Variant: agreement / KL / ECE | Qwen3-0.6B<br>authored144 | Qwen3-0.6B<br>hard | MiniCPM5-2B<br>authored144 | MiniCPM5-2B<br>hard | Qwen3.5-4B<br>authored144 | Qwen3.5-4B<br>hard |
|---|---:|---:|---:|---:|---:|---:|
| `semif` | — / — / 0.450 | — / — / 0.571 | — / — / 0.160 | — / — / 0.453 | — / — / 0.064 | — / — / 0.134 |
| `selected` | 100% / 0 / 0.450 | 100% / 0 / 0.571 | 100% / 0 / 0.160 | 100% / 0 / 0.453 | 100% / 0 / 0.064 | 100% / 0 / 0.134 |
| `graph` | 99.3% / 0.0004 / 0.457 | 100% / 0.0008 / 0.570 | 100% / 0.0003 / 0.159 | 100% / 0.0003 / 0.441 | 100% / 0.0001 / 0.065 | 97.3% / 0.0005 / 0.120 |
| `fla` | | | | | 99.3% / 0.0008 / 0.065 | 98.2% / 0.0011 / 0.112 |
| `compiled` / `fla-compiled` | 98.6% / 0.011 / 0.441 | 100% / 0.004 / 0.566 | 99.3% / 0.002 / 0.151 | 100% / 0.002 / 0.426 | 100% / 0.0007 / 0.058 | 96.4% / 0.0013 / 0.115 |
| `int8` | 90.3% / 0.155 / 0.471 | 86.5% / 0.197 / 0.579 | 95.1% / 0.024 / 0.178 | 97.3% / 0.024 / 0.413 | 88.9% / 0.061 / 0.101 | 89.2% / 0.042 / 0.139¹ |
| `fp8` | 92.4% / 0.135 / 0.494 | | 91.0% / 0.044 / 0.182 | | 95.1% / 0.014 / 0.046² | |

¹ Qwen3.5-4B's BF16 and INT8 copies plus the eager baseline's 4k-token activations do not
fit in 16 GB together, so `fla-int8` ran in its own process and is compared with the
ladder run's saved `semif` probabilities (the eager baseline is bit-identical across runs).
² Its own run, next to `semif` and `fla-compiled`: 22% faster than `fla-compiled`.

- **In-process, the baseline is SemIf.** Qwen3.5-4B's `semif` path agrees with SemIf's
  published BF16 predictions on 144/144 rows and scores 0.813 mean family balanced
  accuracy, SemIf's reported number.
- **`logits_to_keep=1` is already in SemIf.** Computing logits for every position costs
  5–12% on long prompts and nothing measurable on short ones.
- **The selected-token `lm_head` is exact and helps short prompts.** Its logits match the
  full head bit for bit. It saves 6–12% at about 150 tokens, where reading the vocabulary
  matrix (311 MB on Qwen3-0.6B, 1.27 GB on Qwen3.5-4B) is a real share of the pass, and
  nothing once the prompt dominates.
- **CUDA graphs help only while launch overhead matters:** a further 1–9% at 150 tokens,
  nothing on long prompts. Right-padding to a bucket is exact under causal attention: on
  Qwen3-0.6B in fp32, padded and unpadded logits are identical and graph vs eager differ
  by 1e-4 (a 4B model in fp32 does not fit this GPU). The BF16 drift left is kernel-shape
  noise. Every flip on Qwen3.5's hard tier, here and in the fused variants, is a row
  where the baseline's top two options were within 0.05, most of them tied at 0.500.
- **Fused kernels are the biggest parity-safe win,** at 14–44%. On Qwen3.5-4B most of it
  is flash-linear-attention's Gated DeltaNet kernel standing in for Transformers' PyTorch
  reference (−38% on its own). `torch.compile` of that PyTorch reference never finished
  compiling (over 20 minutes on one prompt), so Qwen3.5's compiled and quantized variants
  run on `fla`. Inductor keeps fused intermediates in fp32, so probabilities move a
  little more than with graphs alone, but agreement with SemIf's published predictions
  stays where the eager path's is.
- **Quantization is the only large step left, and it costs calibration.** INT8 W8A8 is
  the fastest variant everywhere: 41–63% under SemIf's path. It also moves 3–14% of the
  choices. On Qwen3.5-4B, the strongest model here, ECE rises from 0.064 to 0.101 and
  balanced accuracy falls from 0.813 to 0.779. MiniCPM5-2B takes it best (95–97%
  agreement), between SemIf's own Q8_0 (97.9%) and Q4_K_M (82.6%) GGUFs above. FP8
  (per-row e4m3) runs without a CUDA graph. Against the fused BF16 path it is slower on
  Qwen3-0.6B, 18% faster on MiniCPM5-2B and 22% faster on Qwen3.5-4B, where it also
  calibrates better than INT8 (95.1% agreement, ECE 0.046). Inductor autotunes the quantized kernels, so their
  agreement moved by up to 3 points between otherwise identical runs; read those rows as
  ±3 points.
- **INT4 weight-only is slower and breaks the readout.** Weight-only formats target
  decode, but every SemIf request is a prefill. INT4 (round-to-nearest, group 128,
  tinygemm) took 23.8 ms against 16.2 ms for BF16 eager on Qwen3-0.6B and agreed on 52% of
  rows (`results/runtime-qwen3-0.6b-authored144-int4-*.json`). INT8 weight-only ran 683 ms
  per row compiled and was dropped.

Pitfalls that turned up along the way:

- Transformers treats CUDA graph capture as tracing and then builds a dense L×L attention
  mask instead of passing `is_causal`, which costs SDPA its flash kernel: graphs ran 2x
  slower than eager at 4k tokens. `runtime.py` registers a mask-free causal SDPA attention
  for the capture.
- torchao's quantize configs call `recommended_inductor_config_setter()` by default, which
  switches the whole process to TF32 and retunes Inductor. It moved Qwen3.5's eager fp32
  Gated DeltaNet baseline by up to 0.25 logit. `runtime.py` passes
  `set_inductor_config=False`.
- Transformers binds Qwen3.5 to flash-linear-attention whenever `fla` is importable, which
  would speed up the baseline too. `runtime.py` hides it at import and swaps the kernel in
  only for the `fla` variants.
- torchao's FP8 path hangs when run on a side CUDA stream, which graph warm-up needs, and
  in a dynamic-shape compile. It runs compiled per bucket, uncaptured, with Dynamo's
  recompile limit raised so later buckets don't fall back to eager.
- Separate graph memory pools per variant added up to an out-of-memory error at 4k tokens
  on the 4B model. All variants share one pool.
- INT4's default torchao packing needs the `mslk` kernel library. `tile_packed_to_4d`
  (tinygemm, in core torch) does not.

## Serving backends vs in-process (2026-09-25)

`backends.py` scores the same rows through `SemIfLM` against an OpenAI-compatible server,
one request at a time, and compares the probabilities with the in-process `semif` run.
Same GPU, same pinned checkpoints (GGUFs as in the parity table above). vLLM 0.30.0 ran
BF16 with `--no-enable-prefix-caching` (`--language-model-only --max-num-seqs 16` for
Qwen3.5); llama.cpp is `ghcr.io/ggml-org/llama.cpp:server-cuda` with `-np 1 -c 8192
--jinja` and `cache_prompt: false`.

```bash
python -m benchmarking.semif.backends --reference qwen3.5-4b --server llamacpp \
    --api-base http://localhost:8089/v1 --label "llama.cpp Q8_0"    # --workload hard
```

Mean latency per question. Servers are timed through `SemIfLM`, HTTP included. In-process is
forward time plus tokenization (0.3–3 ms); a DSPy wrapper around it would add about 1 ms,
and `SemIfLM`'s own DSPy + litellm path measured 4–5 ms against a stub server.

| Backend | Qwen3-0.6B<br>authored144 | Qwen3-0.6B<br>hard | MiniCPM5-2B<br>authored144 | MiniCPM5-2B<br>hard | Qwen3.5-4B<br>authored144 | Qwen3.5-4B<br>hard |
|---|---:|---:|---:|---:|---:|---:|
| In-process, SemIf's path (BF16 eager) | 16.1 ms | 83.2 ms | 38.9 ms | 210 ms | 98.7 ms | 675 ms |
| In-process, fused (`compiled` / `fla-compiled`) | **10.2 ms** | **59.4 ms** | **31.6 ms** | **181 ms** | **55.4 ms** | 384 ms |
| vLLM BF16 | 26.3 ms | 74.9 ms | 48.4 ms | 197 ms | 78.3 ms | **365 ms** |
| llama.cpp Q8_0, defaults | 46.3 ms | 161 ms | 52.7 ms | 262 ms | 209 ms | 719 ms |
| llama.cpp Q8_0, `--cache-ram 0 --ctx-checkpoints 0` | 35.1 ms | 102 ms | 50.6 ms | 229 ms | 84.5 ms | 543 ms |
| llama.cpp Q4_K_M, defaults | | | 50.7 ms | 268 ms | 191 ms | 736 ms |
| llama.cpp Q4_K_M, `--cache-ram 0 --ctx-checkpoints 0` | | | 47.5 ms | 236 ms | 87.0 ms | 566 ms |
| *Quantized:* in-process `fla-int8` / `fla-fp8` | | | | | 38.3 / 43.0 ms | 254 ms / |
| *Quantized:* vLLM FP8 | | | | | 65.3 ms | 256 ms |

Agreement and KL against the in-process `semif` probabilities (authored144 / hard):

| Backend | Qwen3-0.6B | MiniCPM5-2B | Qwen3.5-4B |
|---|---:|---:|---:|
| vLLM BF16 | 97.9 / 99.1% · KL 0.006 | 97.2 / 99.1% · KL 0.003 | 99.3 / 95.5% · KL 0.001 |
| llama.cpp Q8_0 | 97.2 / 99.1% · KL 0.037 / 0.013 | 98.6 / 98.2% · KL 0.003 | 96.5 / 95.5% · KL 0.002 |
| llama.cpp Q4_K_M | | 81.2 / 85.6% · KL 0.21 / 0.14 | 95.8 / 90.1% · KL 0.024 |
| vLLM FP8 | | | 95.8 / 94.6% · KL 0.014 |

- **For one question per state, llama.cpp's defaults cost `SemIfLM` up to 2.5x.** Its
  host-RAM prompt cache and, on Qwen3.5's recurrent layers, its context checkpoints add a
  roughly fixed cost to every request whose prompt differs from the slot's last one.
  With `--cache-ram 0 --ctx-checkpoints 0`, Qwen3.5-4B Q8_0 drops from 209 to 85 ms on
  authored144 and from 719 to 543 ms on the hard tier, with the same probabilities up to
  batch-split rounding. The same holds with `SemIfLM`'s usual `cache_prompt` left on
  (199 vs 85 ms; Qwen3-0.6B 42 vs 30 ms). The JevBench latencies for `SemIfLM` in
  [`benchmarking/jevbench`](../jevbench/README.md) were measured with the defaults.
- **For several questions per state, keep the defaults.** The same caches let a state's
  later questions reuse its prefix. With four questions per long hard-tier state, asked in
  turn with `cache_prompt` on (12 states):

  | Four questions vs one, same state | defaults | `--cache-ram 0` | both off |
  |---|---:|---:|---:|
  | Qwen3.5-4B Q8_0 (hybrid) | 1.07x | 1.99x | 4.05x |
  | Qwen3-0.6B Q8_0 (attention only) | 1.12x | 1.74x | 1.81x |

  On Qwen3.5-4B, turning the caches off makes those four questions 3.7x slower (4.5 s
  against 1.2 s). On Qwen3-0.6B, `--cache-ram 0` alone came out faster in both cases
  (171 vs 290 ms for one question, 298 vs 324 ms for four), but that is one model.
- **vLLM BF16 gives SemIf's numbers.** It agrees with the in-process path about as
  closely as the fused in-process variants do (KL ≤ 0.007), and on long prompts it is as
  fast as anything here: 365 ms on Qwen3.5-4B's hard tier against 384 ms in-process.
- **In-process wins on short prompts, by the server's per-request overhead.** At about 150
  tokens the fused in-process path is 1.4–2.6x faster than the best server (10 vs 26 ms,
  32 vs 48 ms, 55 vs 78 ms). The gap is a fixed per-request serving cost (API handling,
  scheduling, top-logprob extraction) that long prompts amortize. On the hard tier
  in-process leads by 21% on Qwen3-0.6B and 8% on MiniCPM5-2B, and vLLM is 5% ahead on
  Qwen3.5-4B.
- **Q4_K_M buys no speed for SemIf.** Every request is a prefill, which is compute-bound, so
  4-bit weights save memory but not time, and on MiniCPM5-2B they move a fifth of the
  choices.
- **llama.cpp's Qwen3.5 prefill is slower than vLLM's** on long prompts (543 vs 365 ms,
  tuned). On the pure-attention models the gap is 16–36%.
