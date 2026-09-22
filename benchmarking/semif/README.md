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
