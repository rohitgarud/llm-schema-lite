"""In-process SemIf runtime ablation: which systems optimizations make the readout faster?

Scores SemIf's 144-row authored workload (or, with ``--workload hard``, JevBench's public
hard tier of states up to 4k tokens) in-process with PyTorch, once per runtime variant.
Reports latency next to parity with SemIf's path and SemIf's BF16 predictions, and the
calibration (gold NLL, ECE) quantization can cost. Variants run interleaved (every row goes
through every variant before the next row), so a GPU shared with other work slows all
of them alike.

- ``semif``: SemIf's own ``direct.py`` path. ``logits_to_keep=1``, full-vocabulary
  ``lm_head`` on the last position, then the option-letter logits.
- ``allpos``: naive Hugging Face inference. Logits for every position, last row used.
- ``selected``: the ``lm_head`` swapped for the 16 letter rows (A-P), so the last position
  projects onto 16 logits instead of the full vocabulary.
- ``graph``: ``selected``, replayed from CUDA graphs over static, length-bucketed buffers.
- ``compiled``: ``graph`` over a ``torch.compile`` (Inductor) forward, with fused kernels.
- ``fla``: ``graph`` with Qwen3.5's Gated DeltaNet layers on flash-linear-attention's fused
  Triton kernel instead of Transformers' PyTorch reference (needs ``fla``);
  ``fla-compiled`` adds ``torch.compile``, and ``fla-int8`` etc. quantize on top of that.
  On Qwen3.5, ``torch.compile`` of the PyTorch reference ran over 20 minutes on one
  prompt without finishing, so its compiled and quantized rungs need ``fla``.
- ``fp8``, ``int8``, ``int8wo``, ``int4wo``: ``compiled`` on a copy whose decoder linears
  torchao quantized: FP8 or INT8 weights and activations (tensor cores), or INT8 / INT4
  weights only. Needs ``torchao`` (0.17 with torch 2.10). FP8 skips the CUDA graph and
  compiles per bucket: its graph capture and its dynamic-shape compile both hang.

Needs ``torch``, ``transformers`` and ``torchao``, which are not project dependencies::

    uv run --extra dspy --with torch==2.10.0 --with transformers==5.17.0 --with accelerate \\
        --with torchao==0.17.0 python -m benchmarking.semif.runtime --reference qwen3-0.6b \\
        --variants semif,allpos,selected,graph,compiled,int8,fp8
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import functools
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

# Transformers binds Qwen3.5's Gated DeltaNet to flash-linear-attention whenever ``fla`` is
# importable. Keep it hidden so ``semif`` stays SemIf's PyTorch path; ``fla`` opts in.
sys.modules.setdefault("fla", None)  # type: ignore[arg-type]

import transformers  # noqa: E402
import transformers.masking_utils  # noqa: E402
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402

from benchmarking.semif.common import hard_rows, softmax, summarize  # noqa: E402
from benchmarking.semif.parity import (  # noqa: E402
    REFERENCES,
    WORKLOAD,
    fetch_jsonl,
    question,
)
from llm_schema_lite.dspy_integration.adapters.semif_lm import (  # noqa: E402
    LETTERS,
    _messages,
)

# name -> (HF source, pinned revision), SemIf's manifests/models.json
MODELS = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "minicpm5-2b": ("openbmb/MiniCPM5-2B", "12a3808a956f869c767195e9266b59c4d21d92e2"),
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
}
RESULTS = Path(__file__).parent / "results"
Scorer = Callable[[list[int], int], list[float]]  # (prompt ids, option count) -> option logits


class SelectedHead(torch.nn.Module):
    """An ``lm_head`` cut down to a few vocabulary rows; the model's forward is unchanged."""

    def __init__(self, head: torch.nn.Linear, rows: list[int]) -> None:
        super().__init__()
        index = torch.tensor(rows, device=head.weight.device)
        self.weight = torch.nn.Parameter(head.weight[index].detach().clone(), requires_grad=False)
        self.bias = None if head.bias is None else head.bias[index].detach().clone()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden, self.weight, self.bias)


def bucket(n: int) -> int:
    """Round a prompt length up to its CUDA graph's bucket: steps of 16, or 1/8 above 128."""
    step = max(16, 1 << max(0, n.bit_length() - 4))
    return -(-n // step) * step


def _causal_sdpa(
    module: Any, query: Any, key: Any, value: Any, attention_mask: Any, **kw: Any
) -> Any:
    return sdpa_attention_forward(module, query, key, value, None, **kw)


# Transformers builds a dense L x L mask while a CUDA graph is being captured (``is_tracing``
# covers stream capture), which costs SDPA its flash kernel: 2x slower at 4k tokens. A graph
# only ever sees one right-padded causal prompt, so it can rely on ``is_causal`` instead.
transformers.AttentionInterface.register("sdpa_causal", _causal_sdpa)
transformers.masking_utils.AttentionMaskInterface.register("sdpa_causal", lambda *a, **k: None)


@functools.cache
def _graph_pool() -> Any:
    """One CUDA graph memory pool for every scorer, so pools don't add up at 4k tokens.

    Graphs replay one at a time and each output is copied off the GPU at once, so a replay
    may reuse memory another graph's output held: the same reasoning already lets one
    scorer's buckets replay in any order.
    """
    return torch.cuda.graph_pool_handle()


class GraphScorer:
    """Replay one captured CUDA graph per length bucket, with static input buffers.

    The prompt is right-padded to its bucket. Under causal attention the padding cannot
    reach the last real token, whose position is a device tensor passed as
    ``logits_to_keep``, so one graph serves every length in its bucket. With
    ``capture=False`` the same bucketed forward runs uncaptured.
    """

    def __init__(
        self,
        model: Any,
        head: torch.nn.Module,
        pad: int,
        compile: bool = False,
        capture: bool = True,
        dynamic: bool = True,
        patches: dict[tuple[Any, str], Callable[..., Any]] | None = None,
    ) -> None:
        self.model, self.head, self.pad, self.capture = model, head, pad, capture
        self.patches = patches or {}  # (module, attribute) -> replacement, during the readout
        # Inductor fuses the elementwise work (RMSNorm, SwiGLU, RoPE) into fewer kernels;
        # one dynamic-length compile serves every bucket, each still captured as a graph.
        self.forward = torch.compile(self._forward, dynamic=dynamic) if compile else self._forward
        if compile and not dynamic:
            # One compile per bucket: past Dynamo's default of 8, new lengths would run eager.
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 256)
        self.pool = _graph_pool()
        self.graphs: dict[int, tuple[Any, torch.Tensor, torch.Tensor, Any]] = {}

    def _forward(self, ids: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids, use_cache=False, logits_to_keep=last).logits[0, 0]

    @contextlib.contextmanager
    def _readout(self) -> Any:
        """Swap in the selected head and the mask-free causal attention for one forward."""
        original, attention = self.model.lm_head, self.model.config._attn_implementation
        saved = {key: getattr(*key) for key in self.patches}
        self.model.lm_head = self.head
        self.model.set_attn_implementation("sdpa_causal")
        for (module, name), replacement in self.patches.items():
            setattr(module, name, replacement)
        try:
            yield
        finally:
            self.model.lm_head = original
            self.model.set_attn_implementation(attention)
            for (module, name), value in saved.items():
                setattr(module, name, value)

    def _capture(self, length: int) -> tuple[Any, torch.Tensor, torch.Tensor, Any]:
        device = self.model.device
        ids = torch.full((1, length), self.pad, device=device)
        last = torch.full((1,), length - 1, device=device)
        if not self.capture:
            return None, ids, last, None
        with self._readout():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.forward(ids, last)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out = self.forward(ids, last)
        return graph, ids, last, out

    def __call__(self, prompt: list[int], n: int) -> list[float]:
        length = bucket(len(prompt))
        if length not in self.graphs:
            self.graphs[length] = self._capture(length)
        graph, ids, last, out = self.graphs[length]
        ids.fill_(self.pad)
        ids[0, : len(prompt)].copy_(torch.tensor(prompt), non_blocking=True)
        last.fill_(len(prompt) - 1)
        if graph is None:
            with self._readout():
                out = self.forward(ids, last)
        else:
            graph.replay()
        return out[:n].float().tolist()


def load(name: str, device: str = "cuda:0") -> tuple[Any, Any]:
    source, revision = MODELS[name]
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, revision=revision)
    config = transformers.AutoConfig.from_pretrained(source, revision=revision)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:  # as SemIf's load_causal_model
        cls, config = transformers.Qwen3_5ForCausalLM, config.get_text_config()
    model = cls.from_pretrained(
        source, revision=revision, config=config, dtype=torch.bfloat16, device_map=device
    )
    return model.eval(), tokenizer


def letter_ids(tokenizer: Any) -> list[int]:
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(i) != 1 for i in ids):
        raise ValueError("Every option letter must be one token")
    return [i[0] for i in ids]


def encode(tokenizer: Any, row: dict[str, Any]) -> tuple[list[int], str]:
    q = question(row)
    prompt = tokenizer.apply_chat_template(
        _messages(row["state"], q, q["criteria"]),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(prompt, add_special_tokens=False), prompt


def quantized(model: Any, kind: str) -> Any:
    """Quantize the decoder's linear layers onto the GPU with torchao (``lm_head`` stays bf16).

    ``model`` may sit on the CPU: each linear moves to the GPU as it is quantized, so a
    4B model's bf16 copy never has to fit next to the one already there.
    """
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        PerRow,
        quantize_,
    )

    # set_inductor_config=False: by default torchao switches the whole process to TF32 and
    # retunes Inductor, which moved Qwen3.5's eager fp32 Gated DeltaNet baseline by 0.25 logit.
    config = {
        "fp8": lambda: Float8DynamicActivationFloat8WeightConfig(
            granularity=PerRow(), set_inductor_config=False
        ),
        "int8": lambda: Int8DynamicActivationInt8WeightConfig(set_inductor_config=False),
        "int8wo": lambda: Int8WeightOnlyConfig(set_inductor_config=False),
        "int4wo": lambda: Int4WeightOnlyConfig(
            group_size=128, int4_packing_format="tile_packed_to_4d", set_inductor_config=False
        ),
    }[kind]()
    quantize_(
        model,
        config,
        filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and "lm_head" not in fqn,
        device="cuda:0",
    )
    return model.to("cuda:0")


def fla_patches(model: Any) -> dict[tuple[Any, str], Callable[..., Any]]:
    """Route Qwen3.5's Gated DeltaNet prefill to flash-linear-attention's fused Triton kernel."""
    if sys.modules.get("fla", 0) is None:
        del sys.modules["fla"]  # undo the import-time block
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    modeling = sys.modules[type(model).__module__]
    if not hasattr(modeling, "torch_chunk_gated_delta_rule"):
        raise ValueError(f"{type(model).__name__} has no Gated DeltaNet layers for fla")
    return {(modeling, "torch_chunk_gated_delta_rule"): chunk_gated_delta_rule}


def make_variants(
    model: Any,
    slots: list[int],
    names: list[str],
    pad: int = 0,
    reload: Callable[[], Any] | None = None,
) -> dict[str, Scorer]:
    """Build each requested variant's scorer. Variants that swap the head get their own copy."""
    device = model.device
    head = model.lm_head
    variants: dict[str, Scorer] = {}

    def run(ids: list[int], keep: int, custom_head: torch.nn.Module | None) -> torch.Tensor:
        inputs = torch.tensor([ids], device=device)
        model.lm_head = custom_head or head
        try:
            return model(input_ids=inputs, use_cache=False, logits_to_keep=keep).logits[0, -1]
        finally:
            model.lm_head = head

    def full(keep: int) -> Scorer:
        def score(ids: list[int], n: int) -> list[float]:
            vocab = run(ids, keep, None).float()
            return vocab[slots[:n]].tolist()

        return score

    selected_head = SelectedHead(head, slots)

    def selected(ids: list[int], n: int) -> list[float]:
        return run(ids, 1, selected_head).float()[:n].tolist()

    table: dict[str, Callable[[], Scorer]] = {
        "semif": lambda: full(1),
        "allpos": lambda: full(0),
        "selected": lambda: selected,
        "graph": lambda: GraphScorer(model, selected_head, pad),
        "compiled": lambda: GraphScorer(model, selected_head, pad, compile=True),
    }
    table["fla"] = lambda: GraphScorer(model, selected_head, pad, patches=fla_patches(model))
    table["fla-compiled"] = lambda: GraphScorer(
        model, selected_head, pad, compile=True, patches=fla_patches(model)
    )

    def quantized_scorer(kind: str, fla: bool) -> GraphScorer:
        copy = quantized(reload(), kind)  # type: ignore[misc]
        # torchao's FP8 path hangs in CUDA graph capture and in a dynamic-shape compile, so it
        # runs compiled once per bucket, uncaptured.
        return GraphScorer(
            copy,
            selected_head,
            pad,
            compile=True,
            capture=kind != "fp8",
            dynamic=kind != "fp8",
            patches=fla_patches(copy) if fla else None,
        )

    for kind in ("fp8", "int8", "int8wo", "int4wo"):  # each on its own copy of the model
        table[kind] = lambda kind=kind: quantized_scorer(kind, fla=False)
        table[f"fla-{kind}"] = lambda kind=kind: quantized_scorer(kind, fla=True)
    for name in names:
        variants[name] = table[name]()
    return variants


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", choices=sorted(MODELS), required=True, help="Model")
    parser.add_argument("--workload", choices=["authored144", "hard"], default="authored144")
    parser.add_argument("--variants", default="semif,allpos,selected")
    parser.add_argument("--repeats", type=int, default=3, help="Timed passes over the workload")
    parser.add_argument("--tag", default="", help="Suffix for the results file")
    parser.add_argument("--limit", type=int, help="Score only the first N rows (smoke test)")
    args = parser.parse_args(argv)
    names = args.variants.split(",")

    reference: dict[str, Any] = {}
    if args.workload == "hard":
        rows = hard_rows()
    else:
        rows = fetch_jsonl(*WORKLOAD)
        if args.reference in REFERENCES:
            path, sha, *_ = REFERENCES[args.reference]
            reference = {p["id"]: p for p in fetch_jsonl(path, sha)}
    rows = rows[: args.limit]
    model, tokenizer = load(args.reference)
    slots = letter_ids(tokenizer)
    variants = make_variants(
        model,
        slots,
        names,
        tokenizer.pad_token_id or 0,
        reload=lambda: load(args.reference, "cpu")[0],
    )

    encoded = {r["id"]: encode(tokenizer, r) for r in rows}
    hashes = sum(
        hashlib.sha256(encoded[r["id"]][1].encode()).hexdigest()
        == reference[r["id"]]["prompt_sha256"]
        for r in rows
        if r["id"] in reference
    )
    lengths = [len(ids) for ids, _ in encoded.values()]

    logits: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    times: dict[str, list[float]] = {n: [] for n in names}
    with torch.inference_mode():
        for row in rows:  # warm-up: kernels, allocator, every bucket's capture or compilation
            for name, score in variants.items():
                start = time.perf_counter()
                score(encoded[row["id"]][0], len(row["options"]))
                if (took := time.perf_counter() - start) > 2:
                    print(f"warm-up {name} {row['id']}: {took:.0f} s", file=sys.stderr, flush=True)
        for rep in range(args.repeats):
            print(f"pass {rep + 1}/{args.repeats}", file=sys.stderr, flush=True)
            for k, row in enumerate(rows):
                ids, n = encoded[row["id"]][0], len(row["options"])
                order = names[(k + rep) % len(names) :] + names[: (k + rep) % len(names)]
                for name in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    out = variants[name](ids, n)
                    torch.cuda.synchronize()
                    times[name].append(time.perf_counter() - start)
                    logits[name].setdefault(row["id"], out)

    probs = {n: {i: softmax(v) for i, v in logits[n].items()} for n in names}
    ref_probs = {i: reference[i]["probabilities"] for i in probs[names[0]]} if reference else None
    base = probs["semif" if "semif" in probs else names[0]]
    summary = {
        "reference": args.reference,
        "workload": args.workload,
        "model": "{} @ {} (bf16)".format(*MODELS[args.reference]),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "prompt_hash_matches": hashes,
        "prompt_tokens": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "max": max(lengths),
        },
        "repeats": args.repeats,
        "date": datetime.date.today().isoformat(),
        "variants": {n: summarize(rows, probs[n], times[n], ref_probs, base) for n in names},
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"runtime-{args.reference}-{args.workload}{args.tag}-{summary['date']}.json"
    out.write_text(json.dumps({"summary": summary, "probabilities": probs}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
