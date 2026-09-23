"""Local, open-model stand-in for JevLM, following SemIf's direct option-logit readout.

SemIf (https://github.com/TheoLeeCJ/SemIf) puts each option behind a letter, asks the model
for one letter and reads the letters' next-token probabilities. The model generates no
answer text. ``SemIfLM`` runs that readout against any OpenAI-compatible server that
returns ``top_logprobs`` (vLLM, llama.cpp ``llama-server``, SGLang, ...) and answers in
Jev's format, so ``JevAdapter`` works unchanged::

    lm = SemIfLM("openai/Qwen/Qwen3.5-4B", api_base="http://localhost:8000/v1", api_key="x")
    dspy.configure(lm=lm, adapter=JevAdapter())

Each question is one request, a single token long. Probabilities are renormalised over
the letters found in ``top_logprobs``, so they are conditional on the options and
uncalibrated.

Up to 16 options get letters A-P. 17-256 options get two-letter labels AA-PP, read by
the chain rule: a label the tokenizer keeps whole is read from the first token; for a
label it splits, the first letter is pre-filled as the assistant turn and a follow-up
request reads the second (the server must continue a trailing assistant message, as
llama.cpp does). A model may shy away from labels its tokenizer splits: Qwen3 put 0.001
on a correct ``CJ``, which it reads as ``C`` + ``J``.

``dspy.Image`` and other ``dspy.Type`` input fields are carried through as content blocks,
so the readout works on a vision model. The server must return ``top_logprobs`` for a
multimodal request; not every VLM backend does.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import dspy
import litellm
import pydantic
import pydantic_core

LETTERS = "ABCDEFGHIJKLMNOP"
PAIRS = [a + b for a in LETTERS for b in LETTERS]
PREFIX_FLOOR = 0.01  # ponytail: skip follow-ups for first letters under 1% of the mass
PLACEHOLDER = "<<image {}>>"  # stands in for an image in the JSON payload, which stays text
SYSTEM = (  # SemIf's DIRECT_SYSTEM prompt, verbatim
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
SYSTEM_PAIRS = SYSTEM.replace("its uppercase letter", "its two-letter uppercase label")


def _labels(n: int) -> list[str]:
    if not 2 <= n <= len(PAIRS):
        raise ValueError(f"SemIfLM needs 2-{len(PAIRS)} options, got {n}")
    return list(LETTERS[:n]) if n <= len(LETTERS) else PAIRS[:n]


def _options(question: dict[str, Any]) -> dict[str, str]:
    """Map each Jev option key to the description the model reads."""
    if question["type"] == "noul":
        return question.get("criteria") or {"true": "Yes", "false": "No"}
    criteria = question["criteria"]
    if question["type"] == "score":
        return {str(i): desc for i, desc in enumerate(criteria)}
    return dict(criteria)


def _split_media(content: str | list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Separate a user turn into its JSON text and its image blocks.

    DSPy expands a ``dspy.Image`` input into content blocks at the LM boundary, splitting
    the adapter's JSON payload where the image sat - between two string quotes, so the
    text rejoins into valid JSON once each block is replaced by its placeholder.
    """
    if isinstance(content, str):
        return content, []
    media: list[dict[str, Any]] = []
    text = ""
    for block in content:
        if block.get("type") == "text":
            text += block["text"]
        else:
            media.append(block)  # two identical images still get distinct placeholders
            text += PLACEHOLDER.format(len(media))
    return text, media


def _messages(
    state: Any,
    question: dict[str, Any],
    options: dict[str, str],
    media: list[dict[str, Any]] | None = None,
    question_first: bool = False,
) -> list[dict[str, Any]]:
    ins = question["instructions"]
    payload = {
        "criterion": f"{ins['task']}\n{ins['question']}".strip(),
        "options": [
            {"letter": letter, "description": desc}
            for letter, desc in zip(_labels(len(options)), options.values(), strict=True)
        ],
    }
    # SemIf's order puts the evidence first; last, it leaves the question as a shared prefix.
    payload = {**payload, "evidence": state} if question_first else {"evidence": state, **payload}
    text = json.dumps(payload, ensure_ascii=False)
    # Images lead, so the options stay next to the letter the model is about to emit.
    content = [*media, {"type": "text", "text": text}] if media else text
    system = SYSTEM if len(options) <= len(LETTERS) else SYSTEM_PAIRS
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def _top(response: Any) -> list[dict[str, Any]]:
    """The first token's ``top_logprobs``."""
    logprobs = response.choices[0].logprobs
    if isinstance(logprobs, pydantic.BaseModel):
        # model_dump builds a serializer pydantic deferred (a cache hit in a fresh process);
        # to_jsonable_python would pass its placeholder to pydantic-core and fail.
        logprobs = logprobs.model_dump()
    logprobs = pydantic_core.to_jsonable_python(logprobs)
    return logprobs["content"][0]["top_logprobs"] if logprobs else []


def _mass(top: list[dict[str, Any]], labels: list[str]) -> dict[str, float]:
    return {
        label: sum(math.exp(t["logprob"]) for t in top if t["token"].strip() == label)
        for label in labels
    }


def _prefixes(options: dict[str, str], response: Any) -> list[str]:
    """First letters of two-letter labels that carry enough mass to need a follow-up read."""
    labels = _labels(len(options))
    if len(labels[0]) == 1:
        return []
    top = _top(response)
    firsts = _mass(top, sorted({label[0] for label in labels}))
    total = sum(firsts.values()) + sum(_mass(top, labels).values())
    return [f for f, p in firsts.items() if p and p >= PREFIX_FLOOR * total]


def _answer(
    question: dict[str, Any],
    options: dict[str, str],
    response: Any,
    seconds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn the first token's ``top_logprobs`` into a Jev answer over ``options``.

    ``seconds`` maps a pre-filled first letter to the response that read the second.
    """
    labels = _labels(len(options))
    top = _top(response)
    by_label = _mass(top, labels)
    for first, second in (seconds or {}).items():
        p_first = _mass(top, [first])[first]
        p_second = _mass(_top(second), [label[1] for label in labels if label[0] == first])
        norm = sum(p_second.values())
        for letter, p in p_second.items():
            if norm:
                by_label[first + letter] += p_first * p / norm
    mass = dict(zip(options, by_label.values(), strict=True))
    total = sum(mass.values())
    if not total:
        raise ValueError(
            f"No option letter among the model's top logprobs {[t['token'] for t in top]}; "
            "raise top_logprobs, or check that the server returns logprobs."
        )
    probs = {key: p / total for key, p in mass.items()}
    if question["type"] == "noul":
        return {"type": "noul", "noul": probs["true"]}

    best = max(probs, key=probs.__getitem__)
    answer = {"type": question["type"], "confidence": probs[best], "probabilities": probs}
    if question["type"] == "score":
        answer["score"] = sum(int(key) * p for key, p in probs.items())
    else:
        answer["choice"] = best
    return answer


def _prefill(messages: list[dict[str, Any]], first: str) -> list[dict[str, Any]]:
    """Pre-fill ``first`` as the assistant turn so the next token is the label's second."""
    return [*messages, {"role": "assistant", "content": first}]


def _completion(model: str, answers: dict[str, Any]) -> litellm.ModelResponse:
    message = {"role": "assistant", "content": json.dumps(answers)}
    return litellm.ModelResponse(
        model=model, choices=[{"message": message, "finish_reason": "stop"}]
    )


class SemIfLM(dspy.LM):  # type: ignore[misc]
    """Answer JevAdapter decision requests with a local model's option-letter logprobs.

    A regular ``dspy.LM`` otherwise: ``api_base``, ``api_key``, caching (one cache entry
    per question), retries and ``dump_state`` behave as usual. Extra kwargs reach the
    server, e.g. ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}`` so a
    thinking model answers with a letter right away.

    Args:
        model: LiteLLM model id, e.g. ``"openai/<served-model-name>"`` for vLLM.
        top_logprobs: Candidate tokens requested per question; letters outside them get 0.
        question_first: Send the criterion and options before the evidence, not after.
            Requests that differ only in their evidence then share the question as a
            cached prefix: on llama.cpp (``-np 16 --kv-unified``), 16 parallel short
            requests ran about 3x faster. Off by default: it departs from SemIf's prompt,
            and on JevBench it cost Qwen3-0.6B 37 of 231 items, though it made no
            measurable difference on 2B and 4B models.
    """

    def __init__(
        self, model: str, top_logprobs: int = 20, question_first: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(model, **kwargs)
        self.top_logprobs = top_logprobs
        self.question_first = question_first

    def dump_state(self) -> dict[str, Any]:
        return {
            **super().dump_state(),
            "top_logprobs": self.top_logprobs,
            "question_first": self.question_first,
        }

    def _check_truncation(self, results: Any) -> None:
        pass  # Every request stops at one token by design.

    def _requests(
        self, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> tuple[
        list[tuple[str, dict[str, Any], dict[str, str], list[dict[str, Any]]]], dict[str, Any]
    ]:
        """Split a JevAdapter payload into one ``(qid, question, options, messages)`` each."""
        text, media = _split_media(messages[-1]["content"])
        request = json.loads(text)
        requests = []
        for qid, q in request["questions"].items():
            opts = _options(q)
            requests.append(
                (qid, q, opts, _messages(request["state"], q, opts, media, self.question_first))
            )
        kwargs = {
            **kwargs,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }
        return requests, kwargs

    def forward(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> litellm.ModelResponse:
        requests, kw = self._requests(messages or [], kwargs)
        answers = {}
        for qid, q, opts, msgs in requests:
            first = super().forward(messages=msgs, **kw)
            seconds = {
                f: super(SemIfLM, self).forward(messages=_prefill(msgs, f), **kw)
                for f in _prefixes(opts, first)
            }
            answers[qid] = _answer(q, opts, first, seconds)
        return _completion(self.model, answers)

    async def aforward(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> litellm.ModelResponse:
        requests, kw = self._requests(messages or [], kwargs)
        base = super().aforward

        async def answer(q: dict[str, Any], opts: dict[str, str], msgs: list[Any]) -> Any:
            first = await base(messages=msgs, **kw)
            prefixes = _prefixes(opts, first)
            reads = await asyncio.gather(
                *(base(messages=_prefill(msgs, f), **kw) for f in prefixes)
            )
            return _answer(q, opts, first, dict(zip(prefixes, reads, strict=True)))

        results = await asyncio.gather(*(answer(q, opts, msgs) for _, q, opts, msgs in requests))
        answers = {qid: a for (qid, *_), a in zip(requests, results, strict=True)}
        return _completion(self.model, answers)
