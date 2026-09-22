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
SYSTEM = (  # SemIf's DIRECT_SYSTEM prompt, verbatim
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def _options(question: dict[str, Any]) -> dict[str, str]:
    """Map each Jev option key to the description the model reads."""
    if question["type"] == "noul":
        return question.get("criteria") or {"true": "Yes", "false": "No"}
    criteria = question["criteria"]
    if question["type"] == "score":
        return {str(i): desc for i, desc in enumerate(criteria)}
    return dict(criteria)


def _messages(
    state: Any, question: dict[str, Any], options: dict[str, str]
) -> list[dict[str, Any]]:
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"SemIfLM needs 2-{len(LETTERS)} options, got {len(options)}")
    ins = question["instructions"]
    payload = {
        "evidence": state,
        "criterion": f"{ins['task']}\n{ins['question']}".strip(),
        "options": [
            {"letter": letter, "description": desc}
            for letter, desc in zip(LETTERS, options.values(), strict=False)
        ],
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _answer(question: dict[str, Any], options: dict[str, str], response: Any) -> dict[str, Any]:
    """Turn the first token's ``top_logprobs`` into a Jev answer over ``options``."""
    logprobs = response.choices[0].logprobs
    if isinstance(logprobs, pydantic.BaseModel):
        # model_dump builds a serializer pydantic deferred (a cache hit in a fresh process);
        # to_jsonable_python would pass its placeholder to pydantic-core and fail.
        logprobs = logprobs.model_dump()
    logprobs = pydantic_core.to_jsonable_python(logprobs)
    top = logprobs["content"][0]["top_logprobs"] if logprobs else []
    mass = {
        key: sum(math.exp(t["logprob"]) for t in top if t["token"].strip() == letter)
        for key, letter in zip(options, LETTERS, strict=False)
    }
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
    """

    def __init__(self, model: str, top_logprobs: int = 20, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.top_logprobs = top_logprobs

    def dump_state(self) -> dict[str, Any]:
        return {**super().dump_state(), "top_logprobs": self.top_logprobs}

    def _check_truncation(self, results: Any) -> None:
        pass  # Every request stops at one token by design.

    def _requests(
        self, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> tuple[
        list[tuple[str, dict[str, Any], dict[str, str], list[dict[str, Any]]]], dict[str, Any]
    ]:
        """Split a JevAdapter payload into one ``(qid, question, options, messages)`` each."""
        request = json.loads(messages[-1]["content"])
        requests = []
        for qid, q in request["questions"].items():
            opts = _options(q)
            requests.append((qid, q, opts, _messages(request["state"], q, opts)))
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
        answers = {
            qid: _answer(q, opts, super(SemIfLM, self).forward(messages=msgs, **kw))
            for qid, q, opts, msgs in requests
        }
        return _completion(self.model, answers)

    async def aforward(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> litellm.ModelResponse:
        requests, kw = self._requests(messages or [], kwargs)
        responses = await asyncio.gather(
            *(super(SemIfLM, self).aforward(messages=msgs, **kw) for *_, msgs in requests)
        )
        answers = {
            qid: _answer(q, opts, response)
            for (qid, q, opts, _), response in zip(requests, responses, strict=True)
        }
        return _completion(self.model, answers)
