"""DSPy adapter for TypeSafe's Jev, a System One model that answers typed questions.

Jev takes ``{state, questions}`` and returns calibrated probabilities instead of text, so
there is no prompt to render and nothing to repair. The adapter compiles a signature into
that request and decodes the answers back into typed output fields:

- ``bool`` -> ``noul`` (``True`` when the probability reaches the threshold)
- ``Literal`` / ``Enum`` -> ``choice``, or ``score`` when marked as ordinal
- nested ``pydantic.BaseModel`` -> one question per leaf, keyed by its dotted path

Per-field options go in ``json_schema_extra={"jev": {...}}`` through ``Annotated[T,
pydantic.Field(...)]`` (``dspy.OutputField`` drops unknown kwargs): ``type="score"``,
``criteria`` (replaces the default option descriptions) and ``threshold`` (noul only).
The raw answers, with their probabilities and confidences, land on ``prediction.jev``.

Usage::

    dspy.configure(lm=JevLM(), adapter=JevAdapter())
    dspy.Predict(Triage)(message="Payouts failing for 3 days")
"""

from __future__ import annotations

import asyncio
import enum
import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any, Literal, get_args, get_origin

import dspy
import pydantic
import pydantic_core
from dspy.clients.cache import request_cache

_Decode = Callable[[dict[str, Any]], Any]


def _leaves(path: str, annotation: Any, extra: Any, desc: str | None) -> Any:
    """Yield ``(path, annotation, extra, desc)`` for every leaf of a (possibly nested) field."""
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        for name, field in annotation.model_fields.items():
            yield from _leaves(
                f"{path}.{name}", field.annotation, field.json_schema_extra, field.description
            )
    else:
        yield path, annotation, extra, desc


def _options(annotation: Any) -> dict[str, Any]:
    """Map each option's wire key to the Python value it decodes to."""
    if get_origin(annotation) is Literal:
        return {str(v): v for v in get_args(annotation)}
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {str(m.value): m for m in annotation}
    return {}


def _question(
    path: str, annotation: Any, extra: Any, desc: str | None, task: str, threshold: float
) -> tuple[dict[str, Any], _Decode]:
    cfg = (extra or {}).get("jev", {})
    q: dict[str, Any] = {"instructions": {"task": task, "question": desc or path}}

    if annotation is bool:
        cut = cfg.get("threshold", threshold)
        q["type"] = "noul"
        if "criteria" in cfg:
            q["criteria"] = cfg["criteria"]
        return q, lambda a: a["noul"] >= cut

    options = _options(annotation)
    if not options:
        raise TypeError(
            f"JevAdapter cannot answer output field {path!r} of type {annotation!r}: "
            "Jev only returns bool (noul) and Literal/Enum (choice, score) values."
        )

    if cfg.get("type") == "score":
        levels = list(options.values())
        q["type"] = "score"
        q["criteria"] = cfg.get("criteria", list(options))

        def decode_score(a: dict[str, Any]) -> Any:
            # Levels are keyed by int score (0-based live, undocumented): rank the keys.
            keys = sorted(a["probabilities"], key=int)
            return levels[keys.index(max(keys, key=a["probabilities"].__getitem__))]

        return q, decode_score

    q["type"] = "choice"
    q["criteria"] = cfg.get("criteria", {k: k for k in options})
    return q, lambda a: options[a["choice"]]


def _compile(
    signature: type[dspy.Signature], threshold: float
) -> dict[str, tuple[dict[str, Any], _Decode]]:
    """Compile a signature's output fields into ``{question_id: (question, decode)}``."""
    compiled = {}
    for name, field in signature.output_fields.items():
        desc = (field.json_schema_extra or {}).get("desc")
        desc = None if desc == f"${{{name}}}" else desc  # DSPy's placeholder for "no desc"
        for path, annotation, extra, leaf_desc in _leaves(
            name, field.annotation, field.json_schema_extra, desc
        ):
            compiled[path] = _question(
                path, annotation, extra, leaf_desc, signature.instructions, threshold
            )
    return compiled


class JevAdapter(dspy.Adapter):  # type: ignore[misc]
    """Compile a DSPy signature into a Jev decision request and decode its typed answers.

    Demos are ignored (Jev takes no few-shot examples). Instruction optimizers still apply:
    the signature docstring and field descriptions become each question's instructions.

    Args:
        threshold: Default probability at which a ``bool`` (noul) field becomes ``True``.
    """

    def __init__(self, threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold

    def format(
        self, signature: type[dspy.Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        state = {name: inputs[name] for name in signature.input_fields if name in inputs}
        questions = {qid: q for qid, (q, _) in _compile(signature, self.threshold).items()}
        payload = {"state": pydantic_core.to_jsonable_python(state), "questions": questions}
        return [{"role": "user", "content": json.dumps(payload)}]

    def parse(self, signature: type[dspy.Signature], completion: str) -> dict[str, Any]:
        answers = json.loads(completion)
        raw: dict[str, Any] = {}
        for qid, (_, decode) in _compile(signature, self.threshold).items():
            *parents, leaf = qid.split(".")
            node = raw
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = decode(answers[qid])

        return {
            name: pydantic.TypeAdapter(field.annotation).validate_python(raw[name])
            for name, field in signature.output_fields.items()
        }

    def _call_postprocess(
        self, processed_signature: Any, original_signature: Any, outputs: Any, *args: Any
    ) -> Any:
        # Base postprocess keeps only signature fields; attach answers after, like logprobs.
        values = super()._call_postprocess(processed_signature, original_signature, outputs, *args)
        for value, output in zip(values, outputs, strict=True):
            value["jev"] = json.loads(output["text"] if isinstance(output, dict) else output)
        return values


@request_cache(ignored_args_for_cache_key=["api_key"])  # type: ignore[misc]
def _decide(url: str, payload: dict[str, Any], api_key: str) -> dspy.LMResponse:
    """POST one decision request. Cached on (url, payload); a hit comes back with ``cache_hit``."""
    http_request = urllib.request.Request(  # noqa: S310 - caller-configured endpoint
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(http_request) as response:  # noqa: S310
        body = json.load(response)
    return dspy.LMResponse.from_text(
        json.dumps(body["answers"]), model=body.get("model"), usage=body.get("usage")
    )


class JevLM(dspy.BaseLM):  # type: ignore[misc]
    """Send a JevAdapter payload to a Jev decisions endpoint (OpenRouter by default).

    Responses go through DSPy's request cache (``cache=False`` per call or at construction
    to bypass). ``dump_state()`` keeps ``url`` but never the API key, which a loaded program
    re-reads from ``$OPENROUTER_API_KEY``.

    Args:
        model: Jev model id.
        api_key: Bearer token; defaults to ``$OPENROUTER_API_KEY``.
        url: Decisions endpoint. TypeSafe's native one is ``https://api.typesafe.ai/v1/systemone``.
    """

    forward_contract = "typed_lm"

    def __init__(
        self,
        model: str = "typesafe/jev-1.13",
        api_key: str | None = None,
        url: str = "https://openrouter.ai/api/alpha/decisions",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.url = url

    def dump_state(self) -> dict[str, Any]:
        return {**super().dump_state(), "url": self.url}

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        payload = {"model": request.model, **json.loads(request.messages[-1].parts[0].text)}
        per_call = request.config.cache.enabled if request.config.cache else None
        use_cache = self.cache if per_call is None else per_call
        decide = _decide if use_cache else _decide.__wrapped__
        return decide(url=self.url, payload=payload, api_key=self.api_key)

    async def aforward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        # ponytail: blocking HTTP on a worker thread; use an async client if fan-out gets wide.
        return await asyncio.to_thread(self.forward, request)
