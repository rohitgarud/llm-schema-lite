"""Fake ``DummyLM`` subclasses for the offline dspy adapter benchmark.

Three ``DummyLM`` subclasses are defined here and used by the offline smoke test and the
synthetic #1871 reproduction: :class:`RawTextLM`, :class:`JsonObjectOnlyLM`, and
:class:`Issue1871LM`. They **imitate** the patterns in ``tests/dspy_helpers.py``
(``SchemaCapableDummyLM``, ``JsonObjectOnlyDummyLM``) and deliberately do **not** import
from it.

Δ2 construction rule (verbatim): every fake LM in a cell is constructed as
``FakeLM(answers, adapter=<the adapter under test>)``, because ``DummyLM`` renders its
canned answer *through the adapter passed to its constructor* — seeding it with the
default ``ChatAdapter()`` feeds YAML-mode adapters ChatAdapter-shaped text and manufactures
false ``parse_error``s.

``DummyLM`` reports every token count as ``0``, so nothing here (or downstream) may assert
on live token magnitudes.
"""

from __future__ import annotations

from typing import Any

from dspy.dsp.utils.utils import dotdict
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import LMInvalidRequestError

JSON_OBJECT_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}


class RawTextLM(DummyLM):  # type: ignore[misc]
    """DummyLM that returns literal raw strings instead of adapter-rendered answers.

    Needed for the anti-masking test: the first canned string must be unparseable garbage,
    which no adapter-rendered answer could ever be.
    """

    def __init__(self, texts: list[str], **kwargs: Any) -> None:
        """Store the raw texts as a consumable iterator; delegate the rest to DummyLM."""
        super().__init__([], **kwargs)
        self._texts = iter(texts)

    def forward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        """Return the next raw text verbatim in a DummyLM-shaped dotdict response.

        Mirrors dspy/utils/dummies.py — choices[0].message.content = the text,
        usage = zeros, model = "dummy" — bypassing _format_answer_fields entirely.
        """
        text = next(self._texts, "No more responses")
        message = dotdict(content=text, tool_calls=None)
        choices = [dotdict(message=message, finish_reason="stop")]
        return dotdict(
            choices=choices,
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="dummy",
        )


class JsonObjectOnlyLM(DummyLM):  # type: ignore[misc]
    """Advertises response_format but cannot do structured-output schemas — LM Studio's shape.

    Imitates tests/dspy_helpers.py's JsonObjectOnlyDummyLM; not imported from it.
    """

    @property
    def supported_params(self) -> set[str]:
        """Return {"response_format"} instead of DummyLM's empty set."""
        return {"response_format"}

    @property
    def supports_response_schema(self) -> bool:
        """Return False — the capability gap that forces the json_object branch."""
        return False


class Issue1871LM(JsonObjectOnlyLM):
    """Rejects response_format={"type": "json_object"} with HTTP 400, like LM Studio.

    Raises exactly the class dspy's _wrap_litellm_exception produces for a litellm
    BadRequestError. Being an LMError, ChatAdapter and JSONAdapter both re-raise without a
    second call — so the failure is observable, not masked, which is what makes it a valid
    reproduction.
    """

    def forward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        """Raise LMInvalidRequestError iff response_format is json_object.

        Otherwise delegate to DummyLM.forward via the MRO.
        """
        if kwargs.get("response_format") == JSON_OBJECT_RESPONSE_FORMAT:
            raise LMInvalidRequestError(
                "'response_format.type' must be 'json_schema'",
                model="dummy",
                provider="lm_studio",
                status=400,
            )
        return super().forward(prompt=prompt, messages=messages, **kwargs)
