"""Streaming / StreamListener support for StructuredOutputAdapter (offline only)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import mock

import pytest

pytest.importorskip("dspy", minversion="3.3.1")

import dspy  # noqa: E402
from dspy.utils.dummies import DummyLM  # noqa: E402
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices  # noqa: E402

from llm_schema_lite import StreamingNotSupportedError  # noqa: E402
from llm_schema_lite.dspy_integration import (  # noqa: E402
    OutputMode,
    StructuredOutputAdapter,
    register_streaming_support,
)
from tests.dspy_helpers import QA, make_adapter  # noqa: E402

JSON_TOKENS = ['{"', "answer", '":', ' "', "Paris", " is", " the", " capital", '"', "}"]
YAML_TOKENS = ["answer", ":", " Paris", " is", " the", " capital", "\n"]
EXPECTED_ANSWER = "Paris is the capital"


class _SubAdapter(StructuredOutputAdapter):
    """Two-line subclass proving registration is not keyed on one literal class name."""


class _AsyncQA(dspy.Module):  # type: ignore[misc]
    """Async program, so streamify enters the adapter through acall rather than __call__."""

    def __init__(self) -> None:
        """Build the single inner Predict this program delegates to."""
        super().__init__()
        self.predict = dspy.Predict("question->answer")

    async def aforward(self, **kwargs: Any) -> Any:
        """Await the inner Predict so streamify treats this as an async program."""
        return await self.predict.acall(**kwargs)


def _chunk(content: str) -> ModelResponseStream:
    """Build one litellm streaming chunk carrying `content` as delta text."""
    return ModelResponseStream(
        model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=content))]
    )


def _fake_acompletion(tokens: list[str], calls: list[Any]) -> Any:
    """Return an async-generator function standing in for litellm.acompletion(stream=True)."""

    async def _fake(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("model"))
        for token in tokens:
            yield _chunk(token)

    return _fake


@pytest.fixture(scope="module", autouse=True)
def _configure_lm() -> Any:
    """Configure a real, uncached dspy.LM once for the module (dspy.configure is task-bound)."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))
    yield


def collect_stream(
    adapter: Any,
    tokens: list[str] = JSON_TOKENS,
    listeners: bool = True,
    signature: str = "question->answer",
    program: Any = None,
    is_async_program: bool = False,
) -> tuple[list[str], Any, list[Any]]:
    """Run one offline dspy.streamify pass; return (chunks, prediction, recorded lm calls)."""
    calls: list[Any] = []

    async def _run() -> tuple[list[str], Any]:
        listener_list = (
            [dspy.streaming.StreamListener(signature_field_name="answer")] if listeners else []
        )
        inner = dspy.Predict(signature) if program is None else program
        streamed = dspy.streamify(
            inner, stream_listeners=listener_list, is_async_program=is_async_program
        )
        chunks: list[str] = []
        prediction = None
        with dspy.context(adapter=adapter):
            with mock.patch("litellm.acompletion", side_effect=_fake_acompletion(tokens, calls)):
                async for value in streamed(question="capital of France?"):
                    if isinstance(value, dspy.streaming.StreamResponse):
                        chunks.append(value.chunk)
                    elif isinstance(value, dspy.Prediction):
                        prediction = value
        return chunks, prediction

    chunks, prediction = asyncio.run(_run())
    return chunks, prediction, calls


def unwrap(excinfo: Any) -> BaseException:
    """Unwrap an ExceptionGroup down to the real exception streamify wraps it in.

    Duck-typed on `.exceptions` rather than `isinstance(exc, BaseExceptionGroup)`:
    that builtin only exists on Python 3.11+, and this package supports 3.10
    (pyproject requires-python = ">=3.10"; CI runs a 3.10 cell).
    """
    exc = excinfo.value
    while True:
        sub_exceptions = getattr(exc, "exceptions", None)
        if not sub_exceptions:
            return exc
        exc = sub_exceptions[0]


class TestStreamingRegistration:
    """StreamListener.adapter_identifiers gains StructuredOutputAdapter entries."""

    def test_stream_listener_accepts_adapter(self) -> None:
        """A bare StreamListener("answer") already recognises StructuredOutputAdapter."""
        listener = dspy.streaming.StreamListener("answer")
        assert "StructuredOutputAdapter" in listener.adapter_identifiers, (
            f"Known adapters: {sorted(listener.adapter_identifiers)}"
        )

    def test_registered_entry_matches_json_adapter(self) -> None:
        """The registered entry equals (==, not is) the live JSONAdapter entry."""
        listener = dspy.streaming.StreamListener("answer")
        entry = listener.adapter_identifiers["StructuredOutputAdapter"]
        assert entry["start_identifier"] == '"answer":'
        assert entry == listener.adapter_identifiers["JSONAdapter"]

    def test_subclass_is_registered(self) -> None:
        """_SubAdapter.__name__ is registered too, via the recursive subclass walk."""
        listener = dspy.streaming.StreamListener("answer")
        assert _SubAdapter.__name__ in listener.adapter_identifiers, (
            f"Known adapters: {sorted(listener.adapter_identifiers)}"
        )

    def test_upstream_entries_are_untouched(self) -> None:
        """ChatAdapter, JSONAdapter and XMLAdapter all remain present."""
        listener = dspy.streaming.StreamListener("answer")
        for name in ("ChatAdapter", "JSONAdapter", "XMLAdapter"):
            assert name in listener.adapter_identifiers, f"upstream key {name} disappeared"

    def test_register_streaming_support_is_idempotent(self) -> None:
        """A second register_streaming_support() call returns False and adds nothing."""
        assert register_streaming_support() is False
        listener = dspy.streaming.StreamListener("answer")
        keys = [k for k in listener.adapter_identifiers if k == "StructuredOutputAdapter"]
        assert len(keys) == 1


class TestStreamingEndToEnd:
    """JSON/JSONISH per-field streaming, offline, via a patched litellm.acompletion."""

    @pytest.mark.parametrize("mode", [OutputMode.JSONISH, OutputMode.JSON])
    def test_json_modes_stream_field_chunks(self, mode: OutputMode) -> None:
        """AC #1 / AC #2: chunks reassemble to 'Paris is the capital' in both modes."""
        chunks, prediction, calls = collect_stream(make_adapter(mode))
        assert calls, f"{mode}: the fake LM was never called"
        assert chunks, f"{mode}: no StreamResponse chunks were produced"
        assert EXPECTED_ANSWER in "".join(chunks)
        assert prediction is not None and prediction.answer == EXPECTED_ANSWER

    def test_chunks_match_upstream_json_adapter(self) -> None:
        """Our chunk list is identical to dspy.JSONAdapter()'s over the same tokens."""
        ours, _, _ = collect_stream(make_adapter(OutputMode.JSONISH))
        theirs, _, _ = collect_stream(dspy.JSONAdapter())
        assert ours == theirs, f"ours={ours!r} theirs={theirs!r}"

    def test_subclass_streams_end_to_end(self) -> None:
        """_SubAdapter streams correctly too, not just the literal StructuredOutputAdapter."""
        chunks, prediction, calls = collect_stream(_SubAdapter(output_mode=OutputMode.JSONISH))
        assert calls
        assert EXPECTED_ANSWER in "".join(chunks)
        assert prediction is not None and prediction.answer == EXPECTED_ANSWER


class TestYamlStreamingGuard:
    """StreamingNotSupportedError fires before any LM call, in YAML + listener only."""

    def test_sync_streaming_raises_streaming_not_supported(self) -> None:
        """AC #3: message contains 'YAML' and 'streaming'; unwrapped type is our error."""
        with pytest.raises(BaseException) as excinfo:  # noqa: B017
            collect_stream(make_adapter(OutputMode.YAML), tokens=YAML_TOKENS)
        exc = unwrap(excinfo)
        assert isinstance(exc, StreamingNotSupportedError)
        assert isinstance(exc, ValueError)
        assert "YAML" in str(exc)
        assert "streaming" in str(exc)

    def test_guard_fires_before_any_lm_call(self) -> None:
        """The only real proof of 'before any LM call': litellm.acompletion is never called."""
        calls: list[Any] = []

        async def _run() -> None:
            listener = dspy.streaming.StreamListener(signature_field_name="answer")
            streamed = dspy.streamify(dspy.Predict("question->answer"), stream_listeners=[listener])
            with dspy.context(adapter=make_adapter(OutputMode.YAML)):
                with mock.patch(
                    "litellm.acompletion", side_effect=_fake_acompletion(YAML_TOKENS, calls)
                ):
                    async for _ in streamed(question="capital of France?"):
                        pass

        with pytest.raises(BaseException):  # noqa: B017
            asyncio.run(_run())
        assert calls == [], f"the LM was called before the guard fired: {calls!r}"

    def test_async_streaming_raises_streaming_not_supported(self) -> None:
        """Same guard, reached through acall via a real async dspy.Module (_AsyncQA)."""
        with pytest.raises(BaseException) as excinfo:  # noqa: B017
            collect_stream(
                make_adapter(OutputMode.YAML),
                tokens=YAML_TOKENS,
                program=_AsyncQA(),
                is_async_program=True,
            )
        exc = unwrap(excinfo)
        assert isinstance(exc, StreamingNotSupportedError)
        assert "YAML" in str(exc)

    def test_streamify_without_listeners_is_unaffected(self) -> None:
        """YAML + stream_listeners=[] still calls the LM and yields a Prediction."""
        chunks, prediction, calls = collect_stream(
            make_adapter(OutputMode.YAML), tokens=YAML_TOKENS, listeners=False
        )
        assert calls, "YAML streamify without listeners must still reach the LM"
        assert chunks == []
        assert prediction is not None

    def test_non_streaming_yaml_call_is_unaffected(self) -> None:
        """A plain (non-streamify) YAML call under DummyLM never raises the guard."""
        adapter = make_adapter(OutputMode.YAML)
        lm = DummyLM([{"answer": "4"}], adapter=adapter)
        result = adapter(lm, {}, QA, [], {"question": "3+3?"})
        assert result == [{"answer": "4"}]
