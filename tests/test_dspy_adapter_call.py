"""End-to-end tests for __call__/acall under DummyLM, dspy.Predict, and response_format."""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", minversion="3.3.1")

import dspy  # noqa: E402
import pydantic  # noqa: E402
from dspy.utils.dummies import DummyLM  # noqa: E402
from dspy.utils.exceptions import LMError  # noqa: E402

from llm_schema_lite.dspy_integration import OutputMode  # noqa: E402
from llm_schema_lite.dspy_integration.adapters.structured_output_adapter import (  # noqa: E402
    _get_structured_outputs_response_format,
)
from tests.dspy_helpers import (  # noqa: E402
    QA,
    QAOptional,
    SchemaCapableDummyLM,
    assert_message_roles,
    call_sync_and_async,
    make_adapter,
)


class _RaisingLM(DummyLM):  # type: ignore[misc]
    """DummyLM that raises LMError instead of producing an answer."""

    def forward(self, *args, **kwargs):
        """Raise LMError to exercise the missing re-raise clause."""
        raise LMError("boom")


class _DictOut(dspy.Signature):
    """Open-ended mapping output."""

    q: str = dspy.InputField()
    d: dict[str, str] = dspy.OutputField()


class TestPredictEndToEnd:
    """dspy.Predict end-to-end under DummyLM, per mode (D7, D8)."""

    def test_predict_jsonish(self):
        """JSONish mode round-trips through Predict and parses the stub answer."""
        adapter = make_adapter(OutputMode.JSONISH)
        lm = DummyLM([{"answer": "blue", "note": "n"}], adapter=adapter)
        with dspy.context(lm=lm, adapter=adapter):
            prediction = dspy.Predict(QAOptional)(question="colour?")
        assert prediction.answer == "blue"
        assert prediction.note == "n"

    def test_predict_json(self):
        """JSON mode round-trips through Predict and parses the stub answer."""
        adapter = make_adapter(OutputMode.JSON)
        lm = DummyLM([{"answer": "blue", "note": "n"}], adapter=adapter)
        with dspy.context(lm=lm, adapter=adapter):
            prediction = dspy.Predict(QAOptional)(question="colour?")
        assert prediction.answer == "blue"
        assert prediction.note == "n"

    def test_predict_yaml(self):
        """YAML mode round-trips through Predict and parses the stub answer."""
        adapter = make_adapter(OutputMode.YAML)
        lm = DummyLM([{"answer": "blue", "note": "n"}], adapter=adapter)
        with dspy.context(lm=lm, adapter=adapter):
            prediction = dspy.Predict(QAOptional)(question="colour?")
        assert prediction.answer == "blue"
        assert prediction.note == "n"

    def test_call_returns_list_of_dicts(self):
        """__call__ returns a list of dicts keyed by the signature's output fields."""
        adapter = make_adapter(OutputMode.JSONISH)
        lm = DummyLM([{"answer": "4"}], adapter=adapter)
        result = adapter(lm, {}, QA, [{"question": "2+2?", "answer": "4"}], {"question": "3+3?"})
        assert isinstance(result, list) and len(result) == 1, f"Unexpected result: {result!r}"
        assert isinstance(result[0], dict)
        assert set(result[0]) == set(QA.output_fields)

    def test_format_message_roles(self):
        """format() emits system, user, assistant, user for one demo."""
        adapter = make_adapter(OutputMode.JSONISH)
        messages = adapter.format(QA, [{"question": "2+2?", "answer": "4"}], {"question": "3+3?"})
        assert_message_roles(messages, ["system", "user", "assistant", "user"])
        assert messages[-1]["content"].endswith(
            "Respond with a JSON object in the following order of fields: `answer`."
        )


class TestAcallParity:
    """acall matches call for the same LM/signature/demos/inputs (Q4)."""

    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_acall_matches_call(self, mode):
        """acall returns the same result and records the same lm_kwargs as __call__."""
        adapter = make_adapter(mode)
        lm = SchemaCapableDummyLM([{"answer": "4"}, {"answer": "4"}], adapter=adapter)
        sync_result, async_result = call_sync_and_async(adapter, lm, QA, [], {"question": "3+3?"})
        assert (
            sync_result == async_result
        ), f"call/acall diverged for {mode}: {sync_result!r} vs {async_result!r}"
        assert len(lm.lm_kwargs_history) == 2
        # `response_format` is a class built fresh by `pydantic.create_model` on every
        # call, so `==` on the classes is identity comparison and can never hold. Compare
        # a stable projection (model name + field names) instead, which is what "the same
        # lm_kwargs" actually means here. See PLAN PATCH P-2.
        formats = [entry.get("response_format") for entry in lm.lm_kwargs_history]
        projections = [
            None if fmt is None else (fmt.__name__, sorted(fmt.model_fields)) for fmt in formats
        ]
        assert projections[0] == projections[1], (
            f"call/acall recorded different response_format shapes for {mode}: "
            f"{projections[0]!r} vs {projections[1]!r}"
        )


class TestResponseFormat:
    """response_format handling per mode and structured-outputs helper."""

    def test_dummy_lm_never_sets_response_format(self):
        """Plain DummyLM never reaches the response_format branches, in any mode."""
        for mode in OutputMode:
            adapter = make_adapter(mode)
            lm = DummyLM([{"answer": "4"}], adapter=adapter)
            adapter(lm, {}, QA, [], {"question": "3+3?"})
            assert lm.history[-1]["kwargs"].get("response_format") is None, (
                f"response_format unexpectedly set for {mode}: "
                f"{lm.history[-1]['kwargs'].get('response_format')!r}"
            )

    def test_structured_outputs_helper_rejects_dict_output_field(self):
        """The structured-outputs helper rejects open-ended mapping output fields."""
        with pytest.raises(ValueError) as excinfo:
            _get_structured_outputs_response_format(_DictOut)
        assert "open-ended mapping type" in str(excinfo.value)

    @pytest.mark.xfail(
        reason="lsl-2026-09-04-009: JSONAdapter.__call__ re-entry overwrites "
        "response_format; all three modes collapse to DSPyProgramOutputs"
    )
    def test_response_format_differs_per_output_mode(self):
        """Each output mode should reach the LM with its own response_format."""
        recorded = {}
        for mode in OutputMode:
            adapter = make_adapter(mode)
            lm = SchemaCapableDummyLM([{"answer": "4"}], adapter=adapter)
            adapter(lm, {}, QA, [], {"question": "3+3?"})
            recorded[mode] = lm.lm_kwargs_history[-1].get("response_format")

        assert isinstance(recorded[OutputMode.JSON], type) and issubclass(
            recorded[OutputMode.JSON], pydantic.BaseModel
        )
        assert recorded[OutputMode.JSONISH] == {"type": "json_object"}
        assert recorded[OutputMode.YAML] is None

    @pytest.mark.xfail(
        reason="lsl-2026-09-04-009: no LMError re-raise clause, so LMError is swallowed "
        "and retried in JSON mode"
    )
    def test_lm_error_propagates(self):
        """An LMError from the LM should propagate unchanged out of __call__."""
        adapter = make_adapter(OutputMode.JSON)
        lm = _RaisingLM([{"answer": "4"}], adapter=adapter)
        with pytest.raises(LMError):
            adapter(lm, {}, QA, [], {"question": "3+3?"})


class TestStreaming:
    """StreamListener adapter-identifier coverage (offline only)."""

    @pytest.mark.xfail(
        reason="lsl-2026-09-04-016: StreamListener.adapter_identifiers has no "
        "StructuredOutputAdapter entry"
    )
    def test_stream_listener_accepts_adapter(self):
        """StreamListener should recognise StructuredOutputAdapter."""
        listener = dspy.streaming.StreamListener("answer")
        assert (
            "StructuredOutputAdapter" in listener.adapter_identifiers
        ), f"Known adapters: {sorted(listener.adapter_identifiers)}"
