"""End-to-end tests for __call__/acall under DummyLM, dspy.Predict, and response_format."""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytest.importorskip("dspy", minversion="3.3.1")

import dspy  # noqa: E402
import pydantic  # noqa: E402
from dspy.utils.dummies import DummyLM  # noqa: E402
from dspy.utils.exceptions import LMError  # noqa: E402

from llm_schema_lite.dspy_integration import OutputMode  # noqa: E402
from llm_schema_lite.dspy_integration.adapters import structured_output_adapter  # noqa: E402
from llm_schema_lite.dspy_integration.adapters.structured_output_adapter import (  # noqa: E402
    _get_structured_outputs_response_format,
)
from tests.dspy_helpers import (  # noqa: E402
    QA,
    TOOL_ANSWER,
    TOOL_INPUTS,
    FunctionCallingDummyLM,
    JsonObjectOnlyDummyLM,
    QAOptional,
    SchemaCapableDummyLM,
    SchemaCapableRaisingLM,
    ToolCallSig,
    ToolInputSig,
    assert_message_roles,
    call_sync_and_async,
    make_adapter,
    recorded_calls,
    recorded_response_format,
    response_format_projection,
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


class _Tagged(pydantic.BaseModel):
    """Nested model carrying an `x-*` vendor key, plus a property whose name starts `x-`."""

    label: str = pydantic.Field(json_schema_extra={"x-comparison": "exact"})
    x_id: str = pydantic.Field(alias="x-id")


_SENTINEL_RESPONSE_FORMAT = {"type": "sentinel"}
_JSON_OBJECT = {"type": "json_object"}
_SCHEMA_ANSWER = ("DSPyProgramOutputs", ["answer"])
_SCHEMA_EMPTY = ("DSPyProgramOutputs", [])
_ABSENT = object()

_SIG_INPUTS = {
    QA: {"question": "3+3?"},
    _DictOut: {"q": "x"},
    ToolCallSig: TOOL_INPUTS,
    ToolInputSig: TOOL_INPUTS,
}

_SIG_ANSWER = {
    QA: {"answer": "4"},
    _DictOut: {"d": {"a": "b"}},
    ToolCallSig: TOOL_ANSWER,
    ToolInputSig: {"answer": "hi"},
}

# Design v2 §4.2-§4.4 truth table, rows 4-26. Rows 1-3 (gate 0 under a plain DummyLM) are
# covered by TestResponseFormat::test_dummy_lm_never_sets_response_format and by the
# plain-dummy parametrisation of test_json_mode_matches_upstream_json_adapter.
# Columns: row, mode, lm_class, signature, use_native_function_calling, flag, expected.
_RESPONSE_FORMAT_MATRIX = [
    (4, OutputMode.YAML, SchemaCapableDummyLM, QA, True, True, None),
    (5, OutputMode.YAML, SchemaCapableDummyLM, QA, True, False, None),
    (6, OutputMode.YAML, JsonObjectOnlyDummyLM, QA, True, True, None),
    (7, OutputMode.YAML, SchemaCapableDummyLM, _DictOut, True, True, None),
    (8, OutputMode.YAML, SchemaCapableDummyLM, ToolCallSig, True, True, None),
    (9, OutputMode.YAML, SchemaCapableDummyLM, ToolInputSig, True, True, None),
    (10, OutputMode.JSONISH, SchemaCapableDummyLM, QA, True, True, _JSON_OBJECT),
    (11, OutputMode.JSONISH, SchemaCapableDummyLM, QA, True, False, None),
    (12, OutputMode.JSONISH, JsonObjectOnlyDummyLM, QA, True, True, _JSON_OBJECT),
    (13, OutputMode.JSONISH, JsonObjectOnlyDummyLM, QA, True, False, None),
    (14, OutputMode.JSONISH, SchemaCapableDummyLM, _DictOut, True, True, _JSON_OBJECT),
    (15, OutputMode.JSONISH, SchemaCapableDummyLM, ToolCallSig, True, True, None),
    (16, OutputMode.JSONISH, SchemaCapableDummyLM, ToolCallSig, False, True, None),
    (17, OutputMode.JSONISH, SchemaCapableDummyLM, ToolInputSig, True, True, None),
    (18, OutputMode.JSONISH, SchemaCapableDummyLM, ToolCallSig, True, False, None),
    (19, OutputMode.JSON, SchemaCapableDummyLM, QA, True, True, _SCHEMA_ANSWER),
    (20, OutputMode.JSON, SchemaCapableDummyLM, QA, True, False, _SCHEMA_ANSWER),
    (21, OutputMode.JSON, JsonObjectOnlyDummyLM, QA, True, True, _JSON_OBJECT),
    (22, OutputMode.JSON, JsonObjectOnlyDummyLM, QA, True, False, _JSON_OBJECT),
    (23, OutputMode.JSON, SchemaCapableDummyLM, _DictOut, True, True, _JSON_OBJECT),
    (24, OutputMode.JSON, SchemaCapableDummyLM, ToolCallSig, True, True, _SCHEMA_EMPTY),
    (25, OutputMode.JSON, SchemaCapableDummyLM, ToolCallSig, False, True, _JSON_OBJECT),
    (26, OutputMode.JSON, SchemaCapableDummyLM, ToolInputSig, True, True, _SCHEMA_ANSWER),
]


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
        assert sync_result == async_result, (
            f"call/acall diverged for {mode}: {sync_result!r} vs {async_result!r}"
        )
        assert len(lm.lm_kwargs_history) == 2
        projections = [
            response_format_projection(entry.get("response_format"))
            for entry in lm.lm_kwargs_history
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

    def test_structured_outputs_helper_strips_vendor_extension_keys(self):
        """`x-*` keys never reach the provider, but a property *named* `x-...` survives.

        stanfordnlp/dspy#9686: strict-schema providers (Bedrock) 400 on them, the call
        falls back to json_object, and the predictions come back empty.
        """
        sig = dspy.Signature({"q": (str, dspy.InputField()), "out": (_Tagged, dspy.OutputField())})
        schema = _get_structured_outputs_response_format(sig).model_json_schema()
        assert "x-comparison" not in str(schema)
        assert "x-id" in schema["$defs"]["_Tagged"]["properties"]

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

    @pytest.mark.parametrize("call_kind", ["sync", "async"])
    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_lm_error_propagates(self, mode, call_kind):
        """An LMError propagates unchanged after exactly one LM call, in every mode.

        Design D8. At HEAD, JSON mode swallowed the LMError and retried in json_object
        mode, making a second, equally doomed call.
        """
        adapter = make_adapter(mode)
        lm = SchemaCapableRaisingLM([{"answer": "4"}], adapter=adapter)
        with pytest.raises(LMError):
            if call_kind == "sync":
                adapter(lm, {}, QA, [], {"question": "3+3?"})
            else:
                asyncio.run(adapter.acall(lm, {}, QA, [], {"question": "3+3?"}))
        assert len(lm.lm_kwargs_history) == 1, (
            f"{mode}/{call_kind}: expected exactly one LM call, got {len(lm.lm_kwargs_history)}"
        )

    def test_lm_error_propagates_without_response_format_support(self):
        """An LMError from a gate-0 LM propagates unchanged, with no try/except in play."""
        adapter = make_adapter(OutputMode.JSON)
        lm = _RaisingLM([{"answer": "4"}], adapter=adapter)
        with pytest.raises(LMError):
            adapter(lm, {}, QA, [], {"question": "3+3?"})

    @pytest.mark.parametrize(
        ("mode", "flag", "expected"),
        [
            (OutputMode.YAML, True, _SENTINEL_RESPONSE_FORMAT),
            (OutputMode.JSONISH, False, _SENTINEL_RESPONSE_FORMAT),
            (OutputMode.JSONISH, True, _JSON_OBJECT),
            (OutputMode.JSON, True, _SCHEMA_ANSWER),
        ],
        ids=[
            "yaml-survives",
            "jsonish-flag-off-survives",
            "jsonish-overwritten",
            "json-overwritten",
        ],
    )
    def test_caller_supplied_response_format_is_not_popped(self, mode, flag, expected):
        """Omitting response_format means never writing the key, never popping the caller's.

        Design D7. A NONE plan leaves a caller-supplied value untouched; the JSON_OBJECT
        and SCHEMA plans overwrite it, matching upstream.
        """
        adapter = make_adapter(mode, use_json_object_response_format=flag)
        lm = SchemaCapableDummyLM([{"answer": "4"}] * 4, adapter=adapter)
        adapter(lm, {"response_format": _SENTINEL_RESPONSE_FORMAT}, QA, [], {"question": "3+3?"})
        assert response_format_projection(recorded_response_format(lm)) == expected, (
            f"{mode}/flag={flag}: expected {expected!r}, got "
            f"{response_format_projection(recorded_response_format(lm))!r}"
        )


class TestResponseFormatMatrix:
    """The design v2 response_format truth table, and JSON-mode parity with upstream."""

    @pytest.mark.parametrize(
        ("row", "mode", "lm_class", "signature", "nfc", "flag", "expected"),
        _RESPONSE_FORMAT_MATRIX,
        ids=[f"row{entry[0]}" for entry in _RESPONSE_FORMAT_MATRIX],
    )
    def test_response_format_matrix(self, row, mode, lm_class, signature, nfc, flag, expected):
        """Each truth-table row reaches the LM with exactly its expected response_format.

        Both __call__ and acall run against the same LM, so acall parity (AC #5) is
        satisfied by construction: the two recorded projections must both equal the cell.
        """
        adapter = make_adapter(
            mode,
            use_native_function_calling=nfc,
            use_json_object_response_format=flag,
        )
        lm = lm_class([_SIG_ANSWER[signature]] * 8, adapter=adapter)
        call_sync_and_async(adapter, lm, signature, [], _SIG_INPUTS[signature])

        calls = recorded_calls(lm)
        assert len(calls) == 2, (
            f"row {row}: expected one sync + one async LM call, got {len(calls)}"
        )
        projections = [response_format_projection(entry.get("response_format")) for entry in calls]
        assert projections == [expected, expected], (
            f"row {row} ({mode}, {lm_class.__name__}, {signature.__name__}, "
            f"nfc={nfc}, flag={flag}): expected {expected!r} twice, got {projections!r}"
        )

    @pytest.mark.parametrize("nfc", [True, False], ids=["nfc-on", "nfc-off"])
    @pytest.mark.parametrize(
        "lm_class",
        [SchemaCapableDummyLM, JsonObjectOnlyDummyLM, DummyLM],
        ids=["schema-capable", "json-object-only", "plain-dummy"],
    )
    @pytest.mark.parametrize(
        "signature",
        [QA, _DictOut, ToolCallSig, ToolInputSig],
        ids=["qa", "dict-out", "tool-call-sig", "tool-input-sig"],
    )
    def test_json_mode_matches_upstream_json_adapter(self, signature, lm_class, nfc):
        """JSON mode records the same response_format and call count as upstream.

        A differential test, not a restatement: if a future DSPy release changes
        JSONAdapter's gate order or conditions, this fails instead of diverging silently.
        """
        ours = make_adapter(OutputMode.JSON, use_native_function_calling=nfc)
        upstream = dspy.JSONAdapter(use_native_function_calling=nfc)
        answer = _SIG_ANSWER[signature]
        our_lm = lm_class([answer] * 8, adapter=ours)
        upstream_lm = lm_class([answer] * 8, adapter=upstream)

        ours(our_lm, {}, signature, [], _SIG_INPUTS[signature])
        upstream(upstream_lm, {}, signature, [], _SIG_INPUTS[signature])

        our_calls = recorded_calls(our_lm)
        upstream_calls = recorded_calls(upstream_lm)
        assert len(our_calls) == len(upstream_calls), (
            f"{signature.__name__}/{lm_class.__name__}/nfc={nfc}: we made "
            f"{len(our_calls)} LM calls, upstream made {len(upstream_calls)}"
        )
        ours_projection = response_format_projection(our_calls[0].get("response_format"))
        upstream_projection = response_format_projection(upstream_calls[0].get("response_format"))
        assert ours_projection == upstream_projection, (
            f"{signature.__name__}/{lm_class.__name__}/nfc={nfc}: JSON mode diverged "
            f"from upstream JSONAdapter: ours={ours_projection!r} "
            f"upstream={upstream_projection!r}"
        )


class TestParallelToolCalls:
    """parallel_tool_calls pass-through (design D4, truth-table rows 27-30)."""

    @pytest.mark.parametrize(
        ("row", "parallel_tool_calls", "lm_class", "expected"),
        [
            (27, None, FunctionCallingDummyLM, _ABSENT),
            (28, True, FunctionCallingDummyLM, True),
            (29, False, FunctionCallingDummyLM, False),
            (30, True, DummyLM, _ABSENT),
        ],
        ids=["row27-none", "row28-true", "row29-false", "row30-not-supported"],
    )
    def test_parallel_tool_calls_reaches_lm_kwargs(
        self, row, parallel_tool_calls, lm_class, expected
    ):
        """The ctor option reaches lm_kwargs iff set and the LM supports function calling.

        Row 30 pins the vacuous case: a plain DummyLM reports
        supports_function_calling is False, so neither tools nor parallel_tool_calls are
        injected no matter what the option says.
        """
        adapter = make_adapter(OutputMode.JSONISH, parallel_tool_calls=parallel_tool_calls)
        lm = lm_class([TOOL_ANSWER] * 4, adapter=adapter)
        adapter(lm, {}, ToolCallSig, [], TOOL_INPUTS)
        kwargs = recorded_calls(lm)[0]

        if expected is _ABSENT:
            assert "parallel_tool_calls" not in kwargs, (
                f"row {row}: parallel_tool_calls unexpectedly set to "
                f"{kwargs.get('parallel_tool_calls')!r}"
            )
        else:
            assert kwargs["parallel_tool_calls"] is expected, (
                f"row {row}: expected parallel_tool_calls={expected!r}, got "
                f"{kwargs.get('parallel_tool_calls')!r}"
            )

        if lm_class is DummyLM:
            assert "tools" not in kwargs, f"row {row}: tools injected for a non-FC LM"
        else:
            assert len(kwargs["tools"]) == 1, (
                f"row {row}: expected exactly one injected tool, got {kwargs.get('tools')!r}"
            )


class TestNoPrivateDSPySymbol:
    """AC #1, executable: no underscore-prefixed DSPy symbol survives in the adapter."""

    def test_no_private_dspy_symbol_is_referenced(self):
        """The adapter module no longer references JSONAdapter._json_adapter_call_common."""
        source = inspect.getsource(structured_output_adapter)
        assert "_json_adapter_call_common" not in source, (
            "structured_output_adapter.py still references DSPy's private _json_adapter_call_common"
        )
