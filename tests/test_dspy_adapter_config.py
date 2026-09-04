"""Tests for StructuredOutputAdapter construction and OutputMode."""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", minversion="3.3.1")

from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter  # noqa: E402
from tests.dspy_helpers import Extract  # noqa: E402


class TestAdapterConfig:
    """Constructor option set and inherited defaults."""

    def test_default_option_set(self):
        """A default adapter exposes the four documented constructor options."""
        adapter = StructuredOutputAdapter()
        assert adapter.output_mode is OutputMode.JSONISH
        assert adapter.include_input_schemas is True
        assert adapter.use_native_function_calling is True
        assert adapter.callbacks == []

    def test_explicit_output_mode_is_stored(self):
        """Each OutputMode member round-trips through __init__."""
        for mode in OutputMode:
            assert StructuredOutputAdapter(output_mode=mode).output_mode is mode

    def test_inherited_defaults_not_exposed(self):
        """Inherited defaults the constructor does not expose keep their upstream values."""
        adapter = StructuredOutputAdapter()
        assert adapter.parallel_tool_calls is None
        # use_json_adapter_fallback is dead for this class: ChatAdapter.__call__
        # short-circuits on isinstance(self, JSONAdapter).
        assert adapter.use_json_adapter_fallback is True
        assert [t.__name__ for t in adapter.native_response_types] == [
            "Citations",
            "Reasoning",
        ]


class TestOutputMode:
    """OutputMode enum membership and values."""

    def test_members_and_values(self):
        """OutputMode declares exactly json, jsonish and yaml, in that order."""
        assert [m.value for m in OutputMode] == ["json", "jsonish", "yaml"]


class TestIncludeInputSchemas:
    """include_input_schemas is currently inert (XF-01)."""

    @pytest.mark.xfail(
        reason="lsl-2026-09-04-007: include_input_schemas is inert in every mode (research Q2)"
    )
    def test_include_input_schemas_false_omits_input_field_schema(self):
        """include_input_schemas=False should drop the input field's schema note."""
        on = StructuredOutputAdapter(output_mode=OutputMode.JSON, include_input_schemas=True)
        off = StructuredOutputAdapter(output_mode=OutputMode.JSON, include_input_schemas=False)

        input_marker = "Inputs will have the following structure:"
        output_marker = "Outputs will be a JSON object with the following fields."

        on_out = on.format_field_structure(Extract)
        off_out = off.format_field_structure(Extract)

        assert input_marker in on_out
        assert output_marker in on_out
        assert input_marker in off_out
        assert output_marker in off_out

        on_input_section = on_out[on_out.index(input_marker) : on_out.index(output_marker)]
        off_input_section = off_out[off_out.index(input_marker) : off_out.index(output_marker)]

        assert "# note: the value you produce must adhere to the JSON schema:" in on_input_section
        assert "# note:" not in off_input_section
