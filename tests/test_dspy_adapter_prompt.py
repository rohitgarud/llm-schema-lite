"""Tests for format_field_structure and user_message_output_requirements."""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", minversion="3.3.1")

import dspy  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter  # noqa: E402
from tests.dspy_helpers import (  # noqa: E402
    QA,
    Extract,
    assert_has_field_marker,
    assert_mode_header,
    assert_no_dspy_owned_text,
    assert_note_clause,
    make_adapter,
)

SHARED_PREAMBLE = (
    "All interactions will be structured in the following way, "
    "with the appropriate values filled in."
)
INPUTS_HEADER = "Inputs will have the following structure:"
JSON_HEADER = "Outputs will be a JSON object with the following fields."
YAML_HEADER = "Outputs will be in YAML format with the following fields."

NOTE_JSON_SCHEMA = "# note: the value you produce must adhere to the JSON schema:"
NOTE_WILL_FOLLOW = "# note: the value you produce will follow the schema:"
NOTE_PARSEABLE = (
    "# note: the value you produce must be parseable according to the following schema:"
)
NOTE_FLOAT = "# note: the value you produce must be a single float value"


class TestFieldStructure:
    """format_field_structure per output mode, all against Extract."""

    def test_json_mode_anchors(self):
        """JSON mode emits the JSON header, both input markers, and the JSON-schema notes."""
        out = make_adapter(OutputMode.JSON).format_field_structure(Extract)
        assert_mode_header(out, OutputMode.JSON)
        assert_has_field_marker(out, "text")
        assert_has_field_marker(out, "meta")
        assert_note_clause(out, "meta")
        assert_note_clause(out, "person")
        assert_note_clause(out, "score")
        assert NOTE_JSON_SCHEMA in out
        assert NOTE_FLOAT in out

    def test_jsonish_mode_anchors(self):
        """JSONish mode shares the JSON header but uses the simplified-schema note wording."""
        out = make_adapter(OutputMode.JSONISH).format_field_structure(Extract)
        assert_mode_header(out, OutputMode.JSONISH)
        assert NOTE_WILL_FOLLOW in out
        assert NOTE_PARSEABLE in out
        assert NOTE_FLOAT in out

    def test_yaml_mode_anchors(self):
        """YAML mode emits the YAML header and never the JSON header."""
        out = make_adapter(OutputMode.YAML).format_field_structure(Extract)
        assert YAML_HEADER in out
        assert JSON_HEADER not in out

    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_mode_headers_are_mutually_exclusive(self, mode):
        """Exactly one of the two output headers appears, per mode."""
        out = make_adapter(mode).format_field_structure(Extract)
        assert_mode_header(out, mode)

    def test_shared_preamble_present(self):
        """All three modes carry the shared preamble and inputs header."""
        for mode in OutputMode:
            out = make_adapter(mode).format_field_structure(Extract)
            assert SHARED_PREAMBLE in out
            assert INPUTS_HEADER in out

    def test_omits_dspy_owned_blocks(self):
        """format_field_structure never emits DSPy's own system-message blocks."""
        for mode in OutputMode:
            out = make_adapter(mode).format_field_structure(Extract)
            assert_no_dspy_owned_text(out)

    @pytest.mark.xfail(
        reason="lsl-2026-09-04-007: output schema is embedded in a JSON string value, "
        "so its newlines and quotes are escaped"
    )
    def test_output_schema_is_plain_text_not_escaped_json_string(self):
        """The output section should be plain text, not a JSON string literal."""
        out = make_adapter(OutputMode.JSONISH).format_field_structure(Extract)
        section = out[out.index(JSON_HEADER) :]
        assert "\\n" not in section
        assert '\\"' not in section

    @pytest.mark.xfail(
        reason="lsl-2026-09-04-007: no required-marker legend is emitted for the * suffix"
    )
    def test_required_marker_legend_present(self):
        """A legend should explain that the * suffix marks a required field."""
        out = make_adapter(OutputMode.JSONISH).format_field_structure(Extract)
        legend = [ln for ln in out.splitlines() if "*" in ln and "required" in ln.lower()]
        assert legend, f"No '*' required-marker legend line found. Snippet: {out[:200]!r}"

    @pytest.mark.xfail(
        reason="lsl-2026-09-04-007: YAML mode dumps a schema string through yaml.dump "
        "(research Q19.3)"
    )
    def test_yaml_field_structure_is_plain_text(self):
        """YAML mode should render nested models as an indented YAML block."""
        out = make_adapter(OutputMode.YAML).format_field_structure(Extract)
        assert "Address.street" not in out
        assert "\\\n" not in out

    def test_jsonish_schema_comments_have_no_stray_quote(self):
        """JSONish schema comments should not end with a stray double quote."""
        out = make_adapter(OutputMode.JSONISH).format_field_structure(Extract)
        stray_tokens = ['Street:"', 'City:"', 'Country:"', 'Street:\\"', 'Age:\\"']
        for token in stray_tokens:
            assert token not in out, (
                f"Stray-quote token {token!r} present in JSONish schema comment. "
                f"Snippet: {out[:200]!r}"
            )


EXPECTED_EXTRACT_JSON = (
    "Respond with a JSON object in the following order of fields: "
    "`person` (must be formatted as a valid Python Person), then "
    "`score` (must be formatted as a valid Python float)."
)
EXPECTED_EXTRACT_YAML = (
    "Respond with a YAML-style object in the following order of fields: "
    "`person` (must be formatted as a valid Python Person), then "
    "`score` (must be formatted as a valid Python float)."
)
EXPECTED_QA_JSON = "Respond with a JSON object in the following order of fields: `answer`."
EXPECTED_QA_YAML = "Respond with a YAML-style object in the following order of fields: `answer`."


class TestOutputRequirements:
    """user_message_output_requirements per output mode."""

    def test_json_mode(self):
        """JSON mode emits the exact requirements sentence for Extract."""
        assert (
            make_adapter(OutputMode.JSON).user_message_output_requirements(Extract)
            == EXPECTED_EXTRACT_JSON
        )

    def test_jsonish_mode_matches_json_mode(self):
        """JSONish mode emits the byte-identical requirements sentence."""
        assert (
            make_adapter(OutputMode.JSONISH).user_message_output_requirements(Extract)
            == EXPECTED_EXTRACT_JSON
        )

    def test_yaml_mode(self):
        """YAML mode differs only in the object wording."""
        assert (
            make_adapter(OutputMode.YAML).user_message_output_requirements(Extract)
            == EXPECTED_EXTRACT_YAML
        )

    def test_single_output_field_wording(self):
        """A single output field produces the short requirements sentence."""
        assert (
            make_adapter(OutputMode.JSONISH).user_message_output_requirements(QA)
            == EXPECTED_QA_JSON
        )
        assert (
            make_adapter(OutputMode.YAML).user_message_output_requirements(QA) == EXPECTED_QA_YAML
        )


class RecNode(BaseModel):
    """Self-referencing model local to this suite (lsl-2026-09-04-014).

    Per ``tests/dspy_helpers.py:13-18`` the DSPy suite defines its own models rather
    than importing recursive fixtures from ``tests/conftest.py``.
    """

    name: str
    children: list[RecNode] = []


RecNode.model_rebuild()


class RecOutput(dspy.Signature):
    """Produce a recursive tree from a query."""

    query: str = dspy.InputField()
    tree: RecNode = dspy.OutputField()


class TestMaxRecursionDepthForwarding:
    """Tier 2 (anchor): StructuredOutputAdapter.max_recursion_depth reaches simplify_schema."""

    def test_recursive_output_field_prompt_contains_placeholder(self):
        """AC4: a recursive output field renders the compact placeholder, not raw $defs.

        Before phases 1-6 landed, the adapter's `except Exception` at :265 swallowed the
        ConversionError from a recursive model and silently fell back to the full verbose
        JSON schema (which contains a literal `"$defs"` key). A test that only checks for
        the absence of a traceback would pass before this change; the `$defs` assertion is
        what proves the compact rendering path is actually being taken.
        """
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH, max_recursion_depth=2)
        out = adapter.format_field_structure(RecOutput)
        assert "recursive:" in out
        assert '"$defs"' not in out

    def test_adapter_max_recursion_depth_kwarg_is_forwarded(self):
        """Two adapters with different max_recursion_depth render different prompts."""
        shallow = StructuredOutputAdapter(output_mode=OutputMode.JSONISH, max_recursion_depth=1)
        deep = StructuredOutputAdapter(output_mode=OutputMode.JSONISH, max_recursion_depth=3)
        out_shallow = shallow.format_field_structure(RecOutput)
        out_deep = deep.format_field_structure(RecOutput)
        assert out_shallow != out_deep
