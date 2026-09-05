"""Tests for format_field_structure and user_message_output_requirements."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("dspy", minversion="3.3.1")

import dspy  # noqa: E402
from dspy.adapters.json_adapter import JSONAdapter  # noqa: E402
from dspy.adapters.utils import translate_field_type  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from llm_schema_lite import FormatterConfig  # noqa: E402
from llm_schema_lite.dspy_integration import (  # noqa: E402
    OutputMode,
    PromptLayout,
    StructuredOutputAdapter,
)
from tests.dspy_helpers import (  # noqa: E402
    QA,
    Choices,
    Extract,
    HistoryIn,
    ImageListIn,
    ListOut,
    OptImageIn,
    QAOptional,
    ToolCallSig,
    ToolInputSig,
    assert_has_field_marker,
    assert_message_roles,
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
NOTE_INPUT_SCHEMA = "# note: this value follows the schema:"
NOTE_INPUT_JSON_SCHEMA = "# note: this value adheres to the JSON schema:"
NOTE_PARSEABLE = (
    "# note: the value you produce must be parseable according to the following schema:"
)
NOTE_FLOAT = "# note: the value you produce must be a single float value"
NOTE_TOOLCALLS_HINT = '{"tool_calls": [{"name": "...", "args": {...}}]}'


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
        assert NOTE_INPUT_JSON_SCHEMA in out
        json_input_section = out[out.index(INPUTS_HEADER) : out.index(JSON_HEADER)]
        assert "the value you produce must adhere" not in json_input_section

    def test_jsonish_mode_anchors(self):
        """JSONish mode shares the JSON header but uses the simplified-schema note wording."""
        out = make_adapter(OutputMode.JSONISH).format_field_structure(Extract)
        assert_mode_header(out, OutputMode.JSONISH)
        assert NOTE_INPUT_SCHEMA in out
        assert NOTE_PARSEABLE in out
        assert NOTE_FLOAT in out
        assert "the value you produce will follow" not in out

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

    @pytest.mark.parametrize("layout", list(PromptLayout))
    def test_output_schema_is_plain_text_not_escaped_json_string(self, layout):
        """The output section should be plain text, not a JSON string literal."""
        out = make_adapter(OutputMode.JSONISH, layout=layout).format_field_structure(Extract)
        section = out[out.index(JSON_HEADER) :]
        assert "\\n" not in section
        assert '\\"' not in section

    @pytest.mark.parametrize("layout", list(PromptLayout))
    def test_yaml_field_structure_is_plain_text(self, layout):
        """YAML mode renders nested models as plain text, with no escapes."""
        out = make_adapter(OutputMode.YAML, layout=layout).format_field_structure(Extract)
        assert "\\\n" not in out
        assert "\\n" not in out
        assert '\\"' not in out
        assert YAML_HEADER in out

    def test_yaml_field_structure_has_no_hoisted_class_keys(self):
        """The hoisted `Address.street` block is the YAML formatter's, not the adapter's."""
        out = make_adapter(OutputMode.YAML).format_field_structure(Extract)
        assert "Address.street" not in out

    def test_jsonish_schema_comments_have_no_stray_quote(self):
        """JSONish schema comments should not end with a stray double quote."""
        out = make_adapter(OutputMode.JSONISH).format_field_structure(Extract)
        stray_tokens = ['Street:"', 'City:"', 'Country:"', 'Street:\\"', 'Age:\\"']
        for token in stray_tokens:
            assert token not in out, (
                f"Stray-quote token {token!r} present in JSONish schema comment. "
                f"Snippet: {out[:200]!r}"
            )

    @pytest.mark.parametrize("layout", list(PromptLayout))
    @pytest.mark.parametrize("mode,prefix", [(OutputMode.JSONISH, "//"), (OutputMode.YAML, "#")])
    def test_required_marker_legend_appears_exactly_once(self, mode, prefix, layout):
        """The legend line is hoisted to the preamble and printed exactly once."""
        out = make_adapter(mode, layout=layout).format_field_structure(Extract)
        assert out.count(f"{prefix} Fields marked with * are required") == 1

    @pytest.mark.parametrize("layout", list(PromptLayout))
    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_no_legend_when_no_field_uses_the_marker(self, mode, layout):
        """QA has no complex fields, so no schema uses '*' and no legend is emitted."""
        assert "are required" not in make_adapter(mode, layout=layout).format_field_structure(QA)

    @pytest.mark.parametrize("layout", list(PromptLayout))
    def test_json_mode_emits_no_legend(self, layout):
        """JSON mode never simplifies a schema, so it never emits the legend either."""
        assert "are required" not in make_adapter(
            OutputMode.JSON, layout=layout
        ).format_field_structure(Extract)

    @pytest.mark.parametrize("layout", list(PromptLayout))
    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_history_field_carries_no_note_or_schema(self, mode, layout):
        """dspy.History is a Type-adjacent carve-out: bare placeholder, no note."""
        out = make_adapter(mode, layout=layout).format_field_structure(HistoryIn)
        assert "{history}        # note:" not in out
        assert "conversation history is a list of messages" not in out

    @pytest.mark.parametrize("layout", list(PromptLayout))
    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_tool_calls_output_carries_a_schema_note(self, mode, layout):
        """A ToolCalls OUTPUT is no longer a bare placeholder in any mode/layout."""
        out = make_adapter(mode, layout=layout).format_field_structure(ToolCallSig)
        assert "{tool_calls}        # note:" in out

    def test_tool_calls_json_mode_note_stem_matches_upstream(self):
        """JSON mode carries our verbatim JSON-schema note stem for tool_calls."""
        out = make_adapter(OutputMode.JSON).format_field_structure(ToolCallSig)
        assert NOTE_JSON_SCHEMA in out

    def test_tool_calls_json_schema_is_object_equivalent_to_upstream(self):
        """Our tool_calls JSON schema is object-equal to upstream JSONAdapter's (AC-1)."""
        out = make_adapter(OutputMode.JSON).format_field_structure(ToolCallSig)
        ours = json.loads(out.split(NOTE_JSON_SCHEMA, 1)[1].splitlines()[0].strip())
        upstream = translate_field_type("tool_calls", ToolCallSig.output_fields["tool_calls"])
        theirs = json.loads(upstream.split(NOTE_JSON_SCHEMA, 1)[1].splitlines()[0].strip())
        assert ours == theirs

    def test_tool_calls_simplified_schema_in_jsonish(self):
        """JSONish mode renders our simplified ToolCalls schema block."""
        out = make_adapter(OutputMode.JSONISH).format_field_structure(ToolCallSig)
        assert "//Title: ToolCalls" in out
        assert "tool_calls*: [{" in out

    def test_tool_calls_simplified_schema_in_yaml(self):
        """YAML mode renders our simplified ToolCalls schema block."""
        out = make_adapter(OutputMode.YAML).format_field_structure(ToolCallSig)
        assert "# Title: ToolCalls" in out
        assert "tool_calls*:" in out
        assert "- name*: string" in out

    @pytest.mark.parametrize("layout", list(PromptLayout))
    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_tool_input_field_carries_no_note(self, mode, layout):
        """list[dspy.Tool] inputs are carved out: no note on the {tools} placeholder."""
        for signature in (ToolCallSig, ToolInputSig):
            out = make_adapter(mode, layout=layout).format_field_structure(signature)
            assert "{tools}        # note:" not in out

    @pytest.mark.parametrize("layout", list(PromptLayout))
    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_image_list_input_carries_no_note(self, mode, layout):
        """list[dspy.Image] inputs are carved out: no note and no legend (AC-3)."""
        out = make_adapter(mode, layout=layout).format_field_structure(ImageListIn)
        assert "{images}        # note:" not in out
        assert "are required" not in out

    @pytest.mark.parametrize("layout", list(PromptLayout))
    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_optional_image_input_carries_no_note(self, mode, layout):
        """Optional[dspy.Image] inputs are carved out: no note and no legend."""
        out = make_adapter(mode, layout=layout).format_field_structure(OptImageIn)
        assert "{image}        # note:" not in out
        assert "are required" not in out

    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_ordinary_model_inputs_still_carry_notes(self, mode):
        """Ordinary Pydantic inputs keep their notes after the carve-out change."""
        out = make_adapter(mode).format_field_structure(Extract)
        assert_note_clause(out, "meta")
        if mode == OutputMode.JSON:
            assert NOTE_INPUT_JSON_SCHEMA in out
        else:
            assert NOTE_INPUT_SCHEMA in out

    @pytest.mark.parametrize("layout", list(PromptLayout))
    @pytest.mark.parametrize("mode,prefix", [(OutputMode.JSONISH, "//"), (OutputMode.YAML, "#")])
    def test_default_required_marker_legend_unaffected(self, mode, prefix, layout):
        """The default '*' marker still hoists exactly one legend line (C3 guard)."""
        out = make_adapter(mode, layout=layout).format_field_structure(Extract)
        assert out.count(f"{prefix} Fields marked with * are required") == 1

    @pytest.mark.parametrize("layout", list(PromptLayout))
    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_empty_required_marker_emits_no_legend(self, mode, layout):
        """An empty required_marker disables markers, so no legend is emitted (AC-4)."""
        out = make_adapter(
            mode, layout=layout, formatter_config=FormatterConfig(required_marker="")
        ).format_field_structure(Extract)
        assert "are required" not in out


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
EXPECTED_TOOLCALLS_JSON = (
    "Respond with a JSON object in the following order of fields: "
    '`tool_calls` (must be a JSON object like {"tool_calls": [{"name": "...", '
    '"args": {...}}]}).'
)
EXPECTED_TOOLCALLS_YAML = (
    "Respond with a YAML-style object in the following order of fields: "
    '`tool_calls` (must be a JSON object like {"tool_calls": [{"name": "...", '
    '"args": {...}}]}).'
)


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

    def test_tool_calls_requirements_matches_upstream_json_adapter(self):
        """JSON mode reproduces upstream JSONAdapter's ToolCalls sentence byte for byte."""
        assert make_adapter(OutputMode.JSON).user_message_output_requirements(
            ToolCallSig
        ) == JSONAdapter().user_message_output_requirements(ToolCallSig)

    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_tool_calls_requirements_hint_in_all_modes(self, mode):
        """Every mode carries upstream's ToolCalls hint with the mode-correct prefix."""
        sentence = make_adapter(mode).user_message_output_requirements(ToolCallSig)
        assert NOTE_TOOLCALLS_HINT in sentence
        if mode == OutputMode.YAML:
            assert sentence == EXPECTED_TOOLCALLS_YAML
        else:
            assert sentence == EXPECTED_TOOLCALLS_JSON

    def test_non_tool_calls_requirements_unchanged(self):
        """Non-ToolCalls signatures are untouched by the new type_info branch."""
        assert (
            make_adapter(OutputMode.JSON).user_message_output_requirements(Extract)
            == EXPECTED_EXTRACT_JSON
        )
        assert (
            make_adapter(OutputMode.YAML).user_message_output_requirements(Extract)
            == EXPECTED_EXTRACT_YAML
        )
        assert (
            make_adapter(OutputMode.JSON).user_message_output_requirements(QA) == EXPECTED_QA_JSON
        )
        assert (
            make_adapter(OutputMode.YAML).user_message_output_requirements(QA) == EXPECTED_QA_YAML
        )
        assert (
            make_adapter(OutputMode.JSON).user_message_output_requirements(ToolInputSig)
            == "Respond with a JSON object in the following order of fields: `answer`."
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


EXPECTED_PERSON_SCHEMA_JSONISH = (
    "// Person nested under Extract.person; matches the captured fixture.\n"
    "{\n"
    "  name*: string // Full name,\n"
    "  age*: int,\n"
    "  address: { // Three-field address used to produce every Tier-1 captured string.\n"
    "    street*: string,\n"
    "    city*: string,\n"
    "    country: string // (default='US')\n"
    "  } OR null  // (default=null)\n"
    "}"
)


class TestPromptLayouts:
    """The in-scope prompt snapshot cases, per layout (lsl-2026-09-04-007)."""

    @pytest.mark.parametrize("layout", list(PromptLayout))
    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_simple_types_prompt(self, mode, layout):
        """An all-str signature carries no notes and no legend, in any mode or layout."""
        out = make_adapter(mode, layout=layout).format_field_structure(QA)
        assert_mode_header(out, mode)
        assert_has_field_marker(out, "question")
        assert_no_dspy_owned_text(out)
        assert "# note:" not in out
        assert "are required" not in out

    def test_simple_types_json_block_is_byte_identical_to_upstream_shape(self):
        """json_block reproduces upstream JSONAdapter's all-str output block exactly."""
        out = make_adapter(
            OutputMode.JSONISH, layout=PromptLayout.JSON_BLOCK
        ).format_field_structure(QA)
        section = out[out.index(JSON_HEADER) + len(JSON_HEADER) :].strip()
        assert section == '{\n  "answer": "{answer}"\n}'

    @pytest.mark.parametrize("layout", list(PromptLayout))
    def test_nested_model_output_prompt(self, layout):
        """A nested Pydantic output renders as literal multi-line schema text."""
        out = make_adapter(OutputMode.JSONISH, layout=layout).format_field_structure(Extract)
        assert_mode_header(out, OutputMode.JSONISH)
        assert_note_clause(out, "person")
        assert_no_dspy_owned_text(out)
        assert "\\n" not in out
        assert '\\"' not in out
        assert EXPECTED_PERSON_SCHEMA_JSONISH in out
        if layout is PromptLayout.SECTIONS:
            assert_has_field_marker(out, "person")
        else:
            assert '"person": {person}' in out

    @pytest.mark.parametrize("layout", list(PromptLayout))
    def test_single_demo_message_shape(self, layout):
        """The demo assistant turn is one JSON object, unchanged by the layout (I1)."""
        messages = make_adapter(OutputMode.JSONISH, layout=layout).format(
            QA, [{"question": "2+2?", "answer": "4"}], {"question": "3+3?"}
        )
        assert_message_roles(messages, ["system", "user", "assistant", "user"])
        assert messages[2]["content"] == '{\n  "answer": "4"\n}'
        assert messages[-1]["content"].endswith(
            "Respond with a JSON object in the following order of fields: `answer`."
        )

    @pytest.mark.parametrize("layout", list(PromptLayout))
    def test_list_model_output_prompt(self, layout):
        """A list[Model] output renders as a bracketed simplified block, not raw $defs."""
        out = make_adapter(OutputMode.JSONISH, layout=layout).format_field_structure(ListOut)
        assert '"$defs"' not in out
        assert "[\n{\n  name*: string // Full name," in out
        assert "\n}\n]" in out
        assert "\\n" not in out
        assert_note_clause(out, "items")

    @pytest.mark.parametrize("layout", list(PromptLayout))
    def test_optional_literal_enum_output_prompt(self, layout):
        """Enum, Literal and str | None outputs each render their own note shape."""
        out = make_adapter(OutputMode.JSONISH, layout=layout).format_field_structure(Choices)
        assert_mode_header(out, OutputMode.JSONISH)
        assert "# note: the value you produce must be one of: red; blue" in out
        assert "must exactly match (no extra characters) one of: a; b" in out
        assert "string OR null" in out
        assert "\\n" not in out

    @pytest.mark.parametrize("layout", list(PromptLayout))
    def test_optional_str_output_uses_the_dict_path(self, layout):
        """QAOptional.note (str | None) reaches the JSON-schema-dict path, not $defs."""
        out = make_adapter(OutputMode.JSONISH, layout=layout).format_field_structure(QAOptional)
        assert "string OR null" in out
        assert '"anyOf"' not in out
