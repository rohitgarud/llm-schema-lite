"""Tests for format_user_message_content, format_assistant_message_content,
format_field_with_value, and format_finetune_data."""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", minversion="3.3.1")

import dspy  # noqa: E402
from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName  # noqa: E402
from dspy.adapters.json_adapter import JSONAdapter  # noqa: E402

from llm_schema_lite.dspy_integration import OutputMode  # noqa: E402
from tests.dspy_helpers import (  # noqa: E402
    QA,
    Address,
    AliasedAddress,
    AliasIn,
    Extract,
    HistoryIn,
    ImageIn,
    Person,
    Typed,
    Unordered,
    assert_message_roles,
    make_adapter,
)

EXPECTED_REQUIREMENTS_JSON = (
    "Respond with a JSON object in the following order of fields: "
    "`person` (must be formatted as a valid Python Person), then "
    "`score` (must be formatted as a valid Python float)."
)
EXPECTED_REQUIREMENTS_YAML = (
    "Respond with a YAML-style object in the following order of fields: "
    "`person` (must be formatted as a valid Python Person), then "
    "`score` (must be formatted as a valid Python float)."
)
EXPECTED_USER_PREFIX = (
    "[[ ## text ## ]]\nhi\n\n[[ ## meta ## ]]\n"
    '{\n  "street": "a",\n  "city": "b",\n  "country": "US"\n}\n\n'
)
INPUTS = {"text": "hi", "meta": Address(street="a", city="b")}


class TestUserMessage:
    """format_user_message_content renders Pydantic inputs (D12)."""

    def test_renders_pydantic_input_with_main_request(self):
        """JSON and JSONish render the Pydantic input and the JSON requirements sentence."""
        for mode in (OutputMode.JSON, OutputMode.JSONISH):
            result = make_adapter(mode).format_user_message_content(
                Extract, INPUTS, main_request=True
            )
            # Tier 1 despite [[ ## k ## ]] being assembled by ChatAdapter: short,
            # defect-free, and a change here is one we want to see. If dspy-latest
            # reddens on this test alone, demote THIS TEST ONLY to Tier 2.
            assert result == EXPECTED_USER_PREFIX + EXPECTED_REQUIREMENTS_JSON

    def test_yaml_mode_differs_only_in_trailing_sentence(self):
        """YAML mode differs from JSON mode only in the trailing requirements sentence."""
        result = make_adapter(OutputMode.YAML).format_user_message_content(
            Extract, INPUTS, main_request=True
        )
        assert result == EXPECTED_USER_PREFIX + EXPECTED_REQUIREMENTS_YAML

    def test_without_main_request_omits_requirements(self):
        """Without main_request the result carries no 'Respond with' sentence."""
        result = make_adapter(OutputMode.JSONISH).format_user_message_content(Extract, INPUTS)
        assert "Respond with" not in result
        assert "[[ ## meta ## ]]" in result

    def test_pydantic_input_is_indented_and_aliased(self):
        """AliasIn's addr field renders as indented JSON honouring the field's alias."""
        for mode in OutputMode:
            result = make_adapter(mode).format_user_message_content(
                AliasIn,
                {"text": "hi", "addr": AliasedAddress(street_name="Main St", city="Anytown")},
                main_request=True,
            )
            assert result.startswith(
                "[[ ## text ## ]]\nhi\n\n[[ ## addr ## ]]\n"
                '{\n  "streetName": "Main St",\n  "city": "Anytown"\n}\n\n'
            )

    def test_image_input_marker_survives(self):
        """A dspy.Image input still yields byte-identical content blocks to JSONAdapter."""
        img = dspy.Image(url="data:image/png;base64,iVBORw0KGgo=")
        ours = make_adapter(OutputMode.JSONISH).format(ImageIn, [], {"img": img})
        upstream = JSONAdapter().format(ImageIn, [], {"img": img})
        assert ours[-1]["content"] == upstream[-1]["content"]

    def test_history_input_is_not_preserialised(self):
        """A dspy.History input is left to upstream, byte-identical to JSONAdapter."""
        inputs = {
            "history": dspy.History(messages=[{"question": "2+2?", "answer": "4"}]),
            "question": "3+3?",
        }
        assert make_adapter(OutputMode.JSONISH).format_user_message_content(
            HistoryIn, inputs
        ) == JSONAdapter().format_user_message_content(HistoryIn, inputs)


EXPECTED_ASSISTANT_JSON = (
    '{\n  "person": {\n    "name": "Jo",\n    "age": 3,\n'
    '    "address": null\n  },\n  "score": 0.5\n}'
)
OUTPUTS = {"person": Person(name="Jo", age=3), "score": 0.5}


class TestAssistantMessage:
    """format_assistant_message_content per output mode."""

    def test_json_mode(self):
        """JSON mode emits indented JSON with the nested Person serialized."""
        assert (
            make_adapter(OutputMode.JSON).format_assistant_message_content(Extract, OUTPUTS)
            == EXPECTED_ASSISTANT_JSON
        )

    def test_jsonish_mode_matches_json_mode(self):
        """JSONish mode emits byte-identical assistant content."""
        assert (
            make_adapter(OutputMode.JSONISH).format_assistant_message_content(Extract, OUTPUTS)
            == EXPECTED_ASSISTANT_JSON
        )

    def test_yaml_mode_single_field(self):
        """YAML mode emits a single-field YAML document."""
        assert (
            make_adapter(OutputMode.YAML).format_assistant_message_content(QA, {"answer": "4"})
            == "answer: '4'\n"
        )


class TestFieldWithValue:
    """format_field_with_value role-keyword contract and defects."""

    def test_accepts_role_keyword(self):
        """format_field_with_value accepts the role keyword DummyLM passes."""
        fields = {
            FieldInfoWithName(name=name, info=info): "4" for name, info in QA.output_fields.items()
        }
        result = make_adapter(OutputMode.JSONISH).format_field_with_value(fields, role="assistant")
        assert isinstance(result, str) and result, f"Empty result: {result!r}"

    @pytest.mark.xfail(
        reason="no owning ticket - research Q19.4: json.dumps called without "
        "ensure_ascii=False, diverging from JSONAdapter"
    )
    def test_non_ascii_is_not_escaped(self):
        """Non-ASCII output should survive unescaped, as upstream JSONAdapter does."""
        assert (
            make_adapter(OutputMode.JSONISH).format_assistant_message_content(
                QA, {"answer": "café ✓"}
            )
            == '{\n  "answer": "café ✓"\n}'
        )

    @pytest.mark.xfail(
        reason="no owning ticket - research Q19.2: _format_yaml_output calls yaml.dump "
        "without sort_keys=False"
    )
    def test_yaml_preserves_declaration_order(self):
        """YAML output should preserve signature declaration order."""
        assert (
            make_adapter(OutputMode.YAML).format_assistant_message_content(
                Unordered, {"zeta": "z", "alpha": "a"}
            )
            == "zeta: z\nalpha: a\n"
        )


class TestFinetuneData:
    """format_finetune_data composes format() + format_assistant_message_content()
    into an OpenAI chat-format {"messages": [...]} record, per mode."""

    def test_format_finetune_data_returns_messages(self):
        """format_finetune_data should return a {'messages': [...]} payload."""
        result = make_adapter(OutputMode.JSONISH).format_finetune_data(
            QA, [], {"question": "3+3?"}, {"answer": "6"}
        )
        assert "messages" in result
        assert {m["role"] for m in result["messages"]} <= {"system", "user", "assistant"}

    def test_json_mode_record_structure(self):
        """JSON mode returns a well-formed chat record whose tail is the assistant turn."""
        adapter = make_adapter(OutputMode.JSON)
        demos = [dict(INPUTS, **OUTPUTS)]
        record = adapter.format_finetune_data(Extract, demos, INPUTS, OUTPUTS)
        assert set(record) == {"messages"}
        assert record["messages"][-1]["role"] == "assistant"
        assert isinstance(record["messages"][-1]["content"], str)
        assert record["messages"][:-1] == adapter.format(Extract, demos, INPUTS)
        assert_message_roles(
            record["messages"], ["system", "user", "assistant", "user", "assistant"]
        )
        assert record["messages"][-1]["content"] == adapter.format_assistant_message_content(
            Extract, OUTPUTS
        )

    def test_jsonish_mode_record_structure(self):
        """JSONish mode returns a well-formed chat record whose tail is the assistant turn."""
        adapter = make_adapter(OutputMode.JSONISH)
        demos = [dict(INPUTS, **OUTPUTS)]
        record = adapter.format_finetune_data(Extract, demos, INPUTS, OUTPUTS)
        assert set(record) == {"messages"}
        assert record["messages"][-1]["role"] == "assistant"
        assert isinstance(record["messages"][-1]["content"], str)
        assert record["messages"][:-1] == adapter.format(Extract, demos, INPUTS)
        assert_message_roles(
            record["messages"], ["system", "user", "assistant", "user", "assistant"]
        )
        assert record["messages"][-1]["content"] == adapter.format_assistant_message_content(
            Extract, OUTPUTS
        )

    def test_yaml_mode_record_structure(self):
        """YAML mode returns a well-formed chat record whose tail is the assistant turn."""
        adapter = make_adapter(OutputMode.YAML)
        demos = [dict(INPUTS, **OUTPUTS)]
        record = adapter.format_finetune_data(Extract, demos, INPUTS, OUTPUTS)
        assert set(record) == {"messages"}
        assert record["messages"][-1]["role"] == "assistant"
        assert isinstance(record["messages"][-1]["content"], str)
        assert record["messages"][:-1] == adapter.format(Extract, demos, INPUTS)
        assert_message_roles(
            record["messages"], ["system", "user", "assistant", "user", "assistant"]
        )
        assert record["messages"][-1]["content"] == adapter.format_assistant_message_content(
            Extract, OUTPUTS
        )

    def test_json_mode_round_trips_through_parse(self):
        """Extract's outputs are both required and non-nullable, so
        apply_output_field_defaults cannot invent one and this equality is not vacuous."""
        adapter = make_adapter(OutputMode.JSON)
        record = adapter.format_finetune_data(Extract, [], INPUTS, OUTPUTS)
        assert adapter.parse(Extract, record["messages"][-1]["content"]) == OUTPUTS

    def test_jsonish_mode_round_trips_through_parse(self):
        """Same non-vacuous round-trip in JSONish mode; see the JSON-mode test for why
        Extract rather than QA or QAOptional is used."""
        adapter = make_adapter(OutputMode.JSONISH)
        record = adapter.format_finetune_data(Extract, [], INPUTS, OUTPUTS)
        assert adapter.parse(Extract, record["messages"][-1]["content"]) == OUTPUTS

    def test_yaml_mode_round_trips_through_parse(self):
        """Same non-vacuous round-trip in YAML mode; see the JSON-mode test for why
        Extract rather than QA or QAOptional is used."""
        adapter = make_adapter(OutputMode.YAML)
        record = adapter.format_finetune_data(Extract, [], INPUTS, OUTPUTS)
        assert adapter.parse(Extract, record["messages"][-1]["content"]) == OUTPUTS

    def test_jsonish_mode_round_trips_with_non_default_typed_values(self):
        """Typed's declared defaults (count=0, tier='a') are deliberately avoided so this
        equality still discriminates on a signature that does have fillable fields."""
        adapter = make_adapter(OutputMode.JSONISH)
        outputs = {"answer": "x", "count": 7, "tier": "b"}
        record = adapter.format_finetune_data(Typed, [], {"question": "3+3?"}, outputs)
        assert adapter.parse(Typed, record["messages"][-1]["content"]) == outputs

    def test_history_input_produces_full_role_sequence(self):
        """A History input with two prior turns yields alternating user/assistant turns."""
        adapter = make_adapter(OutputMode.JSONISH)
        inputs = {
            "history": dspy.History(
                messages=[
                    {"question": "2+2?", "answer": "4"},
                    {"question": "1+1?", "answer": "2"},
                ]
            ),
            "question": "3+3?",
        }
        record = adapter.format_finetune_data(HistoryIn, [], inputs, {"answer": "6"})
        assert_message_roles(
            record["messages"],
            ["system", "user", "assistant", "user", "assistant", "user", "assistant"],
        )
        assert all(isinstance(m["content"], str) for m in record["messages"])

    def test_image_input_preserves_multimodal_content_blocks(self):
        """A dspy.Image user turn keeps its list-valued content; the record is not normalised."""
        adapter = make_adapter(OutputMode.JSONISH)
        img = dspy.Image(url="data:image/png;base64,iVBORw0KGgo=")
        record = adapter.format_finetune_data(ImageIn, [], {"img": img}, {"caption": "a cat"})
        assert_message_roles(record["messages"], ["system", "user", "assistant"])
        assert isinstance(record["messages"][1]["content"], list)
        assert isinstance(record["messages"][-1]["content"], str)

    def test_matches_upstream_chat_adapter_envelope(self):
        """Canary: our record equals ChatAdapter's for the same inputs. If dspy-latest
        reddens on this test alone, demote or delete THIS TEST ONLY, never the
        implementation."""
        adapter = make_adapter(OutputMode.JSONISH)
        inputs, outputs = {"question": "3+3?"}, {"answer": "6"}
        assert ChatAdapter.format_finetune_data(
            adapter, QA, [], inputs, outputs
        ) == adapter.format_finetune_data(QA, [], inputs, outputs)
