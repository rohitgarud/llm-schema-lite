"""Tests for StructuredOutputAdapter.parse (_parse_json / _parse_yaml), all against QAOptional."""

from __future__ import annotations

import pydantic
import pytest

pytest.importorskip("dspy", minversion="3.3.1")

from dspy.utils.exceptions import AdapterParseError  # noqa: E402

from llm_schema_lite import ParseConfig  # noqa: E402
from llm_schema_lite.dspy_integration import OutputMode  # noqa: E402
from tests.dspy_helpers import QAOptional, Typed, make_adapter  # noqa: E402


class TestParseJSON:
    """parse() JSON-branch success cases."""

    def test_plain_json(self):
        """A plain JSON object parses to the exact output-field dict."""
        adapter = make_adapter(OutputMode.JSONISH)
        assert adapter.parse(QAOptional, '{"answer":"x","note":"y"}') == {
            "answer": "x",
            "note": "y",
        }

    def test_fenced_json(self):
        """A fenced ```json block parses to the same dict."""
        adapter = make_adapter(OutputMode.JSONISH)
        completion = '```json\n{"answer":"x","note":"y"}\n```'
        assert adapter.parse(QAOptional, completion) == {
            "answer": "x",
            "note": "y",
        }

    def test_json_mode_matches_jsonish_mode(self):
        """JSON and JSONish modes share _parse_json and return identical dicts."""
        adapter = make_adapter(OutputMode.JSON)
        assert adapter.parse(QAOptional, '{"answer":"x","note":"y"}') == {
            "answer": "x",
            "note": "y",
        }
        completion = '```json\n{"answer":"x","note":"y"}\n```'
        assert adapter.parse(QAOptional, completion) == {
            "answer": "x",
            "note": "y",
        }


class TestParseYAML:
    """parse() YAML-branch success case."""

    def test_yaml_document(self):
        """A YAML document parses to the exact output-field dict."""
        assert make_adapter(OutputMode.YAML).parse(QAOptional, "answer: x\nnote: y\n") == {
            "answer": "x",
            "note": "y",
        }

    def test_yaml_with_trailing_prose(self):
        """YAML with trailing prose after the document still parses."""
        completion = "answer: x\nnote: y\n\nThat is my answer!"
        assert make_adapter(OutputMode.YAML).parse(QAOptional, completion) == {
            "answer": "x",
            "note": "y",
        }

    def test_yaml_mode_parses_json_fenced_block(self):
        """A fenced ```json block in YAML mode is rescued via the JSON fallback."""
        completion = '```json\n{"answer":"x","note":"y"}\n```'
        assert make_adapter(OutputMode.YAML).parse(QAOptional, completion) == {
            "answer": "x",
            "note": "y",
        }

    def test_yaml_mode_parses_prose_wrapped_json(self):
        """Prose-wrapped JSON in YAML mode is rescued via the JSON fallback."""
        completion = 'Here you go:\n{"answer":"x","note":"y"}\nHope that helps'
        assert make_adapter(OutputMode.YAML).parse(QAOptional, completion) == {
            "answer": "x",
            "note": "y",
        }


class TestParseErrors:
    """parse() failure cases."""

    def test_missing_required_field_raises(self):
        """A missing output field raises AdapterParseError naming the fields it did find."""
        with pytest.raises(AdapterParseError) as excinfo:
            make_adapter(OutputMode.JSONISH).parse(QAOptional, '{"note":"y"}')
        assert "Expected to find output fields in the LM response" in str(excinfo.value)
        assert "Actual output fields parsed from the LM response: [note]" in str(excinfo.value)

    def test_non_dict_response_raises(self):
        """A JSON scalar raises AdapterParseError with the serialization message."""
        with pytest.raises(AdapterParseError) as excinfo:
            make_adapter(OutputMode.JSONISH).parse(QAOptional, "42")
        assert "LM response cannot be serialized to a JSON object." in str(excinfo.value)

    def test_missing_optional_field_uses_default(self):
        """A missing optional output field should fall back to its default."""
        assert make_adapter(OutputMode.JSONISH).parse(QAOptional, '{"answer":"x"}') == {
            "answer": "x",
            "note": None,
        }

    def test_garbage_yaml_raises_adapter_parse_error(self):
        """Garbage YAML should surface a DSPy AdapterParseError, not a core exception."""
        with pytest.raises(AdapterParseError):
            make_adapter(OutputMode.YAML).parse(QAOptional, "!!! not yaml or json")

    def test_garbage_json_raises_adapter_parse_error(self):
        """Unparsable JSON in JSONISH mode raises AdapterParseError, not a ConversionError."""
        with pytest.raises(AdapterParseError) as excinfo:
            make_adapter(OutputMode.JSONISH).parse(QAOptional, "!!! not yaml or json")
        assert "LM response cannot be serialized to a JSON object." in str(excinfo.value)


class TestParseConfig:
    """parse() behaviour with an explicit ParseConfig."""

    def test_partial_invalid_optional_field_takes_default(self):
        """An invalid optional field is dropped and refilled with its real default."""
        adapter = make_adapter(OutputMode.JSONISH, parse_config=ParseConfig(partial=True))
        completion = '{"answer":"x","count":"abc","tier":"a"}'
        assert adapter.parse(Typed, completion) == {"answer": "x", "count": 0, "tier": "a"}

    def test_without_parse_config_behaviour_unchanged(self):
        """The same invalid input without parse_config leaks pydantic.ValidationError."""
        adapter = make_adapter(OutputMode.JSONISH)
        completion = '{"answer":"x","count":"abc","tier":"a"}'
        with pytest.raises(pydantic.ValidationError):
            adapter.parse(Typed, completion)

    def test_missing_optional_with_explicit_default(self):
        """A field with an explicit OutputField(default=...) is filled when omitted."""
        adapter = make_adapter(OutputMode.JSONISH)
        assert adapter.parse(Typed, '{"answer":"x","tier":"a"}') == {
            "answer": "x",
            "count": 0,
            "tier": "a",
        }

    def test_none_value_for_optional_field_survives_parse_config(self):
        """note stays None under parse_config; it must never become the string "None"."""
        adapter = make_adapter(OutputMode.JSONISH, parse_config=ParseConfig(partial=True))
        assert adapter.parse(QAOptional, '{"answer":"x","note":null}') == {
            "answer": "x",
            "note": None,
        }

    def test_partial_rescues_case_insensitive_literal(self):
        """The coercion rescue matches "A" to the Literal member "a" under partial=True."""
        adapter = make_adapter(OutputMode.JSONISH, parse_config=ParseConfig(partial=True))
        completion = '{"answer":"x","count":1,"tier":"A"}'
        assert adapter.parse(Typed, completion) == {"answer": "x", "count": 1, "tier": "a"}

    def test_partial_required_field_still_raises(self):
        """partial=True never masks a genuinely missing required field."""
        adapter = make_adapter(OutputMode.JSONISH, parse_config=ParseConfig(partial=True))
        with pytest.raises(AdapterParseError) as excinfo:
            adapter.parse(Typed, '{"count":"abc"}')
        assert excinfo.value.parsed_result == {"count": 0, "tier": "a"}
