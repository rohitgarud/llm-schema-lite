"""Tests for StructuredOutputAdapter.parse (_parse_json / _parse_yaml), all against QAOptional."""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", minversion="3.3.1")

from dspy.utils.exceptions import AdapterParseError  # noqa: E402

from llm_schema_lite.dspy_integration import OutputMode  # noqa: E402
from tests.dspy_helpers import QAOptional, make_adapter  # noqa: E402


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


class TestParseErrors:
    """parse() failure cases."""

    def test_missing_required_field_raises(self):
        """A missing output field raises AdapterParseError naming the expected fields."""
        with pytest.raises(AdapterParseError) as excinfo:
            make_adapter(OutputMode.JSONISH).parse(QAOptional, '{"answer":"x"}')
        assert "Expected to find output fields in the LM response" in str(excinfo.value)

    def test_non_dict_response_raises(self):
        """A JSON scalar raises AdapterParseError with the serialization message."""
        with pytest.raises(AdapterParseError) as excinfo:
            make_adapter(OutputMode.JSONISH).parse(QAOptional, "42")
        assert "LM response cannot be serialized to a JSON object." in str(excinfo.value)

    @pytest.mark.xfail(reason="lsl-2026-09-04-008: _parse_json never applies output-field defaults")
    def test_missing_optional_field_uses_default(self):
        """A missing optional output field should fall back to its default."""
        assert make_adapter(OutputMode.JSONISH).parse(QAOptional, '{"answer":"x"}') == {
            "answer": "x",
            "note": None,
        }

    @pytest.mark.xfail(
        reason="lsl-2026-09-04-008: _parse_yaml falls back to _parse_json and leaks "
        "llm_schema_lite ConversionError"
    )
    def test_garbage_yaml_raises_adapter_parse_error(self):
        """Garbage YAML should surface a DSPy AdapterParseError, not a core exception."""
        with pytest.raises(AdapterParseError):
            make_adapter(OutputMode.YAML).parse(QAOptional, "!!! not yaml or json")
