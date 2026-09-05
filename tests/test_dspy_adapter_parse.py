"""Tests for StructuredOutputAdapter.parse (_parse_json / _parse_yaml), all against QAOptional."""

from __future__ import annotations

import pydantic
import pytest

pytest.importorskip("dspy", minversion="3.3.1")

import dspy  # noqa: E402
from dspy.utils.exceptions import AdapterParseError  # noqa: E402

from llm_schema_lite import FormatterConfig, ParseConfig  # noqa: E402
from llm_schema_lite.dspy_integration import OutputMode  # noqa: E402
from tests.dspy_helpers import QA, QAOptional, Typed, make_adapter  # noqa: E402


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


ARRAY_VARIANTS = {
    "plain": '[{"answer": "x"}]',
    "indented": '[\n  {\n    "answer": "x"\n  }\n]',
    "fenced": '```json\n[{"answer": "x"}]\n```',
    "fenced_indented": '```json\n[\n  {\n    "answer": "x"\n  }\n]\n```',
}


class TestParseArrayWrapped:
    """A top-level JSON array carrying the object parses as upstream JSONAdapter does."""

    @pytest.mark.parametrize("mode", list(OutputMode))
    @pytest.mark.parametrize("variant", sorted(ARRAY_VARIANTS))
    def test_array_wrapped_object_unwraps(self, mode, variant):
        """AC 2: array-wrapped objects unwrap in every mode and every text layout."""
        adapter = make_adapter(mode)
        assert adapter.parse(QA, ARRAY_VARIANTS[variant]) == {"answer": "x"}

    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_multi_element_array_takes_first_dict(self, mode):
        """Only the first dict is used; later elements are silently ignored (upstream)."""
        adapter = make_adapter(mode)
        assert adapter.parse(QA, '[{"answer": 1}, {"answer": 2}]') == {"answer": "1"}

    def test_array_without_output_fields_fails_completeness(self):
        """A first dict lacking the output fields fails completeness -- upstream's error too."""
        adapter = make_adapter(OutputMode.JSONISH)
        with pytest.raises(AdapterParseError):
            adapter.parse(QA, '[{"nope": 1}, {"answer": "x"}]')

    @pytest.mark.parametrize("completion", ["[]", "[1,2,3]"])
    def test_empty_and_scalar_arrays_keep_serialization_message(self, completion):
        """A list with no dict anywhere falls through to the verbatim dict-guard message."""
        adapter = make_adapter(OutputMode.JSONISH)
        with pytest.raises(AdapterParseError) as excinfo:
            adapter.parse(QA, completion)
        assert "LM response cannot be serialized to a JSON object." in str(excinfo.value)

    def test_nested_array_unwraps(self):
        """Upstream's balanced-brace scan finds the object at any nesting; so must we."""
        adapter = make_adapter(OutputMode.JSONISH)
        assert adapter.parse(QA, '[[{"answer": "x"}]]') == {"answer": "x"}

    def test_array_skips_non_dict_elements(self):
        """Non-dict elements are skipped, not rejected."""
        adapter = make_adapter(OutputMode.JSONISH)
        assert adapter.parse(QA, '[null, {"answer": "x"}]') == {"answer": "x"}

    @pytest.mark.parametrize(
        "completion",
        [
            '[{"answer": "x"}]',
            '[\n  {\n    "answer": "x"\n  }\n]',
            '```json\n[{"answer": "x"}]\n```',
            '```json\n[\n  {\n    "answer": "x"\n  }\n]\n```',
            '[{"answer": 1}, {"answer": 2}]',
            '[[{"answer": "x"}]]',
            '[null, {"answer": "x"}]',
        ],
    )
    def test_upstream_parity_on_arrays(self, completion):
        """Array handling matches dspy.JSONAdapter exactly (design v2 section 7, rows 1-4,6,9,10).

        Scoped to ARRAYS ONLY. The marker half of this ticket is a deliberate divergence
        from upstream and must never be asserted for parity.
        """
        adapter = make_adapter(OutputMode.JSONISH)
        assert adapter.parse(QA, completion) == dspy.JSONAdapter().parse(QA, completion)


class TestParseMarkerKeys:
    """The adapter absorbs the required marker its own prompt renders (AC 1)."""

    @pytest.mark.parametrize("mode", list(OutputMode))
    @pytest.mark.parametrize(
        "parse_config",
        [None, ParseConfig(), ParseConfig(partial=True)],
        ids=["no_config", "default_config", "partial_config"],
    )
    def test_marked_key_accepted_in_every_mode_and_config(self, mode, parse_config):
        """AC 1: a marked reply key maps onto its output field, in all 9 combinations."""
        adapter = make_adapter(mode, parse_config=parse_config)
        assert adapter.parse(QA, '{"answer*": "x"}') == {"answer": "x"}

    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_verbatim_key_never_altered(self, mode):
        """A key that matches an output field verbatim is never touched."""
        adapter = make_adapter(mode)
        assert adapter.parse(QA, '{"answer": "x"}') == {"answer": "x"}

    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_marker_collision_prefers_verbatim_key(self, mode):
        """With both forms present, the verbatim key's value wins."""
        adapter = make_adapter(mode)
        assert adapter.parse(QA, '{"answer": "V", "answer*": "M"}') == {"answer": "V"}

    def test_unknown_marked_key_still_fails_completeness(self):
        """Stripping never invents an output field that was not declared."""
        adapter = make_adapter(OutputMode.JSONISH)
        with pytest.raises(AdapterParseError):
            adapter.parse(QA, '{"bogus*": 1}')

    def test_key_equal_to_marker_is_not_remapped(self):
        """A key that IS the marker strips to "", which is falsy: never remapped."""
        adapter = make_adapter(OutputMode.JSONISH)
        with pytest.raises(AdapterParseError):
            adapter.parse(QA, '{"*": "x"}')

    @pytest.mark.parametrize("mode", list(OutputMode))
    def test_custom_formatter_marker_round_trips(self, mode):
        """The adapter strips the marker its OWN formatter renders.

        This is THE regression test for the render/parse loop -- the one that would have
        caught the original defect. parse_config is None here on purpose.
        """
        adapter = make_adapter(mode, formatter_config=FormatterConfig(required_marker="!"))
        assert adapter.parse(QA, '{"answer!": "x"}') == {"answer": "x"}

    def test_default_marker_still_stripped_under_custom_formatter_marker(self):
        """An explicit ParseConfig adds a SECOND candidate; both markers are accepted."""
        adapter = make_adapter(
            OutputMode.JSONISH,
            formatter_config=FormatterConfig(required_marker="!"),
            parse_config=ParseConfig(),
        )
        assert adapter.parse(QA, '{"answer*": "x"}') == {"answer": "x"}
        assert adapter.parse(QA, '{"answer!": "x"}') == {"answer": "x"}

    def test_empty_parse_config_marker_disables_stripping(self):
        """strip_required_marker="" is the explicit off-switch and beats the base marker."""
        adapter = make_adapter(
            OutputMode.JSONISH, parse_config=ParseConfig(strip_required_marker="")
        )
        with pytest.raises(AdapterParseError):
            adapter.parse(QA, '{"answer*": "x"}')

    def test_native_yaml_marked_key(self):
        """A native YAML reply preserves the marker verbatim; the adapter strips it."""
        adapter = make_adapter(OutputMode.YAML)
        assert adapter.parse(QA, "answer*: x") == {"answer": "x"}

    def test_chain_of_thought_marked_keys(self):
        """reasoning is an ordinary required output field; its marker is stripped too."""
        adapter = make_adapter(OutputMode.JSONISH)
        signature = dspy.ChainOfThought(QA).predict.signature
        assert adapter.parse(signature, '{"reasoning*": "r", "answer*": "x"}') == {
            "reasoning": "r",
            "answer": "x",
        }

    def test_marked_key_inside_array_wrapper(self):
        """The array unwrap and the marker strip compose (design v2 section 7, row 12)."""
        adapter = make_adapter(OutputMode.JSONISH)
        assert adapter.parse(QA, '[{"answer*": "x"}]') == {"answer": "x"}
