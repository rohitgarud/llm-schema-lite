"""Tests for StructuredOutputAdapter.parse (JSON and YAML extraction), all against QAOptional."""

from __future__ import annotations

import enum
import logging
from typing import Literal

import pydantic
import pytest

pytest.importorskip("dspy", minversion="3.3.1")

import dspy  # noqa: E402
from dspy.utils.exceptions import AdapterParseError  # noqa: E402

from llm_schema_lite import FormatterConfig, ParseConfig  # noqa: E402
from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter  # noqa: E402
from llm_schema_lite.dspy_integration.adapters.structured_output_adapter import (
    _merge_record_lists,
    _null_empty_objects,
    _prune_null_list_items,
    _renest_hoisted_fields,
    _unwrap_single_item_lists,
    _wrap_scalars_in_lists,
)  # noqa: E402
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
        """JSON and JSONish modes share _extract_json and return identical dicts."""
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


class _Rating(enum.Enum):
    POOR = 1
    GOOD = 2


class _Color(enum.Enum):
    RED = "crimson"


class _Rescued(dspy.Signature):
    """Rate and describe the laptop."""

    review: str = dspy.InputField()
    rating: _Rating = dspy.OutputField()
    color: _Color | None = dspy.OutputField()
    series_model: str | None = dspy.OutputField()


class TestValueRescues:
    """The coercion rescue reaches Optional[X] fields and int enums sent as text."""

    def test_optional_field_rescued_against_inner_type(self):
        """14 into `str | None` and "RED" (by name) into `_Color | None` parse as for bare X."""
        completion = '{"rating": 2, "color": "RED", "series_model": 14}'
        adapter = make_adapter(OutputMode.JSONISH, parse_config=ParseConfig())
        assert adapter.parse(_Rescued, completion) == {
            "rating": _Rating.GOOD,
            "color": _Color.RED,
            "series_model": "14",
        }
        with pytest.raises(pydantic.ValidationError):
            make_adapter(OutputMode.JSONISH).parse(_Rescued, completion)

    def test_int_enum_sent_as_text(self):
        """ "2" for an int-valued enum becomes the member valued 2; null stays null."""
        completion = '{"rating": "2", "color": null, "series_model": null}'
        adapter = make_adapter(OutputMode.JSONISH, parse_config=ParseConfig())
        assert adapter.parse(_Rescued, completion) == {
            "rating": _Rating.GOOD,
            "color": None,
            "series_model": None,
        }
        with pytest.raises(ValueError):
            make_adapter(OutputMode.JSONISH).parse(_Rescued, completion)

    def test_list_into_optional_str_is_not_stringified(self):
        """["a"] into `str | None` still fails; it never becomes the string "['a']"."""
        completion = '{"rating": 2, "color": null, "series_model": ["a"]}'
        adapter = make_adapter(OutputMode.JSONISH, parse_config=ParseConfig())
        with pytest.raises(pydantic.ValidationError):
            adapter.parse(_Rescued, completion)


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


class _PruneContact(pydantic.BaseModel):
    email: str
    phone: str | None = None


class _PruneRec(pydantic.BaseModel):
    name: str
    contacts: list[_PruneContact] = pydantic.Field(default_factory=list)


class TestPruneNullListItemsRescue:
    """The all-null-list-item rescue: opt-in, rescue-only, information-preserving.

    Small models answer an empty list with a placeholder object of nulls rather than
    `[]`. Coercion cannot repair that without inventing a value; dropping an item whose
    every field is None can only remove information that was never there.
    """

    @staticmethod
    def _sig():
        return dspy.Signature(
            {"text": (str, dspy.InputField()), "record": (_PruneRec, dspy.OutputField())}
        )

    PLACEHOLDER = '{"record": {"name": "Ada", "contacts": [{"email": null, "phone": null}]}}'

    def test_rescue_recovers_the_record(self):
        out = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, parse_config=ParseConfig()
        ).parse(self._sig(), self.PLACEHOLDER)
        assert out["record"].name == "Ada"
        assert out["record"].contacts == []

    def test_is_off_when_parse_config_is_none(self):
        """`parse_config is None` must stay upstream-equivalent -- a default-on semantic
        repair would silently change what every existing caller gets back."""
        with pytest.raises((pydantic.ValidationError, AdapterParseError, ValueError)):
            StructuredOutputAdapter(output_mode=OutputMode.JSONISH).parse(
                self._sig(), self.PLACEHOLDER
            )

    def test_a_valid_reply_is_never_touched(self):
        """Rescue-only: a list that validates never reaches the pruner, so a legitimate
        all-optional item survives."""
        good = '{"record": {"name": "Ada", "contacts": [{"email": "a@b.c", "phone": null}]}}'
        out = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, parse_config=ParseConfig()
        ).parse(self._sig(), good)
        assert len(out["record"].contacts) == 1
        assert out["record"].contacts[0].phone is None


class TestPruneNullListItemsUnit:
    """`_prune_null_list_items` in isolation -- the narrowings that keep it safe."""

    def test_drops_only_all_null_items(self):
        value = [{"a": None, "b": None}, {"a": 1, "b": None}]
        assert _prune_null_list_items(value) == [{"a": 1, "b": None}]

    def test_empty_dict_is_not_a_null_placeholder(self):
        assert _prune_null_list_items([{}]) == [{}]

    def test_returns_the_same_object_when_nothing_is_dropped(self):
        """Identity signals 'no change' so the caller can skip a pointless re-parse."""
        value = {"xs": [{"a": 1}]}
        assert _prune_null_list_items(value) is value

    def test_recurses_into_nesting(self):
        value = {"outer": [{"inner": [{"a": None}], "keep": 1}]}
        assert _prune_null_list_items(value) == {"outer": [{"inner": [], "keep": 1}]}

    def test_scalars_and_plain_lists_pass_through(self):
        assert _prune_null_list_items([1, None, "x"]) == [1, None, "x"]


class _UnwrapName(pydantic.BaseModel):
    family: str
    given: list[str] | None = None


class _UnwrapRec(pydantic.BaseModel):
    name: _UnwrapName | None
    contacts: list[_PruneContact] = pydantic.Field(default_factory=list)


class _UnwrapDoc(pydantic.BaseModel):
    name: _UnwrapName


class TestUnwrapSingleItemListRescue:
    """A single object wrapped in a one-item list, where the schema wants the object.

    In JSON mode qwen3.5:0.8b sent the whole financial-NER record so (`{"entities": [{...}]}`)
    on 29 of 30 cases; on clinical notes it also wraps nested `name` / `address` objects.
    """

    @staticmethod
    def _sig():
        return dspy.Signature(
            {"text": (str, dspy.InputField()), "record": (_UnwrapRec, dspy.OutputField())}
        )

    WRAPPED = (
        '{"record": {"name": [{"family": "Doe"}], "contacts": [{"email": "a@b.c", "phone": null}]}}'
    )

    def test_rescue_unwraps_the_object_and_leaves_a_real_list_alone(self):
        out = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, parse_config=ParseConfig()
        ).parse(self._sig(), self.WRAPPED)
        assert out["record"].name.family == "Doe"
        assert len(out["record"].contacts) == 1

    def test_unwraps_a_whole_output_field(self):
        """The financial-NER shape: the entire record in a one-item list, in JSON mode."""
        sig = dspy.Signature(
            {"text": (str, dspy.InputField()), "record": (_UnwrapName, dspy.OutputField())}
        )
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON, parse_config=ParseConfig())
        out = adapter.parse(sig, '{"record": [{"family": "Doe"}]}')
        assert out["record"].family == "Doe"

    def test_is_off_when_parse_config_is_none(self):
        with pytest.raises((pydantic.ValidationError, AdapterParseError, ValueError)):
            StructuredOutputAdapter(output_mode=OutputMode.JSONISH).parse(self._sig(), self.WRAPPED)

    def test_composes_with_the_null_item_prune(self):
        both = (
            '{"record": {"name": [{"family": "Doe"}], '
            '"contacts": [{"email": null, "phone": null}]}}'
        )
        out = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, parse_config=ParseConfig()
        ).parse(self._sig(), both)
        assert out["record"].name.family == "Doe"
        assert out["record"].contacts == []


class TestUnwrapSingleItemListsUnit:
    """`_unwrap_single_item_lists` in isolation -- where it may and may not unwrap."""

    def test_unwraps_where_the_schema_wants_an_object(self):
        value = {"name": [{"family": "Doe"}], "contacts": []}
        assert _unwrap_single_item_lists(value, _UnwrapRec) == {
            "name": {"family": "Doe"},
            "contacts": [],
        }

    def test_never_unwraps_a_list_typed_field(self):
        value = {"name": None, "contacts": [{"email": "a@b.c"}]}
        assert _unwrap_single_item_lists(value, _UnwrapRec) is value

    def test_leaves_two_items_for_validation_to_reject(self):
        value = {"name": [{"family": "A"}, {"family": "B"}]}
        assert _unwrap_single_item_lists(value, _UnwrapRec) is value

    def test_reaches_models_nested_in_a_list(self):
        value = [{"name": [{"family": "X"}]}]
        assert _unwrap_single_item_lists(value, list[_UnwrapDoc]) == [{"name": {"family": "X"}}]

    def test_a_wrapped_scalar_is_not_an_object(self):
        """`name: ['Rudolf']` (seen in YAML mode) has no object to unwrap to."""
        value = {"name": ["Rudolf"]}
        assert _unwrap_single_item_lists(value, _UnwrapRec) is value


class _RenestHeader(pydantic.BaseModel):
    claim_id: str | None
    channel: str | None


class _RenestPolicy(pydantic.BaseModel):
    policy_number: str | None = None
    channel: str | None = None


class _RenestClaim(pydantic.BaseModel):
    header: _RenestHeader
    notes: str | None = None


class _RenestAmbiguous(pydantic.BaseModel):
    header: _RenestHeader | None = None
    policy: _RenestPolicy | None = None


class TestRenestHoistedFieldsRescue:
    """A nested object's fields hoisted into its parent (insurance-claims' `header`)."""

    @staticmethod
    def _sig():
        return dspy.Signature(
            {"text": (str, dspy.InputField()), "claim": (_RenestClaim, dspy.OutputField())}
        )

    FLAT = '{"claim": {"claim_id": "CLM-1", "channel": "Email", "notes": "n"}}'

    def test_rescue_moves_the_fields_back(self):
        out = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, parse_config=ParseConfig()
        ).parse(self._sig(), self.FLAT)
        assert out["claim"].header.claim_id == "CLM-1"
        assert out["claim"].notes == "n"

    def test_reply_level_keys_move_into_the_output_field(self):
        """qwen3.5:0.8b's JSONISH insurance shape: the header's fields under `claim`, and
        `claim`'s own fields beside it at the reply root."""
        reply = '{"claim": {"claim_id": "CLM-1", "channel": "Email"}, "notes": "n"}'
        out = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, parse_config=ParseConfig()
        ).parse(self._sig(), reply)
        assert out["claim"].header.channel == "Email"
        assert out["claim"].notes == "n"

    def test_is_off_when_parse_config_is_none(self):
        with pytest.raises((pydantic.ValidationError, AdapterParseError, ValueError)):
            StructuredOutputAdapter(output_mode=OutputMode.JSONISH).parse(self._sig(), self.FLAT)


class TestRenestHoistedFieldsUnit:
    """`_renest_hoisted_fields` in isolation -- when a key may move and when it must not."""

    def test_a_key_the_target_already_holds_never_moves(self):
        value = {"header": {"claim_id": "x", "channel": None}, "channel": "Email"}
        assert _renest_hoisted_fields(value, _RenestClaim) is value

    def test_a_missing_key_joins_a_present_target(self):
        value = {"header": {"claim_id": "x"}, "channel": "Email"}
        assert _renest_hoisted_fields(value, _RenestClaim) == {
            "header": {"claim_id": "x", "channel": "Email"}
        }

    def test_a_key_the_parent_owns_never_moves(self):
        value = {"claim_id": "x", "notes": "n"}
        assert _renest_hoisted_fields(value, _RenestClaim) == {
            "notes": "n",
            "header": {"claim_id": "x"},
        }

    def test_a_key_two_targets_claim_stays_put(self):
        """`channel` fits both `header` and `policy`: picking one would be a guess."""
        value = {"claim_id": "x", "channel": "Email"}
        assert _renest_hoisted_fields(value, _RenestAmbiguous) == {
            "channel": "Email",
            "header": {"claim_id": "x"},
        }

    def test_reaches_models_nested_in_a_list(self):
        value = [{"claim_id": "x", "channel": None}]
        assert _renest_hoisted_fields(value, list[_RenestClaim]) == [
            {"header": {"claim_id": "x", "channel": None}}
        ]


class TestNestedMarkerRescue:
    """The prompt's `*` required marker copied into nested keys (qwen3.5:0.8b, YAML)."""

    # Fenced, as the model sent it.
    MARKED = "```yaml\nclaim:\n  header*:\n    claim_id*: CLM-1\n    channel*: Email\n```"

    @staticmethod
    def _sig():
        return dspy.Signature(
            {"text": (str, dspy.InputField()), "claim": (_RenestClaim, dspy.OutputField())}
        )

    def test_rescue_strips_nested_markers(self):
        out = StructuredOutputAdapter(
            output_mode=OutputMode.YAML, parse_config=ParseConfig()
        ).parse(self._sig(), self.MARKED)
        assert out["claim"].header.claim_id == "CLM-1"

    def test_is_off_when_parse_config_is_none(self):
        with pytest.raises((pydantic.ValidationError, AdapterParseError, ValueError)):
            StructuredOutputAdapter(output_mode=OutputMode.YAML).parse(self._sig(), self.MARKED)


class _NullEmp(pydantic.BaseModel):
    company: str
    years: int | None = None


class _NullRec(pydantic.BaseModel):
    name: str
    employment: _NullEmp | None = None


class _NullReq(pydantic.BaseModel):
    employment: _NullEmp


class TestNullEmptyObjectsRescue:
    """An absent optional object answered as an object of nulls (qwen3.5:0.8b, YAML)."""

    ALL_NULL = '{"record": {"name": "Ada", "employment": {"company": null, "years": null}}}'

    @staticmethod
    def _sig():
        return dspy.Signature(
            {"text": (str, dspy.InputField()), "record": (_NullRec, dspy.OutputField())}
        )

    def test_rescue_nulls_the_object(self):
        out = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, parse_config=ParseConfig()
        ).parse(self._sig(), self.ALL_NULL)
        assert out["record"].employment is None

    def test_is_off_when_parse_config_is_none(self):
        with pytest.raises((pydantic.ValidationError, AdapterParseError, ValueError)):
            StructuredOutputAdapter(output_mode=OutputMode.JSONISH).parse(
                self._sig(), self.ALL_NULL
            )

    def test_a_slot_that_does_not_admit_null_is_left_alone(self):
        value = {"employment": {"company": None}}
        assert _null_empty_objects(value, _NullReq) is value

    def test_an_empty_object_is_not_all_null(self):
        value = {"name": "Ada", "employment": {}}
        assert _null_empty_objects(value, _NullRec) is value


class TestMissingEnvelopeRescue:
    """The output field's contents sent without its key (qwen3.5:0.8b, pii, JSON mode)."""

    BARE = '{"name": "Ada", "employment": null}'

    @staticmethod
    def _sig():
        return dspy.Signature(
            {"text": (str, dspy.InputField()), "record": (_NullRec, dspy.OutputField())}
        )

    def test_rescue_moves_the_contents_under_the_field(self):
        out = StructuredOutputAdapter(
            output_mode=OutputMode.JSON, parse_config=ParseConfig()
        ).parse(self._sig(), self.BARE)
        assert out["record"].name == "Ada"

    def test_is_off_when_parse_config_is_none(self):
        with pytest.raises(AdapterParseError):
            StructuredOutputAdapter(output_mode=OutputMode.JSON).parse(self._sig(), self.BARE)

    def test_a_failed_retry_keeps_the_original_error(self):
        """`y` moves under `b`, which then fails validation (`x` is ambiguous and stays):
        the caller must still see the missing-field AdapterParseError."""
        sig = dspy.Signature(
            {
                "text": (str, dspy.InputField()),
                "a": (_TwoA, dspy.OutputField()),
                "b": (_TwoB, dspy.OutputField()),
            }
        )
        with pytest.raises(AdapterParseError):
            StructuredOutputAdapter(output_mode=OutputMode.JSON, parse_config=ParseConfig()).parse(
                sig, '{"x": "v", "y": "w"}'
            )


class _TwoA(pydantic.BaseModel):
    x: str


class _TwoB(pydantic.BaseModel):
    x: str
    y: str | None = None


class _WrapRec(pydantic.BaseModel):
    Company: list[str] | None = None
    contacts: list[_PruneContact] = pydantic.Field(default_factory=list)


class TestWrapScalarsInListsRescue:
    """A one-entity list answered as the entity itself (granite3.1-moe:1b, financial-NER)."""

    LONE = '{"record": {"Company": "Apple"}}'

    @staticmethod
    def _sig():
        return dspy.Signature(
            {"text": (str, dspy.InputField()), "record": (_WrapRec, dspy.OutputField())}
        )

    def test_rescue_wraps_the_scalar(self):
        out = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, parse_config=ParseConfig()
        ).parse(self._sig(), self.LONE)
        assert out["record"].Company == ["Apple"]

    def test_is_off_when_parse_config_is_none(self):
        with pytest.raises((pydantic.ValidationError, AdapterParseError, ValueError)):
            StructuredOutputAdapter(output_mode=OutputMode.JSONISH).parse(self._sig(), self.LONE)

    def test_never_builds_a_list_of_objects_from_a_scalar(self):
        value = {"contacts": "a@b.c"}
        assert _wrap_scalars_in_lists(value, _WrapRec) is value

    def test_keeps_the_value_whole(self):
        assert _wrap_scalars_in_lists({"Company": "Apple Inc"}, _WrapRec) == {
            "Company": ["Apple Inc"]
        }


class _MergeEntities(pydantic.BaseModel):
    Company: list[str] | None = None
    Date: list[str] | None = None


class TestMergeRecordListsRescue:
    """An object of lists answered as one record per entity (llama3.2:1b, financial-NER)."""

    RECORDS = '{"entities": [{"Company": "Apple", "Date": "2021"}, {"Company": ["Intel"]}]}'

    @staticmethod
    def _sig():
        return dspy.Signature(
            {"text": (str, dspy.InputField()), "entities": (_MergeEntities, dspy.OutputField())}
        )

    def test_rescue_concatenates_each_field(self):
        out = StructuredOutputAdapter(
            output_mode=OutputMode.JSON, parse_config=ParseConfig()
        ).parse(self._sig(), self.RECORDS)
        assert out["entities"] == _MergeEntities(Company=["Apple", "Intel"], Date=["2021"])

    def test_is_off_when_parse_config_is_none(self):
        with pytest.raises((pydantic.ValidationError, AdapterParseError, ValueError)):
            StructuredOutputAdapter(output_mode=OutputMode.JSON).parse(self._sig(), self.RECORDS)

    def test_runs_before_the_null_item_prune(self):
        """Pruning first would leave one record, after the unwrap has already run."""
        both = '{"entities": [{"Company": "Apple"}, {"Company": null, "Date": null}]}'
        out = StructuredOutputAdapter(
            output_mode=OutputMode.JSON, parse_config=ParseConfig()
        ).parse(self._sig(), both)
        assert out["entities"] == _MergeEntities(Company=["Apple"], Date=None)

    def test_only_where_every_field_is_a_list(self):
        value = [{"family": "A"}, {"family": "B"}]
        assert _merge_record_lists(value, _UnwrapName) is value

    def test_never_where_the_slot_also_admits_a_list(self):
        value = [{"Company": ["A"]}, {"Company": ["B"]}]
        assert _merge_record_lists(value, _MergeEntities | list[_MergeEntities]) is value


class _SalClaim(pydantic.BaseModel):
    kind: Literal["auto", "home"] | None = None
    amount: int


class _SalContact(pydantic.BaseModel):
    kind: Literal["home", "work"]
    value: str


class _SalRec(pydantic.BaseModel):
    name: str
    claim: _SalClaim | None = None
    contacts: list[_SalContact] = pydantic.Field(default_factory=list)


class TestPartialSalvage:
    """`partial=True` keeps the valid part of a rejected field instead of dropping it all."""

    BAD_ENUM = '{"record": {"name": "Ada", "claim": {"kind": "boat", "amount": 3}}}'

    @staticmethod
    def _parse(completion, annotation=_SalRec, partial=True):
        sig = dspy.Signature(
            {"text": (str, dspy.InputField()), "record": (annotation, dspy.OutputField())}
        )
        adapter = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, parse_config=ParseConfig(partial=partial)
        )
        return adapter.parse(sig, completion)["record"]

    def test_nulls_an_invalid_nested_enum_and_keeps_the_rest(self, caplog):
        with caplog.at_level(logging.DEBUG):
            record = self._parse(self.BAD_ENUM)
        assert record == _SalRec(name="Ada", claim=_SalClaim(kind=None, amount=3))
        assert "null claim.kind" in caplog.text

    def test_is_off_without_partial(self):
        with pytest.raises(pydantic.ValidationError):
            self._parse(self.BAD_ENUM, partial=False)

    def test_drops_a_list_item_whose_required_value_is_invalid(self):
        record = self._parse(
            '{"record": {"name": "Ada", "contacts": '
            '[{"kind": "home", "value": "1"}, {"kind": "fax", "value": "2"}]}}'
        )
        assert record.contacts == [_SalContact(kind="home", value="1")]

    def test_nothing_valid_left_drops_the_field(self):
        """A required enum at the field's top level cannot be nulled: the field is dropped
        and refilled by its default, as before."""
        record = self._parse('{"record": {"kind": "fax", "value": "2"}}', _SalContact | None)
        assert record is None


class TestReplyKeyRescues:
    """Fields wrapped under one extra key (dspy#8539) and near-miss keys (dspy#8377)."""

    ACTORS = dspy.Signature(
        {
            "text": (str, dspy.InputField()),
            "reasoning": (str, dspy.OutputField()),
            "actors": (list[str], dspy.OutputField()),
            "details": (str, dspy.OutputField()),
        }
    )
    REACT = dspy.Signature(
        {
            "q": (str, dspy.InputField()),
            "next_thought": (str, dspy.OutputField()),
            "next_tool_name": (str, dspy.OutputField()),
            "next_tool_args": (dict, dspy.OutputField()),
        }
    )
    NAMES = dspy.Signature(
        {
            "text": (str, dspy.InputField()),
            "first_name": (str, dspy.OutputField()),
            "last_name": (str, dspy.OutputField()),
        }
    )
    WRAPPED = '{"json_input": {"reasoning": "r", "actors": ["A"], "details": "d"}}'
    WRAPPED_TEXT = (
        '{"json": "{\\n \\"reasoning\\": \\"r\\",\\n \\"actors\\": [\\"B\\"],\\n'
        ' \\"details\\": \\"d\\"\\n}\\n</invoke>"}'
    )
    NEAR_MISS = '{"next_thought": "t", "next_tool_name": "search", "tool_args": {"q": "x"}}'
    AMBIGUOUS = '{"name": "Ada"}'

    @staticmethod
    def _parse(sig, completion, parse_config=None):
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON, parse_config=parse_config)
        return adapter.parse(sig, completion)

    def test_unwraps_an_object_envelope(self):
        out = self._parse(self.ACTORS, self.WRAPPED, ParseConfig())
        assert out == {"reasoning": "r", "actors": ["A"], "details": "d"}

    def test_unwraps_a_json_text_envelope(self):
        out = self._parse(self.ACTORS, self.WRAPPED_TEXT, ParseConfig())
        assert out == {"reasoning": "r", "actors": ["B"], "details": "d"}

    def test_renames_a_near_miss_key(self):
        out = self._parse(self.REACT, self.NEAR_MISS, ParseConfig())
        assert out["next_tool_args"] == {"q": "x"}

    def test_renames_a_key_that_differs_only_in_case_and_separators(self):
        out = self._parse(self.NAMES, '{"First-Name": "Ada", "last_name": "L"}', ParseConfig())
        assert out == {"first_name": "Ada", "last_name": "L"}

    def test_an_ambiguous_near_miss_still_fails(self):
        with pytest.raises(AdapterParseError):
            self._parse(self.NAMES, self.AMBIGUOUS, ParseConfig())

    def test_an_envelope_without_output_fields_still_fails(self):
        with pytest.raises(AdapterParseError):
            self._parse(self.ACTORS, '{"json": "{\\"other\\": 1}"}', ParseConfig())

    @pytest.mark.parametrize("case", ["WRAPPED", "WRAPPED_TEXT", "NEAR_MISS", "AMBIGUOUS"])
    def test_all_are_off_when_parse_config_is_none(self, case):
        sig = {"WRAPPED": self.ACTORS, "WRAPPED_TEXT": self.ACTORS, "NEAR_MISS": self.REACT}
        with pytest.raises(AdapterParseError):
            self._parse(sig.get(case, self.NAMES), getattr(self, case))


class TestExtractionFixes:
    """Reply shapes the shared extraction used to misread."""

    def test_leading_think_block_is_not_parsed_as_the_answer(self):
        adapter = make_adapter(OutputMode.JSON)
        completion = '<think>Format is {"answer": ...}</think>\n{"answer": "Paris"}'
        assert adapter.parse(QA, completion) == {"answer": "Paris"}
