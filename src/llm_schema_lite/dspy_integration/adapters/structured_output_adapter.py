import copy
import dataclasses
import enum
import inspect
import json
import logging
import re
import textwrap
import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import dspy
import pydantic
import yaml
from dspy.adapters.chat_adapter import FieldInfoWithName
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types import Type as DSPyType
from dspy.adapters.types.history import History as DSPyHistory
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.adapters.utils import (
    apply_output_field_defaults,
    get_annotation_name,
    parse_value,
    serialize_for_json,
)
from dspy.clients.base_lm import BaseLM
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError, LMError
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

# Use llm_schema_lite for schema simplification and robust parsing
from llm_schema_lite import (
    CoercionMetadata,
    ConversionError,
    FormatterConfig,
    ParseConfig,
    StreamingNotSupportedError,
    coerce_to_schema,
    loads,
    simplify_schema,
)
from llm_schema_lite.parsers import normalize_marker_keys
from llm_schema_lite.parsers.base import _strip_leading_reasoning
from llm_schema_lite.parsers.schema_parser import normalize_marker_keys_recursive

logger = logging.getLogger(__name__)

# Exact literal from upstream JSONAdapter's non-dict-response message; used at two call
# sites (the _extract_json ConversionError wrap and the shared dict guard) so that
# test_non_dict_response_raises keeps matching verbatim.
_JSON_PARSE_ERROR_MESSAGE = "LM response cannot be serialized to a JSON object."

# Names both formats actually attempted in YAML mode by the time this is raised
# (see _extract_yaml).
_YAML_PARSE_ERROR_MESSAGE = "LM response cannot be parsed as YAML or JSON."

# Bound on _unwrap_array_reply's recursion into nested lists. Deliberately separate from
# FormatterConfig.max_recursion_depth (a render-side cap) and from MARKER_WALK_MAX_DEPTH
# (parsers/schema_parser.py) -- each bounds a different walk.
_ARRAY_UNWRAP_MAX_DEPTH = 8

# Bound on _salvage_leaves: one validation per nulled or dropped node, so a reply with
# more invalid leaves than this is dropped whole, as it was before partial salvaged. The
# longest salvage over the captured small-model replies took 142 steps.
_SALVAGE_MAX_STEPS = 256

# Names the offending mode and both escape hatches; asserted on by
# tests/test_dspy_adapter_streaming.py (must contain "YAML" and "streaming").
_YAML_STREAMING_ERROR_MESSAGE = (
    "Unsupported output mode for streaming: YAML. Per-field streaming requires "
    'OutputMode.JSON or OutputMode.JSONISH; YAML output has no `"field":` boundary '
    "for DSPy's StreamListener to detect. Either switch output_mode, or call "
    "dspy.streamify() without stream_listeners."
)


class OutputMode(enum.Enum):
    """
    Output format modes for structured responses.

    - JSON: LLM outputs JSON, schema uses full model_json_schema() (verbose)
    - JSONISH: LLM outputs JSON, schema uses simplified BAML-like format (token-efficient)
    - YAML: LLM outputs YAML, schema uses simplified YAML format (token-efficient).
      The rendered schema is YAML-flavoured prompt text, not a serialization format.
    """

    JSON = "json"
    JSONISH = "jsonish"
    YAML = "yaml"


class PromptLayout(str, enum.Enum):
    """Output-block layout for :meth:`StructuredOutputAdapter.format_field_structure`.

    - SECTIONS (default): each output field gets its own ``[[ ## field ## ]]`` block,
      identical in form to how the input section already renders. No field's schema is
      ever routed through ``json.dumps``.
    - JSON_BLOCK: a single ``{ "field": ... }``-shaped block, unescaped. A placeholder
      keeps its surrounding quotes iff the field carries no schema text, and loses them
      when a multi-line schema follows.

    Both layouts leave the input section, the preamble, the headers and the
    required-marker legend untouched; only the *output* block's shape changes.
    Accepted as either the enum member or the plain string value.
    """

    SECTIONS = "sections"
    JSON_BLOCK = "json_block"


@dataclasses.dataclass(frozen=True, slots=True)
class _FieldBlock:
    """Neutral, layout-independent description of one signature field.

    Produced once per field by ``StructuredOutputAdapter._describe``; consumed by both
    ``_render_sections`` and ``_render_json_block`` so the two renderers share 100% of
    the schema-derivation logic.

    Attributes:
        name: the field's name, used verbatim in ``[[ ## name ## ]]`` markers and as the
            JSON key in JSON_BLOCK layout.
        note_text: text placed after ``        # note: `` (eight spaces), or None when the
            field carries no note at all.
        schema_text: the multi-line rendered schema printed on the following line(s), or
            None when everything is already inside ``note_text``.
        legend_needed: True iff this field's schema used the ``*`` required marker in
            field-key position and therefore wants the once-per-prompt legend line.
    """

    name: str
    note_text: str | None
    schema_text: str | None
    legend_needed: bool = False


class _ResponseFormatPlan(enum.Enum):
    """What, if anything, this call should put in lm_kwargs["response_format"]."""

    NONE = "none"  # never write the key
    JSON_OBJECT = "json_object"  # write {"type": "json_object"}
    SCHEMA = "schema"  # build the Pydantic model; fall back on failure


def _same_or_new(old: Any, new: Any) -> Any:
    """``new`` if any item differs by identity from ``old``'s (same keys/length), else ``old``.

    Every structural repair returns its input BY IDENTITY when nothing changed, so the caller
    can skip a pointless re-parse; this is that check for one rebuilt dict or list.
    """
    olds, news = (old.values(), new.values()) if isinstance(old, dict) else (old, new)
    return new if any(n is not o for o, n in zip(olds, news, strict=True)) else old


def _prune_null_list_items(value: Any) -> Any:
    """Drop list items in which EVERY field is ``None``, recursively.

    Rescue-only, and reached only from the ``except`` branch of the ``parse_value`` loop
    in ``_build_output_fields`` -- so it can only improve an outcome that already failed.

    **Why this repair and not coercion.** Small models routinely answer an empty list
    with a placeholder instead of ``[]``::

        {"contacts": [{"email": null, "phone": null}]}   # what the model sent
        {"contacts": []}                                 # what the text supports

    ``email: str`` is required, so this fails validation and the whole record scores
    zero. Coercion cannot help: satisfying ``str`` would mean *inventing* a value, which
    turns a wrong answer into a passing one -- the one direction the parser must never
    fail in. Dropping the item is information-preserving instead: an item whose every
    field is ``None`` carries nothing, so removing it cannot destroy a value the model
    actually extracted.

    Measured on ``qwen3.5:0.8b`` over 30 labeled extraction cases (JSONISH): field
    accuracy 0.745 -> 0.929, 24/30 -> 30/30 parsed, all 6 validation errors gone. On
    ``granite3.1-moe:1b``, which does not make this mistake, the numbers are unchanged
    (0.865, 29/30) -- the repair is inert where it is not needed.

    **The narrowing that keeps it honest.** An empty dict is NOT pruned, and a list item
    is pruned only when it has at least one field and all of them are ``None``. For a
    model whose fields are *all* optional an all-null item can be legitimate, which is
    exactly why this never runs on the success path: if the value validated, it is
    returned untouched and this function is never called.

    Returns ``value`` itself (identity-comparable) when nothing was dropped, so the
    caller can skip a pointless re-parse.
    """
    if isinstance(value, dict):
        return _same_or_new(value, {k: _prune_null_list_items(v) for k, v in value.items()})
    if isinstance(value, list):
        kept: list[Any] = []
        dropped = False
        for item in value:
            item_pruned = _prune_null_list_items(item)
            if (
                isinstance(item_pruned, dict)
                and item_pruned
                and all(field is None for field in item_pruned.values())
            ):
                dropped = True
                continue
            if item_pruned is not item:
                dropped = True
            kept.append(item_pruned)
        return kept if dropped else value
    return value


def _is_list_type(annotation: Any) -> bool:
    return annotation is list or get_origin(annotation) is list


def _members(annotation: Any) -> tuple[list[Any], bool]:
    """Strip ``Annotated`` and split a ``Union``: its non-None members, and whether None was one.

    One level only: a member comes back as written (it may itself be ``Annotated``). An
    annotation that is neither is its own only member.
    """
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = get_args(annotation)
        return [arg for arg in args if arg is not type(None)], type(None) in args
    return [annotation], False


def _walk(value: Any, annotation: Any, fn: Any) -> Any:
    """Apply ``fn(child, child_annotation)`` to every child the schema types.

    A child is a field of the dict ``value`` when ``annotation`` admits an object (keys the
    model does not declare are kept as-is), else an item of the list ``value`` when it
    admits a list; anything else is a leaf. Returns ``value`` itself when no child changed.
    """
    model = _object_model(annotation)
    if model is not None:
        if not isinstance(value, dict):
            return value
        fields = model.model_fields
        return _same_or_new(
            value, {k: fn(v, fields[k].annotation) if k in fields else v for k, v in value.items()}
        )
    item_type = _list_item_type(annotation)
    if item_type is not None and isinstance(value, list):
        return _same_or_new(value, [fn(v, item_type) for v in value])
    return value


def _unwrap_single_item_lists(value: Any, annotation: Any) -> Any:
    """Replace a one-item list with its item wherever the schema expects an object.

    Rescue-only, like :func:`_prune_null_list_items`, and run just before it in the same
    step so the two compose. Small models wrap a single object in a one-item list, most
    often the whole record in JSON mode: ``qwen3.5:0.8b`` answered
    ``{"entities": [{"Company": [...], ...}]}`` on 29 of 30 financial-NER cases. Replaying
    the same completions with and without this repair (JSON, SECTIONS, ``ParseConfig()``):
    9/30 -> 26/30 parsed and field accuracy 0.209 -> 0.606 there, 0/30 -> 4/30 on clinical
    notes. JSONISH and YAML replies were unchanged on every corpus measured -- they do not
    produce the shape.

    **Why this repair is safe.** A list holding exactly one object, where exactly one object
    is allowed, has a single reading - unwrapping it invents nothing.

    **Schema-guided, unlike the prune.** It unwraps only where the annotation expects a
    pydantic model and admits no list, so a legitimate one-item list (``contacts: [...]``)
    is never touched, and a list of two or more items is left for validation to reject -
    picking one would be inventing.

    Returns ``value`` itself (identity-comparable) when nothing changed. Unlike the other
    repairs it reads only the annotation's own members, never through an ``Annotated``
    member of a Union.
    """
    members, _ = _members(annotation)
    listy = next((m for m in members if _is_list_type(m)), None)
    if isinstance(value, list) and listy is not None:
        return _walk(value, listy, _unwrap_single_item_lists)
    # ponytail: first model member only; a union of two models is never unwrapped into
    # the second. Resolve by trying each member if a real schema ever needs it.
    model = next(
        (m for m in members if isinstance(m, type) and issubclass(m, pydantic.BaseModel)),
        None,
    )
    if model is None:
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return _walk(value[0], model, _unwrap_single_item_lists)
    return _walk(value, model, _unwrap_single_item_lists)


def _merge_record_lists(value: Any, annotation: Any) -> Any:
    """Merge a list of records into one object where the schema wants an object of lists.

    Rescue-only; the several-item twin of :func:`_unwrap_single_item_lists`. Asked for one
    object whose every field is a list -- financial-NER's ``entities`` -- small models send
    one record per entity instead::

        {"entities": [{"Company": "Apple", "Date": "2021"}, {"Company": "Intel"}]}
        {"entities": {"Company": ["Apple", "Intel"], "Date": ["2021"]}}

    Replaying 1,350 captured completions (three small models x five corpora x three modes,
    ``ParseConfig()``) with and without it: financial-NER went from 126 to 216 of 270
    parsed and recall on non-null gold 0.232 -> 0.407 (``llama3.2:1b`` JSON 0.016 ->
    0.422); no other corpus changed and no case got worse. The rescued records bring their
    inventions too: categories filled where the gold is empty rose 217 -> 543 of 1,323,
    half of the new ones placeholder strings ("not explicitly mentioned") of the kind
    replies that parse unrepaired carry as well. It runs before the prune, which could cut
    two records to one after the unwrap ran.

    **Why this repair is safe.** When every field is a list, the records have one reading:
    each field is the concatenation of that field across them, in order. A lone value is
    appended whole, a null adds nothing (a field null in every record stays null), and a
    slot that also admits a list is left alone -- there the list may be the answer. It
    keeps exactly what the model wrote, right or wrong; without it the reply raises.

    Returns ``value`` itself (identity-comparable) when nothing changed.
    """
    model = _object_model(annotation)
    if (
        model is not None
        and model.model_fields
        and _list_item_type(annotation) is None
        and isinstance(value, list)
        and len(value) > 1
        and all(isinstance(item, dict) for item in value)
        and all(_list_item_type(f.annotation) is not None for f in model.model_fields.values())
    ):
        merged: dict[str, Any] = {}
        for record in value:
            for k, x in record.items():
                if x is None:
                    merged.setdefault(k, None)
                else:
                    merged[k] = (merged.get(k) or []) + (x if isinstance(x, list) else [x])
        value = merged
    return _walk(value, annotation, _merge_record_lists)


def _object_model(annotation: Any) -> type[pydantic.BaseModel] | None:
    """The pydantic model an annotation admits as an object (through Optional/Annotated)."""
    members, _ = _members(annotation)
    if members[0] is not annotation:  # not a leaf: search the members, depth-first
        return next((m for m in map(_object_model, members) if m is not None), None)
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        return annotation
    return None


def _list_item_type(annotation: Any) -> Any:
    """The item type of a list annotation (through Optional/Annotated), else None."""
    members, _ = _members(annotation)
    if members[0] is not annotation:  # not a leaf: search the members, depth-first
        return next((t for t in map(_list_item_type, members) if t is not None), None)
    if _is_list_type(annotation) and get_args(annotation):
        return get_args(annotation)[0]
    return None


def _move_strays(item: dict[str, Any], annotations: dict[str, Any]) -> dict[str, Any]:
    """Move each key no annotation names into the one object field whose model has it.

    A target qualifies when it is absent, null, or a dict that lacks the key. A key two
    targets could take stays put. Returns ``item`` itself when nothing moved.
    """
    owners: dict[str, list[str]] = {}
    for name, annotation in annotations.items():
        target, current = _object_model(annotation), item.get(name)
        if target is None or not (current is None or isinstance(current, dict)):
            continue
        for k in item:
            if k in annotations or k not in target.model_fields:
                continue
            if isinstance(current, dict) and k in current:
                continue
            owners.setdefault(k, []).append(name)
    moves = {k: names[0] for k, names in owners.items() if len(names) == 1}
    if not moves:
        return item
    out = {k: v for k, v in item.items() if k not in moves}
    for k, name in moves.items():
        out[name] = {**(out.get(name) or {}), k: item[k]}
    return out


def _unwrap_envelope(item: dict[str, Any], annotations: dict[str, Any]) -> dict[str, Any]:
    """The dict a reply wrapped its fields in under one extra key, else ``item`` itself.

    ``{"json_input": {...}}`` or ``{"json": "<JSON text>"}`` (stanfordnlp/dspy#8539): one
    key, none an annotation names, and a value that is (or parses to) a dict holding one.
    """
    if len(item) != 1 or item.keys() & annotations.keys():
        return item
    (inner,) = item.values()
    if isinstance(inner, str):
        try:
            inner = loads(inner, mode="json", repair=True)
        except ConversionError:
            return item
    return inner if isinstance(inner, dict) and inner.keys() & annotations.keys() else item


def _rename_near_misses(item: dict[str, Any], annotations: dict[str, Any]) -> dict[str, Any]:
    """Rename each unknown key to the absent field it nearly names (stanfordnlp/dspy#8377).

    Near: equal once case, ``_``, ``-`` and spaces are dropped, or one is the other with
    whole ``_`` words added only at the front or only at the back (``tool_args`` for
    ``next_tool_args``). Only a one-to-one match renames: ``name`` against both
    ``first_name`` and ``last_name`` stays put. Returns ``item`` itself when nothing moved.
    """

    def near(a: str, b: str) -> bool:
        if re.sub(r"[\s_-]", "", a.lower()) == re.sub(r"[\s_-]", "", b.lower()):
            return True
        short, long = sorted((a.split("_"), b.split("_")), key=len)
        return long[: len(short)] == short or long[-len(short) :] == short

    absent = [name for name in annotations if name not in item]
    pairs = [(k, n) for k in item if k not in annotations for n in absent if near(k, n)]
    keys, names = [k for k, _ in pairs], [n for _, n in pairs]
    renames = {k: n for k, n in pairs if keys.count(k) == 1 and names.count(n) == 1}
    return {renames.get(k, k): v for k, v in item.items()} if renames else item


def _renest_hoisted_fields(value: Any, annotation: Any) -> Any:
    """Move a nested object's fields back under it when the model hoisted them a level up.

    Rescue-only, like the other structural repairs. Small models flatten a nested object
    into its parent -- ``{"claim_id": ..., "channel": ..., "policy_details": {...}}`` where
    the schema wants ``{"header": {"claim_id": ..., "channel": ...}, ...}`` -- so the
    required ``header`` is reported missing and the hoisted values are dropped as unknown
    keys. ``_build_output_fields`` applies the same move one level higher, to reply keys
    that are not output fields.

    **Why this repair is safe.** It only moves values the model sent; it never creates
    one. A key moves only when the parent model has no field of that name, the target
    object does not already hold it, and exactly one target could take it - an ambiguous
    key stays where it is for validation to reject.

    Returns ``value`` itself (identity-comparable) when nothing changed.
    """
    model = _object_model(annotation)
    if model is not None and isinstance(value, dict):
        value = _move_strays(value, {k: f.annotation for k, f in model.model_fields.items()})
    return _walk(value, annotation, _renest_hoisted_fields)


def _admits_none(annotation: Any) -> bool:
    return _members(annotation)[1]


def _null_empty_objects(value: Any, annotation: Any) -> Any:
    """Replace an object whose every field is null with null, where the schema allows null.

    Rescue-only; the object-slot twin of :func:`_prune_null_list_items`. Small models answer
    an absent optional object with its fields all null --
    ``employment: {company: null, role: null, years: null}`` -- which fails a required
    ``company: str``. The object carries nothing, so nulling it destroys nothing; a slot that
    does not admit null is left for validation to reject.

    Returns ``value`` itself (identity-comparable) when nothing changed.
    """
    walked = _walk(value, annotation, _null_empty_objects)
    model = _object_model(annotation)
    if model is None or not isinstance(walked, dict):
        return walked
    fields = model.model_fields
    empty = [
        k
        for k, v in walked.items()
        if k in fields
        and isinstance(v, dict)
        and v
        and all(x is None for x in v.values())
        and _object_model(fields[k].annotation) is not None
        and _admits_none(fields[k].annotation)
    ]
    return {**walked, **dict.fromkeys(empty)} if empty else walked


def _wrap_scalars_in_lists(value: Any, annotation: Any) -> Any:
    """Wrap a lone scalar in a one-item list where the schema wants a list of scalars.

    Rescue-only. ``granite3.1-moe:1b`` answers a one-entity list with the entity itself --
    ``"Company": "Apple"`` for ``Company: list[str]`` -- on most financial-NER replies.

    **Why this repair is safe.** A scalar where a list of that scalar is expected has one
    reading: a list holding it. The value is kept whole (never split), and a list of
    objects is never built from a scalar.

    Returns ``value`` itself (identity-comparable) when nothing changed.
    """
    if isinstance(value, (str, int, float)) and _object_model(annotation) is None:
        item_type = _list_item_type(annotation)
        if item_type is not None and _object_model(item_type) is None:
            return [value]
    return _walk(value, annotation, _wrap_scalars_in_lists)


def _strip_nested_markers(value: Any, annotation: Any, markers: list[str]) -> Any:
    """Strip the prompt's required marker from keys BELOW an output field's top level.

    The reply's own top-level keys are stripped on every parse by ``_normalize_reply_keys``;
    nested ones were not, and ``qwen3.5:0.8b`` copies the marker into them in YAML mode
    (``header*:`` / ``claim_id*:``), so the required ``header`` goes missing. Rescue-only,
    and schema-guided by the same walk ``SchemaParser`` uses, so a key genuinely named
    ``name*`` is protected by its verbatim-first rule.

    Returns ``value`` itself when nothing was stripped.
    """
    if not markers or not isinstance(value, (dict, list)):
        return value
    try:
        schema = TypeAdapter(annotation).json_schema()
    except pydantic.PydanticUserError:
        return value
    stripped = value
    for marker in markers:
        stripped = normalize_marker_keys_recursive(stripped, schema, marker)
    return value if stripped == value else stripped


def _salvage_leaves(value: Any, annotation: Any) -> tuple[Any, list[str]] | None:
    """Null what fails validation and keep the rest -- ``ParseConfig(partial=True)`` only.

    Validates ``value`` with ``parse_value``; on error, follows the first error's ``loc``
    through the data (skipping segments that do not index it -- union and validator tags)
    and nulls the node it reaches, adding a ``missing`` key as null. A node inside a list
    is dropped from it instead, and a node null does not satisfy is retried one level up.
    Repeats until the value validates, the failure climbs to the root, or
    ``_SALVAGE_MAX_STEPS`` runs out -- the last two return None.

    **Why opt-in.** The structural repairs only reshape what the model sent; this deletes
    part of it, so the caller gets a record the model did not write, and it keeps every
    value that validated, right or wrong, so a record rescued from one bad enum also
    brings back whatever the model invented beside it. Replaying 1,350 captured
    completions (three small models x five corpora x three modes) with it: recall on
    non-null gold insurance-claims 0.259 -> 0.455, patient-notes 0.038 -> 0.323, synthetic
    0.603 -> 0.705, no case worse -- but values where the gold is null rose with it,
    patient-notes 0.022 -> 0.369 of null-gold fields.

    Returns ``(parsed, steps)`` -- the ``parse_value`` result and each ``null <path>`` /
    ``drop <path>`` taken -- or None. Never mutates ``value``.
    """
    value = copy.deepcopy(value)
    steps: list[str] = []
    tried: set[tuple[Any, ...]] = set()
    for _ in range(_SALVAGE_MAX_STEPS):
        try:
            return parse_value(value, annotation), steps
        except pydantic.ValidationError as exc:
            error = exc.errors()[0]
        except ValueError:
            return None
        path: list[Any] = []
        node, loc = value, error["loc"]
        for i, seg in enumerate(loc):
            indexes = (isinstance(node, dict) and seg in node) or (
                isinstance(node, list) and isinstance(seg, int) and seg < len(node)
            )
            if indexes:
                path.append(seg)
                node = node[seg]
            elif error["type"] == "missing" and i == len(loc) - 1 and isinstance(node, dict):
                path.append(seg)
        while path and tuple(path) in tried:
            path.pop()
        if not path:
            return None
        tried.add(tuple(path))
        *head, last = path
        parent = value
        for seg in head:
            parent = parent[seg]
        where = "".join(f"[{s}]" if isinstance(s, int) else f".{s}" for s in path).lstrip(".")
        if isinstance(parent, list):
            del parent[last]
            # Indices past the dropped item shifted: forget what was tried under this list.
            tried = {t for t in tried if t[: len(head)] != tuple(head)}
            steps.append(f"drop {where}")
        else:
            parent[last] = None
            steps.append(f"null {where}")
    return None


class StructuredOutputAdapter(JSONAdapter):  # type: ignore[misc]
    """
    Unified adapter for structured output with multiple format support.

    Key Features:
    - JSON mode: Standard JSON output with verbose schemas
        (compatible with OpenAI structured outputs)
    - JSONish mode: JSON output with simplified BAML-like
        schemas (60-85% token reduction from verbose JSON schemas)
    - YAML mode: YAML output with simplified schemas. The rendered schema is
      YAML-flavoured prompt text, not a serialization format.
    - Simplified schemas for complex input fields (Pydantic models)
    - Robust parsing with fallback mechanisms

    Args:
        callbacks: Optional list of callbacks for monitoring
        use_native_function_calling: Whether to use native function calling
        output_mode: Output format mode (JSON, JSONISH, or YAML)
        include_input_schemas: Whether to include simplified schemas for complex input types
        max_recursion_depth: How many times a recursive $ref's body is rendered before
            it is replaced by a placeholder, forwarded to FormatterConfig. Default 2.
        formatter_config: Optional FormatterConfig forwarded unchanged to simplify_schema.
            When given it wins entirely and max_recursion_depth is ignored; when None a default
            FormatterConfig(max_recursion_depth=self.max_recursion_depth) is used, i.e. all
            metadata on. Pass FormatterConfig(include_descriptions=False) for terser prompts.
        prompt_layout: PromptLayout.SECTIONS (default) renders each output field as its
            own [[ ## field ## ]] block; PromptLayout.JSON_BLOCK renders one JSON-shaped
            block with the schema inlined, unescaped. Input fields always use the
            sectioned form regardless of this option.
        use_json_object_response_format: JSONish mode only. When True (default) the adapter
            sends response_format={"type": "json_object"} so the model is constrained to emit
            a JSON object. Set False for OpenAI-compatible local servers (LM Studio, some
            Ollama builds) that advertise response_format but reject the json_object type
            (see stanfordnlp/dspy#1871). Ignored in JSON mode, which reproduces upstream
            JSONAdapter's structured-outputs behaviour, and in YAML mode, which never sets
            response_format. Even in JSONish mode, response_format is omitted when the
            signature carries dspy.Tool / dspy.ToolCalls fields.
        parallel_tool_calls: Forwarded unchanged to the DSPy adapter base. When not None and
            native function calling is active on an LM that supports it, DSPy sets
            lm_kwargs["parallel_tool_calls"]. None (default) leaves the provider option unset.
        parse_config: Optional ParseConfig. When None (default), parsing is
            upstream-JSONAdapter-equivalent: a field that fails parse_value leaks its
            ValidationError. When supplied, a field that parse_value rejects is first
            offered to the coercion rescue (see _coerce_field_value), then to the
            structural repairs; if those also fail and parse_config.partial is True, its
            invalid values are nulled and the rest kept (see _salvage_leaves), and only a
            field with nothing valid left is dropped and refilled by
            apply_output_field_defaults. Coercion/salvage/drop events are logged at DEBUG
            only; the Prediction shape is unchanged.
            parse_config no longer gates whether marker stripping happens -- that is
            unconditional and sourced from _effective_formatter_config().required_marker.
            An explicit parse_config.strip_required_marker adds a second marker
            candidate, and "" disables stripping entirely.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = True,
        output_mode: OutputMode = OutputMode.JSONISH,
        include_input_schemas: bool = True,
        max_recursion_depth: int = 2,
        formatter_config: FormatterConfig | None = None,
        prompt_layout: PromptLayout = PromptLayout.SECTIONS,
        use_json_object_response_format: bool = True,
        parallel_tool_calls: bool | None = None,
        parse_config: ParseConfig | None = None,
    ):
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            parallel_tool_calls=parallel_tool_calls,
        )
        self.output_mode = output_mode
        self.include_input_schemas = include_input_schemas
        self.max_recursion_depth = max_recursion_depth
        self.formatter_config = formatter_config
        self.prompt_layout = prompt_layout
        self.use_json_object_response_format = use_json_object_response_format
        self.parse_config = parse_config
        # parallel_tool_calls is stored by Adapter.__init__ (dspy/adapters/base.py:73);
        # do not re-assign it here.

    # ==================== Core Call Methods ====================

    def _plan_response_format(self, lm: BaseLM, signature: type[Signature]) -> _ResponseFormatPlan:
        """Decide what this call should put in lm_kwargs["response_format"].

        Pure: makes no LM call, mutates nothing and raises nothing. The gate order below
        is load-bearing and must not be reordered for readability.

        Args:
            lm: The language model the call will be issued against.
            signature: The DSPy signature being served.

        Returns:
            NONE to leave lm_kwargs["response_format"] untouched, JSON_OBJECT to write
            {"type": "json_object"}, or SCHEMA to build the structured-outputs model.
        """
        # Gate 0 - upstream's first early exit (dspy/adapters/json_adapter.py:57). An LM
        # that does not advertise response_format never receives one, in any mode.
        if "response_format" not in lm.supported_params:
            return _ResponseFormatPlan.NONE

        if self.output_mode == OutputMode.YAML:
            # YAML never sets response_format.
            return _ResponseFormatPlan.NONE

        if self.output_mode == OutputMode.JSONISH:
            if not self.use_json_object_response_format:
                return _ResponseFormatPlan.NONE
            # BROAD predicate (design D5): tools + json_object is the combination that
            # 400s on OpenAI-compatible servers, so JSONish backs off for *any*
            # tool-carrying signature, not just the ToolCalls-output case upstream checks.
            # JSONish deliberately does NOT consult _has_open_ended_mapping: an open-ended
            # mapping is still a JSON object, and no schema is ever sent in this mode.
            if _has_tool_fields(signature):
                return _ResponseFormatPlan.NONE
            return _ResponseFormatPlan.JSON_OBJECT

        # JSON mode - must equal upstream JSONAdapter 3.3.1 exactly. Same three
        # conditions, same left-to-right order, NARROW predicate (design D5), and no
        # consultation of use_json_object_response_format.
        if (
            _has_open_ended_mapping(signature)
            or (not self.use_native_function_calling and _has_tool_calls_output(signature))
            or not lm.supports_response_schema
        ):
            return _ResponseFormatPlan.JSON_OBJECT

        return _ResponseFormatPlan.SCHEMA

    def _guard_streaming_mode(self) -> None:
        """Reject per-field streaming in YAML mode before the LM request is issued.

        Raises:
            StreamingNotSupportedError: if output_mode is OutputMode.YAML and this call is
                part of a dspy.streamify() run with at least one stream listener attached.
                Non-streaming YAML calls, and YAML streamify() calls with an empty
                stream_listeners list, are both unaffected.
        """
        if (
            self.output_mode is OutputMode.YAML
            and dspy.settings.send_stream is not None
            and bool(dspy.settings.stream_listeners)
        ):
            raise StreamingNotSupportedError(_YAML_STREAMING_ERROR_MESSAGE)

    def __call__(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Synchronous call with format-specific response_format handling."""
        self._guard_streaming_mode()
        # Dispatch past JSONAdapter.__call__ to ChatAdapter.__call__: upstream re-derives
        # and overwrites lm_kwargs["response_format"], which we replace wholesale here.
        # We remain a JSONAdapter subclass on purpose (isinstance contracts).
        chat_call = super(JSONAdapter, self).__call__

        plan = self._plan_response_format(lm, signature)

        if plan is _ResponseFormatPlan.SCHEMA:
            try:
                lm_kwargs["response_format"] = _get_structured_outputs_response_format(
                    signature, self.use_native_function_calling
                )
                return chat_call(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]
            except LMError:
                # Provider/backend failure: propagate. Retrying in json_object mode would
                # issue a second, equally doomed LM call (upstream json_adapter.py:86-89).
                raise
            except Exception:
                logger.warning("Failed to use structured output format, falling back to JSON mode.")
                lm_kwargs["response_format"] = {"type": "json_object"}
                return chat_call(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]

        if plan is _ResponseFormatPlan.JSON_OBJECT:
            lm_kwargs["response_format"] = {"type": "json_object"}

        return chat_call(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]

    async def acall(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Asynchronous call with format-specific response_format handling."""
        self._guard_streaming_mode()
        # Dispatch past JSONAdapter.acall to ChatAdapter.acall - see __call__'s comment.
        # The try/except cannot be shared with __call__: the async LM exception surfaces
        # at `await`, not at coroutine creation.
        chat_acall = super(JSONAdapter, self).acall

        plan = self._plan_response_format(lm, signature)

        if plan is _ResponseFormatPlan.SCHEMA:
            try:
                lm_kwargs["response_format"] = _get_structured_outputs_response_format(
                    signature, self.use_native_function_calling
                )
                return await chat_acall(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]
            except LMError:
                # Provider/backend failure: propagate. Retrying in json_object mode would
                # issue a second, equally doomed LM call (upstream json_adapter.py:113-116).
                raise
            except Exception:
                logger.warning("Failed to use structured output format, falling back to JSON mode.")
                lm_kwargs["response_format"] = {"type": "json_object"}
                return await chat_acall(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]

        if plan is _ResponseFormatPlan.JSON_OBJECT:
            lm_kwargs["response_format"] = {"type": "json_object"}

        return await chat_acall(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]

    # ==================== Schema & Field Formatting ====================

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Build the system-prompt field-structure block.

        Every field - input or output - is first reduced to a ``_FieldBlock`` by
        ``_describe``, then rendered to text. ``self.prompt_layout`` governs only the
        OUTPUT block's renderer; the input block always uses ``_render_sections``.

        This method never calls ``self.format_field_with_value`` (design invariant I1),
        which is what keeps the schema out of a ``json.dumps`` string value.
        """
        parts: list[str] = []
        parts.append(
            "All interactions will be structured in the following way,"
            " with the appropriate values filled in."
        )

        input_blocks = [
            self._describe(name, info, "input") for name, info in signature.input_fields.items()
        ]
        output_blocks = [
            self._describe(name, info, "output") for name, info in signature.output_fields.items()
        ]

        if any(block.legend_needed for block in (*input_blocks, *output_blocks)):
            parts.append(self._legend_line())

        # The header only earns its tokens when some input block actually carries a note
        # or a schema. For an all-bare signature (every input a plain str) it introduces
        # nothing but `[[ ## name ## ]]` markers, so it is dropped - as it is when
        # include_input_schemas=False has already emptied every block.
        if any(
            block.note_text is not None or block.schema_text is not None for block in input_blocks
        ):
            parts.append("Inputs will have the following structure:")
        parts.append(self._render_sections(input_blocks))

        if self.output_mode == OutputMode.YAML:
            parts.append("Outputs will be in YAML format with the following fields.")
        else:
            parts.append("Outputs will be a JSON object with the following fields.")

        parts.append(self._render(output_blocks))
        return "\n\n".join(parts).strip()

    # ---------- describe/render helpers (lsl-2026-09-04-007) ----------

    def _effective_formatter_config(self) -> FormatterConfig:
        """Return the explicit formatter_config, else the max_recursion_depth default.

        Precedence is unchanged from before this ticket: an explicit ``formatter_config``
        wins entirely and ``max_recursion_depth`` is ignored.
        """
        if self.formatter_config is not None:
            return self.formatter_config
        return FormatterConfig(max_recursion_depth=self.max_recursion_depth)

    def _legend_line(self) -> str:
        """Return the once-per-prompt required-marker legend line for this mode."""
        marker = self._effective_formatter_config().required_marker
        comment_prefix = "#" if self.output_mode == OutputMode.YAML else "//"
        return f"{comment_prefix} Fields marked with {marker} are required"

    @staticmethod
    def _legend_needed(schema_text: str, legend_line: str, marker: str) -> bool:
        """Report whether ``schema_text`` wants the required-marker legend.

        True iff a whole line equals ``legend_line`` (the literal the formatters emit
        per simplified schema), OR any line carries the marker in field-key position.
        The second disjunct is required because the JSON-schema-dict path emits ``*``
        markers without ever emitting a legend line of its own. An empty marker means
        "no field is markable," so no legend can ever be needed.

        The field-key pattern tolerates an optional leading YAML sequence dash
        (``- name*: string``): this is not an adapter artifact being tolerated, it is
        recognizing a shape the YAML formatter itself emits natively for every root
        array of objects (and has always emitted for nested ones), so this closes a
        pre-existing latent gap in the marker scan, not merely a root-shape one.
        """
        if not marker:
            return False
        lines = schema_text.splitlines()
        if any(line.strip() == legend_line for line in lines):
            return True
        pattern = re.compile(r"^\s*(?:-\s+)?[^\s:]+" + re.escape(marker) + r"\s*:")
        return any(pattern.match(line) for line in lines)

    @staticmethod
    def _is_dspy_special_annotation(annotation: Any) -> bool:
        """True iff the annotation is, or contains, a dspy.Type or dspy.History.

        Uses the public ``DSPyType.extract_custom_type_from_annotation`` classmethod,
        which walks arbitrarily nested annotations (``list[X]``, ``Optional[X]``,
        ``dict[str, X]``) to find any wrapped ``dspy.Type`` subclass (Image, Audio,
        Tool, ToolCalls, Code). ``dspy.History`` is not a ``Type`` subclass, so it is
        checked separately via ``inspect.isclass``/``issubclass`` - this second check
        is load-bearing, not redundant.
        """
        if DSPyType.extract_custom_type_from_annotation(annotation):
            return True
        return inspect.isclass(annotation) and issubclass(annotation, DSPyHistory)

    def _describe(
        self,
        name: str,
        field_info: FieldInfo,
        role: Literal["input", "output"],
    ) -> _FieldBlock:
        """Build the neutral, layout-independent record for one signature field."""
        field_type = field_info.annotation

        # 1. include_input_schemas gate: evaluated per field, before any schema work.
        if role == "input" and not self.include_input_schemas:
            return _FieldBlock(name=name, note_text=None, schema_text=None)

        # 2. Carve-out for dspy.Type-wrapping / dspy.History annotations, EXCEPT a
        #    ToolCalls OUTPUT field, which falls through to the normal schema chain
        #    (upstream JSONAdapter renders a full schema for ToolCalls).
        if self._is_dspy_special_annotation(field_type) and not (
            role == "output" and field_type == ToolCalls
        ):
            return _FieldBlock(name=name, note_text=None, schema_text=None)

        voice = "the value you produce " if role == "output" else "this value "

        # Constraints passed as InputField/OutputField kwargs (ge=, max_length=, ...) live
        # in FieldInfo.metadata, not the annotation (stanfordnlp/dspy#10195). Folding them
        # back in lets the schema chain render them; a constrained scalar thereby skips the
        # note-only branches below. Tuple form because `*` in a subscript needs 3.11.
        if field_info.metadata:
            field_type = Annotated[(field_type, *field_info.metadata)]  # type: ignore[assignment]

        # 3. Scalar / enum / literal branches - note only, never a schema block.
        if field_type is str:
            return _FieldBlock(name=name, note_text=None, schema_text=None)
        if field_type is bool:
            return _FieldBlock(
                name=name, note_text=f"{voice}must be True or False", schema_text=None
            )
        if field_type in (int, float):
            return _FieldBlock(
                name=name,
                note_text=f"{voice}must be a single {field_type.__name__} value",
                schema_text=None,
            )
        if inspect.isclass(field_type) and issubclass(field_type, enum.Enum):
            enum_vals = "; ".join(str(member.value) for member in field_type)
            return _FieldBlock(
                name=name, note_text=f"{voice}must be one of: {enum_vals}", schema_text=None
            )
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:  # type: ignore[union-attr]
            args = "; ".join(str(x) for x in field_type.__args__)  # type: ignore[union-attr]
            return _FieldBlock(
                name=name,
                note_text=(f"{voice}must exactly match (no extra characters) one of: {args}"),
                schema_text=None,
            )

        # 4. Complex types.
        format_type: Literal["jsonish", "typescript", "yaml"] = (
            "yaml" if self.output_mode == OutputMode.YAML else "jsonish"
        )
        note_stem, schema_text = self._resolve_schema_text(field_type, role, format_type)
        legend_needed = False
        if schema_text is not None:
            legend_line = self._legend_line()
            marker = self._effective_formatter_config().required_marker
            legend_needed = self._legend_needed(schema_text, legend_line, marker)
            schema_text = "\n".join(
                line for line in schema_text.splitlines() if line.strip() != legend_line
            )
        return _FieldBlock(
            name=name,
            note_text=f"{voice}{note_stem}",
            schema_text=schema_text,
            legend_needed=legend_needed,
        )

    def _resolve_schema_text(
        self,
        annotation: Any,
        role: Literal["input", "output"],
        format_type: Literal["jsonish", "typescript", "yaml"],
    ) -> tuple[str, str | None]:
        """Resolve a complex annotation to ``(note_stem, schema_text)``.

        ``note_stem`` is the sentence fragment placed after the role voice by
        ``_describe``. ``schema_text`` is the multi-line schema to print on the
        following line(s), or None when the tier embeds everything in the stem.

        Tiers, each catching Exception, logging at DEBUG and falling through, so no
        exception ever escapes:

          1. ``simplify_schema(annotation, ...)`` - the bare-BaseModel path.
          2. ``TypeAdapter(annotation).json_schema()`` fed back into ``simplify_schema``
             as a dict, with a root-array unwrap. Covers list[Model], Model | None,
             str | None and dict[str, Model].
          3. the verbose JSON schema, byte-for-byte the current silent fallback. This
             is also the tier entered directly in OutputMode.JSON.
          4. ``must be a valid <name>`` - the terminal fallback.
        """
        simplified_stem = (
            "follows the schema:"
            if role == "input"
            else "must be parseable according to the following schema:"
        )
        if self.output_mode != OutputMode.JSON:
            config = self._effective_formatter_config()
            try:
                simplified = simplify_schema(annotation, config=config, format_type=format_type)
                return simplified_stem, simplified.to_string()
            except Exception as exc:
                logger.debug(f"simplify_schema failed for {annotation}: {exc}")

            # Tier 2: simplify_schema also accepts a JSON-schema dict, which covers
            # list[Model], Model | None, str | None and dict[str, Model].
            try:
                schema = TypeAdapter(annotation).json_schema()
                text = simplify_schema(schema, config=config, format_type=format_type).to_string()
                return simplified_stem, text
            except Exception as exc:
                logger.debug(f"json-schema-dict path failed for {annotation}: {exc}")

        verbose_stem = (
            "adheres to the JSON schema:" if role == "input" else "must adhere to the JSON schema:"
        )
        try:
            schema = TypeAdapter(annotation).json_schema()
            return f"{verbose_stem} {json.dumps(schema, ensure_ascii=False)}", None
        except Exception as exc:
            logger.debug(f"json_schema failed for {annotation}: {exc}")

        return f"must be a valid {get_annotation_name(annotation)}", None

    def _render(self, blocks: list[_FieldBlock]) -> str:
        """Dispatch ``blocks`` to the renderer selected by ``self.prompt_layout``.

        Used for the OUTPUT block only; the input block always calls
        ``_render_sections`` directly, in both layouts.
        """
        if self.prompt_layout == PromptLayout.JSON_BLOCK:
            return self._render_json_block(blocks)
        # YAML gets its own renderer rather than the JSON one's `"name":` prefix. The
        # prefix is not valid YAML, and bolting it on measured as a regression
        # (0.849 -> 0.760 field accuracy on qwen3.5:0.8b) before this renderer existed.
        if self.output_mode == OutputMode.YAML:
            return self._render_yaml_sections(blocks)
        return self._render_sections(blocks, bind_field_name=True)

    def _render_sections(self, blocks: list[_FieldBlock], bind_field_name: bool = False) -> str:
        """Render ``blocks`` as ``[[ ## name ## ]]`` sections joined by a blank line.

        ``bind_field_name`` prefixes a schema with the key it must be emitted under, and
        is passed only by ``_render`` -- i.e. only for OUTPUT blocks. It exists because an
        unprefixed schema block is the dominant failure mode on small models: the schema
        opens a brace at column 0 on its own line, directly after the last thing in the
        prompt, so the model reads it as the response *envelope* and emits the record
        bare::

            {"name": "Ada", "age": 36}          # what the model sent
            {"record": {"name": "Ada", ...}}    # what DSPy needs to find the field

        The reply is valid JSON with every value correct, but the output field is
        missing, so ``parse`` raises ``AdapterParseError`` and a perfect extraction
        scores zero. Measured on ``qwen3.5:0.8b`` over 30 labeled extraction cases:
        JSONISH went 0/30 parsed -> 30/30 parsed (0.000 -> 0.745 field accuracy) with
        this prefix, and JSON mode was unchanged (0.886 -> 0.880, same 30/30). ``_render``
        withholds it for YAML, where the same prefix measured as a regression.

        Input blocks are deliberately left alone: the model does not produce them, so
        they never carry the envelope ambiguity, and prefixing them would churn the
        committed prompt-cost numbers for no behavioural gain.
        """
        rendered: list[str] = []
        for block in blocks:
            text = f"[[ ## {block.name} ## ]]\n{{{block.name}}}"
            if block.note_text is not None:
                text += f"{' ' * 8}# note: {block.note_text}"
            if block.schema_text is not None:
                if bind_field_name:
                    text += f'\n"{block.name}": {block.schema_text.lstrip()}'
                else:
                    text += f"\n{block.schema_text}"
            rendered.append(text)
        return "\n\n".join(rendered).strip()

    def _render_yaml_sections(self, blocks: list[_FieldBlock]) -> str:
        """Render OUTPUT ``blocks`` as a YAML mapping, one ``name:`` key per field.

        The ``[[ ## name ## ]]`` marker form used everywhere else is not YAML, and in
        YAML mode nothing overrides it: JSON and JSONISH send
        ``response_format={"type": "json_object"}``, which forces the reply back into
        shape no matter what the structure block demonstrated, while YAML sends no
        response format at all. So the demonstration *is* the specification, and a
        marker demonstration under a "respond with YAML" instruction asks the model for
        two incompatible things. Small models follow the demonstration and reply::

            [[ ## record ## ]]
            name: Ada

        which the YAML parser rejects - ``sola-yaml-sections`` scored 0/18 on
        ``qwen3:8b`` in the outcomes arm before this. Emitting a real YAML skeleton
        instead also solves the envelope ambiguity ``_render_sections`` documents, for
        the same reason and by the same means: the schema is nested under the key it
        must be produced under rather than starting at column 0 on its own line.

        The schema body is indented as a whole, which is safe precisely because YAML
        indentation is relative - shifting every line by two columns nests the mapping
        without disturbing the formatter's own structure. Comment lines shift with it
        and stay comments.
        """
        rendered: list[str] = []
        for block in blocks:
            if block.schema_text is None:
                text = f"{block.name}: {{{block.name}}}"
            else:
                text = f"{block.name}:"
            if block.note_text is not None:
                text += f"{' ' * 8}# note: {block.note_text}"
            if block.schema_text is not None:
                text += "\n" + textwrap.indent(block.schema_text.strip("\n"), "  ")
            rendered.append(text)
        return "\n".join(rendered).strip()

    def _render_json_block(self, blocks: list[_FieldBlock]) -> str:
        """Render ``blocks`` as one ``{ "name": ... }``-shaped block, unescaped.

        A placeholder keeps its surrounding quotes iff the field carries no schema
        text, which reproduces upstream JSONAdapter's block byte-for-byte for an
        all-str signature. Schema text is appended verbatim and left-aligned, never
        re-indented under its JSON key - re-indenting would corrupt the formatter's
        own significant indentation.
        """
        entries: list[str] = []
        for block in blocks:
            note_suffix = (
                f"{' ' * 8}# note: {block.note_text}" if block.note_text is not None else ""
            )
            if block.schema_text is None:
                entries.append(f'  "{block.name}": "{{{block.name}}}{note_suffix}"')
            else:
                entries.append(
                    f'  "{block.name}": {{{block.name}}}{note_suffix}\n{block.schema_text}'
                )
        if not entries:
            return "{}"
        return "{\n" + ",\n".join(entries) + "\n}"

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        """Upstream JSONAdapter's sentence; YAML mode asks for a YAML-style object instead.

        Only the leading "a JSON object" is swapped, so the ToolCalls hint ("must be a JSON
        object like ...") survives in every mode.
        """
        message: str = super().user_message_output_requirements(signature)
        if self.output_mode == OutputMode.YAML:
            return message.replace("a JSON object", "a YAML-style object", 1)
        return message

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Pre-serialise plain Pydantic input values, then delegate to ChatAdapter.

        Upstream's chain (ChatAdapter.format_user_message_content ->
        dspy.adapters.utils.format_field_value -> serialize_for_json) passes neither
        ``by_alias`` nor ``indent``, so a Pydantic input renders as compact,
        non-aliased single-line JSON. Replacing qualifying values with a pre-rendered
        ``str`` makes ``format_field_value`` pass them through untouched.

        The dspy.Type / dspy.History guard is load-bearing: dspy.Image, dspy.Audio,
        dspy.Tool, ToolCalls and dspy.Code are all pydantic.BaseModel subclasses, and
        wrapping their ``<<CUSTOM-TYPE-START-IDENTIFIER>>`` marker in JSON string
        quotes would corrupt the content blocks that
        ``_expand_legacy_custom_type_markers_in_chat_message`` builds afterwards.
        dspy.History is a BaseModel but not a dspy.Type, so it needs its own clause.
        """
        patched: dict[str, Any] = dict(inputs)
        for key, value in inputs.items():
            if (
                isinstance(value, pydantic.BaseModel)
                and not isinstance(value, DSPyType)
                and not isinstance(value, DSPyHistory)
            ):
                patched[key] = value.model_dump_json(indent=2, by_alias=True)
        return super().format_user_message_content(  # type: ignore[no-any-return]
            signature, patched, prefix, suffix, main_request
        )

    def format_field_with_value(
        self, fields_with_values: dict[FieldInfoWithName, Any], role: str = "user"
    ) -> str:
        """
        Format field values according to role.

        - User role: upstream JSONAdapter's [[ ## field ## ]] format, unchanged
        - Assistant role: JSON for JSON/JSONish modes (unlike upstream, without
          ensure_ascii=False), YAML for YAML mode
        """
        if role == "user":
            return super().format_field_with_value(fields_with_values, role)  # type: ignore[no-any-return]
        d = {k.name: v for k, v in fields_with_values.items()}
        if self.output_mode == OutputMode.YAML:
            return self._format_yaml_output(d)
        return json.dumps(serialize_for_json(d), indent=2)

    # ==================== Format-Specific Output Methods ====================

    def _format_yaml_output(self, data: dict[str, Any]) -> str:
        """Format output as YAML-style."""
        try:
            serialized = serialize_for_json(data)
            return yaml.dump(serialized, default_flow_style=False, allow_unicode=True)  # type: ignore[no-any-return, unused-ignore]
        except Exception as e:
            logger.warning(f"Failed to format as YAML: {e}")
            return json.dumps(serialize_for_json(data), indent=2)

    # ==================== Parsing Methods ====================

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """
        Parse completion based on output mode with robust fallback.

        - JSON mode: Parse JSON
        - JSONish mode: Parse JSON (same as JSON, just different schema in prompt)
        - YAML mode: Parse YAML, fallback to JSON

        Mode-specific extraction via _extract_json / _extract_yaml, then the mode-shared
        _build_output_fields pipeline.
        """
        if self.output_mode == OutputMode.YAML:
            raw = self._extract_yaml(signature, completion)
        else:
            raw = self._extract_json(signature, completion)
            # A reply cut off mid-string at max_tokens (stanfordnlp/dspy#1727) is closed by
            # the repair, so the value it was writing would pass as complete. Drop it.
            if self.parse_config is not None and _cut_mid_string(completion):
                raw = _drop_last_leaf(raw)
        return self._build_output_fields(signature, completion, raw)

    def _extract_json(self, signature: type[Signature], completion: str) -> Any:
        """Mode-specific extraction for JSON/JSONish.

        Returns whatever core `loads` produced -- dict, scalar, or list; dict-ness is
        enforced by the shared pipeline (_build_output_fields), not here, so that a
        non-dict result (e.g. the int 42) reaches the shared dict guard rather than
        being intercepted at this layer. This is what keeps test_non_dict_response_raises
        passing -- do not add an isinstance(..., dict) check here.

        Args:
            signature: The DSPy signature being parsed against.
            completion: The raw LM completion text.

        Returns:
            The value `loads(completion, mode="json", repair=True)` produced, unchanged.

        Raises:
            AdapterParseError: if core `loads` raises ConversionError. Chained `from` the
                ConversionError. Never raises ConversionError itself.
        """
        try:
            return loads(completion, mode="json", repair=True)
        except ConversionError as exc:
            raise AdapterParseError(
                adapter_name="StructuredOutputAdapter",
                signature=signature,
                lm_response=completion,
                message=_JSON_PARSE_ERROR_MESSAGE,
            ) from exc

    def _extract_yaml(self, signature: type[Signature], completion: str) -> Any:
        """Mode-specific extraction for YAML, with a narrow JSON rescue.

        Tries YAML first; on ConversionError ONLY (never a bare `except Exception`)
        retries as JSON. No semantic check (dict-ness, completeness) is ever inside
        this method's try -- both live in the shared pipeline downstream. This is the
        fix for the old blanket `except Exception`, which also caught the completeness
        check's own correct AdapterParseError and discarded it.

        Args:
            signature: The DSPy signature being parsed against.
            completion: The raw LM completion text.

        Returns:
            The value produced by whichever of the two `loads` calls succeeded,
            unchanged.

        Raises:
            AdapterParseError: if BOTH the YAML and the JSON-rescue `loads` calls raise
                ConversionError. Chained `from` the YAML ConversionError specifically
                (the mode the caller actually asked for), never from the JSON one.
        """
        try:
            return loads(completion, mode="yaml", repair=True)
        except ConversionError as yaml_exc:
            try:
                result = loads(completion, mode="json", repair=True)
            except ConversionError:
                raise AdapterParseError(
                    adapter_name="StructuredOutputAdapter",
                    signature=signature,
                    lm_response=completion,
                    message=_YAML_PARSE_ERROR_MESSAGE,
                ) from yaml_exc
            logger.debug("YAML extraction failed; JSON rescue succeeded for completion")
            return result

    def _marker_candidates(self) -> list[str]:
        """Ordered, de-duplicated markers this adapter will try to strip from reply keys.

        Base candidate is self._effective_formatter_config().required_marker -- the
        marker this adapter's own prompt renders. Falsy means nothing is marked in the
        prompt, so nothing is added. If self.parse_config is not None, its
        strip_required_marker either disables stripping entirely (an explicit "" returns
        [] regardless of the base candidate) or appends a second candidate,
        de-duplicated against the base.

        Returns:
            The ordered marker candidates to try, base marker first. [] means marker
            stripping is fully disabled for this call.
        """
        candidates: list[str] = []
        base = self._effective_formatter_config().required_marker
        if base:
            candidates.append(base)
        if self.parse_config is not None:
            override = self.parse_config.strip_required_marker
            if override == "":
                return []
            if override and override not in candidates:
                candidates.append(override)
        return candidates

    def _normalize_reply_keys(
        self, signature: type[Signature], raw: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply each marker candidate to the reply's top-level keys.

        Sequential passes over self._marker_candidates(), each via
        normalize_marker_keys(raw, signature.output_fields.keys(), marker). Sequential
        passes are strictly more conservative than a merged multi-marker pass, because
        each pass re-evaluates the "stripped name not already in data" guard against the
        PREVIOUS pass's output, which can only contain more known keys. A no-op when
        _marker_candidates() is empty.

        Args:
            signature: The DSPy signature being parsed against -- supplies the known key
                universe (signature.output_fields.keys()), which is the same set the
                output-field filter downstream tests, so stripping and filtering can
                never disagree about what a known key is.
            raw: The dict produced after the dict guard, before the output-field filter.

        Returns:
            A new dict with the same values, with markers from every candidate stripped
            in order. Never mutates `raw`.
        """
        for marker in self._marker_candidates():
            raw = normalize_marker_keys(raw, signature.output_fields.keys(), marker)
        return raw

    def _build_output_fields(
        self, signature: type[Signature], completion: str, raw: Any
    ) -> dict[str, Any]:
        """The shared field pipeline: byte-identical semantics for JSON and YAML.

        Mirrors upstream JSONAdapter.parse (DSPy 3.3.1) with two rescues as the only
        insertions -- _coerce_field_value, then the structural repair: reply-level strays
        moved into the field (_move_strays), nested markers stripped
        (_strip_nested_markers), then _unwrap_single_item_lists, _merge_record_lists,
        _renest_hoisted_fields, _null_empty_objects, _wrap_scalars_in_lists and
        _prune_null_list_items -- plus one fallback before the missing-field error: an
        output field sent without its key is moved under it (_move_strays), fields wrapped
        under one extra key are unwrapped (_unwrap_envelope), or a near-miss key is
        renamed (_rename_near_misses), and the reply parsed once more. With
        parse_config.partial, a field still rejected is offered to
        _salvage_leaves before it is dropped. parse funnels every mode through it after
        the mode-specific extraction.

        Invariants that any change to this method must preserve:
          - self.parse_config is None => structurally upstream-equivalent. The first
            line of the except branch below re-raises before any new machinery is
            reachable; this must not depend on config values happening to be inert.
            The prune rescue sits BELOW that re-raise and behind allow_coercion for the
            same reason: it is a semantic repair, and a default-on repair would silently
            change what every existing caller gets back.
          - Defaults from apply_output_field_defaults are inserted as already-typed
            Python values and are never fed back through parse_value.
          - Return type is dict[str, Any] keyed by output-field name; CoercionMetadata
            goes only to the logger, never into the returned dict or the Prediction.
          - Both _unwrap_array_reply and self._normalize_reply_keys sit strictly ABOVE the
            parse_value loop and change only WHICH keys/shape enter it, never how a value
            is parsed, rescued, or raised.

        Args:
            signature: The DSPy signature being parsed against.
            completion: The raw LM completion text (for AdapterParseError's lm_response).
            raw: Whatever the mode-specific _extract_* method returned -- dict, scalar,
                or list.

        Returns:
            dict[str, Any] with exactly signature.output_fields.keys() as keys.

        Raises:
            AdapterParseError: if `raw` is not a dict, or if a required field is missing
                after coercion and defaults. pydantic.ValidationError / ValueError leak
                through from parse_value when self.parse_config is None and no rescue is
                attempted, matching upstream.
        """
        raw = _unwrap_array_reply(raw)

        if not isinstance(raw, dict):
            raise AdapterParseError(
                adapter_name="StructuredOutputAdapter",
                signature=signature,
                lm_response=completion,
                message=_JSON_PARSE_ERROR_MESSAGE,
            )

        raw = self._normalize_reply_keys(signature, raw)

        fields = {k: v for k, v in raw.items() if k in signature.output_fields}
        # With parse_config, constraints passed as OutputField kwargs (le=1.0, max_length=)
        # are checked too (stanfordnlp/dspy#10195); upstream checks only the annotation.
        output_annotations = {
            name: Annotated[(f.annotation, *f.metadata)]
            if self.parse_config is not None and f.metadata
            else f.annotation
            for name, f in signature.output_fields.items()
        }

        out: dict[str, Any] = {}
        for k, v in fields.items():
            annotation = output_annotations[k]
            try:
                out[k] = parse_value(v, annotation)
                continue
            except (pydantic.ValidationError, ValueError):
                if self.parse_config is None:
                    raise
                rescue = self._coerce_field_value(k, v, annotation)
                if rescue is not None:
                    coerced_value, metadata = rescue
                    try:
                        out[k] = parse_value(coerced_value, annotation)
                    except (pydantic.ValidationError, ValueError):
                        pass
                    else:
                        logger.debug(f"Coercion rescued output field {k!r}: {metadata}")
                        continue
                best = v
                if self.parse_config.allow_coercion:
                    # Unwrap first: pruning first would turn `[{"family": null}]` under an
                    # object-typed field into `[]`, which no longer has anything to unwrap.
                    # Reply keys that are not output fields may be this field's own,
                    # hoisted out of it: `{"claim": {<header>}, "policy_details": ...}`.
                    rooted = _strip_nested_markers(
                        _move_strays(raw, output_annotations).get(k, v),
                        annotation,
                        self._marker_candidates(),
                    )
                    repaired = _unwrap_single_item_lists(rooted, annotation)
                    repaired = _merge_record_lists(repaired, annotation)
                    repaired = _renest_hoisted_fields(repaired, annotation)
                    repaired = _null_empty_objects(repaired, annotation)
                    repaired = _wrap_scalars_in_lists(repaired, annotation)
                    repaired = _prune_null_list_items(repaired)
                    best = repaired
                    if repaired is not v:
                        try:
                            out[k] = parse_value(repaired, annotation)
                        except (pydantic.ValidationError, ValueError):
                            pass
                        else:
                            logger.debug(f"Structural rescue repaired output field {k!r}")
                            continue
                if self.parse_config.partial:
                    salvaged = _salvage_leaves(best, annotation)
                    if salvaged is not None:
                        out[k], steps = salvaged
                        if self.parse_config.log_coercions:
                            logger.debug(f"Partial salvage kept output field {k!r}: {steps}")
                        continue
                    logger.debug(f"Dropping unparseable output field {k!r} (partial=True)")
                    continue
                raise

        out = apply_output_field_defaults(signature, out)

        if out.keys() != signature.output_fields.keys():
            # A reply that sent a field's contents without its key (the PII fields bare at
            # the top instead of under `pii`), wrapped every field under one extra key, or
            # misnamed a field fails here. The first repair that changes the reply is
            # parsed once more. Each retry has less left to repair (fewer unknown keys, or
            # an envelope gone for good), so this cannot loop. A retry that still fails
            # raises THIS error: a rescue never changes how a reply fails.
            if self.parse_config is not None and self.parse_config.allow_coercion:
                for repair in (_move_strays, _unwrap_envelope, _rename_near_misses):
                    retry = repair(raw, output_annotations)
                    if retry is not raw:
                        try:
                            return self._build_output_fields(signature, completion, retry)
                        except (AdapterParseError, pydantic.ValidationError, ValueError):
                            break
            raise AdapterParseError(
                adapter_name="StructuredOutputAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=out,
            )

        return out

    def _coerce_field_value(
        self, name: str, value: Any, annotation: Any
    ) -> tuple[Any, list[CoercionMetadata]] | None:
        """Attempt schema-aware coercion of ONE field value that parse_value rejected.

        Rescue-only by design: only ever called from the except branch of the
        parse_value loop in _build_output_fields, so it can only improve an outcome
        that was already a failure -- it never touches a value that already parsed
        cleanly. That inversion is what removes the corruption class where
        `str | None` holding null became the literal string "None" and dict[str, Any]
        values were silently stringified.

        The scalar-only predicate below is hand-maintained; WIDENING IT RE-ADMITS THAT
        CORRUPTION. tests/test_dspy_adapter_parse.py::TestParseConfig::
        test_none_value_for_optional_field_survives_parse_config exists specifically to
        fail if it is widened -- point any future change at that test by name. An
        Optional[X] field holding a scalar is rescued against X; null, dicts and lists never
        are, so none of them is stringified.

        Never raises: any failure at any step declines the rescue, logs at DEBUG, and
        returns None. The rescue must never be the reason a parse fails.

        Args:
            name: The output field's name (used as the coerce_to_schema wrapper key and
                in debug logging).
            value: The raw value that parse_value rejected.
            annotation: signature.output_fields[name].annotation.

        Returns:
            (coerced_value, metadata) if coercion was attempted and coerce_to_schema ran
            without raising -- the caller re-runs parse_value on coerced_value and only
            keeps it if that succeeds. None if the rescue declined to act (exotic
            annotation, non-scalar predicate, or an internal exception) -- the caller
            must treat None exactly like "coercion did not help".
        """
        members, admits_none = _members(annotation)
        if not isinstance(value, dict | list | None):
            # What one member of a union accepts bare, e.g. an Enum member by name: "RED"
            # for "crimson". A plain X already failed exactly this, so it is not retried.
            for member in members if admits_none or len(members) > 1 else []:
                try:
                    return parse_value(value, member), []
                except Exception:
                    pass
            # An Enum member named or valued in another case: "red" for RED = "crimson". Its
            # value, not the member, is returned: a bare Enum re-parses by value or name.
            enums = [m for m in members if inspect.isclass(m) and issubclass(m, enum.Enum)]
            if isinstance(value, str) and enums:
                key = value.strip().casefold()
                hits = {
                    m
                    for e in enums
                    for m in e
                    if key in (m.name.casefold(), str(m.value).casefold())
                }
                if len(hits) == 1:
                    return hits.pop().value, []
            if admits_none and len(members) == 1:
                annotation = members[0]
        try:
            field_schema = TypeAdapter(annotation).json_schema()
        except Exception as exc:
            logger.debug(f"Coercion rescue declined for {name!r}: json_schema failed: {exc}")
            return None

        if not (
            field_schema.get("type") in {"string", "integer", "number", "boolean"}
            and not ({"anyOf", "$ref", "$defs"} & field_schema.keys())
        ):
            return None

        try:
            coerced, metadata = coerce_to_schema(
                {name: value},
                {"type": "object", "properties": {name: field_schema}},
                self.parse_config,
            )
        except Exception as exc:
            logger.debug(f"Coercion rescue declined for {name!r}: coerce_to_schema: {exc}")
            return None

        return coerced.get(name, value), metadata

    # ==================== Fine-tuning Support ====================

    def format_finetune_data(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        """Format one example as an OpenAI chat-format fine-tuning record.

        Returns ``{"messages": [...]}`` whose final message is the assistant turn for
        the active output_mode, produced by ``format_assistant_message_content``.

        Composed locally rather than delegated to ``ChatAdapter.format_finetune_data``
        (reachable via ``super(JSONAdapter, self)``, as ``__call__``/``acall`` already
        do): that method's body belongs to ``ChatAdapter`` and describes *its* wire
        format, so delegation would keep matching us only incidentally and could
        silently start emitting wrong training data if upstream ever hardens it around
        ChatAdapter's own field-marker contract. Local composition mirrors upstream's
        shape today without borrowing its body.

        Known, deliberate properties, not defects: a multimodal user turn (e.g.
        dspy.Image) keeps its list-of-content-blocks ``content`` unchanged; the
        assistant turn carries no ``[[ ## completed ## ]]`` marker, so it is
        byte-identical to the corresponding demo assistant turn.
        """
        messages = self.format(signature=signature, demos=demos, inputs=inputs)
        assistant = {
            "role": "assistant",
            "content": self.format_assistant_message_content(signature=signature, outputs=outputs),
        }
        return {"messages": messages + [assistant]}


# ==================== Helper Functions ====================


def _unwrap_array_reply(raw: Any, _depth: int = 0) -> Any:
    """Return the first dict reachable through a top-level list, else `raw` unchanged.

    Pure, module-level, no `self`. Mirrors the OUTCOME of upstream JSONAdapter's
    balanced-brace regex rescue for the array shapes both packages agree on: the first
    dict found by scanning a list left-to-right, recursing into nested lists up to
    _ARRAY_UNWRAP_MAX_DEPTH, is returned. Non-dict, non-list elements are skipped, not
    rejected. A list containing no dict anywhere (`[]`, `[1,2,3]`) falls through
    unchanged so it reaches the existing dict guard in _build_output_fields with the
    verbatim _JSON_PARSE_ERROR_MESSAGE.

    Not dispatched on non-list input: `raw` is returned unchanged for a dict, a scalar,
    or any other shape, so the caller's own dict guard remains the single place that
    decides pass/fail.

    Args:
        raw: Whatever the mode-specific extraction (_extract_json / _extract_yaml)
            produced.
        _depth: Recursion counter for nested lists. Callers never pass this.

    Returns:
        The first dict reachable per the rule above, or `raw` unchanged.
    """
    if not isinstance(raw, list) or _depth > _ARRAY_UNWRAP_MAX_DEPTH:
        return raw
    for element in raw:
        if isinstance(element, dict):
            if len(raw) > 1:
                logger.debug(f"Unwrapping first dict from a {len(raw)}-element array reply")
            return element
        if isinstance(element, list):
            inner = _unwrap_array_reply(element, _depth + 1)
            if isinstance(inner, dict):
                return inner
    return raw


def _cut_mid_string(completion: str) -> bool:
    """Whether a JSON reply stops inside a string before its first object or array closes.

    That is a reply cut off at max_tokens mid-value (stanfordnlp/dspy#1727). A reply that
    closes its structure is never cut, whatever follows it.
    """
    # ponytail: a reply cut mid-number or between values is not detected; it parses as
    # before. A finish_reason check would catch those, but parse() never sees one.
    text = _strip_leading_reasoning(completion)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return False
    depth = 0
    in_string = escaped = False
    for char in text[min(starts) :]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return False
    return in_string


def _drop_last_leaf(value: Any) -> Any:
    """``value`` without its last leaf in document order: the one a cut reply was writing.

    The key (or list item) is removed rather than nulled, so a required field goes missing
    and fails as missing instead of reading as ``"None"``. Never mutates ``value``.
    """
    if isinstance(value, dict) and value:
        *_, key = value
        if isinstance(value[key], dict | list) and value[key]:
            return {**value, key: _drop_last_leaf(value[key])}
        return {k: v for k, v in value.items() if k != key}
    if isinstance(value, list) and value:
        if isinstance(value[-1], dict | list) and value[-1]:
            return [*value[:-1], _drop_last_leaf(value[-1])]
        return value[:-1]
    return value


def _has_open_ended_mapping(signature: SignatureMeta) -> bool:
    """Return True if any output field is an open-ended mapping (``dict[...]``).

    Vendored from DSPy 3.3.1 ``json_adapter.py:28-38``. Structured Outputs require
    explicit properties, so such fields are incompatible. The name deliberately matches
    upstream's so the two can be diffed mechanically.

    Args:
        signature: The DSPy signature to inspect.

    Returns:
        True if at least one output field is annotated with a ``dict`` origin.
    """
    return any(get_origin(f.annotation) is dict for f in signature.output_fields.values())


def _has_tool_calls_output(signature: SignatureMeta) -> bool:
    """Return True if any OUTPUT field is annotated ``dspy.ToolCalls``.

    NARROW predicate - upstream's inline check (``json_adapter.py:60``). Used by JSON
    mode only, where byte-for-byte upstream parity is a hard requirement.

    Args:
        signature: The DSPy signature to inspect.

    Returns:
        True if at least one output field is annotated ``ToolCalls``.
    """
    return any(f.annotation == ToolCalls for f in signature.output_fields.values())


def _has_tool_fields(signature: SignatureMeta) -> bool:
    """Return True if the signature carries tools at all.

    That is: a ``dspy.Tool`` or ``list[dspy.Tool]`` INPUT field, or a ``dspy.ToolCalls``
    OUTPUT field.

    BROAD predicate - ours, not upstream's. Used by JSONish mode only. A ``Tool`` input
    alone is enough for DSPy to inject ``lm_kwargs["tools"]``
    (``dspy/adapters/base.py:114``), and tools + ``json_object`` is the provider
    combination that 400s, so JSONish backs off for the input-only case too.

    Args:
        signature: The DSPy signature to inspect.

    Returns:
        True if the signature declares tools on either side.
    """
    if _has_tool_calls_output(signature):
        return True
    for field in signature.input_fields.values():
        annotation = field.annotation
        if annotation == Tool:
            return True
        args = getattr(annotation, "__args__", ())
        if get_origin(annotation) is list and args and args[0] == Tool:
            return True
    return False


def _get_structured_outputs_response_format(
    signature: SignatureMeta,
    use_native_function_calling: bool = True,
) -> type[pydantic.BaseModel]:
    """
    Builds a Pydantic model from a DSPy signature's output_fields for structured outputs.
    Vendored: kept as our own copy even after DSPy 3.3.1's upstream helper changed shape,
    per lsl-2026-09-04-009 decision D6 (the package always owns this builder).
    """
    for name, field in signature.output_fields.items():
        annotation = field.annotation
        if get_origin(annotation) is dict:
            raise ValueError(
                f"Field '{name}' has an open-ended mapping type which is not supported by Structured Outputs."  # noqa: E501
            )

    fields = {}
    for name, field in signature.output_fields.items():
        annotation = field.annotation
        if use_native_function_calling and annotation == ToolCalls:
            continue
        default = field.default if hasattr(field, "default") else ...
        fields[name] = (annotation, default)

    pydantic_model = pydantic.create_model(
        "DSPyProgramOutputs",
        __config__=pydantic.ConfigDict(extra="forbid"),
        **fields,  # type: ignore
    )

    schema = pydantic_model.model_json_schema()

    # Remove DSPy-specific metadata
    for prop in schema.get("properties", {}).values():
        prop.pop("json_schema_extra", None)

    def strip_extensions(node: Any) -> None:
        """Recursively drop Pydantic ``x-*`` vendor keys, which strict-schema providers
        (e.g. Bedrock) reject with a 400 (stanfordnlp/dspy#9686). The keys of a
        ``properties``/``$defs`` mapping are names, not keywords, so they are kept."""
        if isinstance(node, list):
            for item in node:
                strip_extensions(item)
        elif isinstance(node, dict):
            for key in [k for k in node if k.startswith("x-")]:
                del node[key]
            for key, value in node.items():
                if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
                    for sub_schema in value.values():
                        strip_extensions(sub_schema)
                else:
                    strip_extensions(value)

    strip_extensions(schema)

    def enforce_required(schema_part: dict[str, Any]) -> None:
        """Recursively enforce required fields for OpenAI Structured Outputs."""
        if schema_part.get("type") == "object":
            props = schema_part.get("properties")
            if props is not None:
                schema_part["required"] = list(props.keys())
                schema_part["additionalProperties"] = False
                for sub_schema in props.values():
                    if isinstance(sub_schema, dict):
                        enforce_required(sub_schema)
            else:
                schema_part["properties"] = {}
                schema_part["required"] = []
                schema_part["additionalProperties"] = False
        if schema_part.get("type") == "array" and isinstance(schema_part.get("items"), dict):
            enforce_required(schema_part["items"])
        for key in ("$defs", "definitions"):
            if key in schema_part:
                for def_schema in schema_part[key].values():
                    enforce_required(def_schema)

    enforce_required(schema)
    pydantic_model.model_json_schema = lambda *args, **kwargs: schema

    return pydantic_model  # type: ignore[no-any-return]
