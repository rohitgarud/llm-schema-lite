import dataclasses
import enum
import inspect
import json
import logging
import re
from typing import Any, Literal, get_origin

import dspy
import pydantic
from dspy.adapters.chat_adapter import FieldInfoWithName
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types import Type as DSPyType
from dspy.adapters.types.history import History as DSPyHistory
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.adapters.utils import (
    apply_output_field_defaults,
    format_field_value,
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
      EXPERIMENTAL - see lsl-2026-09-04-006 for known formatter defects.
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


class StructuredOutputAdapter(JSONAdapter):  # type: ignore[misc]
    """
    Unified adapter for structured output with multiple format support.

    Key Features:
    - JSON mode: Standard JSON output with verbose schemas
        (compatible with OpenAI structured outputs)
    - JSONish mode: JSON output with simplified BAML-like
        schemas (60-85% token reduction from verbose JSON schemas)
    - YAML mode (experimental): YAML output with simplified schemas. Known formatter
      defects are tracked in lsl-2026-09-04-006.
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
            offered to the coercion rescue (see _coerce_field_value); if that also
            fails and parse_config.partial is True, the field is dropped and refilled
            by apply_output_field_defaults. Coercion/drop events are logged at DEBUG
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

    def _comment_prefix(self) -> str:
        """Return the formatter's comment prefix for the current output mode."""
        return "#" if self.output_mode == OutputMode.YAML else "//"

    def _legend_line(self) -> str:
        """Return the once-per-prompt required-marker legend line for this mode."""
        marker = self._effective_formatter_config().required_marker
        return f"{self._comment_prefix()} Fields marked with {marker} are required"

    @staticmethod
    def _legend_needed(schema_text: str, legend_line: str, marker: str) -> bool:
        """Report whether ``schema_text`` wants the required-marker legend.

        True iff a whole line equals ``legend_line`` (the literal the formatters emit
        per simplified schema), OR any line carries the marker in field-key position.
        The second disjunct is required because the JSON-schema-dict path emits ``*``
        markers without ever emitting a legend line of its own.
        """
        lines = schema_text.splitlines()
        if any(line.strip() == legend_line for line in lines):
            return True
        pattern = re.compile(r"^\s*[^\s:]+" + re.escape(marker) + r"\s*:")
        return any(pattern.match(line) for line in lines)

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

        # 2. Unconditional carve-out for dspy.Type subclasses and dspy.History, which
        #    upstream JSONAdapter also emits nothing for (dspy.Image, dspy.Audio,
        #    dspy.Tool, ToolCalls, dspy.Code are Type subclasses; History is not).
        if inspect.isclass(field_type) and issubclass(field_type, DSPyType | DSPyHistory):
            return _FieldBlock(name=name, note_text=None, schema_text=None)

        voice = "the value you produce " if role == "output" else "this value "

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
            # list[Model], Model | None, str | None and dict[str, Model]. The array
            # unwrap duplicates a container token the jsonish/YAML formatters will own
            # once the root-array rendering follow-up lands; delete it then.
            try:
                schema = TypeAdapter(annotation).json_schema()
                if schema.get("type") == "array" and "items" in schema:
                    inner = dict(schema["items"])
                    if "$defs" in schema:
                        inner["$defs"] = schema["$defs"]
                    body = simplify_schema(
                        inner, config=config, format_type=format_type
                    ).to_string()
                    if format_type == "yaml":
                        text = "- " + body.replace("\n", "\n  ")
                    else:
                        text = "[\n" + body + "\n]"
                else:
                    text = simplify_schema(
                        schema, config=config, format_type=format_type
                    ).to_string()
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
        return self._render_sections(blocks)

    def _render_sections(self, blocks: list[_FieldBlock]) -> str:
        """Render ``blocks`` as ``[[ ## name ## ]]`` sections joined by a blank line."""
        rendered: list[str] = []
        for block in blocks:
            text = f"[[ ## {block.name} ## ]]\n{{{block.name}}}"
            if block.note_text is not None:
                text += f"{' ' * 8}# note: {block.note_text}"
            if block.schema_text is not None:
                text += f"\n{block.schema_text}"
            rendered.append(text)
        return "\n\n".join(rendered).strip()

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
        """Specify output format requirements based on mode."""

        def type_info(v: Any) -> str:
            return (
                f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
                if v.annotation is not str
                else ""
            )

        base_message = "Respond with "

        if self.output_mode == OutputMode.YAML:
            base_message += "a YAML-style object "
        else:
            # Both JSON and JSONish output JSON
            base_message += "a JSON object "

        base_message += "in the following order of fields: "
        base_message += ", then ".join(
            f"`{f}`{type_info(v)}" for f, v in signature.output_fields.items()
        )
        base_message += "."
        return base_message

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

        - User role: Always uses DSPy's [[ ## field ## ]] format (unchanged)
        - Assistant role: JSON for JSON/JSONish modes, YAML for YAML mode
        """
        if role == "user":
            # Input formatting - keep DSPy standard format
            output = []
            for field, field_value in fields_with_values.items():
                formatted_field_value = format_field_value(field_info=field.info, value=field_value)
                output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")
            return "\n\n".join(output).strip()
        else:
            # Output formatting - based on mode
            d = {k.name: v for k, v in fields_with_values.items()}

            if self.output_mode in (OutputMode.JSON, OutputMode.JSONISH):
                # Both JSON and JSONish output JSON format
                return json.dumps(serialize_for_json(d), indent=2)
            elif self.output_mode == OutputMode.YAML:
                return self._format_yaml_output(d)
            else:
                # Fallback to JSON
                return json.dumps(serialize_for_json(d), indent=2)

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: Any = None,
    ) -> str:
        """Format assistant message content based on output mode."""
        fields_with_values = {
            FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
            for k, v in signature.output_fields.items()
        }
        return self.format_field_with_value(fields_with_values, role="assistant")

    # ==================== Format-Specific Output Methods ====================

    def _format_yaml_output(self, data: dict[str, Any]) -> str:
        """Format output as YAML-style."""
        try:
            import yaml

            serialized = serialize_for_json(data)
            return yaml.dump(serialized, default_flow_style=False, allow_unicode=True)  # type: ignore[no-any-return, unused-ignore]
        except ImportError:
            logger.warning("PyYAML not installed, falling back to JSON-like YAML")
            # Fallback to your current implementation
            serialized = serialize_for_json(data)
            lines = []
            for key, value in serialized.items():
                if isinstance(value, dict | list):
                    lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
                else:
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)
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

        Delegates extraction to _extract_json / _extract_yaml and field-building to the
        shared _build_output_fields pipeline; parse/_parse_json/_parse_yaml keep their
        existing names so nothing that references them by name breaks.
        """
        # JSON and JSONish both parse as JSON
        if self.output_mode in (OutputMode.JSON, OutputMode.JSONISH):
            return self._parse_json(signature, completion)
        elif self.output_mode == OutputMode.YAML:
            return self._parse_yaml(signature, completion)
        else:
            # Fallback
            return self._parse_json(signature, completion)

    def _parse_json(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse JSON completion (used for both JSON and JSONish modes).

        Thin wrapper: mode-specific extraction via _extract_json, then the mode-shared
        field pipeline via _build_output_fields.
        """
        raw = self._extract_json(signature, completion)
        return self._build_output_fields(signature, completion, raw)

    def _parse_yaml(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse YAML completion (with a narrow JSON-extraction rescue).

        Thin wrapper: mode-specific extraction via _extract_yaml, then the mode-shared
        field pipeline via _build_output_fields. Unlike the previous implementation, no
        `except Exception` here ever swallows a completeness-check AdapterParseError
        raised further down the pipeline -- that defect is what this rewrite removes
        structurally, not just at this call site.
        """
        raw = self._extract_yaml(signature, completion)
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

        Mirrors upstream JSONAdapter.parse (DSPy 3.3.1) with the coercion rescue
        (_coerce_field_value) as the only insertion. Both _parse_json and _parse_yaml
        funnel through it after their mode-specific extraction.

        Invariants that any change to this method must preserve:
          - self.parse_config is None => structurally upstream-equivalent. The first
            line of the except branch below re-raises before any new machinery is
            reachable; this must not depend on config values happening to be inert.
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

        out: dict[str, Any] = {}
        for k, v in fields.items():
            annotation = signature.output_fields[k].annotation
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
                if self.parse_config.partial:
                    logger.debug(f"Dropping unparseable output field {k!r} (partial=True)")
                    continue
                raise

        out = apply_output_field_defaults(signature, out)

        if out.keys() != signature.output_fields.keys():
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
        fail if it is widened -- point any future change at that test by name.

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
