import dataclasses
import enum
import inspect
import json
import logging
import re
from typing import Any, Literal, get_origin

import pydantic
from dspy.adapters.chat_adapter import FieldInfoWithName
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types import Type as DSPyType
from dspy.adapters.types.history import History as DSPyHistory
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    parse_value,
    serialize_for_json,
)
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

# Use llm_schema_lite for schema simplification and robust parsing
from llm_schema_lite import FormatterConfig, loads, simplify_schema

logger = logging.getLogger(__name__)


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
    ):
        super().__init__(
            callbacks=callbacks, use_native_function_calling=use_native_function_calling
        )
        self.output_mode = output_mode
        self.include_input_schemas = include_input_schemas
        self.max_recursion_depth = max_recursion_depth
        self.formatter_config = formatter_config
        self.prompt_layout = prompt_layout

    # ==================== Core Call Methods ====================

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Synchronous call with format-specific handling."""
        result = self._json_adapter_call_common(
            lm, lm_kwargs, signature, demos, inputs, super().__call__
        )
        if result:
            return result  # type: ignore[no-any-return]

        # For JSON mode, try structured outputs (OpenAI native)
        if self.output_mode == OutputMode.JSON:
            try:
                structured_output_model = _get_structured_outputs_response_format(
                    signature, self.use_native_function_calling
                )
                lm_kwargs["response_format"] = structured_output_model
                return super().__call__(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]
            except Exception:
                logger.warning("Failed to use structured output format, falling back to JSON mode.")
                lm_kwargs["response_format"] = {"type": "json_object"}
                return super().__call__(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]
        else:
            # For JSONish and YAML modes
            if self.output_mode == OutputMode.JSONISH:
                lm_kwargs["response_format"] = {"type": "json_object"}
            # For YAML, we don't set response_format (let LLM output YAML naturally)
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Asynchronous call with format-specific handling."""
        result = self._json_adapter_call_common(
            lm, lm_kwargs, signature, demos, inputs, super().acall
        )
        if result:
            return await result  # type: ignore[no-any-return]

        # For JSON mode, try structured outputs (OpenAI native)
        if self.output_mode == OutputMode.JSON:
            try:
                structured_output_model = _get_structured_outputs_response_format(
                    signature, self.use_native_function_calling
                )
                lm_kwargs["response_format"] = structured_output_model
                return await super().acall(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]
            except Exception:
                logger.warning("Failed to use structured output format, falling back to JSON mode.")
                lm_kwargs["response_format"] = {"type": "json_object"}
                return await super().acall(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]
        else:
            # For JSONish and YAML modes
            if self.output_mode == OutputMode.JSONISH:
                lm_kwargs["response_format"] = {"type": "json_object"}
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]

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
        """Parse JSON completion (used for both JSON and JSONish modes)."""
        # Parse with llm-schema-lite for robustness
        # (includes markdown extraction and JSON object extraction)
        fields = loads(completion, mode="json", repair=True)

        if not isinstance(fields, dict):
            raise AdapterParseError(
                adapter_name="StructuredOutputAdapter",
                signature=signature,
                lm_response=completion,
                message="LM response cannot be serialized to a JSON object.",
            )

        # Filter to only output fields
        fields = {k: v for k, v in fields.items() if k in signature.output_fields}

        # Cast values to expected types
        for k, v in fields.items():
            if k in signature.output_fields:
                fields[k] = parse_value(v, signature.output_fields[k].annotation)

        # Validate all fields present
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="StructuredOutputAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )

        return fields

    def _parse_yaml(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """
        Parse YAML completion.
        Convert YAML to dict then process normally.
        """
        try:
            # Use llm-schema-lite for robust YAML parsing with markdown extraction
            fields = loads(completion, mode="yaml", repair=True)

            if not isinstance(fields, dict):
                raise ValueError("YAML did not parse to a dictionary")

            # Filter and cast
            fields = {k: v for k, v in fields.items() if k in signature.output_fields}

            for k, v in fields.items():
                if k in signature.output_fields:
                    fields[k] = parse_value(v, signature.output_fields[k].annotation)

            if fields.keys() != signature.output_fields.keys():
                raise AdapterParseError(
                    adapter_name="StructuredOutputAdapter",
                    signature=signature,
                    lm_response=completion,
                    parsed_result=fields,
                )

            return fields
        except Exception as e:
            logger.debug(f"YAML parsing failed: {e}, falling back to JSON parsing")
            # Fallback to JSON parsing
            return self._parse_json(signature, completion)

    # ==================== Fine-tuning Support ====================

    def format_finetune_data(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        """Format data for fine-tuning (not yet implemented)."""
        raise NotImplementedError("Fine-tuning data formatting not yet implemented")


# ==================== Helper Functions ====================


def _get_structured_outputs_response_format(
    signature: SignatureMeta,
    use_native_function_calling: bool = True,
) -> type[pydantic.BaseModel]:
    """
    Builds a Pydantic model from a DSPy signature's output_fields for structured outputs.
    (Copied from JSONAdapter for compatibility with DSPy 3.0.3)
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
