import enum
import inspect
import json
import logging
from typing import Any, Literal, get_origin

import pydantic
from dspy.adapters.chat_adapter import FieldInfoWithName
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    parse_value,
    serialize_for_json,
)
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.signatures.utils import get_dspy_field_type
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

# Import your schema-lite formatters
from llm_schema_lite import loads, simplify_schema

logger = logging.getLogger(__name__)


class OutputMode(enum.Enum):
    """
    Output format modes for structured responses.

    - JSON: LLM outputs JSON, schema uses full model_json_schema() (verbose)
    - JSONISH: LLM outputs JSON, schema uses simplified BAML-like format (token-efficient)
    - YAML: LLM outputs YAML, schema uses simplified YAML format (token-efficient)
    """

    JSON = "json"
    JSONISH = "jsonish"
    YAML = "yaml"


class StructuredOutputAdapter(JSONAdapter):  # type: ignore[misc]
    """
    Unified adapter for structured output with multiple format support.

    Key Features:
    - JSON mode: Standard JSON output with verbose schemas
        (compatible with OpenAI structured outputs)
    - JSONish mode: JSON output with simplified BAML-like
        schemas (60-85% token reduction from verbose JSON schemas)
    - YAML mode: YAML output with simplified schemas
    - Simplified schemas for complex input fields (Pydantic models)
    - Robust parsing with fallback mechanisms

    Args:
        callbacks: Optional list of callbacks for monitoring
        use_native_function_calling: Whether to use native function calling
        output_mode: Output format mode (JSON, JSONISH, or YAML)
        include_input_schemas: Whether to include simplified schemas for complex input types
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = True,
        output_mode: OutputMode = OutputMode.JSONISH,
        include_input_schemas: bool = True,
    ):
        super().__init__(
            callbacks=callbacks, use_native_function_calling=use_native_function_calling
        )
        self.output_mode = output_mode
        self.include_input_schemas = include_input_schemas

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
        """
        Format field structure with optional schema simplification.

        - JSON mode: Uses full model_json_schema() (verbose but compatible)
        - JSONish mode: Uses simplified BAML-like schema (token-efficient)
        - YAML mode: Uses simplified YAML schema (token-efficient)
        """
        parts = []
        parts.append(
            "All interactions will be structured in the following way,"
            " with the appropriate values filled in."
        )

        def format_signature_fields_for_instructions(
            fields: dict[str, FieldInfo], role: str
        ) -> str:
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): self._translate_field_type(
                        field_name, field_info
                    )
                    for field_name, field_info in fields.items()
                },
                role=role,
            )

        parts.append("Inputs will have the following structure:")
        parts.append(format_signature_fields_for_instructions(signature.input_fields, role="user"))

        # Output message based on mode
        if self.output_mode == OutputMode.YAML:
            parts.append("Outputs will be in YAML format with the following fields.")
        else:
            # Both JSON and JSONish output JSON
            parts.append("Outputs will be a JSON object with the following fields.")

        parts.append(
            format_signature_fields_for_instructions(signature.output_fields, role="assistant")
        )
        return "\n\n".join(parts).strip()

    def _translate_field_type(self, field_name: str, field_info: FieldInfo) -> str:
        """
        Translate field type with mode-specific schema representation.
        This is the key method that differentiates JSON vs JSONish vs YAML modes.

        - JSON mode: Uses full JSON schema (verbose)
        - JSONish mode: Uses simplified schema (BAML-like)
        - YAML mode: Uses simplified schema (YAML-style)
        """
        field_type = field_info.annotation

        # For input fields or string types, use minimal description
        if field_type is str:
            desc = ""
        elif field_type is bool:
            desc = "must be True or False"
        elif field_type in (int, float):
            desc = f"must be a single {field_type.__name__} value"
        elif inspect.isclass(field_type) and issubclass(field_type, enum.Enum):
            enum_vals = "; ".join(str(member.value) for member in field_type)
            desc = f"must be one of: {enum_vals}"
        elif hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:  # type: ignore[union-attr]
            desc = f"must exactly match (no extra characters) one of: {'; '.join([str(x) for x in field_type.__args__])}"  # type: ignore[union-attr] # noqa: E501
        else:
            # Complex types - this is where mode-specific logic applies
            desc = self._get_complex_type_description(field_type, field_info)  # type: ignore[arg-type]

        desc = (" " * 8) + f"# note: the value you produce {desc}" if desc else ""
        return f"{{{field_name}}}{desc}"

    def _get_complex_type_description(self, field_type: type[Any], field_info: FieldInfo) -> str:
        """
        Get description for complex types with mode-specific schema representation.

        Key difference:
        - JSON mode: Always uses full model_json_schema() (verbose)
        - JSONish/YAML mode: Uses simplified schema from llm-schema-lite (token-efficient)
        """
        is_input_field = get_dspy_field_type(field_info) == "input"

        # For JSON mode, always use full JSON schema (no simplification)
        if self.output_mode == OutputMode.JSON:
            try:
                schema = TypeAdapter(field_type).json_schema()
                return f"must adhere to the JSON schema: {json.dumps(schema, ensure_ascii=False)}"
            except Exception:
                return f"must be a valid {get_annotation_name(field_type)}"

        # For JSONish and YAML modes, use simplified schemas
        # Also for input fields when include_input_schemas is enabled
        should_simplify = self.output_mode in (OutputMode.JSONISH, OutputMode.YAML) or (
            is_input_field and self.include_input_schemas
        )

        if should_simplify:
            try:
                # Determine format type for simplification
                if self.output_mode == OutputMode.YAML:
                    format_type = "yaml"
                else:
                    format_type = "jsonish"  # Default for JSONish mode and input fields

                # Simplify using llm-schema-lite
                format_type_literal: Literal["jsonish", "typescript", "yaml"] = format_type  # type: ignore
                simplified = simplify_schema(
                    field_type, format_type=format_type_literal, include_metadata=False
                )
                schema_str = simplified.to_string()

                # Adjust wording for input vs output fields
                if is_input_field:
                    return f"will follow the schema: {schema_str}"
                else:
                    return f"must be parseable according to the following schema: {schema_str}"
            except Exception as e:
                logger.debug(f"Failed to simplify schema for {field_type}: {e}")
                # Fallback to full JSON schema
                try:
                    schema = TypeAdapter(field_type).json_schema()
                    return (
                        f"must adhere to the JSON schema: {json.dumps(schema, ensure_ascii=False)}"
                    )
                except Exception:
                    return f"must be a valid {get_annotation_name(field_type)}"
        else:
            # Fallback for other cases
            try:
                schema = TypeAdapter(field_type).json_schema()
                return f"must adhere to the JSON schema: {json.dumps(schema, ensure_ascii=False)}"
            except Exception:
                return f"must be a valid {get_annotation_name(field_type)}"

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
        fields = loads(completion, mode="json")

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
            fields = loads(completion, mode="yaml")

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
