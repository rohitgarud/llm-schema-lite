"""Base formatter abstract class for schema formatters."""

import json
import re
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, Literal

from ..schema_normalization import normalize_schema_titles
from .config import FormatterConfig

ContainerKind = Literal["mapping", "tuple", "list", "any", "object", "scalar"]

_COMPOSITION_KEYS = ("$ref", "enum", "const", "anyOf", "oneOf", "allOf", "not")

DEFERRED_OPEN: Final[str] = "⟪"  # U+27EA MATHEMATICAL LEFT DOUBLE ANGLE BRACKET
DEFERRED_CLOSE: Final[str] = "⟫"  # U+27EB MATHEMATICAL RIGHT DOUBLE ANGLE BRACKET
DEFERRED_TAG: Final[str] = "lsl"  # literal infix, before the per-instance nonce


@dataclass(frozen=True)
class ContainerShape:
    """What shape a property schema is, with the sub-schemas a renderer needs.

    Purely descriptive: carries no rendered text and no formatter state.
    """

    kind: ContainerKind
    value_schema: dict[str, Any] | None = None
    key_schema: dict[str, Any] | None = None
    prefix_schemas: tuple[dict[str, Any], ...] = ()
    rest_schema: dict[str, Any] | None = None
    item_schema: dict[str, Any] | None = None


def _as_schema(value: Any) -> dict[str, Any] | None:
    """Return ``value`` when it is a non-empty schema dict, else ``None``."""
    if isinstance(value, dict) and value:
        return value
    return None


def classify_container(schema: Any) -> ContainerShape:
    """Classify one JSON-Schema node. Pure; never raises; never reads formatter state."""
    # Rule 1 -- nothing to classify.
    if not isinstance(schema, dict) or not schema:
        return ContainerShape(kind="any")

    # Rule 2 -- composition / reference keywords own their own renderers.
    for key in _COMPOSITION_KEYS:
        if key in schema:
            return ContainerShape(kind="scalar")

    prefix_items = schema.get("prefixItems")
    items = schema.get("items")

    # Rule 3 -- 2020-12 tuple. prefixItems beats items.
    if isinstance(prefix_items, list) and prefix_items:
        return ContainerShape(
            kind="tuple",
            prefix_schemas=tuple(s for s in prefix_items if isinstance(s, dict)),
            rest_schema=_as_schema(items),
        )

    # Rule 4 -- draft-7 tuple: a list-valued ``items`` is never a recursion target.
    if isinstance(items, list) and items:
        additional_items = schema.get("additionalItems")
        return ContainerShape(
            kind="tuple",
            prefix_schemas=tuple(s for s in items if isinstance(s, dict)),
            rest_schema=additional_items if isinstance(additional_items, dict) else None,
        )

    # Rule 5 -- homogeneous list.
    if schema.get("type") == "array":
        return ContainerShape(kind="list", item_schema=_as_schema(items))

    additional_properties = schema.get("additionalProperties")
    is_object_ish = schema.get("type") == "object" or "additionalProperties" in schema
    # Decision C1: a schema that declares ``properties`` (or ``patternProperties``)
    # is an object, never a mapping. This clause must not be loosened.
    is_open = is_object_ish and not schema.get("properties") and not schema.get("patternProperties")

    # Rule 6 -- mapping with a typed value.
    if is_open and isinstance(additional_properties, dict) and additional_properties:
        return ContainerShape(
            kind="mapping",
            value_schema=additional_properties,
            key_schema=_as_schema(schema.get("propertyNames")),
        )

    # Rule 7 -- mapping open to any value.
    if is_open and additional_properties is True:
        return ContainerShape(
            kind="mapping",
            value_schema=None,
            key_schema=_as_schema(schema.get("propertyNames")),
        )

    # Rule 8 -- closed / structured object.
    if schema.get("type") == "object":
        return ContainerShape(kind="object")

    # Rule 9 -- typed leaf.
    if "type" in schema:
        return ContainerShape(kind="scalar")

    # Rule 10 -- typeless but non-empty: ``Any`` carrying only annotations.
    return ContainerShape(kind="any")


def format_literal_value(value: Any) -> str:
    """Render one enum/const/Literal value as it appears inside a ``one of: ...`` list.

    Args:
        value: A raw value from a JSON Schema ``enum``/``const`` list, as produced by
            ``model_json_schema()`` (``bool``, ``int``, ``float``, ``str``, or ``None``).

    Returns:
        ``"true"``/``"false"`` for a bool (tested BEFORE ``int``, since ``bool`` is an
        ``int`` subclass in Python); ``"null"`` for ``None``; ``str(value)`` for a plain
        ``int``/``float``; ``json.dumps(value, ensure_ascii=False)`` for a ``str``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(str(value), ensure_ascii=False)


def infer_json_type(values: Sequence[Any]) -> str | None:
    """Infer a single JSON Schema type name for an enum node that carries no ``type`` key.

    Args:
        values: The raw ``enum`` value list (or a single-element list built from a
            ``const``). ``None`` entries are ignored for the purpose of inference.

    Returns:
        The shared JSON Schema type name (``"string"``, ``"integer"``, ``"number"``,
        ``"boolean"``) when every non-``None`` value has the same Python type, else
        ``None`` (heterogeneous, or nothing but ``None`` values).
    """
    names: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            names.add("boolean")
        elif isinstance(value, int):
            names.add("integer")
        elif isinstance(value, float):
            names.add("number")
        elif isinstance(value, str):
            names.add("string")
        else:
            return None
        if len(names) > 1:
            return None
    if len(names) != 1:
        return None
    return names.pop()


class BaseFormatter(ABC):
    """
    Abstract base class for schema formatters.

    All schema formatters must inherit from this class and implement
    the required methods.

    Usage:
        There are two common patterns for implementing formatters:

        Pattern A (TypeScript/YAML formatters):
            Subclasses may call process_schema() and use its return value,
            then implement transform_schema() to format the processed data.
            This pattern leverages the shared processing logic in the base class.

        Pattern B (JSONish formatter):
            Subclasses may ignore process_schema() and implement transform_schema()
            entirely with their own logic. This pattern is useful when the formatter
            requires fundamentally different processing approach.
    """

    # Common regex pattern for $ref processing
    # REF_PATTERN = re.compile(r"#/\$defs/(.+)$", re.IGNORECASE)
    REF_PATTERN = re.compile(r"#/(?:definitions|\$defs)/(.+)$", re.IGNORECASE)

    # Common metadata mapping for all formatters
    METADATA_MAP: dict[str, Callable[[Any], str]] = {
        "title": lambda v: str(v),
        "default": lambda v: f"(defaults to {v})",
        "description": lambda v: str(v),
        "pattern": lambda v: f"pattern: {v}",
        "minimum": lambda v: f"min: {v}",
        "maximum": lambda v: f"max: {v}",
        "minLength": lambda v: f"minLength: {v}",
        "maxLength": lambda v: f"maxLength: {v}",
        "format": lambda v: f"format: {v}",
        "multipleOf": lambda v: f"multipleOf: {v}",
        "const": lambda v: f"const: {v}",
        "if": lambda v: f"if: {v}",
        "then": lambda v: f"then: {v}",
        "else": lambda v: f"else: {v}",
        "contains": lambda v: f"contains: {v}",
        "dependencies": lambda v: f"dependencies: {v}",
        "patternProperties": lambda v: f"patternProperties: {v}",
        "propertyNames": lambda v: f"propertyNames: {v}",
        "unevaluatedProperties": lambda v: f"unevaluatedProperties: {v}",
        "minItems": lambda v: f"minItems: {v}",
        "maxItems": lambda v: f"maxItems: {v}",
        "minProperties": lambda v: f"minProperties: {v}",
        "maxProperties": lambda v: f"maxProperties: {v}",
        "exclusiveMinimum": lambda v: f"exclusiveMin: {v}",
        "exclusiveMaximum": lambda v: f"exclusiveMax: {v}",
        "uniqueItems": lambda v: "unique items" if v else "",
        "additionalItems": lambda v: f"additionalItems: {v}" if isinstance(v, dict) else "",
    }

    def __init__(
        self,
        schema: dict[str, Any],
        config: FormatterConfig | None = None,
        include_metadata: bool | None = None,
    ):
        """
        Initialize the formatter.

        Args:
            schema: JSON schema from Pydantic model_json_schema.
            config: FormatterConfig for customizing formatter behavior.
            include_metadata: Deprecated. Use config.include_metadata instead.
        """
        schema = normalize_schema_titles(schema)
        self.schema = schema
        # Handle config - use provided config or create default
        self.config = config if config is not None else FormatterConfig()

        # Handle backward compatibility for include_metadata parameter
        if include_metadata is not None:
            # If legacy parameter is provided, it takes precedence
            self.config.include_metadata = include_metadata

        # Backward compatibility: expose include_metadata as instance attribute
        self.include_metadata = self.config.include_metadata

        # Metadata inclusion configuration for fine-grained control
        self._metadata_inclusion = self.config.metadata_inclusion

        self.defs = schema.get("$defs", schema.get("definitions", {}))
        self.properties = schema.get("properties", {})
        self.required_fields = set(schema.get("required", []))
        self._ref_cache: dict[str, str] = {}
        # Optional cache of processed schema data; may be set by subclasses in
        # transform_schema() for reuse. TypeScript/YAML check this before re-processing.
        self._processed_data: dict[str, Any] | None = None

        # Unconditional safety net; also tiers anyOf/oneOf member caps.
        self._global_expansion_budget = 150  # Max total $ref expansions across entire schema
        self._global_expansion_count = 0  # Track total expansions
        self._ref_expansion_path: list[str] = []  # Active $ref expansion path (the depth counter)
        self._truncation_epoch = 0  # Monotonic count of recursion truncations
        self._root_ref_key: str | None = None  # def name adopted by _adopt_root_ref()
        self._nested_required_stack: list[set[str]] = []

        # Deferred-comment mechanism: the comment body for an enum/const's "one of: ..."
        # travels out of band in this slot table so it can survive json.dumps /
        # _remove_quotes and PyYAML's plain-scalar quoting rules.
        self._deferred_bodies: list[str] = []
        self._deferred_index: dict[str, int] = {}
        self._deferred_nonce: str = secrets.token_hex(4)
        self._deferred_pattern: re.Pattern[str] = re.compile(
            f"{DEFERRED_OPEN}{DEFERRED_TAG}{self._deferred_nonce}\\.(\\d+){DEFERRED_CLOSE}"
        )

        # Pre-warm cache for common patterns

    def _adopt_root_ref(self) -> str | None:
        """Adopt a root-level ``$ref``'s definition as the effective root.

        Pydantic emits ``{"$defs": ..., "$ref": "#/$defs/T"}`` (no ``properties``) for
        every root model that participates in a cycle. Without this, YAML/TypeScript
        fall into their "no properties" branch and render ``{}`` / ``interface Schema {}``.

        Mutates ``self.properties`` / ``self.required_fields`` and returns the def name,
        or returns None when the schema is not a bare root ``$ref`` to an object def.
        """
        ref_str = self.schema.get("$ref", "")
        if not ref_str or self.properties:
            return None
        ref_match = self.REF_PATTERN.search(ref_str)
        if not ref_match:
            return None
        ref_key = ref_match.group(1)
        ref_def = self.defs.get(ref_key)
        if not isinstance(ref_def, dict) or not ref_def.get("properties"):
            return None
        self.properties = ref_def["properties"]
        self.required_fields = set(ref_def.get("required", []))
        return ref_key

    @contextmanager
    def _expanding(self, ref_key: str | None) -> Iterator[None]:
        """Treat a block rendered outside ``process_ref`` as an expansion of ``ref_key``.

        Used for the adopted root ``$ref`` body and for each ``$defs`` section, so those
        bodies count toward ``max_recursion_depth`` exactly like an inline expansion.
        """
        if ref_key is None:
            yield
            return
        self._ref_expansion_path.append(ref_key)
        try:
            yield
        finally:
            if self._ref_expansion_path and self._ref_expansion_path[-1] == ref_key:
                self._ref_expansion_path.pop()

    def recursion_placeholder(self, type_name: str) -> str:
        """Placeholder token emitted where a recursive $ref is truncated."""
        return f"object  {self.comment_prefix} recursive: {type_name}"

    def _reset_ref_state(self) -> None:
        """Reset per-render $ref expansion state so the depth budget is deterministic."""
        self._ref_cache.clear()
        self._ref_expansion_path.clear()
        self._global_expansion_count = 0
        self._truncation_epoch = 0

    def process_schema(self) -> dict[str, Any]:
        """
        Process the schema and return the appropriate data structure.

        This method handles schema processing logic. It determines the appropriate
        processing method based on the schema structure.

        Returns:
            Dictionary containing processed schema data.
        """
        # Handle different top-level schema types
        if "$ref" in self.schema and not self.schema.get("properties"):
            # Handle $ref at top level
            ref_result = self.process_ref(self.schema)
            if ref_result == "object" or not ref_result:
                # Fallback for failed ref resolution
                ref_path = self.schema.get("$ref", "")
                if ref_path:
                    ref_match = self.REF_PATTERN.search(ref_path)
                    if ref_match:
                        ref_key = ref_match.group(1)
                        ref_def = self.defs.get(ref_key)
                        if ref_def and isinstance(ref_def, dict):
                            if "properties" in ref_def:
                                # Process properties from resolved definition
                                processed_props = self.process_properties(ref_def["properties"])
                                return {"schema": self.dict_to_string(processed_props, indent=0)}
                            elif "enum" in ref_def:
                                # Handle enum in resolved definition
                                return {"schema": self.process_enum(ref_def)}
                            elif "oneOf" in ref_def:
                                return {"schema": self.process_oneof(ref_def)}
                            elif "anyOf" in ref_def:
                                return {"schema": self.process_anyof(ref_def)}
                            elif "allOf" in ref_def:
                                return {"schema": self.process_allof(ref_def)}
                            elif "type" in ref_def:
                                return {"schema": self.process_type_value(ref_def)}
                        return {"schema": f"object  {self.comment_prefix}$ref: {ref_path}"}
                    return {"schema": f"object  {self.comment_prefix}$ref resolution failed"}
                return {"schema": "object"}
            else:
                return {"schema": ref_result}
        elif "properties" in self.schema and self.schema.get("properties"):
            # Handle object schemas with properties
            schema_type = self.schema.get("type")
            if schema_type in ("array", "string", "number", "integer", "boolean", "null"):
                # Let type handling take precedence for non-object types
                return {"schema": self.process_type_value(self.schema)}
            else:
                # Check if we have schema-level features that need to be included
                has_schema_features = any(
                    key in self.schema
                    for key in [
                        "dependencies",
                        "if",
                        "then",
                        "else",
                        "patternProperties",
                        "propertyNames",
                        "unevaluatedProperties",
                    ]
                )

                if has_schema_features:
                    # Use transform_schema() to include schema-level features
                    return {"schema": self.transform_schema()}
                else:
                    # Return processed properties dict for internal use
                    return self.process_properties(self.schema.get("properties", {}))
        elif "type" in self.schema:
            # Handle schemas with type but no properties
            if self.schema.get("type") == "object":
                return {"schema": self.transform_schema()}
            else:
                return {"schema": self.process_type_value(self.schema)}
        elif "oneOf" in self.schema:
            return {"schema": self.process_oneof(self.schema)}
        elif "anyOf" in self.schema:
            return {"schema": self.process_anyof(self.schema)}
        elif "allOf" in self.schema:
            return {"schema": self.process_allof(self.schema)}
        else:
            # Fallback for unknown schema types
            return {"schema": "object"}

    #     # Pre-warm cache for common patterns
    #     self._warm_cache()

    # def _warm_cache(self) -> None:
    #     """Pre-process common reference patterns."""
    #     for ref_key, ref_def in self.defs.items():
    #         # Only cache simple types that are NOT enums
    #         if (
    #             "type" in ref_def
    #             and ref_def["type"] in ["string", "integer", "number", "boolean"]
    #             and "enum" not in ref_def
    #         ):
    #             self._ref_cache[ref_key] = self.TYPE_MAP.get(ref_def["type"], ref_def["type"])

    def _is_problematic_schema(self, schema: dict[str, Any]) -> bool:
        """Detect schemas that are likely to cause issues."""
        # Check for very large schemas
        if len(str(schema)) > 50000:  # Very large schemas
            return True

        # Check for schemas with many definitions
        defs = schema.get("$defs", schema.get("definitions", {}))
        if len(defs) > 100:  # Too many definitions
            return True

        # Check for schemas with deep nesting
        def _check_depth(obj: Any, current_depth: int = 0, max_depth: int = 10) -> bool:
            if current_depth > max_depth:
                return True
            if isinstance(obj, dict):
                for value in obj.values():
                    if _check_depth(value, current_depth + 1, max_depth):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if _check_depth(item, current_depth + 1, max_depth):
                        return True
            return False

        return _check_depth(schema)

    def _resolve_nested_definition_path(self, ref_path: str) -> dict[str, Any] | None:
        """
        Priority 2: Resolve nested definition paths like #/definitions/636d/full.

        Args:
            ref_path: The $ref path (e.g., "#/definitions/636d/full")

        Returns:
            The resolved definition or None if not found
        """
        # Remove the leading #/ if present
        if ref_path.startswith("#/"):
            ref_path = ref_path[2:]

        # Split the path into parts
        parts = ref_path.split("/")

        # Start with the root definitions
        current = None
        if parts[0] in ("definitions", "$defs"):
            current = self.defs
            parts = parts[1:]  # Skip the definitions/$defs part
        else:
            return None

        # Navigate through the path
        for part in parts:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current if isinstance(current, dict) else None

    def get_available_metadata(self, value: dict[str, Any]) -> list[str]:
        """
        Get available metadata keys for a property.

        Args:
            value: The field definition containing metadata.

        Returns:
            List of available metadata keys.
        """
        available = [
            k
            for k in self.METADATA_MAP.keys()
            if k in value and not (k == "default" and value[k] is None)
        ]

        # Also check for underscore-prefixed versions (e.g., _format, _uniqueItems)
        for k in self.METADATA_MAP.keys():
            underscore_key = f"_{k}"
            if underscore_key in value and k not in available:
                available.append(k)  # Add without underscore for processing

        return available

    def format_metadata_parts(
        self, value: dict[str, Any], exclude: tuple[str, ...] = ()
    ) -> list[str]:
        """
        Format metadata parts for a property.

        Args:
            value: The field definition containing metadata.
            exclude: Metadata keys to omit from the result even if otherwise
                available and included by config (e.g. keys already rendered
                elsewhere by the caller, to avoid duplication).

        Returns:
            List of formatted metadata strings.
        """
        available_metadata = self.get_available_metadata(value)

        # Filter available metadata based on metadata_inclusion config
        filtered_metadata = [
            k for k in available_metadata if self._should_include_metadata(k) and k not in exclude
        ]

        formatted_parts = []

        for k in filtered_metadata:
            # Check both regular and underscore-prefixed keys
            actual_key = k if k in value else f"_{k}"

            if k == "contains":
                formatted_parts.append(f"contains: {self._format_contains(value[actual_key])}")
            elif k == "additionalItems" and classify_container(value).kind == "tuple":
                # Skip: the tuple renderer already consumed this as the variadic tail.
                continue
            elif k == "additionalItems":
                formatted_parts.append(
                    f"additionalItems: {self._format_type_simple(value[actual_key])}"
                )
            elif k == "if" and ("then" in value or "_then" in value):
                # Handle conditional logic as a single unit
                if_schema = value.get("if", {})
                then_schema = value.get("then", {})
                else_schema = value.get("else")
                if else_schema is not None:
                    formatted_parts.append(
                        self._format_conditional(if_schema, then_schema, else_schema)
                    )
                else:
                    formatted_parts.append(self._format_conditional(if_schema, then_schema))
            elif k in ["then", "else"] and ("if" in value or "_if" in value):
                # Skip these as they're handled with "if"
                continue
            elif (
                k in ["uniqueItems", "minItems", "maxItems"]
                and "type" in value
                and value["type"] == "array"
            ):
                # Skip these for arrays as they're integrated into the type description
                continue
            elif k == "propertyNames" and classify_container(value).kind == "mapping":
                # Skip: the mapping renderer already turned this into the key token.
                continue
            elif k in ["minLength", "maxLength"] and "type" in value and value["type"] == "string":
                # Skip these for strings as they're integrated into the type description
                continue
            elif (
                k in ["minimum", "maximum"]
                and "type" in value
                and value["type"] in ["number", "integer"]
            ):
                # Skip these for numbers as they're integrated into the type description
                continue
            else:
                formatted_parts.append(self.METADATA_MAP[k](value[actual_key]))

        return formatted_parts

    def format_field_name(self, field_name: str) -> str:
        """
        Format field name with required/optional indicator if applicable.

        Args:
            field_name: The name of the field.

        Returns:
            Field name with marker if required/optional.
        """
        if field_name in self.required_fields:
            return f"{field_name}{self.config.required_marker}"
        return f"{field_name}{self.config.optional_marker}"

    def _should_include_metadata(self, key: str) -> bool:
        """
        Check if a metadata key should be included in output.

        This method respects the metadata_inclusion configuration to provide
        fine-grained control over which metadata keywords appear in the output.

        Args:
            key: The metadata key to check (e.g., "pattern", "format", "examples").

        Returns:
            True if the metadata key should be included, False otherwise.
        """
        # If include_metadata is False, exclude everything
        if not self.include_metadata:
            return False

        # Check the metadata_inclusion config, defaulting to True if not specified
        return self._metadata_inclusion.get(key, True)

    def get_required_fields_comment(self) -> str:
        """
        Get a comment explaining the required field notation.

        Returns:
            Comment string explaining marker notation for required fields.
        """
        if not self.include_metadata:
            return ""
        if not self.required_fields:
            return ""
        marker = self.config.required_marker
        return f"{self.comment_prefix} Fields marked with {marker} are required"

    def get_schema_info_comment(self) -> str:
        """
        Get a comment containing schema title and description if present.

        Returns:
            Comment string with schema title and description, or empty string if neither present.
        """
        if not self.include_metadata:
            return ""

        comments = []

        if "title" in self.schema and self.schema["title"]:
            comments.append(f"Title: {self.schema['title']}")

        if "description" in self.schema and self.schema["description"]:
            comments.append(f"Description: {self.schema['description']}")

        if comments:
            return f"{self.comment_prefix} {', '.join(comments)}"
        return ""

    @property
    @abstractmethod
    def TYPE_MAP(self) -> dict[str, str]:
        """Type mapping dictionary for the formatter."""
        pass

    @property
    @abstractmethod
    def comment_prefix(self) -> str:
        """Comment prefix for the formatter (e.g., '//' for JSONish/TypeScript, '#' for YAML)."""
        pass

    def process_ref(self, ref: dict[str, Any]) -> str:
        """
        Process a $ref reference to a definition.

        Args:
            ref: Dictionary containing the $ref key.

        Returns:
            Processed reference representation.
        """
        ref_str: str = ref.get("$ref", "")
        if not ref_str:
            return "object"

        # Safely extract ref key with null check
        ref_match = self.REF_PATTERN.search(ref_str)
        if not ref_match:
            return "object"  # Fallback for invalid ref

        ref_key = ref_match.group(1)

        # Unconditional safety net.
        if self._global_expansion_count >= self._global_expansion_budget:
            return "object"  # Hit global budget limit

        # The one truncation contract: same-type re-entries on the active path.
        # Consulted only on re-entry, so the first expansion of any $ref is unconditional.
        reentries = self._ref_expansion_path.count(ref_key)
        if reentries >= 1 and reentries >= self.config.max_recursion_depth:
            self._truncation_epoch += 1
            return self.recursion_placeholder(ref_key)

        # Check cache first (only untruncated renderings are ever cached)
        if ref_key in self._ref_cache:
            return self._ref_cache[ref_key]

        entry_epoch = self._truncation_epoch
        self._global_expansion_count += 1
        self._ref_expansion_path.append(ref_key)

        try:
            # Priority 2: Try to resolve nested definition paths first
            ref_def = None
            if "/" in ref_key:
                # This might be a nested path like "636d/full"
                ref_def = self._resolve_nested_definition_path(f"definitions/{ref_key}")
                if not ref_def:
                    ref_def = self._resolve_nested_definition_path(f"$defs/{ref_key}")

            # Fallback to simple lookup
            if not ref_def:
                ref_def = self.defs.get(ref_key)

            if not ref_def:
                return "object"  # Fallback for missing definition

            if isinstance(ref_def, bool):
                # Handle boolean values in JSON Schema: true means any value, false means no value
                ref_str = "any" if ref_def else "never"
                if self._truncation_epoch == entry_epoch:
                    self._ref_cache[ref_key] = ref_str
                return ref_str

            # Handle different definition types with better structure preservation
            # Prioritize properties when present, as it gives more concrete structure
            if "properties" in ref_def and ref_def["properties"]:
                # Handle object definitions with properties. The referenced definition owns
                # its own ``required`` list; expose it so subclasses that mark required
                # fields inside inline literals do not consult the ROOT required list.
                self._nested_required_stack.append(set(ref_def.get("required", [])))
                try:
                    processed_properties = self.process_properties(ref_def["properties"])
                finally:
                    self._nested_required_stack.pop()
                ref_str = self.dict_to_string(processed_properties, indent=2)
            elif "enum" in ref_def:
                # Handle enum definitions
                ref_str = self.process_enum(ref_def)
            elif "oneOf" in ref_def:
                # Handle oneOf definitions - preserve structure
                ref_str = self.process_oneof(ref_def)
            elif "anyOf" in ref_def:
                # Handle anyOf definitions - preserve structure
                ref_str = self.process_anyof(ref_def)
            elif "allOf" in ref_def:
                # Handle allOf definitions - preserve structure
                ref_str = self.process_allof(ref_def)
            elif "type" in ref_def:
                # Handle type definitions with constraints
                ref_str = self.process_type_value(ref_def)
            elif "$ref" in ref_def:
                # Handle nested $ref references
                ref_str = self.process_ref(ref_def)
            elif "const" in ref_def:
                # Handle const definitions
                ref_str = str(ref_def["const"])
            elif "pattern" in ref_def:
                # Handle pattern-only definitions (like regex patterns)
                ref_str = f"string (pattern: {ref_def['pattern']})"
            elif "format" in ref_def:
                # Handle format-only definitions
                ref_str = f"string (format: {ref_def['format']})"
            else:
                # For complex definitions that don't match patterns, try to preserve some structure
                if isinstance(ref_def, dict) and len(ref_def) > 0:
                    # Try to extract meaningful information
                    if "description" in ref_def:
                        ref_str = f"object //{ref_def['description']}"
                    elif "title" in ref_def:
                        ref_str = f"object //{ref_def['title']}"
                    else:
                        ref_str = "object"
                else:
                    ref_str = "object"

            # Taint-and-skip: never cache a rendering that truncated.
            if self._truncation_epoch == entry_epoch:
                self._ref_cache[ref_key] = ref_str
            return ref_str
        finally:
            if self._ref_expansion_path and self._ref_expansion_path[-1] == ref_key:
                self._ref_expansion_path.pop()

    @property
    def deferred_comment_gap(self) -> str:
        """Whitespace between a rendered token and its hoisted comment marker.

        Returns:
            A single space. ``YAMLFormatter`` overrides this to two spaces to match its
            existing ``"  # ..."`` convention.
        """
        return " "

    def defer_comment(self, body: str) -> str:
        """Register ``body`` in the slot table and return the marker token for it.

        Content-keyed: a ``body`` already present reuses its existing slot index rather
        than allocating a new one.

        Args:
            body: The comment text to render once the marker is hoisted, e.g.
                ``'one of: "US", "CA"'``.

        Returns:
            ``f"{DEFERRED_OPEN}{DEFERRED_TAG}{self._deferred_nonce}.{index}{DEFERRED_CLOSE}"``.
        """
        index = self._deferred_index.get(body)
        if index is None:
            index = len(self._deferred_bodies)
            self._deferred_bodies.append(body)
            self._deferred_index[body] = index
        return "".join(
            (DEFERRED_OPEN, DEFERRED_TAG, self._deferred_nonce, ".", str(index), DEFERRED_CLOSE)
        )

    def append_deferred_comment(self, representation: str, extra: str) -> str:
        """Fold ``extra`` into the slot body referenced by the first marker in ``representation``.

        Args:
            representation: A string that may contain zero or one deferred-comment marker.
            extra: Text to fold into that marker's slot body.

        Returns:
            ``representation`` unchanged when ``extra`` is empty or no marker is present;
            otherwise ``representation`` with its first marker token replaced by a new
            token referencing the updated slot body.
        """
        if not extra:
            return representation
        match = self._deferred_pattern.search(representation)
        if match is None:
            return representation
        body = self._deferred_bodies[int(match.group(1))]
        token = self.defer_comment(f"{body}; {extra}")
        return representation.replace(match.group(0), token, 1)

    def carries_deferred_comment(self, representation: object) -> bool:
        """True iff ``representation`` is a string containing at least one deferred marker.

        Args:
            representation: Value to test. Typed ``object`` so callers holding a
                ``str | dict[str, Any] | list[Any]`` union need not narrow it first.

        Returns:
            Whether this instance's ``_deferred_pattern`` matches anywhere in it.
        """
        return (
            isinstance(representation, str)
            and self._deferred_pattern.search(representation) is not None
        )

    def hoist_deferred_comments(self, text: str) -> str:
        """Resolve every deferred marker in ``text`` to a real trailing comment, per line.

        Args:
            text: Fully rendered formatter output, one or more lines.

        Returns:
            ``text`` with every marker removed and its slot body re-emitted as a trailing
            comment via ``_hoist_deferred_line``; never contains ``DEFERRED_OPEN``.
        """
        if DEFERRED_OPEN not in text:
            return text
        return "\n".join(self._hoist_deferred_line(line) for line in text.split("\n"))

    def _hoist_deferred_line(self, line: str) -> str:
        """Apply the multi-fragment hoist rule to one physical line.

        Args:
            line: One line of rendered output, with zero or more markers.

        Returns:
            ``line`` unchanged if it carries no marker; otherwise ``line`` with every marker
            removed and one trailing ``f"{gap}{comment_prefix} {frag}"`` appended before any
            trailing comma.
        """
        matches = list(self._deferred_pattern.finditer(line))
        if not matches:
            return line

        # 1-2. Collect every slot body in source order, dropping exact duplicates.
        bodies: list[str] = []
        seen: set[str] = set()
        for match in matches:
            body = self._deferred_bodies[int(match.group(1))]
            if body in seen:
                continue
            seen.add(body)
            bodies.append(body)

        # 3-4. Join survivors, then strip every token out of the line.
        frag = "; ".join(bodies)
        head = self._deferred_pattern.sub("", line)

        # 5. Hold any trailing comma aside so the comment lands before it.
        trailing_comma = ""
        if head.rstrip().endswith(","):
            head = head.rstrip()
            head, trailing_comma = head[:-1], ","
        head = head.rstrip()

        # 6. Absorb any pre-existing comment on the line into the fragment.
        prefix = self.comment_prefix
        if prefix in head:
            head, _, existing = head.partition(prefix)
            head = head.rstrip()
            existing = existing.strip()
            if existing and existing not in frag:
                frag = f"{frag}; {existing}"

        # 7. Emit the single trailing comment.
        return "".join((head, self.deferred_comment_gap, prefix, " ", frag, trailing_comma))

    def _extract_enum_metadata(
        self, enum_value: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Extract x-enum-descriptions and x-enum-aliases from schema.

        Args:
            enum_value: Schema node that may contain enum and extension fields.

        Returns:
            Tuple of (descriptions, aliases). descriptions maps value -> str;
            aliases maps canonical value -> list of alias strings.
        """
        descriptions = enum_value.get("x-enum-descriptions", {})
        aliases = enum_value.get("x-enum-aliases", {})
        if not isinstance(descriptions, dict):
            descriptions = {}
        if not isinstance(aliases, dict):
            aliases = {}
        return descriptions, aliases

    def enum_type_token(self, node: dict[str, Any]) -> str:
        """Base type token for an enum/const node.

        Args:
            node: The schema node carrying ``enum`` (or the synthetic single-element
                ``enum`` list built from a ``const``).

        Returns:
            A ``TYPE_MAP``-mapped token derived from ``node["type"]`` -- including the list
            and nullable-list forms -- or, when ``type`` is absent, ``infer_json_type``
            applied to the enum values; ``"any"`` when that inference is heterogeneous or
            the value list is empty.
        """
        enum_type = node.get("type")

        if enum_type is None:
            inferred = infer_json_type(node.get("enum", []))
            if inferred is None:
                return "any"
            return self.TYPE_MAP.get(inferred, inferred)

        if isinstance(enum_type, list):
            if len(enum_type) == 1:
                single_type = str(enum_type[0])
                return self.TYPE_MAP.get(single_type, single_type)
            if "null" in enum_type and len(enum_type) == 2:
                non_null_type = str(next(t for t in enum_type if t != "null"))
                return self.TYPE_MAP.get(non_null_type, non_null_type)
            non_null_types = [str(t) for t in enum_type if t != "null"]
            fallback = non_null_types[0] if non_null_types else "string"
            return self.TYPE_MAP.get(fallback, fallback)

        enum_type_str = str(enum_type)
        return self.TYPE_MAP.get(enum_type_str, enum_type_str)

    def build_enum_comment(self, node: dict[str, Any], values: list[Any]) -> str:
        """Build the ``"one of: ..."`` comment body for an enum/const node.

        Args:
            node: The schema node, consulted via ``_extract_enum_metadata`` for
                ``x-enum-descriptions`` / ``x-enum-aliases``.
            values: The enum's value list, or the single-element list built from a ``const``.

        Returns:
            ``"one of: "`` followed by each value's ``format_literal_value``, joined by the
            fixed ``", "`` (never ``union_separator``), with any per-value description
            and/or alias list folded in.
        """
        descriptions, aliases = self._extract_enum_metadata(node)
        parts: list[str] = []
        for value in values:
            literal = format_literal_value(value)
            if isinstance(value, bool):
                canonical = "true" if value else "false"
            else:
                canonical = str(value)
            part = literal
            if canonical in descriptions and descriptions[canonical]:
                part = f"{part} ({descriptions[canonical]}"
                if canonical in aliases and aliases[canonical]:
                    part = f"{part}; aliases: {', '.join(aliases[canonical])}"
                part = f"{part})"
            elif canonical in aliases and aliases[canonical]:
                part = f"{part} (aliases: {', '.join(aliases[canonical])})"
            parts.append(part)
        return "".join(("one of: ", ", ".join(parts)))

    def _get_title_description_default_value(
        self, value: dict[str, Any]
    ) -> tuple[str, str, str, str]:
        """Base default: no metadata extraction.

        ``JSONishFormatter`` and ``YAMLFormatter`` both override this with their own
        formatter-specific extraction and stay byte-identical; this default exists only so
        the shared ``process_enum``/``process_const`` can call it under mypy for a formatter
        that does not override it.

        Args:
            value: Schema node (unused by this default).

        Returns:
            ``("", "", "", "")``.
        """
        return "", "", "", ""

    def process_enum(self, enum_value: dict[str, Any]) -> str:
        """
        Process an enum field.

        Args:
            enum_value: Dictionary containing enum definition.

        Returns:
            The base type token immediately followed by a deferred-comment marker whose
            slot holds the ``"one of: ..."`` body (plus any description / default /
            example folded in with ``"; "``). ``"string"`` for an empty enum, with no
            marker minted. The gap before the comment is inserted later, by
            ``_hoist_deferred_line``.
        """
        enum_list = enum_value.get("enum", [])
        if not enum_list:
            return "string"  # Fallback for empty enum

        token = self.enum_type_token(enum_value)
        body = self.build_enum_comment(enum_value, enum_list)

        # ``title`` is deliberately read and DISCARDED: appending it is what leaked the
        # Pydantic enum class name (``Role:``, ``PriorityWithMetadata:``) into output.
        _title, description, default_value, example = self._get_title_description_default_value(
            enum_value
        )
        for extra in (description, default_value, example):
            stripped = extra.strip()
            if stripped:
                body = "; ".join((body, stripped))

        return f"{token}{self.defer_comment(body)}"

    def process_const(self, const_value: dict[str, Any]) -> str:
        """
        Process a const field (single literal value).

        Args:
            const_value: Dictionary containing const definition.

        Returns:
            The same representation ``process_enum`` produces for a single-element enum:
            a ``const`` is an enum of exactly one value.
        """
        synthesised = {**const_value, "enum": [const_value.get("const")]}
        return self.process_enum(synthesised)

    def render_type_token(self, schema: dict[str, Any]) -> str:
        """One-line type token for a nested position (tuple element / mapping value)."""
        return self.process_property(schema)

    def key_token(self, shape: ContainerShape) -> str:
        """Rendered key *type* of a MAPPING (bare word, e.g. ``"string"``). No angle brackets.

        Pure query: reads ``self.defs`` / ``self.TYPE_MAP`` and mutates nothing.
        """
        key_schema = shape.key_schema
        if key_schema is None:
            return "string"

        ref = key_schema.get("$ref")
        if isinstance(ref, str):
            ref_match = self.REF_PATTERN.search(ref)
            if ref_match:
                ref_def = self.defs.get(ref_match.group(1))
                if isinstance(ref_def, dict):
                    ref_type = ref_def.get("type")
                    if isinstance(ref_type, str):
                        return str(self.TYPE_MAP.get(ref_type, ref_type))
            return "string"

        key_type = key_schema.get("type")
        if isinstance(key_type, str):
            return str(self.TYPE_MAP.get(key_type, key_type))

        return "string"

    def render_mapping(self, shape: ContainerShape) -> str:
        """One-line mapping token, e.g. ``"{ <string>: int }"``."""
        if shape.value_schema is not None:
            value_token = self.render_type_token(shape.value_schema)
        else:
            value_token = "any"
        return f"{{ <{self.key_token(shape)}>: {value_token} }}"

    def render_tuple(self, shape: ContainerShape) -> str:
        """One-line tuple token, e.g. ``"[int, string]"`` / ``"[int, string, ...string]"``."""
        tokens = [self.render_type_token(s) for s in shape.prefix_schemas]
        if shape.rest_schema is not None:
            tokens.append(f"...{self.render_type_token(shape.rest_schema)}")
        return "[" + ", ".join(tokens) + "]"

    def process_type_value(self, type_value: dict[str, Any]) -> str:
        """
        Process a type field.

        Args:
            type_value: Dictionary containing type definition.

        Returns:
            Formatted type representation.
        """
        # Check if this is an empty schema (any valid JSON)
        if not type_value or len(type_value) == 0:
            return "any"

        # Safely get type with fallback
        type_name = type_value.get("type")

        # If no type specified but has other constraints, treat as "any"
        if type_name is None:
            # Has constraints but no type -> could be any type with those constraints
            # For simplicity, show as "any" for now
            return "any"

        # Handle array of types (union types like ["string", "null"])
        if isinstance(type_name, list):
            if len(type_name) == 1:
                type_name = type_name[0]
            elif "null" in type_name and len(type_name) == 2:
                # Handle nullable types like ["string", "null"] -> "string?"
                non_null_type = next(t for t in type_name if t != "null")
                type_str = self.TYPE_MAP.get(non_null_type, non_null_type)
                return f"{type_str}?"  # Mark as nullable
            else:
                # Multiple non-null types - treat as union
                type_strs = [self.TYPE_MAP.get(t, t) for t in type_name if t != "null"]
                return self.config.union_separator.join(s for s in type_strs if s is not None)

        # Now type_name is guaranteed to be a string
        type_str = self.TYPE_MAP.get(type_name, type_name)

        # Add validation constraints to type description (only if metadata is enabled)
        if self.include_metadata:
            if type_name == "string":
                constraints = []

                # Add length constraints
                length_range = self._format_validation_range(
                    type_value, "minLength", "maxLength", " chars"
                )
                if length_range:
                    constraints.append(length_range)

                # Add pattern constraints
                if "pattern" in type_value:
                    pattern = type_value["pattern"]
                    # Truncate very long patterns for readability
                    if len(pattern) > 50:
                        pattern = pattern[:47] + "..."
                    constraints.append(f"pattern: {pattern}")

                # Add format constraints
                if "format" in type_value:
                    constraints.append(f"format: {type_value['format']}")

                if constraints:
                    type_str = f"{type_str} ({', '.join(constraints)})"
            elif type_name in ["number", "integer"]:
                range_info = self._format_validation_range(type_value, "minimum", "maximum")
                if range_info:
                    type_str = f"{type_str} ({range_info})"

        if type_str == "array":
            shape = classify_container(type_value)
            if shape.kind == "tuple":
                return self.render_tuple(shape)

            # Safely handle array items (list-only from here on)
            items = type_value.get("items")
            if not items:
                type_str = "array"  # Fallback for array without items
            elif isinstance(items, bool):
                # Handle boolean items (true means any type, false means no items)
                type_str = "array" if items else "array"
            elif isinstance(items, dict) and "type" in items:
                # For object items, process the full structure
                if items["type"] == "object" and "properties" in items:
                    processed_properties = self.process_properties(items["properties"])
                    items_structure = self.dict_to_string(processed_properties, indent=2)
                    type_str = f"[\n{items_structure}\n]"
                else:
                    items_type = self.process_type_value(items)
                    type_str = f"{items_type}[]"
            elif isinstance(items, dict) and "$ref" in items:
                items_type = self.process_ref(items)
                type_str = f"[{items_type}]"
            elif isinstance(items, dict) and "anyOf" in items:
                items_type = self.process_anyof(items)
                type_str = f"[{items_type}]"
            elif isinstance(items, dict) and "allOf" in items:
                items_type = self.process_allof(items)
                type_str = f"[{items_type}]"
            elif isinstance(items, dict) and "oneOf" in items:
                items_type = self.process_oneof(items)
                type_str = f"[{items_type}]"
            else:
                type_str = "array"  # Fallback for unknown array item type

            type_str += self.format_array_constraints(type_value)

            if "contains" in type_value:
                type_str += self.process_contains(type_value)
            # process_unique_items is no longer called from any array path (orphaned;
            # kept for API stability per R4).

        return type_str  # type: ignore[no-any-return]

    def process_anyof(self, anyof: dict[str, Any]) -> str:
        """
        Process an anyOf field (union types).

        Args:
            anyof: Dictionary containing anyOf definition.

        Returns:
            Formatted union type representation.
        """
        # Safely get anyOf list
        anyof_list = anyof.get("anyOf", [])
        if not anyof_list:
            return "string"  # Fallback for empty anyOf

        item_types = []
        array_found = False
        for item in anyof_list:
            # Skip non-dictionary items (like booleans)
            if not isinstance(item, dict):
                continue

            if "enum" in item:
                item_types.append(self.process_enum(item))
            elif "const" in item:
                item_types.append(str(item["const"]))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "type" in item:
                if item["type"] == "array":
                    array_found = True

                if array_found and item["type"] == "null":
                    item_types.append("[]")
                else:
                    item_types.append(self.process_type_value(item))
            elif "properties" in item:
                # Handle object schemas in anyOf
                processed_props = self.process_properties(item["properties"])
                items_structure = self.dict_to_string(processed_props, indent=2)
                item_types.append(f"{{\n{items_structure}\n}}")
            else:
                # Unknown anyOf item, skip it
                continue

        # Limit the number of union types to prevent excessive expansion
        # Be very aggressive to prevent recursive anyOf explosion
        if self._global_expansion_count > 100:
            max_items = 2  # Very aggressive for deep recursion
        elif self._global_expansion_count > 30:
            max_items = 3  # Aggressive (lowered threshold from 50)
        elif self._global_expansion_count > 10:
            max_items = 4  # Moderate (new tier)
        else:
            max_items = 5  # Conservative start (reduced from 6)

        if len(item_types) > max_items:
            return f"anyOf: {len(item_types)} options"
        else:
            return self.config.union_separator.join(item_types) if item_types else "string"

    def process_oneof(self, oneof: dict[str, Any]) -> str:
        """Process oneOf (exclusive choice) schemas."""
        oneof_list = oneof.get("oneOf", [])
        if not oneof_list:
            return "string"

        item_types = []
        for item in oneof_list:
            # Skip non-dictionary items (like booleans)
            if not isinstance(item, dict):
                continue

            if "allOf" in item:
                # Handle allOf inside oneOf
                item_types.append(self.process_allof(item))
            elif "anyOf" in item:
                # Handle anyOf inside oneOf
                item_types.append(self.process_anyof(item))
            elif "enum" in item:
                # Check enum before type to preserve enum constraints
                item_types.append(self.process_enum(item))
            elif "type" in item:
                item_types.append(self.process_type_value(item))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "properties" in item:
                # Handle object schemas in oneOf
                processed_props = self.process_properties(item["properties"])
                items_structure = self.dict_to_string(processed_props, indent=2)
                item_types.append(f"{{\n{items_structure}\n}}")
            elif "const" in item:
                item_types.append(str(item["const"]))

        # Preserve oneOf structure but limit to reasonable number of options
        # Be very aggressive to prevent recursive oneOf explosion
        if self._global_expansion_count > 100:
            max_items = 3  # Very aggressive for deep recursion
        elif self._global_expansion_count > 30:
            max_items = 4  # Aggressive (lowered threshold from 50)
        elif self._global_expansion_count > 10:
            max_items = 5  # Moderate (new tier)
        else:
            max_items = 6  # Conservative start (reduced from 8)

        if len(item_types) > max_items:
            return f"oneOf: {len(item_types)} options"
        elif item_types:
            return f"oneOf: {self.config.union_separator.join(item_types)}"
        else:
            return "string"

    def process_allof(self, allof: dict[str, Any]) -> str:
        """Process allOf (intersection) schemas."""
        allof_list = allof.get("allOf", [])
        if not allof_list:
            return "string"

        # For allOf, we process each schema and combine them
        # This is a simplified approach - in practice, allOf is complex
        item_types = []
        for item in allof_list:
            if isinstance(item, dict):
                if "type" in item and "properties" in item:
                    # Handle object schemas with properties in allOf
                    processed_props = self.process_properties(item["properties"])
                    items_structure = self.dict_to_string(processed_props, indent=2)
                    item_types.append(f"{{\n{items_structure}\n}}")
                elif "type" in item:
                    item_types.append(self.process_type_value(item))
                elif "$ref" in item:
                    item_types.append(self.process_ref(item))
                elif "properties" in item:
                    # Handle object schemas in allOf
                    processed_props = self.process_properties(item["properties"])
                    item_types.append(self.dict_to_string(processed_props, indent=2))
                elif "description" in item and not item.get("type"):
                    # Skip items that only have description
                    continue
                else:
                    item_types.append("object")

        # Limit allOf combinations to prevent excessive expansion
        if len(item_types) > 3:  # Reduced from 5 to prevent over-expansion
            return f"allOf: {len(item_types)} schemas"
        elif item_types:
            return f"allOf: {' & '.join(item_types)}"
        else:
            return "object"

    def process_not(self, not_schema: dict[str, Any]) -> str:
        """Process not (negation) schemas."""
        not_def = not_schema.get("not", {})
        if not_def:
            return f"not: {self.process_type_value(not_def)}"
        return "string"

    def process_property(self, _property: Any) -> str:
        """
        Process a single property from the schema.

        Args:
            _property: Property definition from the schema.

        Returns:
            Processed property representation as a string.
        """
        # Handle non-dictionary property values (like booleans, strings, numbers)
        if not isinstance(_property, dict):
            if isinstance(_property, bool):
                return "bool"
            elif isinstance(_property, str):
                return "string"
            elif isinstance(_property, int | float):
                return "number"
            else:
                return "any"

        # Empty schema {} means any valid JSON
        if not _property:
            return "any"

        if "$ref" in _property:
            prop_str = self.process_ref(_property)
        elif "const" in _property:
            prop_str = self.process_const(_property)
        elif "enum" in _property:
            prop_str = self.process_enum(_property)
        elif "anyOf" in _property:
            prop_str = self.process_anyof(_property)
        elif "oneOf" in _property:
            prop_str = self.process_oneof(_property)
        elif "allOf" in _property:
            prop_str = self.process_allof(_property)
        elif "not" in _property:
            prop_str = self.process_not(_property)
        elif "type" in _property:
            shape = classify_container(_property)
            if shape.kind == "mapping":
                prop_str = self.render_mapping(shape)
            elif shape.kind == "tuple":
                prop_str = self.render_tuple(shape)
            elif shape.kind == "any":
                prop_str = "any"
            # Check if this is an object with nested properties that should be expanded
            elif (
                _property.get("type") == "object"
                and "properties" in _property
                and _property["properties"]
            ):
                # Expand nested object properties
                nested_props = self.process_properties(_property["properties"])
                prop_str = self.dict_to_string(nested_props, indent=1)
            # Check if this is an object with patternProperties
            elif _property.get("type") == "object" and "patternProperties" in _property:
                # Process patternProperties and show the structure
                pattern_props = _property["patternProperties"]
                pattern_results = []
                for pattern, pattern_def in list(pattern_props.items())[:2]:  # Limit to 2
                    if isinstance(pattern_def, dict):
                        if "$ref" in pattern_def:
                            pattern_type = self.process_ref(pattern_def)
                        elif "properties" in pattern_def:
                            nested_props = self.process_properties(pattern_def["properties"])
                            pattern_type = self.dict_to_string(nested_props, indent=1)
                        elif "type" in pattern_def:
                            pattern_type = self.process_type_value(pattern_def)
                        else:
                            pattern_type = "object"
                    else:
                        pattern_type = str(pattern_def)
                    pattern_results.append(f"[{pattern}]: {pattern_type}")
                prop_str = f"object  //pattern: {', '.join(pattern_results)}"
            else:
                prop_str = self.process_type_value(_property)
        else:
            # Fallback: a typeless, non-empty schema is `Any` (classify_container rule 10),
            # e.g. a field annotated only with a description or a title.
            prop_str = "any"

        return self.add_metadata(prop_str, _property)

    @abstractmethod
    def add_metadata(self, representation: str, value: dict[str, Any]) -> str:
        """
        Add metadata comments to a field representation.

        Args:
            representation: The base field representation.
            value: The field definition containing metadata.

        Returns:
            Field representation with metadata comments.
        """
        pass

    def dict_to_string(self, value: Any, indent: int = 1) -> str:
        """
        Convert a dictionary or list to a formatted string representation.

        Args:
            value: The value to convert (dict, list, or primitive).
            indent: Current indentation level.

        Returns:
            Formatted string representation.
        """
        # Default implementation - subclasses can override
        return str(value)

    def process_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """
        Process multiple properties from the schema.

        Args:
            properties: Dictionary of property definitions.

        Returns:
            Dictionary of processed properties with required field names formatted.
        """
        processed_properties = {}
        for prop_name, value in properties.items():
            # Format field name with required indicator
            formatted_name = self.format_field_name(prop_name)
            processed_properties[formatted_name] = self.process_property(value)
        return processed_properties

    def process_additional_properties(
        self, schema: dict[str, Any], show_structure: bool = True
    ) -> str:
        """Process additionalProperties constraint.

        Args:
            schema: Schema containing additionalProperties
            show_structure: If False, return simpler comment (structure shown elsewhere)

        Returns:
            Formatted additionalProperties comment.
        """
        additional_props = schema.get("additionalProperties")
        if additional_props is False:
            return " //no additional properties"
        elif isinstance(additional_props, dict) and additional_props:
            if not schema.get("properties"):
                # Pure mapping (classify_container rule 6/C1): the value type is rendered
                # structurally by the caller's mapping renderer, never as a comment.
                return ""
            if not show_structure:
                # Structure shown via placeholder key, just indicate it's allowed
                return " //any properties allowed"

            type_str = self.process_type_value(additional_props)
            required = additional_props.get("required", [])
            props = additional_props.get("properties", {})
            if isinstance(props, dict) and props:
                prop_details = []
                for prop_name, prop_def in props.items():
                    if isinstance(prop_def, dict):
                        prop_type = self.process_type_value(prop_def)
                    else:
                        prop_type = str(prop_def)
                    if prop_name in required:
                        prop_details.append(f"{prop_name}* (required): {prop_type}")
                    else:
                        prop_details.append(f"{prop_name}: {prop_type}")
                return f" //additional: {type_str} with {', '.join(prop_details)}"
            if required:
                return f" //additional: {type_str} with required {', '.join(required)}"
            return f" //additional: {type_str}"
        return ""

    def process_pattern_properties(self, schema: dict[str, Any]) -> str:
        """Process patternProperties constraint."""
        pattern_props = schema.get("patternProperties", {})
        if pattern_props:
            patterns = []
            for pattern, definition in pattern_props.items():
                # Process pattern definition properly (handle $ref, properties, etc.)
                if isinstance(definition, dict):
                    if "$ref" in definition:
                        pattern_type = self.process_ref(definition)
                    elif "properties" in definition:
                        # Expand properties for pattern
                        nested_props = self.process_properties(definition["properties"])
                        pattern_type = self.dict_to_string(nested_props, indent=1)
                    elif "type" in definition:
                        pattern_type = self.process_type_value(definition)
                    else:
                        pattern_type = "object"
                else:
                    pattern_type = str(definition)
                patterns.append(f"{pattern}: {pattern_type}")
            return f" //patternProperties: {', '.join(patterns)}"
        return ""

    def process_dependencies(self, schema: dict[str, Any]) -> str:
        """Process dependencies constraint."""
        dependencies = schema.get("dependencies", {})
        if dependencies:
            deps = []
            for prop, deps_list in dependencies.items():
                if isinstance(deps_list, list):
                    deps.append(f"{prop} requires {', '.join(deps_list)}")
                else:
                    deps.append(f"{prop} requires {deps_list}")
            return f" //dependencies: {', '.join(deps)}"
        return ""

    def process_conditional(self, schema: dict[str, Any]) -> str:
        """Process if/then/else conditional schemas."""
        if_clause = schema.get("if")
        then_clause = schema.get("then")
        else_clause = schema.get("else")

        if if_clause and then_clause:
            condition = self.process_type_value(if_clause)
            consequence = self.process_type_value(then_clause)
            result = f"if {condition} then {consequence}"

            if else_clause:
                alternative = self.process_type_value(else_clause)
                result += f" else {alternative}"

            return f" //{result}"
        return ""

    def process_contains(self, schema: dict[str, Any]) -> str:
        """Process contains constraint for arrays."""
        contains = schema.get("contains")
        if contains:
            return f" //contains: {self._format_contains(contains)}"
        return ""

    def process_unique_items(self, schema: dict[str, Any]) -> str:
        """Process uniqueItems constraint."""
        unique = schema.get("uniqueItems")
        if unique:
            return " //unique items"
        return ""

    def array_constraint_tokens(self, schema: dict[str, Any]) -> list[str]:
        """Ordered, already-gated constraint words for an array-ish schema.

        Order is fixed: uniqueness first, then the length range. Every token is already
        filtered through ``_should_include_metadata`` (itself False whenever
        ``include_metadata`` is False), so callers must not re-gate.
        """
        tokens: list[str] = []

        is_unique = schema.get("uniqueItems") or schema.get("_uniqueItems")
        if is_unique and self._should_include_metadata("uniqueItems"):
            tokens.append("unique")

        has_min = "minItems" in schema and self._should_include_metadata("minItems")
        has_max = "maxItems" in schema and self._should_include_metadata("maxItems")
        if has_min and has_max:
            tokens.append(f"{schema['minItems']}-{schema['maxItems']} items")
        elif has_min:
            tokens.append(f">= {schema['minItems']} items")
        elif has_max:
            tokens.append(f"<= {schema['maxItems']} items")

        return tokens

    def format_array_constraints(self, schema: dict[str, Any]) -> str:
        """``" (a, b)"`` for a non-empty token list, ``""`` otherwise. Never a comment."""
        tokens = self.array_constraint_tokens(schema)
        return f" ({', '.join(tokens)})" if tokens else ""

    def process_property_names(self, schema: dict[str, Any]) -> str:
        """Process propertyNames constraint."""
        prop_names = schema.get("propertyNames")
        if prop_names:
            return f" //propertyNames: {self.process_type_value(prop_names)}"
        return ""

    def process_unevaluated_properties(self, schema: dict[str, Any]) -> str:
        """Process unevaluatedProperties constraint."""
        uneval_props = schema.get("unevaluatedProperties")
        if uneval_props is False:
            return " //no unevaluated properties"
        elif isinstance(uneval_props, dict):
            return f" //unevaluated: {self.process_type_value(uneval_props)}"
        return ""

    def _format_contains(self, contains_schema: Any) -> str:
        """Format contains constraint in user-friendly way."""
        if isinstance(contains_schema, dict):
            if "enum" in contains_schema:
                return f"string ({', '.join(contains_schema['enum'])})"
            elif "type" in contains_schema:
                return str(contains_schema["type"])
            else:
                return str(contains_schema)
        return str(contains_schema)

    def _format_type_simple(self, schema: Any) -> str:
        """Extract simple type from schema."""
        if isinstance(schema, dict):
            return str(schema.get("type", "any"))
        return str(schema)

    def _format_validation_range(
        self, schema: dict[str, Any], min_key: str, max_key: str, unit: str = ""
    ) -> str:
        """Format validation range constraints."""
        min_val = schema.get(min_key)
        max_val = schema.get(max_key)

        if min_val is not None and max_val is not None:
            return f"{min_val}-{max_val}{unit}"
        elif min_val is not None:
            return f"≥{min_val}{unit}"
        elif max_val is not None:
            return f"≤{max_val}{unit}"
        return ""

    def _format_conditional(
        self,
        if_schema: dict[str, Any],
        then_schema: dict[str, Any],
        else_schema: dict[str, Any] | None = None,
    ) -> str:
        """Format conditional logic in user-friendly way."""
        if_desc = self._describe_condition(if_schema)
        then_desc = self._describe_schema(then_schema)

        if else_schema:
            else_desc = self._describe_schema(else_schema)
            return f"if {if_desc} then {then_desc} else {else_desc}"
        else:
            return f"if {if_desc} then {then_desc}"

    def _describe_condition(self, condition: dict[str, Any]) -> str:
        """Describe a condition in user-friendly way."""
        if "properties" in condition:
            props = condition["properties"]
            if len(props) == 1:
                prop_name, prop_schema = next(iter(props.items()))
                if "minimum" in prop_schema:
                    return f"{prop_name} ≥ {prop_schema['minimum']}"
                elif "maximum" in prop_schema:
                    return f"{prop_name} ≤ {prop_schema['maximum']}"
                elif "pattern" in prop_schema:
                    return f"{prop_name} matches {prop_schema['pattern']}"
            return f"condition on {', '.join(props.keys())}"
        return "condition"

    def _describe_schema(self, schema: dict[str, Any]) -> str:
        """Describe a schema in user-friendly way."""
        if "required" in schema:
            return f"requires {', '.join(schema['required'])}"
        elif "properties" in schema:
            return f"object with {', '.join(schema['properties'].keys())}"
        return "schema"

    @abstractmethod
    def transform_schema(self) -> str:
        """
        Transform the schema into the desired format.

        Returns:
            Formatted schema as a string.
        """
        pass
