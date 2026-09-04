"""JSONish formatter for transforming Pydantic schemas into BAML-like format."""

import copy
import json
from typing import Any

from .base import BaseFormatter, classify_container
from .config import FormatterConfig


class JSONishFormatter(BaseFormatter):
    """
    Transforms Pydantic schema into a JSONish (BAML-like) representation.

    This formatter creates a JSON-like format (not valid JSON) with inline comments
    for metadata, optimized for LLM consumption with minimal token usage.

    Example output:
        {
         name: string  //description, minLength: 1,
         age: int  //min: 0, max: 120
        }
    """

    def __init__(
        self,
        schema: dict[str, Any],
        config: "FormatterConfig | None" = None,
        include_metadata: bool | None = None,
    ):
        """
        Initialize the JSONish formatter.

        Args:
            schema: JSON schema from Pydantic model_json_schema.
            config: FormatterConfig for customizing formatter behavior.
            include_metadata: Deprecated. Use config.include_metadata instead.
        """
        # Set default union_separator for JSONish format if not explicitly provided
        if config is None:
            config = FormatterConfig(union_separator=" OR ")
        elif config.union_separator == FormatterConfig().union_separator:
            # User didn't override union_separator, use JSONish default
            config.union_separator = " OR "

        super().__init__(schema, config, include_metadata)
        # Trial-specific state
        self.processed_ref_cache: dict[str, dict[str, Any] | str | list[Any]] = {}
        self.pending_postfix: dict[str, str] = {}
        self.pending_recursion: dict[str, str] = {}
        self.pending_prefix: dict[str, str] = {}
        self.simplified_schema: str | None = None

    @property
    def TYPE_MAP(self) -> dict[str, str]:
        """Type mapping for JSONish format."""
        return {"number": "float", "integer": "int", "boolean": "bool"}

    @property
    def comment_prefix(self) -> str:
        """Comment prefix for JSONish format."""
        return "//"

    @property
    def deferred_comment_gap(self) -> str:
        """One space before a hoisted ``//`` comment.

        Matches ``_extract_description``'s ``f" {self.comment_prefix} ..."`` and is immune
        to ``transform_schema``'s trailing ``.replace("  ", " ")``.
        """
        return " "

    def add_metadata(self, representation: str, value: dict[str, Any]) -> str:
        """
        Add metadata comments to a field representation.

        Note: The new implementation handles metadata inline during processing,
        so this method is a pass-through to satisfy the base class interface.

        Args:
            representation: The base field representation.
            value: The field definition containing metadata.

        Returns:
            Field representation (unchanged).
        """
        return representation

    def _extract_description(self, schema: dict[str, Any]) -> str:
        """
        Extract description from any schema node with comment prefix.

        Args:
            schema: Schema dictionary to extract description from.

        Returns:
            Formatted description string with // comment prefix or empty string.
        """
        if "description" in schema and schema["description"] is not None:
            return f" {self.comment_prefix} {schema['description']}"
        return ""

    def _get_options_format_pattern(self, value: dict[str, Any]) -> tuple[str, str]:
        """Extract format and pattern from schema value.

        Note: this previously also returned an `options` element derived from an
        `OPTIONS: ...` string built from `value["enum"]`. That branch was proven
        unreachable (every caller of `process_types` checks `enum` before reaching
        this code) and was removed.
        """
        format_ = ""
        # Filter pattern based on metadata_inclusion config
        if "pattern" in value and value["pattern"] and self._should_include_metadata("pattern"):
            pattern = f" (PATTERN: {value['pattern']})"
        else:
            pattern = ""
        # Filter format based on metadata_inclusion config
        if self._should_include_metadata("format"):
            if "format" in value and value["format"]:
                format_ = f" (FORMAT: {value['format']})"
            elif "_format" in value and value["_format"]:
                format_ = f" (FORMAT: {value['_format']})"
        return format_, pattern

    def _get_fields_dependencies(self, schema: dict[str, Any], field_name: str) -> str:
        """Extract dependencies for a field from schema."""
        if "dependencies" in schema and schema["dependencies"]:
            if field_name in schema["dependencies"]:
                dependencies = schema["dependencies"][field_name]
                if isinstance(dependencies, list):
                    return f"(DEPENDS ON: {', '.join(dependencies)})"
                else:
                    # TODO: handle validations
                    return f"(DEPENDS ON: {dependencies})"
        return ""

    def process_ref(  # type: ignore[override]
        self, value: dict[str, Any], key: str | None = None
    ) -> str | dict[str, Any] | list[Any]:
        """
        Process a $ref reference using trial's logic.

        Args:
            value: Dictionary containing the $ref key.
            key: Optional property key for postfix tracking.

        Returns:
            Processed reference representation.
        """
        _ref = value["$ref"].split("/")[-1]
        defs = self.schema.get("$defs", self.schema.get("definitions", {}))
        if _ref in defs:
            _def = defs[_ref]
        else:
            return "object"

        # Same rule as BaseFormatter.process_ref: same-type re-entries on the active path.
        reentries = self._ref_expansion_path.count(_ref)
        if reentries >= 1 and reentries >= self.config.max_recursion_depth:
            self._truncation_epoch += 1
            if key is not None:
                self.pending_recursion[key] = f"recursive: {_ref}"
                return "object"
            # No field key to hang a trailing comment on (nested array element, mapping
            # value, deep anyOf member). A `//` here would swallow the rest of the line,
            # so use the block form, which is terminated and cannot.
            return f"object /* recursive: {_ref} */"

        if _ref in self.processed_ref_cache:
            output = self.processed_ref_cache[_ref]
        else:
            entry_epoch = self._truncation_epoch
            self._ref_expansion_path.append(_ref)
            try:
                output = self._process_schema_recursive(_def)
            finally:
                if self._ref_expansion_path and self._ref_expansion_path[-1] == _ref:
                    self._ref_expansion_path.pop()
            if self._truncation_epoch == entry_epoch:
                self.processed_ref_cache[_ref] = output
        if "default" in value and self._should_include_metadata("default"):
            if isinstance(output, str):
                output = output + f" (default='{value['default']}')"

        # Include description from resolved definition if available
        if (
            isinstance(_def, dict)
            and "description" in _def
            and _def["description"]
            and self._should_include_metadata("description")
        ):
            def_description = f" {self.comment_prefix} {_def['description']}"
            if self.carries_deferred_comment(output):
                # Plain-scalar invariant: never append a bare `//` comment to a
                # marker-bearing representation. Suffixes appended later (` []` from a
                # container, ` OR null` from an anyOf) would land *after* that comment
                # and be swallowed into it by the hoist. Route the text into the
                # marker's slot instead -- and only when `process_enum` has not already
                # folded this same `$defs` description in.
                resolved = self.hoist_deferred_comments(str(output))
                if str(_def["description"]) not in resolved:
                    output = self.append_deferred_comment(str(output), str(_def["description"]))
            elif isinstance(output, str) and def_description not in output:
                output = str(output) + def_description
            elif isinstance(output, dict | list) and key is not None:
                self.pending_prefix[key] = def_description.strip()

        if isinstance(output, dict | list):
            return copy.deepcopy(output)
        return str(output) if not isinstance(output, str) else output

    def process_anyof(  # type: ignore[override]
        self, value: dict[str, Any], key: str | None = None
    ) -> str | dict[str, Any] | list[Any]:
        """
        Process anyOf union types.

        Args:
            value: Dictionary containing anyOf definition.
            key: Optional property key for postfix tracking.

        Returns:
            Formatted union type representation (string or dict for nested serialization).
        """
        comment = ""
        title, description, default_value, example = self._get_title_description_default_value(
            value
        )
        if self._is_root_schema(value):
            title, description = "", ""
        anyof_list = value.get("anyOf", [])
        items: list[dict[str, Any] | str | list[Any]] = []
        for item in anyof_list:
            # Include description from individual anyOf items inline
            item_desc = self._extract_description(item)
            if isinstance(item, dict) and item.get("$ref"):
                processed_item = self.process_ref(item, key)
            else:
                processed_item = self._process_schema_recursive(item)
            if item_desc:
                if isinstance(processed_item, dict | list):
                    item_str = (
                        self._jsonish_dump(processed_item, 0)
                        if isinstance(processed_item, dict | list)
                        else str(processed_item)
                    )
                    processed_item = item_str + item_desc
                else:
                    processed_item = str(processed_item) + item_desc
            items.append(processed_item)

        if description or default_value:
            comment = f" {self.comment_prefix}"
        if len(items) == 2 and isinstance(items[0], dict | list) and items[1] == "null":
            if key is not None:
                self.pending_postfix[key] = (
                    f"OR null {comment}{title}{description}{default_value}{example}"
                )
            first_item = items[0]
            if isinstance(first_item, dict | list):
                return first_item
            return str(first_item)
        else:
            str_items = [
                self._jsonish_dump(item, 0) if isinstance(item, dict | list) else str(item)
                for item in items
            ]
            sep = self.config.union_separator
            output = sep.join(str_items) if len(str_items) > 1 else str_items[0]

        return f"{output}{comment}{title}{description}{default_value}{example}"

    def process_oneof(  # type: ignore[override]
        self, value: dict[str, Any], key: str | None = None
    ) -> str | dict[str, Any] | list[Any]:
        """
        Process oneOf exclusive choice types.

        Args:
            value: Dictionary containing oneOf definition.
            key: Optional property key for postfix tracking.

        Returns:
            Formatted exclusive choice representation (string or dict).
        """
        comment = ""
        title, description, default_value, example = self._get_title_description_default_value(
            value
        )
        if self._is_root_schema(value):
            title, description = "", ""
        oneof_list = value.get("oneOf", [])
        items: list[dict[str, Any] | str | list[Any]] = []
        for item in oneof_list:
            # Include description from individual oneOf items inline
            item_desc = self._extract_description(item)
            if isinstance(item, dict) and item.get("$ref"):
                processed_item = self.process_ref(item, key)
            else:
                processed_item = self._process_schema_recursive(item)
            if item_desc:
                if isinstance(processed_item, dict | list):
                    item_str = (
                        self._jsonish_dump(processed_item, 0)
                        if isinstance(processed_item, dict | list)
                        else str(processed_item)
                    )
                    processed_item = item_str + item_desc
                else:
                    processed_item = str(processed_item) + item_desc
            items.append(processed_item)

        if description or default_value:
            comment = f" {self.comment_prefix}"
        if len(items) == 2 and isinstance(items[0], dict | list) and items[1] == "null":
            if key is not None:
                self.pending_postfix[key] = (
                    f"ONE OF: {comment}{title}{description}{default_value}{example}"
                )
            first_item = items[0]
            if isinstance(first_item, dict | list):
                return first_item
            return str(first_item)
        else:
            str_items = [
                self._jsonish_dump(item, 0) if isinstance(item, dict | list) else str(item)
                for item in items
            ]
            sep = self.config.union_separator
            one_of_output = "ONE OF: " + sep.join(str_items) if len(str_items) > 1 else str_items[0]

        return f"{one_of_output}{comment}{title}{description}{default_value}{example}"

    def _merge_allof_objects(
        self, items: list[dict[str, Any] | str | list[Any]]
    ) -> dict[str, Any] | None:
        """If all items are dicts, merge them by key (shallow merge). Otherwise return None."""
        dicts = [x for x in items if isinstance(x, dict)]
        if len(dicts) != len(items):
            return None
        merged: dict[str, Any] = {}
        for d in dicts:
            for k, v in d.items():
                if k == "__additional_properties__":
                    merged[k] = v
                else:
                    merged[k] = v
        return merged

    def process_allof(  # type: ignore[override]
        self, value: dict[str, Any], key: str | None = None
    ) -> str | dict[str, Any] | list[Any]:
        """
        Process allOf intersection types.

        Args:
            value: Dictionary containing allOf definition.
            key: Optional property key for postfix tracking.

        Returns:
            Formatted intersection representation (string or merged dict).
        """
        comment = ""
        # If this allOf node has its own description (e.g. same as parent items.description),
        # skip adding it again to avoid duplication when already shown in array header
        value_has_description = "description" in value and value.get("description")
        title, description, default_value, example = self._get_title_description_default_value(
            value
        )
        if self._is_root_schema(value):
            title, description = "", ""
        if value_has_description:
            description = ""
        allof_list = value.get("allOf", [])
        items: list[dict[str, Any] | str | list[Any]] = []
        for item in allof_list:
            # Include description from individual allOf items inline
            item_desc = self._extract_description(item)
            if isinstance(item, dict) and item.get("$ref"):
                processed_item = self.process_ref(item, key)
            else:
                processed_item = self._process_schema_recursive(item)
            if item_desc:
                if isinstance(processed_item, dict | list):
                    item_str = (
                        self._jsonish_dump(processed_item, 0)
                        if isinstance(processed_item, dict | list)
                        else str(processed_item)
                    )
                    processed_item = item_str + item_desc
                else:
                    processed_item = str(processed_item) + item_desc
            items.append(processed_item)
        if description or default_value:
            comment = f" {self.comment_prefix}"

        if len(items) == 2 and isinstance(items[0], dict | list) and items[1] == "null":
            if key is not None:
                self.pending_postfix[key] = (
                    f"AND null {comment}{title}{description}{default_value}{example}"
                )
            first_item = items[0]
            if isinstance(first_item, dict | list):
                return first_item
            return str(first_item)
        else:
            merged = self._merge_allof_objects(items)
            if merged is not None:
                return merged
            str_items = [
                self._jsonish_dump(item, 0) if isinstance(item, dict | list) else str(item)
                for item in items
            ]
            output = " AND ".join(str_items) if len(str_items) > 1 else str_items[0]

        return f"{output}{comment}{title}{description}{default_value}{example}"

    def process_enum(self, enum_value: dict[str, Any], key: str | None = None) -> str:
        """Thin shim: delegate to the shared ``BaseFormatter.process_enum``.

        Args:
            enum_value: Dictionary containing the enum definition. Renamed from ``value``
                to match the base signature, avoiding a ``# type: ignore[override]``.
            key: Unused. Kept only so the existing positional call sites need no change.

        Returns:
            ``super().process_enum(enum_value)``.
        """
        return super().process_enum(enum_value)

    def process_const(self, enum_value: dict[str, Any], key: str | None = None) -> str:
        """Thin shim: delegate to the shared ``BaseFormatter.process_const``.

        Args:
            enum_value: Dictionary containing the const definition.
            key: Unused. Kept only so the existing positional call site needs no change.

        Returns:
            ``super().process_const(enum_value)``.
        """
        return super().process_const(enum_value)

    def _is_root_schema(self, value: dict[str, Any]) -> bool:
        """Return True if value is the root schema (skip duplicating title/description)."""
        return value is self.schema

    def _additional_properties_key(self, schema: dict[str, Any]) -> str:
        """Sentinel dict key for an additionalProperties comment: root schema vs. nested."""
        return (
            "__root_additional_properties__"
            if self._is_root_schema(schema)
            else "__additional_properties__"
        )

    def process_types(
        self, value: dict[str, Any], key: str | None = None
    ) -> str | dict[str, Any] | list[Any]:
        """
        Process type fields with constraints.

        Args:
            value: Dictionary containing type definition.
            key: Optional property key for postfix tracking.

        Returns:
            Formatted type representation (string, list for arrays, or dict for object).
        """
        comment = ""
        title, description, default_value, example = self._get_title_description_default_value(
            value
        )
        if self._is_root_schema(value):
            title, description = "", ""
        format_, pattern = self._get_options_format_pattern(value)

        if "type" in value:
            if value["type"] == "string":
                type_name = "string"

                length_range = ""
                # Filter length constraints based on metadata_inclusion config
                check_min_len = self._should_include_metadata("minLength")
                check_max_len = self._should_include_metadata("maxLength")
                if check_min_len or check_max_len:
                    has_min = "minLength" in value and check_min_len
                    has_max = "maxLength" in value and check_max_len
                    if has_min and has_max:
                        length_range = f" ({value['minLength']}-{value['maxLength']} chars)"
                    elif has_min:
                        length_range += f" (>= {value['minLength']} chars)"
                    elif has_max:
                        length_range += f" (<= {value['maxLength']} chars)"
                if title or description or default_value or example:
                    comment = f" {self.comment_prefix}"
                return f"{type_name}{pattern}{format_}{length_range}{comment}{title}{description}{default_value}{example}"  # noqa: E501
            elif value["type"] in ["number", "integer"]:
                type_name = "float" if value["type"] == "number" else "int"
                value_range = ""
                # Filter range constraints based on metadata_inclusion config
                check_min = self._should_include_metadata("minimum")
                check_max = self._should_include_metadata("maximum")
                if check_min or check_max:
                    has_min = "minimum" in value and check_min
                    has_max = "maximum" in value and check_max
                    if has_min and has_max:
                        value_range = f" ({value['minimum']} to {value['maximum']})"
                    elif has_min:
                        value_range += f" (>= {value['minimum']})"
                    elif has_max:
                        value_range += f" (<= {value['maximum']})"
                if title or description or default_value or example:
                    comment = f" {self.comment_prefix}"
                return f"{type_name}{format_}{pattern}{value_range}{comment}{title}{description}{default_value}{example}"  # noqa: E501
            elif value["type"] == "boolean":
                type_name = "bool"
                if title or description or default_value or example:
                    comment = f" {self.comment_prefix}"
                return f"{type_name}{comment}{title}{description}{default_value}{example}"
            elif value["type"] == "array":
                shape = classify_container(value)
                if shape.kind == "tuple":
                    tuple_str = self.render_tuple(shape)
                    # Length-suffix suppression (design 4.2): when
                    # minItems == maxItems == len(prefix_schemas), the length token is
                    # redundant with the positional list and must not be appended.
                    if not (
                        value.get("minItems") == value.get("maxItems") == len(shape.prefix_schemas)
                    ):
                        tuple_str += self.format_array_constraints(value)
                    if title or description or default_value or example:
                        comment = f" {self.comment_prefix}"
                        if key is not None:
                            self.pending_postfix[key] = (
                                f"{comment}{title}{description}{default_value}{example}"
                            )
                    return tuple_str

                items_range = self.format_array_constraints(value)

                items: dict[str, Any] | str | list[Any] = {}
                if "items" in value and isinstance(value["items"], dict) and not value["items"]:
                    # ``items: {}`` is ``list[Any]`` -- classify_container rule 1 calls the
                    # element ``any``, so the list must render ``any []``, not ``[]``.
                    items = "any"
                elif "items" in value and value["items"]:
                    item_schema = value["items"]
                    if isinstance(item_schema, dict) and item_schema.get("$ref"):
                        items = self.process_ref(item_schema, key)
                    else:
                        items = self._process_schema_recursive(item_schema)
                # Note: items.description is now handled in transform_schema array header
                if items and isinstance(items, dict | list):
                    if items_range or title or description or default_value or example:
                        comment = f" {self.comment_prefix}"
                        if key is not None:
                            self.pending_postfix[key] = (
                                f"{comment}{title}{description}{items_range}{default_value}{example}"
                            )
                    if isinstance(items, dict):
                        return [items]
                    return items
                elif items and isinstance(items, str | int | float | bool):
                    # ``items_range`` is rendered before the comment marker, so it must
                    # not on its own justify emitting one.
                    comment = (
                        f" {self.comment_prefix}"
                        if (title or description or default_value or example)
                        else ""
                    )
                    return f"{items} []{items_range}{comment}{title}{description}{default_value}{example}"  # noqa: E501
                else:
                    comment = (
                        f" {self.comment_prefix}"
                        if (title or description or default_value or example)
                        else ""
                    )
                    return f"[]{items_range}{comment}{title}{description}{default_value}{example}"  # noqa: E501
            elif value["type"] == "object":
                if title or description or default_value or example:
                    comment = f" {self.comment_prefix}"
                    if key is not None:
                        self.pending_postfix[key] = (
                            f"{comment}{title}{description}{default_value}{example}"
                        )
                result = self._process_schema_recursive(value)
                return result if isinstance(result, dict) else str(result)
            elif value["type"] == "null":
                return "null"
            elif isinstance(value["type"], list):
                if title or description or default_value or example:
                    comment = f" {self.comment_prefix}"

                if len(value["type"]) == 1:
                    return f"{value['type'][0]} {format_}{pattern}{comment}{title}{description}{default_value}{example}"  # noqa: E501
                elif len(value["type"]) == 2 and "null" in value["type"]:
                    if "array" in value["type"]:
                        array_items: dict[str, Any] | str | list[Any] = {}
                        if title or description or default_value or example:
                            comment = f" {self.comment_prefix}"
                            if key is not None:
                                self.pending_postfix[key] = (
                                    f"or null {comment}{title}{description}{default_value}{example}"
                                )
                        if "items" in value and value["items"]:
                            array_items = self._process_schema_recursive(value["items"])
                        if isinstance(array_items, dict | list):
                            if isinstance(array_items, dict):
                                return [array_items]
                            return array_items
                        elif isinstance(array_items, str | int | float | bool):
                            return f"{array_items} []"
                    return f"{value['type'][0]} {format_}{pattern} or null {comment}{title}{description}{default_value}{example}"  # noqa: E501
                else:
                    return f"{', '.join(value['type'])} {format_}{pattern}{comment}{title}{description}{default_value}{example}"  # noqa: E501

        return ""

    def _render_mapping_value(
        self, value_schema: dict[str, Any]
    ) -> str | dict[str, Any] | list[Any]:
        """Dispatch a MAPPING's value schema through the same per-schema routing used for an
        object's own properties ($ref / anyOf / oneOf / allOf / enum / const / type / empty),
        without the required-marker bookkeeping that only applies to named object properties.

        Returns "any" for an empty/None schema (bare ``dict`` / ``additionalProperties: true``).
        """
        if not value_schema:
            return "any"
        if value_schema.get("$ref"):
            return self.process_ref(value_schema)
        if value_schema.get("anyOf"):
            return self.process_anyof(value_schema)
        if value_schema.get("oneOf"):
            return self.process_oneof(value_schema)
        if value_schema.get("allOf"):
            return self.process_allof(value_schema)
        if "const" in value_schema:
            return self.process_const(value_schema)
        if value_schema.get("enum"):
            return self.process_enum(value_schema)
        if "type" in value_schema:
            return self.process_types(value_schema)
        return "any"

    def _process_schema_recursive(self, schema: dict[str, Any]) -> dict[str, Any] | str | list[Any]:
        """
        Recursively process schema structure (trial's implementation).

        Args:
            schema: Schema dictionary to process.

        Returns:
            Processed schema as dict or string.
        """
        output: dict[str, Any] = {}
        required = schema.get("required", [])

        # Base-case: an empty schema ``{}`` means "any valid JSON value" (design 4.4).
        # Without this, ``Optional[Any]``'s ``anyOf: [{}, {"type": "null"}]`` drops its
        # first member and renders ``{} OR null`` instead of ``any OR null``.
        if not schema:
            return "any"

        # Base-case: an empty object schema like {"type": "object"} should render as
        # an empty object rather than recursing via process_types("object") and back here.
        if (
            schema.get("type") == "object"
            and not schema.get("properties")
            and not schema.get("anyOf")
            and not schema.get("oneOf")
            and not schema.get("allOf")
            and not schema.get("enum")
            and not schema.get("$ref")
        ):
            shape = classify_container(schema)
            if shape.kind == "mapping":
                key = f"<{self.key_token(shape)}>"
                output[key] = (
                    self._render_mapping_value(shape.value_schema)
                    if shape.value_schema is not None
                    else "any"
                )
                # No __additional_properties__ / "any properties allowed" comment for a
                # mapping (design 4.1 / 6.2) -- the <string> key + value say everything.
                return output

            # additionalProperties is False, or absent: unchanged legacy comment path.
            additional_props_comment = self.process_additional_properties(schema)
            if additional_props_comment:
                output[self._additional_properties_key(schema)] = additional_props_comment
            return output

        if "properties" in schema and schema["properties"]:
            for prop_name, value in schema["properties"].items():
                comment = ""
                field_dependencies = self._get_fields_dependencies(schema, prop_name)

                processed_prop_name = prop_name
                if prop_name in required:
                    processed_prop_name = f"{prop_name}{self.config.required_marker}"
                else:
                    processed_prop_name = f"{prop_name}{self.config.optional_marker}"
                if "$ref" in value and value["$ref"]:
                    output[processed_prop_name] = self.process_ref(value, processed_prop_name)
                elif "anyOf" in value and value["anyOf"]:
                    result = self.process_anyof(value, processed_prop_name)
                    output[processed_prop_name] = result
                elif "oneOf" in value and value["oneOf"]:
                    result = self.process_oneof(value, processed_prop_name)
                    output[processed_prop_name] = result
                elif "allOf" in value and value["allOf"]:
                    result = self.process_allof(value, processed_prop_name)
                    output[processed_prop_name] = result
                elif "const" in value:
                    # Priority: const before type/enum (single literal value)
                    output[processed_prop_name] = self.process_const(value, processed_prop_name)
                elif "enum" in value and value["enum"]:
                    # Priority: enum before type (multiple literal values)
                    output[processed_prop_name] = self.process_enum(value, processed_prop_name)
                elif "type" in value:
                    type_result: str | dict[str, Any] | list[Any] = self.process_types(
                        value, processed_prop_name
                    )
                    if isinstance(type_result, dict | list):
                        output[processed_prop_name] = type_result
                    elif isinstance(type_result, str):
                        output[processed_prop_name] = type_result
                    else:
                        output[processed_prop_name] = str(type_result)
                elif "properties" in value and value["properties"]:
                    nested = self._process_schema_recursive(value)
                    output[processed_prop_name] = nested
                else:
                    title, description, default_value, example = (
                        self._get_title_description_default_value(value)
                    )
                    if description or title or default_value or example:
                        comment_part = (
                            f" {self.comment_prefix}{title}{description}{default_value}{example}"
                        ).strip()
                        output[processed_prop_name] = (
                            f"any {comment_part}" if comment_part else "any"
                        )
                    else:
                        # Empty schema {} means any valid JSON value
                        if isinstance(value, dict) and not value:
                            output[processed_prop_name] = "any"
                        else:
                            output[processed_prop_name] = str(value)

                if (field_dependencies) and processed_prop_name not in self.pending_postfix:
                    comment = f" {self.comment_prefix}"
                    self.pending_postfix[processed_prop_name] = f"{comment}{field_dependencies}"

            # Set additionalProperties comment for object with properties if present
            additional_props_comment = self.process_additional_properties(schema)
            if additional_props_comment:
                output[self._additional_properties_key(schema)] = additional_props_comment

        elif "anyOf" in schema and schema["anyOf"]:
            return self.process_anyof(schema)
        elif "oneOf" in schema and schema["oneOf"]:
            return self.process_oneof(schema)
        elif "enum" in schema and schema["enum"]:
            return self.process_enum(schema)
        elif "type" in schema and schema["type"]:
            schema_type_result: str | dict[str, Any] | list[Any] = self.process_types(schema)
            if isinstance(schema_type_result, dict | list):
                return schema_type_result
            return (
                str(schema_type_result)
                if not isinstance(schema_type_result, str)
                else schema_type_result
            )
        elif "allOf" in schema and schema["allOf"]:
            return self.process_allof(schema)
        elif "$ref" in schema and schema["$ref"]:
            return self.process_ref(schema)

        return output

    def get_required_fields_comment(self) -> str:
        """
        Get a comment explaining the required field notation.

        Returns:
            Comment string explaining marker notation for required fields.
        """
        if not self.include_metadata:
            return ""
        if not self.schema.get("required", None):
            return ""
        marker = self.config.required_marker
        return f"{self.comment_prefix} Fields marked with {marker} are required\n"

    def get_schema_info_comment(self) -> str:
        """
        Get a comment containing schema title and description if present.

        Returns:
            Comment string with schema title and description, or empty string if neither present.
        """
        return self.get_info_comment(self.schema)

    def get_info_comment(self, schema: dict[str, Any]) -> str:
        """
        Get a comment containing schema title and description if present.

        Args:
            schema: Schema dictionary to extract info from.

        Returns:
            Comment string with schema title and description, or empty string if neither present.
        """
        if not self.include_metadata:
            return ""

        comments = []

        if "title" in schema and schema["title"] and self._should_include_metadata("title"):
            comments.append(f"{self.comment_prefix}Title: {schema['title']}")

        if (
            "description" in schema
            and schema["description"]
            and self._should_include_metadata("description")
        ):
            comments.append(f"{self.comment_prefix} {schema['description']}")

        if comments:
            return "\n".join(comments) + "\n"
        return ""

    def _jsonish_dump(
        self, obj: dict[str, Any] | list[Any] | Any, indent: int = 0, is_root: bool = False
    ) -> str:
        """
        Serialize object to JSONish format using json.dumps with postprocessing.

        Args:
            obj: Object to serialize (dict, list, or primitive).
            indent: Current indentation level (used for consistent spacing).
            is_root: True when serializing the top-level object.

        Returns:
            Serialized JSONish string.
        """
        # Step 1: Use standard json.dumps with indentation
        # Note: indent parameter is converted to match json.dumps expected behavior
        json_output = json.dumps(obj, indent=2, ensure_ascii=False)

        # Step 2: Process __additional_properties__ and convert to comments
        json_output = self._process_additional_properties(json_output, is_root=is_root)

        # Step 3: Remove quotes from keys and string values (JSONish style)
        json_output = self._remove_quotes(json_output)

        # Step 5: Apply final spacing normalization
        json_output = self._normalize_spacing(json_output)

        return json_output

    def _remove_quotes(self, json_string: str) -> str:
        """
        Remove JSON string-delimiter quotes from keys and values in JSONish format.

        A `"` character is removed iff it is an unescaped JSON string delimiter (i.e. the
        scanner is not currently inside a string when it is reached); every other character,
        including escape sequences such as `\\"`, `\\\\`, and `\\n`, is emitted verbatim.
        This method does not unescape anything -- see lsl-2026-09-04-002 for why partial
        unescaping is deliberately not performed here.

        Structural lines (`{`, `}`, `[`, `]`, or blank) and lines injected by
        `_process_additional_properties` (which start with `self.comment_prefix` and are not
        `json.dumps` output) are passed through unchanged.

        Args:
            json_string: JSON string with quotes, as produced by `json.dumps` and
                post-processed by `_process_additional_properties`.

        Returns:
            JSONish string with delimiter quotes removed and all other characters,
            including escaped content quotes and backslashes, preserved verbatim.
        """
        lines = json_string.split("\n")
        result_lines = []

        for line in lines:
            # Skip lines that are only whitespace or braces
            stripped = line.strip()
            if stripped in ["{", "}", "[", "]", ""]:
                result_lines.append(line)
                continue

            # Extract leading whitespace
            leading_space = len(line) - len(line.lstrip())
            content = line.strip()

            # Lines injected by _process_additional_properties (e.g. "// Root: ...") are not
            # json.dumps output; the scanner's precondition does not hold for them, so pass
            # them through unchanged rather than scanning.
            if content.startswith(self.comment_prefix):
                result_lines.append(" " * leading_space + content)
                continue

            scanned = self._scan_remove_string_delimiters(content)

            # Reconstruct line
            processed = " " * leading_space + scanned
            result_lines.append(processed)

        return "\n".join(result_lines)

    def _scan_remove_string_delimiters(self, content: str) -> str:
        """
        Remove unescaped JSON string-delimiter quotes from one line's content.

        Two-state scan (outside a string / inside a string) over `content`. A `"` reached
        while outside a string opens one and is dropped; a `"` reached while inside a string
        closes it and is dropped. A `\\` reached while inside a string consumes itself and the
        following character verbatim (so an escaped `\\"` is never seen as a closing
        delimiter). No unescaping is performed: everything other than a delimiter quote is
        emitted as-is.

        Args:
            content: One stripped, non-structural, non-comment-only line of `json.dumps`
                output (leading/trailing whitespace already removed by the caller).

        Returns:
            `content` with its unescaped string-delimiter quotes removed and every other
            character, including escape sequences, preserved verbatim.
        """
        in_string = False
        i = 0
        n = len(content)
        out: list[str] = []

        while i < n:
            char = content[i]

            if in_string and char == "\\":
                if i + 1 < n:
                    out.append(char)
                    out.append(content[i + 1])
                    i += 2
                else:
                    out.append(char)
                    i += 1
                continue

            if char == '"':
                in_string = not in_string
                i += 1
                continue

            out.append(char)
            i += 1

        return "".join(out)

    def _process_additional_properties(self, json_string: str, is_root: bool = False) -> str:
        """
        Convert __additional_properties__ to inline comments.

        Args:
            json_string: JSON string potentially containing __additional_properties__.
            is_root: True when processing the top-level object.

        Returns:
            JSON string with __additional_properties__ converted to comments.
        """
        import re

        lines = json_string.split("\n")
        result_lines = []
        i = 0
        root_comment_used = False

        while i < len(lines):
            line = lines[i]

            # Check if line contains one of the additionalProperties sentinel keys
            if "__additional_properties__" in line or "__root_additional_properties__" in line:
                # Extract the value
                # Pattern: "key": "value" or "key": "value",
                sentinel = (
                    "__root_additional_properties__"
                    if "__root_additional_properties__" in line
                    else "__additional_properties__"
                )
                match = re.search(rf'"{sentinel}"\s*:\s*"([^"]*)"', line)
                if match:
                    comment_value = match.group(1)

                    # The ``// Root:`` prefix is driven by which sentinel key matched, not by
                    # the ``is_root`` parameter, so a nested model never inherits it.
                    if sentinel == "__root_additional_properties__" and not root_comment_used:
                        # Extract content after // if present
                        rest = comment_value.strip()
                        if rest.startswith("//"):
                            suffix = rest[2:].strip()
                        else:
                            suffix = rest
                        comment = f" // Root: {suffix}"
                        root_comment_used = True
                    else:
                        comment = f" {comment_value}"

                    # Find the next closing brace and add comment before it
                    # Skip current line (don't add it to result)
                    j = i + 1
                    while j < len(lines):
                        if "}" in lines[j]:
                            # Add lines between __additional_properties__ and closing brace
                            for k in range(i + 1, j):
                                result_lines.append(lines[k])

                            # Add comment before closing brace
                            brace_line = lines[j]
                            stripped_brace = brace_line.strip()
                            if stripped_brace in ["}", "},"]:
                                # Insert comment on its own line before brace
                                indent = len(brace_line) - len(brace_line.lstrip())
                                result_lines.append(" " * indent + comment.strip())
                            result_lines.append(lines[j])
                            i = j
                            break
                        j += 1
                else:
                    # Pattern not matched, keep line as is
                    result_lines.append(line)
            else:
                result_lines.append(line)

            i += 1

        return "\n".join(result_lines)

    def _normalize_spacing(self, json_string: str) -> str:
        """
        Apply final spacing normalization.

        Args:
            json_string: JSONish string.

        Returns:
            Normalized JSONish string with single spaces.
        """
        # Replace double spaces with single spaces, but preserve leading indentation
        lines = json_string.split("\n")
        result_lines = []

        for line in lines:
            # Extract leading whitespace
            leading_space = len(line) - len(line.lstrip())
            content = line.lstrip()

            # Replace multiple spaces with single space in content
            while "  " in content:
                content = content.replace("  ", " ")

            # Reconstruct line with original indentation
            result_lines.append(" " * leading_space + content)

        return "\n".join(result_lines)

    def _delimiter_balance(self, line: str) -> int:
        """Net ``{``/``[`` minus ``}``/``]`` for one line, ignoring any trailing comment."""
        code = line.split(self.comment_prefix, 1)[0]
        return code.count("{") - code.count("}") + code.count("[") - code.count("]")

    def _collapse_array_object_brackets(self, output_string: str) -> str:
        """Collapse ``[``/``{`` and ``}``/``]`` line pairs into compact ``[{`` / ``}]``."""
        while True:
            lines = output_string.split("\n")
            collapsed = self._collapse_array_object_brackets_once(lines)
            if collapsed is None:
                return output_string
            output_string = "\n".join(collapsed)

    def _collapse_array_object_brackets_once(self, lines: list[str]) -> list[str] | None:
        """One collapse pass. Returns None when no collapsible pair is found."""
        for i in range(len(lines) - 1):
            if not lines[i].rstrip().endswith("[") or lines[i + 1].strip() != "{":
                continue
            depth = 0
            for j in range(i + 1, len(lines)):
                stripped = lines[j].strip()
                depth += self._delimiter_balance(stripped)
                if depth == 0:
                    if stripped != "}" or j + 1 >= len(lines):
                        return None
                    closer = lines[j + 1].strip()
                    if closer not in ("]", "],"):
                        return None
                    indent = " " * (len(lines[j + 1]) - len(lines[j + 1].lstrip()))
                    interior = [
                        line[2:] if line.startswith("  ") else line for line in lines[i + 2 : j]
                    ]
                    return (
                        lines[:i]
                        + [lines[i].rstrip() + "{"]
                        + interior
                        + [indent + "}" + closer]
                        + lines[j + 2 :]
                    )
            return None
        return None

    def _apply_pending_prefix(self, output_string: str) -> str:
        """Append opening-line comments (e.g. a ``$defs`` docstring) to block openers."""
        if not self.pending_prefix:
            return output_string
        lines = output_string.split("\n")
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            for key, prefix in self.pending_prefix.items():
                if stripped.startswith(f"{key}:") or stripped.startswith(f"{key}*:"):
                    if line.rstrip().endswith("{"):
                        lines[idx] = f"{line.rstrip()} {prefix}"
                    break
        return "\n".join(lines)

    def _join_postfix(self, line: str, postfix: str) -> str:
        """Append ``postfix`` to ``line``, keeping at most one comment marker on the line.

        When both the line and the postfix already carry ``comment_prefix``, the postfix's
        comment body is folded into the line's existing comment as a comma-separated
        continuation, and any segment already present on the line is dropped.
        """
        marker = self.comment_prefix
        if marker not in postfix or marker not in line:
            return f"{line} {postfix}"
        head, _, body = postfix.partition(marker)
        head = head.strip()
        if head and head in line:
            head = ""
        segments = [s.strip() for s in body.split(",") if s.strip() and s.strip() not in line]
        out = f"{line} {head}" if head else line
        return f"{out}, " + ", ".join(segments) if segments else out

    def _merge_pending_recursion(self) -> None:
        """Fold recursion notes into ``pending_postfix`` without overwriting an existing one."""
        for key, note in self.pending_recursion.items():
            existing = self.pending_postfix.get(key)
            if not existing:
                self.pending_postfix[key] = f"{self.comment_prefix} {note}"
            elif self.comment_prefix in existing:
                self.pending_postfix[key] = f"{existing.rstrip()}, {note}"
            else:
                self.pending_postfix[key] = f"{existing.rstrip()} {self.comment_prefix} {note}"
        self.pending_recursion.clear()

    def _apply_pending_postfix(self, output_string: str) -> str:
        """
        Apply pending postfix comments to output string.

        Args:
            output_string: The formatted output string.

        Returns:
            Output string with postfix comments applied.
        """
        if not self.pending_postfix:
            return output_string

        lines = output_string.split("\n")

        result_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this line contains a property that has a pending postfix
            property_found = False
            for key, postfix in self.pending_postfix.items():
                # Check both with and without asterisk
                key_with_asterisk = f"{key}*"

                # Check if this line starts with the key (with or without asterisk)
                stripped = line.lstrip()
                if stripped.startswith(f"{key}:") or stripped.startswith(f"{key_with_asterisk}:"):
                    # Count braces/brackets to find the matching closing one. A line whose
                    # delimiters are already balanced (e.g. ``k: {},``) is a single-line
                    # value and must take the else-branch below.
                    open_count = self._delimiter_balance(line)
                    if open_count > 0:
                        j = i + 1
                        processed_lines = [line]

                        # Find the line with the matching closing brace/bracket
                        while j < len(lines) and open_count > 0:
                            next_line = lines[j]
                            open_count += self._delimiter_balance(next_line)
                            processed_lines.append(next_line)
                            if open_count == 0:
                                # Found the closing brace/bracket
                                # Process nested content recursively before adding postfix
                                nested_content = "\n".join(
                                    processed_lines[1:-1]
                                )  # Exclude first and last line
                                if nested_content.strip():
                                    processed_nested = self._apply_pending_postfix(nested_content)
                                    # Reconstruct processed_lines with processed nested content
                                    processed_lines = (
                                        [processed_lines[0]]
                                        + processed_nested.split("\n")
                                        + [processed_lines[-1]]
                                    )

                                # Append postfix on closing line
                                closing_line = processed_lines[-1].rstrip()
                                if closing_line.endswith(","):
                                    closing_line = (
                                        self._join_postfix(closing_line[:-1].rstrip(), postfix)
                                        + ","
                                    )
                                else:
                                    closing_line = self._join_postfix(closing_line, postfix)
                                processed_lines[-1] = closing_line
                                # Skip this line in the next iteration
                                i = j
                                property_found = True
                                break
                            j += 1

                        if property_found:
                            # Add processed lines only on the success path; otherwise fall
                            # through so the line is emitted exactly once.
                            result_lines.extend(processed_lines)
                            break
                    else:
                        closing_line = line.rstrip()
                        if closing_line.endswith(","):
                            closing_line = (
                                self._join_postfix(closing_line[:-1].rstrip(), postfix) + ","
                            )
                        else:
                            closing_line = self._join_postfix(closing_line, postfix)
                        result_lines.append(closing_line)
                        property_found = True
                        break

            if not property_found:
                result_lines.append(line)

            i += 1

        return "\n".join(result_lines)

    def transform_schema(self) -> str:
        """
        Transform the schema into a simplified string representation.

        Returns:
            Formatted schema as a string.
        """
        if self.simplified_schema is not None:
            return self._add_prefix(self.simplified_schema)
        self._reset_ref_state()
        self.processed_ref_cache.clear()
        output = self._process_schema_recursive(self.schema)
        output_string = ""
        if output and isinstance(output, dict):
            output_string = self._jsonish_dump(output, indent=0, is_root=True)
            output_string = self._collapse_array_object_brackets(output_string)
        else:
            output_string = str(output)
        if self.schema.get("type") == "array":
            # Include items description in the array header if present (inline comment)
            items_desc = ""
            items_schema = self.schema.get("items")
            if isinstance(items_schema, dict) and items_schema.get("description"):
                items_desc = f" {self.comment_prefix} {items_schema['description']}"
            output_string = f"// Array of (items):{items_desc}\n{output_string}"
        output_string = f"{self.get_info_comment(self.schema)}{self.get_required_fields_comment()}{output_string}"  # noqa: E501
        notes = self.schema.get("notes")
        links = self.schema.get("links")
        if notes and self.include_metadata:
            notes_str = notes if isinstance(notes, str) else "\n".join(str(n) for n in notes)
            output_string = (
                f"{output_string}\n{self.comment_prefix} Notes:\n{self.comment_prefix} {notes_str}"
            )
        if links and self.include_metadata and isinstance(links, list):
            link_lines = []
            for link in links:
                if isinstance(link, dict):
                    href = link.get("href", link.get("url", ""))
                    method = link.get("method", "")
                    rel = link.get("rel", "")
                    part = f" [{method}]" if method else ""
                    part += f" rel={rel}" if rel else ""
                    link_lines.append(f"{href}{part}")
                else:
                    link_lines.append(str(link))
            if link_lines:
                output_string = f"{output_string}\n{self.comment_prefix} Links:\n" + "\n".join(
                    f"{self.comment_prefix} {L}" for L in link_lines
                )

        self._merge_pending_recursion()
        output_string = self._apply_pending_postfix(output_string)
        output_string = self._apply_pending_prefix(output_string)
        output_string = self.hoist_deferred_comments(output_string)
        self.simplified_schema = output_string.replace("  ", " ")
        return self._add_prefix(output_string)

    def _add_prefix(self, output_string: str) -> str:
        """Add prefix to output if configured."""
        if self.config.prefix:
            return self.config.prefix + output_string
        return output_string

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """
        Estimate token count for the simplified schema.

        Args:
            encoding: Tokenizer encoding to use (default: cl100k_base for GPT-4).

        Returns:
            Estimated token count.

        Raises:
            ImportError: If tiktoken is not installed.
        """
        try:
            import tiktoken

            enc = tiktoken.get_encoding(encoding)
            return len(enc.encode(self.transform_schema()))
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for token counting. Install it with: pip install tiktoken"
            ) from e

    def compare_tokens(
        self,
        original_schema: dict[str, Any] | None = None,
        simplified_schema: str | None = None,
        encoding: str = "cl100k_base",
    ) -> dict[str, Any]:
        """
        Compare token counts between original and simplified schemas.

        Args:
            original_schema: Original schema dict (uses stored if not provided).
            simplified_schema: Simplified schema string (uses generated if not provided).
            encoding: Tokenizer encoding to use.

        Returns:
            Dictionary with original, simplified, and reduction metrics.
        """
        try:
            import tiktoken

            enc = tiktoken.get_encoding(encoding)
            schema_to_compare = original_schema or self.schema

            original_str = json.dumps(schema_to_compare)
            simplified_str = simplified_schema or self.transform_schema()

            original_token_count = len(enc.encode(original_str))
            simplified_token_count = len(enc.encode(simplified_str))
            reduction_percent = (
                (original_token_count - simplified_token_count) / original_token_count * 100
            )

            return {
                "original_tokens": original_token_count,
                "simplified_tokens": simplified_token_count,
                "tokens_saved": original_token_count - simplified_token_count,
                "reduction_percent": round(reduction_percent, 2),
            }
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for token comparison. Install it with: pip install tiktoken"
            ) from e
