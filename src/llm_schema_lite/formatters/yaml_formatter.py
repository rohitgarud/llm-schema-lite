"""YAML-style formatter for transforming Pydantic schemas.

This formatter creates a clean YAML-like representation with Python-style
type hints, optionally including metadata as inline comments.
Feature parity with JSONish formatter for metadata, enums, unions, types,
dependencies, and $ref default.
"""

import json
from typing import Any

import yaml

from .base import BaseFormatter


class YAMLFormatter(BaseFormatter):
    """
    Transforms Pydantic schema into YAML-style format.

    This formatter creates a clean YAML-like representation with Python-style
    type hints, optionally including metadata as inline comments.

    Example output:
        name: str  # Product name, minLength: 1
        age: int  # min: 0, max: 120
        tags: list[str]  # Product tags
    """

    @property
    def TYPE_MAP(self) -> dict[str, str]:
        """Type mapping for YAML format (aligned with JSONish: string, int, float, bool)."""
        return {
            "string": "string",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
            "null": "None",
        }

    def _get_title_description_default_value(
        self, value: dict[str, Any]
    ) -> tuple[str, str, str, str]:
        """Extract title, description, default value, and example (JSONish-style for metadata)."""
        title = ""
        description = ""
        default_value = ""
        example = ""
        if "title" in value and value["title"] is not None:
            title = f" {value['title']}:"
        if "description" in value and value["description"] is not None:
            description = f" {value['description']}"
        if "id" in value and value["id"] is not None and value["id"] != "":
            id_ = value["id"]
            if isinstance(id_, str):
                id_ = f"'{id_}'"
            description += f" (id: {id_})"
        if "$comment" in value and value["$comment"] is not None:
            comment = value["$comment"]
            if isinstance(comment, str):
                comment = f"'{comment}'"
            description += f" (COMMENT: {comment})"
        if "default" in value:
            default = value["default"]
            if default is None:
                default = "null"
            elif isinstance(default, str):
                default = f"'{default}'"
            elif isinstance(default, bool):
                default = "true" if default else "false"
            default_value = f" (default={default})"
        if "example" in value and value["example"] is not None:
            example = f" (EXAMPLE: {value['example']})"
        elif "examples" in value and value["examples"] is not None:
            examples = value["examples"]
            if isinstance(examples, list):
                examples = [json.dumps(ex, indent=2).replace('"', "") for ex in examples]
                example = f" (EXAMPLES: {', '.join(examples)})"
            else:
                example = f" (EXAMPLES: {examples})"
        return title, description, default_value, example

    def _get_fields_dependencies(self, schema: dict[str, Any], field_name: str) -> str:
        """Extract dependencies for a field from schema (JSONish wording)."""
        if "dependencies" in schema and schema["dependencies"]:
            if field_name in schema["dependencies"]:
                dependencies = schema["dependencies"][field_name]
                if isinstance(dependencies, list):
                    return f"(DEPENDS ON: {', '.join(dependencies)})"
                return f"(DEPENDS ON: {dependencies})"
        return ""

    @property
    def comment_prefix(self) -> str:
        """Comment prefix for YAML format."""
        return "#"

    def _dump_yaml(self, data: dict[str, Any]) -> str:
        """
        Dump a dictionary to YAML format.

        Args:
            data: Dictionary to serialize to YAML.

        Returns:
            YAML string representation.
        """
        result = yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        return str(result).rstrip()

    def process_anyof(self, anyof: dict[str, Any]) -> str:
        """
        Process anyOf (union types) with OR keyword (JSONish parity).
        Optional null as postfix: two-item [T, null] -> "T OR null".
        """
        anyof_list = anyof.get("anyOf", [])
        if not anyof_list:
            return "string"

        item_types = []
        for item in anyof_list:
            if not isinstance(item, dict):
                continue
            if "enum" in item:
                item_types.append(self.process_enum(item))
            elif "const" in item:
                item_types.append(str(item["const"]))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "type" in item:
                type_name = item.get("type")
                if type_name == "null" or (isinstance(type_name, list) and "null" in type_name):
                    item_types.append("null")
                else:
                    item_types.append(self.process_type_value(item))
            elif "oneOf" in item:
                item_types.append(self.process_oneof(item))
            elif "allOf" in item:
                item_types.append(self.process_allof(item))

        return " OR ".join(item_types) if item_types else "string"

    def process_enum(self, enum_value: dict[str, Any]) -> str:
        """
        Process enum (JSONish parity): single value -> value; multiple -> OPTIONS: a | b | c.
        """
        enum_list = enum_value.get("enum", [])
        if not enum_list:
            return "string"
        if len(enum_list) == 1:
            return str(enum_list[0])
        return f"OPTIONS: {'| '.join(str(v) for v in enum_list)}"

    def process_oneof(self, oneof: dict[str, Any]) -> str:
        """
        Process oneOf with ONE OF: ... OR ... (JSONish parity).
        """
        oneof_list = oneof.get("oneOf", [])
        if not oneof_list:
            return "string"
        item_types = []
        for item in oneof_list:
            if not isinstance(item, dict):
                continue
            if "enum" in item:
                item_types.append(self.process_enum(item))
            elif "const" in item:
                item_types.append(str(item["const"]))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "type" in item:
                type_name = item.get("type")
                if type_name == "null" or (isinstance(type_name, list) and "null" in type_name):
                    item_types.append("null")
                else:
                    item_types.append(self.process_type_value(item))
            elif "anyOf" in item:
                item_types.append(self.process_anyof(item))
            elif "allOf" in item:
                item_types.append(self.process_allof(item))
        if not item_types:
            return "string"
        if len(item_types) > 1:
            return "ONE OF: " + " OR ".join(item_types)
        return str(item_types[0])

    def process_allof(self, allof: dict[str, Any]) -> str:
        """
        Process allOf with AND (JSONish parity). Expands object schemas with properties
        so merged allOf (e.g. name + age) appears in output.
        """
        allof_list = allof.get("allOf", [])
        if not allof_list:
            return "string"
        item_types = []
        for item in allof_list:
            if not isinstance(item, dict):
                continue
            if "type" in item and "properties" in item and item.get("type") == "object":
                # Expand object schemas so allOf merge shows field names (base-formatter behavior)
                processed_props = self.process_properties(item["properties"])
                items_structure = self.dict_to_string(processed_props, indent=2)
                item_types.append(f"{{\n{items_structure}\n}}")
            elif "properties" in item:
                processed_props = self.process_properties(item["properties"])
                item_types.append(self.dict_to_string(processed_props, indent=2))
            elif "enum" in item:
                item_types.append(self.process_enum(item))
            elif "const" in item:
                item_types.append(str(item["const"]))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "type" in item:
                type_name = item.get("type")
                if type_name == "null" or (isinstance(type_name, list) and "null" in type_name):
                    item_types.append("null")
                else:
                    item_types.append(self.process_type_value(item))
            elif "anyOf" in item:
                item_types.append(self.process_anyof(item))
            elif "oneOf" in item:
                item_types.append(self.process_oneof(item))
        if not item_types:
            return "string"
        if len(item_types) > 3:
            return f"allOf: {len(item_types)} schemas"
        if len(item_types) > 1:
            return " AND ".join(item_types)
        return str(item_types[0])

    def _format_string_constraints_jsonish(self, type_value: dict[str, Any]) -> str:
        """Format string constraints like JSONish: length, PATTERN, FORMAT."""
        parts = []
        min_len = type_value.get("minLength")
        max_len = type_value.get("maxLength")
        if min_len is not None and max_len is not None:
            parts.append(f"({min_len}-{max_len} chars)")
        elif min_len is not None:
            parts.append(f"(>= {min_len} chars)")
        elif max_len is not None:
            parts.append(f"(<= {max_len} chars)")
        if type_value.get("pattern"):
            parts.append(f"(PATTERN: {type_value['pattern']})")
        if type_value.get("format"):
            parts.append(f"(FORMAT: {type_value['format']})")
        elif type_value.get("_format"):
            parts.append(f"(FORMAT: {type_value['_format']})")
        return " ".join(parts)

    def _format_number_range_jsonish(self, type_value: dict[str, Any]) -> str:
        """Format number range like JSONish: (min to max), (>= min), (<= max)."""
        min_val = type_value.get("minimum")
        max_val = type_value.get("maximum")
        if min_val is not None and max_val is not None:
            return f"({min_val} to {max_val})"
        if min_val is not None:
            return f"(>= {min_val})"
        if max_val is not None:
            return f"(<= {max_val})"
        return ""

    def process_type_value(self, type_value: dict[str, Any]) -> str:
        """
        Process type with JSONish constraint phrasing: string (min-max chars), (PATTERN: ...),
        number (min to max), array UNIQUE / (min-max UNIQUE items), object {} or expanded.
        """
        type_name = type_value.get("type", "string")

        # Handle array of types (union types like ["string", "null"])
        if isinstance(type_name, list):
            if len(type_name) == 1:
                type_name = type_name[0]
            elif "null" in type_name and len(type_name) == 2:
                non_null_type = next(t for t in type_name if t != "null")
                type_str = self.TYPE_MAP.get(non_null_type, non_null_type)
                return f"{type_str} OR null"
            else:
                type_strs = [self.TYPE_MAP.get(t, t) for t in type_name if t != "null"]
                return " OR ".join(s for s in type_strs if s is not None)

        type_str = self.TYPE_MAP.get(type_name, type_name)

        if type_name == "string":
            extra = self._format_string_constraints_jsonish(type_value)
            if extra:
                type_str = f"{type_str} {extra}".strip()
        elif type_name in ["number", "integer"]:
            if self.include_metadata:
                range_info = self._format_number_range_jsonish(type_value)
                if range_info:
                    type_str = f"{type_str} {range_info}"
        elif type_name == "boolean":
            pass
        elif type_name == "array":
            items = type_value.get("items")
            if not items:
                type_str = "list[Any]"
            elif isinstance(items, bool):
                type_str = "list[Any]"
            elif isinstance(items, dict) and "type" in items:
                items_type = self.process_type_value(items)
                type_str = f"list[{items_type}]"
            elif isinstance(items, dict) and "$ref" in items:
                items_type = self.process_ref(items)
                type_str = f"list[{items_type}]"
            elif isinstance(items, dict) and "anyOf" in items:
                items_type = self.process_anyof(items)
                type_str = f"list[{items_type}]"
            elif isinstance(items, dict) and "oneOf" in items:
                items_type = self.process_oneof(items)
                type_str = f"list[{items_type}]"
            else:
                type_str = "list[Any]"

            if self.include_metadata:
                unique_items = type_value.get("_uniqueItems") or type_value.get("uniqueItems")
                unique_str = "UNIQUE " if unique_items else ""
                min_i = type_value.get("minItems")
                max_i = type_value.get("maxItems")
                if min_i is not None and max_i is not None:
                    type_str = f"{type_str} ({min_i}-{max_i} {unique_str}items)".strip()
                elif min_i is not None:
                    type_str = f"{type_str} (>= {min_i} {unique_str}items)".strip()
                elif max_i is not None:
                    type_str = f"{type_str} (<= {max_i} {unique_str}items)".strip()
                elif unique_str:
                    type_str = f"{type_str} ({unique_str}items)".strip()
                if "contains" in type_value:
                    type_str += self.process_contains(type_value)
            return self.add_metadata(type_str, type_value)
        elif type_name == "object":
            if (
                "properties" not in type_value
                and "patternProperties" not in type_value
                and not isinstance(type_value.get("additionalProperties"), dict)
            ):
                return "{}"
            return self.add_metadata("object", type_value)

        return self.add_metadata(str(type_str), type_value)

    def add_metadata(self, representation: str, value: dict[str, Any]) -> str:
        """
        Add metadata comments to a field representation (JSONish parity:
        title, description, id, $comment, default, example/examples).
        Always appends additionalProperties when present (even without include_metadata).
        """
        # Always show additionalProperties when present (e.g. "additional: string")
        if isinstance(value, dict) and value.get("additionalProperties") is not None:
            additional_comment = self.process_additional_properties(value)
            if additional_comment:
                representation = (
                    f"{representation}{additional_comment}"
                    if representation
                    else additional_comment.strip()
                )
        if not self.include_metadata:
            return representation

        title, description, default_value, example = self._get_title_description_default_value(
            value
        )
        parts = []
        if title:
            parts.append(title.strip())
        if description:
            parts.append(description.strip())
        if default_value:
            parts.append(default_value.strip())
        if example:
            parts.append(example.strip())

        # Base METADATA_MAP-style parts for pattern, format, etc. (when not in type)
        available_metadata = self.get_available_metadata(value)
        if available_metadata:
            metadata_parts = self.format_metadata_parts(value)
            if metadata_parts:
                parts.extend(metadata_parts)

        if not parts:
            return representation
        return f"{representation}  # {', '.join(parts)}"

    def process_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """
        Process properties and append per-field (DEPENDS ON: ...) when present (JSONish parity).
        For object properties with complex additionalProperties and no fixed properties,
        emit a dict with <key> placeholder.
        """
        self._nested_placeholder_count = 0
        processed_properties: dict[str, Any] = {}
        for prop_name, value in properties.items():
            formatted_name = self.format_field_name(prop_name)
            if isinstance(value, dict) and value.get("type") == "object":
                additional_props_raw = value.get("additionalProperties")
                is_object_with_additional = (
                    isinstance(additional_props_raw, dict)
                    and additional_props_raw.get("type") == "object"
                    and "additionalProperties" in additional_props_raw
                )
                is_complex = (
                    isinstance(additional_props_raw, dict)
                    and additional_props_raw
                    and (
                        "properties" in additional_props_raw
                        or "anyOf" in additional_props_raw
                        or "oneOf" in additional_props_raw
                        or "allOf" in additional_props_raw
                        or is_object_with_additional
                    )
                )
                if (
                    is_complex
                    and isinstance(additional_props_raw, dict)
                    and not (value.get("properties"))
                ):
                    additional_props = additional_props_raw
                    if "properties" in additional_props and additional_props["properties"]:
                        inner_required = set(additional_props.get("required", []))
                        inner = {}
                        for pname, pdef in additional_props["properties"].items():
                            fname = f"{pname}*" if pname in inner_required else pname
                            inner[fname] = self.process_property(pdef)
                        processed_properties[formatted_name] = {"<key>": inner}
                    else:
                        processed_properties[formatted_name] = {
                            "<key>": self.process_property(additional_props)
                        }
                    self._nested_placeholder_count += 1
                    continue
            prop_str = self.process_property(value)
            dep = self._get_fields_dependencies(self.schema, prop_name)
            if dep and self.include_metadata:
                prop_str = f"{prop_str}  # {dep}"
            processed_properties[formatted_name] = prop_str
        return processed_properties

    def process_ref(self, ref: dict[str, Any]) -> str:
        """
        Process $ref and append (default=...) when ref has default (JSONish parity).
        """
        result = super().process_ref(ref)
        if "default" in ref and result:
            default = ref["default"]
            if isinstance(default, str):
                result = f"{result} (default='{default}')"
            elif default is None:
                result = f"{result} (default=null)"
            elif isinstance(default, bool):
                result = f"{result} (default={'true' if default else 'false'})"
            else:
                result = f"{result} (default={default})"
        return result

    def get_schema_info_comment(self) -> str:
        """Schema title/description comment with JSONish-style wording and # prefix."""
        if not self.include_metadata:
            return ""
        comments = []
        if "title" in self.schema and self.schema["title"]:
            comments.append(f"Title: {self.schema['title']}")
        if "description" in self.schema and self.schema["description"]:
            comments.append(f"Description: {self.schema['description']}")
        if comments:
            return f"{self.comment_prefix} " + ", ".join(comments) + "\n"
        return ""

    def get_required_fields_comment(self) -> str:
        """Required fields comment (JSONish wording): Fields marked with * are required."""
        if not self.schema.get("required", None):
            return ""
        return f"{self.comment_prefix} Fields marked with * are required\n"

    def process_additional_properties(
        self, schema: dict[str, Any], show_structure: bool = True
    ) -> str:
        """Emit additionalProperties with # prefix for YAML (JSONish semantics)."""
        additional_props = schema.get("additionalProperties")
        if additional_props is False:
            return f" {self.comment_prefix} no additional properties"
        if isinstance(additional_props, dict) and additional_props:
            if not show_structure:
                return f" {self.comment_prefix} any properties allowed"
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
                details = ", ".join(prop_details)
                return f" {self.comment_prefix} additional: {type_str} with {details}"
            if required:
                req_str = ", ".join(required)
                return f" {self.comment_prefix} additional: {type_str} with required {req_str}"
            return f" {self.comment_prefix} additional: {type_str}"
        return ""

    def dict_to_string(self, value: Any, indent: int = 1) -> str:
        """
        Convert a dictionary to YAML-style key: value lines.

        This is used for formatting processed properties dict.

        Args:
            value: The value to convert (dict, list, or primitive).
            indent: Current indentation level (used for recursion depth
                tracking in nested structures).

        Returns:
            Formatted string representation as YAML key: value lines.
        """
        if isinstance(value, dict):
            if not value:  # Empty dict
                return "{}"

            # Format as YAML key: value pairs
            lines = []
            for k, v in value.items():
                lines.append(f"{k}: {v}")
            return "\n".join(lines)
        elif isinstance(value, list):
            # For arrays, show the item type
            if value and isinstance(value[0], dict):
                return f"list[{self.dict_to_string(value[0], indent + 1)}]"
            else:
                return "list"
        else:
            return str(value)

    def transform_schema(self) -> str:
        """
        Transform schema into YAML-style format.

        Returns:
            YAML-style schema definition as a string.
        """
        # First branch: if _processed_data is set, build from cache
        if hasattr(self, "_processed_data") and self._processed_data:
            all_sections = []

            # Process nested definitions first (from $defs) - same as main flow
            for def_name, def_schema in self.defs.items():
                if "properties" in def_schema:
                    nested_props = def_schema["properties"]
                    nested_required = set(def_schema.get("required", []))

                    # Build dict for this $def (with per-field DEPENDS ON from def_schema)
                    def_dict = {}
                    for prop_name, prop_def in nested_props.items():
                        prop_type = self.process_property(prop_def)
                        dep = self._get_fields_dependencies(def_schema, prop_name)
                        if dep and self.include_metadata:
                            prop_type = f"{prop_type}  # {dep}"
                        formatted_prop_name = (
                            f"{prop_name}*" if prop_name in nested_required else prop_name
                        )
                        def_dict[f"{def_name}.{formatted_prop_name}"] = prop_type

                    # Dump to YAML and optionally prepend section header
                    section_str = self._dump_yaml(def_dict)
                    if self.include_metadata:
                        section_str = f"# {def_name}\n{section_str}"
                        # Add additionalProperties comment if present
                        additional_props_comment = self.process_additional_properties(def_schema)
                        if additional_props_comment:
                            section_str += f"\n{additional_props_comment}"

                    all_sections.append(section_str)

            # Build main content from cached processed data
            main_parts = []

            # Add schema info comment if present
            schema_info_comment = self.get_schema_info_comment()
            if schema_info_comment:
                main_parts.append(schema_info_comment)

            # Add required fields comment if there are required fields
            required_comment = self.get_required_fields_comment()
            if required_comment:
                main_parts.append(required_comment)

            # Use cached processed data for main content
            main_parts.append(self._dump_yaml(self._processed_data))

            # Add additionalProperties comment (short if structure already in cached data)
            if "<key>" in self._processed_data:
                additional_props_comment = self.process_additional_properties(
                    self.schema, show_structure=False
                )
            elif self.include_metadata:
                additional_props_comment = self.process_additional_properties(self.schema)
            else:
                additional_props_comment = ""
            if additional_props_comment:
                main_parts.append(additional_props_comment)

            # Combine nested sections with main content
            if all_sections:
                all_sections.append("\n".join(main_parts))
                return "\n\n".join(all_sections)

            return "\n".join(main_parts)

        # Second branch: no properties - handle schema-level-only cases
        if not self.properties:
            # Check for complex additionalProperties in empty object schemas
            if self.schema.get("type") == "object":
                additional_props = self.schema.get("additionalProperties")
                is_object_with_additional = False
                if isinstance(additional_props, dict):
                    is_object_with_additional = (
                        additional_props.get("type") == "object"
                        and "additionalProperties" in additional_props
                    )
                is_complex = (
                    isinstance(additional_props, dict)
                    and additional_props
                    and (
                        "properties" in additional_props
                        or "anyOf" in additional_props
                        or "oneOf" in additional_props
                        or "allOf" in additional_props
                        or is_object_with_additional
                    )
                )

                if is_complex and isinstance(additional_props, dict):
                    # Build nested structure for placeholder key
                    output_dict: dict[str, Any]
                    if "properties" in additional_props and additional_props["properties"]:
                        inner_required = set(additional_props.get("required", []))
                        inner: dict[str, Any] = {}
                        for prop_name, prop_def in additional_props["properties"].items():
                            formatted_name = (
                                f"{prop_name}*" if prop_name in inner_required else prop_name
                            )
                            inner[formatted_name] = self.process_property(prop_def)
                        output_dict = {"<key>": inner}
                    else:
                        output_dict = {"<key>": self.process_property(additional_props)}

                    additional_comment = self.process_additional_properties(
                        self.schema, show_structure=False
                    )
                    result = self._dump_yaml(output_dict)
                    if additional_comment:
                        result += f"\n{additional_comment}"
                    return result

            # Handle schema-level features even when there are no properties
            schema_level_features = ""

            if "patternProperties" in self.schema:
                schema_level_features += self.process_pattern_properties(self.schema)

            if "dependencies" in self.schema:
                schema_level_features += self.process_dependencies(self.schema)

            if "if" in self.schema or "then" in self.schema or "else" in self.schema:
                schema_level_features += self.process_conditional(self.schema)

            if "propertyNames" in self.schema:
                schema_level_features += self.process_property_names(self.schema)

            if "unevaluatedProperties" in self.schema:
                schema_level_features += self.process_unevaluated_properties(self.schema)

            # Add additionalProperties to schema-level features
            additional_props = self.process_additional_properties(self.schema)
            if additional_props:
                schema_level_features += additional_props

            # Handle schema with type but no properties
            if "type" in self.schema:
                type_content = self.process_type_value(self.schema)
                # For object type with no properties, return {} instead of "object"/"dict"
                if type_content in ("object", "dict") and not schema_level_features:
                    return "{}"
                # Add schema-level features as comments (e.g. additionalProperties)
                if schema_level_features:
                    return (
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{type_content}"
                    )
                return type_content
            elif "oneOf" in self.schema:
                oneof_content = self.process_oneof(self.schema)
                if schema_level_features and self.include_metadata:
                    return (
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{oneof_content}"
                    )
                else:
                    return oneof_content
            elif "anyOf" in self.schema:
                anyof_content = self.process_anyof(self.schema)
                if schema_level_features and self.include_metadata:
                    return (
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{anyof_content}"
                    )
                else:
                    return anyof_content
            elif "allOf" in self.schema:
                allof_content = self.process_allof(self.schema)
                if schema_level_features and self.include_metadata:
                    return (
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{allof_content}"
                    )
                else:
                    return allof_content

            # Return schema-level features as comments if present
            if schema_level_features and self.include_metadata:
                return f"# Schema-level constraints: {schema_level_features.strip()}"
            else:
                return "{}"

        # Third branch: main flow with properties
        all_sections = []

        # Process nested definitions first (from $defs)
        for def_name, def_schema in self.defs.items():
            if "properties" in def_schema:
                nested_props = def_schema["properties"]
                nested_required = set(def_schema.get("required", []))

                # Build dict for this $def (with per-field DEPENDS ON from def_schema)
                def_dict = {}
                for prop_name, prop_def in nested_props.items():
                    prop_type = self.process_property(prop_def)
                    dep = self._get_fields_dependencies(def_schema, prop_name)
                    if dep and self.include_metadata:
                        prop_type = f"{prop_type}  # {dep}"
                    formatted_prop_name = (
                        f"{prop_name}*" if prop_name in nested_required else prop_name
                    )
                    def_dict[f"{def_name}.{formatted_prop_name}"] = prop_type

                # Dump to YAML and optionally prepend section header
                section_str = self._dump_yaml(def_dict)
                if self.include_metadata:
                    section_str = f"# {def_name}\n{section_str}"
                    # Add additionalProperties comment if present
                    additional_props_comment = self.process_additional_properties(def_schema)
                    if additional_props_comment:
                        section_str += f"\n{additional_props_comment}"

                all_sections.append(section_str)

        # Build main content parts
        main_parts = []

        # Add schema info comment if present
        schema_info_comment = self.get_schema_info_comment()
        if schema_info_comment:
            main_parts.append(schema_info_comment)

        # Add required fields comment if there are required fields
        required_comment = self.get_required_fields_comment()
        if required_comment:
            main_parts.append(required_comment)

        # Process properties and cache the result
        processed_properties = self.process_properties(self.properties)

        # Check for complex additionalProperties and add placeholder key if needed
        additional_props = self.schema.get("additionalProperties")
        is_object_with_additional = False
        if isinstance(additional_props, dict):
            is_object_with_additional = (
                additional_props.get("type") == "object"
                and "additionalProperties" in additional_props
            )
        is_complex_additional = (
            isinstance(additional_props, dict)
            and additional_props
            and (
                "properties" in additional_props
                or "anyOf" in additional_props
                or "oneOf" in additional_props
                or "allOf" in additional_props
                or is_object_with_additional
            )
        )
        if is_complex_additional and isinstance(additional_props, dict):
            if "properties" in additional_props and additional_props["properties"]:
                inner_required = set(additional_props.get("required", []))
                inner = {}
                for prop_name, prop_def in additional_props["properties"].items():
                    formatted_name = f"{prop_name}*" if prop_name in inner_required else prop_name
                    inner[formatted_name] = self.process_property(prop_def)
                processed_properties["<key>"] = inner
            else:
                processed_properties["<key>"] = self.process_property(additional_props)

        # Dump processed properties to YAML
        main_content = self._dump_yaml(processed_properties)
        if getattr(self, "_nested_placeholder_count", 0) > 0:
            main_content += "\n# any properties allowed"
        main_parts.append(main_content)

        # Add additionalProperties comment: for complex use short "any properties allowed"
        if is_complex_additional:
            additional_props_comment = self.process_additional_properties(
                self.schema, show_structure=False
            )
        elif self.include_metadata:
            additional_props_comment = self.process_additional_properties(self.schema)
        else:
            additional_props_comment = ""
        if additional_props_comment:
            main_parts.append(additional_props_comment)

        # Set _processed_data for future calls (caching)
        self._processed_data = processed_properties

        # If there are nested sections, combine them
        if all_sections:
            all_sections.append("\n".join(main_parts))
            return "\n\n".join(all_sections)

        return "\n".join(main_parts)
