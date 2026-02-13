"""JSONish formatter for transforming Pydantic schemas into BAML-like format."""

import json
from typing import Any

from .base import BaseFormatter


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

    def __init__(self, schema: dict[str, Any], include_metadata: bool = True):
        """
        Initialize the JSONish formatter.

        Args:
            schema: JSON schema from Pydantic model_json_schema.
            include_metadata: Whether to include metadata in the output.
        """
        super().__init__(schema, include_metadata)
        # Trial-specific state
        self.processed_ref_cache: dict[str, dict[str, Any] | str] = {}
        self.pending_postfix: dict[str, str] = {}
        self.simplified_schema: str | None = None

    @property
    def TYPE_MAP(self) -> dict[str, str]:
        """Type mapping for JSONish format."""
        return {"number": "float", "integer": "int", "boolean": "bool"}

    @property
    def comment_prefix(self) -> str:
        """Comment prefix for JSONish format."""
        return "//"

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

    def _get_title_description_default_value(
        self, value: dict[str, Any]
    ) -> tuple[str, str, str, str]:
        """Extract title, description, default value, and example from schema value."""
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

    def _get_options_format_pattern(self, value: dict[str, Any]) -> tuple[str, str, str]:
        """Extract options, format, and pattern from schema value."""
        options = ""
        format_ = ""
        pattern = ""
        if "pattern" in value and value["pattern"]:
            pattern = f" (PATTERN: {value['pattern']})"
        if "enum" in value and value["enum"]:
            # Convert enum values to strings to handle both string and numeric enums
            enum_strs = [str(v) for v in value["enum"]]
            options = f" (OPTIONS: {'| '.join(enum_strs)})"
        if "format" in value and value["format"]:
            format_ = f" (FORMAT: {value['format']})"
        elif "_format" in value and value["_format"]:
            format_ = f" (FORMAT: {value['_format']})"
        return options, format_, pattern

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

    def process_ref(self, value: dict[str, Any], key: str | None = None) -> str:
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

        if _ref in self.processed_ref_cache:
            output = self.processed_ref_cache[_ref]
        else:
            output = self._process_schema_recursive(_def)
            self.processed_ref_cache[_ref] = output
        if "default" in value:
            if isinstance(output, str):
                output = output + f" (default='{value['default']}')"
        return str(output) if not isinstance(output, str) else output

    def process_anyof(  # type: ignore[override]
        self, value: dict[str, Any], key: str | None = None
    ) -> str | dict[str, Any]:
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
        items: list[dict[str, Any] | str] = []
        for item in anyof_list:
            items.append(self._process_schema_recursive(item))

        if description or default_value:
            comment = f" {self.comment_prefix}"
        if len(items) == 2 and isinstance(items[0], dict | list) and items[1] == "null":
            if key is not None:
                self.pending_postfix[key] = (
                    f"OR null {comment}{title}{description}{default_value}{example}"
                )
            first_item = items[0]
            if isinstance(first_item, dict):
                return first_item
            return str(first_item)
        else:
            str_items = [
                self._jsonish_dump(item, 0) if isinstance(item, dict | list) else str(item)
                for item in items
            ]
            output = " OR ".join(str_items) if len(str_items) > 1 else str_items[0]

        return f"{output}{comment}{title}{description}{default_value}{example}"

    def process_oneof(  # type: ignore[override]
        self, value: dict[str, Any], key: str | None = None
    ) -> str | dict[str, Any]:
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
        items: list[dict[str, Any] | str] = []
        for item in oneof_list:
            items.append(self._process_schema_recursive(item))

        if description or default_value:
            comment = f" {self.comment_prefix}"
        if len(items) == 2 and isinstance(items[0], dict | list) and items[1] == "null":
            if key is not None:
                self.pending_postfix[key] = (
                    f"ONE OF: {comment}{title}{description}{default_value}{example}"
                )
            first_item = items[0]
            if isinstance(first_item, dict):
                return first_item
            return str(first_item)
        else:
            str_items = [
                self._jsonish_dump(item, 0) if isinstance(item, dict | list) else str(item)
                for item in items
            ]
            output = "ONE OF: " + " OR ".join(str_items) if len(str_items) > 1 else str_items[0]

        return f"{output}{comment}{title}{description}{default_value}{example}"

    def _merge_allof_objects(self, items: list[dict[str, Any] | str]) -> dict[str, Any] | None:
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
    ) -> str | dict[str, Any]:
        """
        Process allOf intersection types.

        Args:
            value: Dictionary containing allOf definition.
            key: Optional property key for postfix tracking.

        Returns:
            Formatted intersection representation (string or merged dict).
        """
        comment = ""
        title, description, default_value, example = self._get_title_description_default_value(
            value
        )
        if self._is_root_schema(value):
            title, description = "", ""
        allof_list = value.get("allOf", [])
        items: list[dict[str, Any] | str] = []
        for item in allof_list:
            items.append(self._process_schema_recursive(item))
        if description or default_value:
            comment = f" {self.comment_prefix}"

        if len(items) == 2 and isinstance(items[0], dict | list) and items[1] == "null":
            if key is not None:
                self.pending_postfix[key] = (
                    f"AND null {comment}{title}{description}{default_value}{example}"
                )
            first_item = items[0]
            if isinstance(first_item, dict):
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

    def process_enum(self, value: dict[str, Any], key: str | None = None) -> str:
        """
        Process enum fields.

        Args:
            value: Dictionary containing enum definition.
            key: Optional property key for postfix tracking.

        Returns:
            Formatted enum representation.
        """
        comment = ""
        title, description, default_value, example = self._get_title_description_default_value(
            value
        )
        enum_list = value.get("enum", [])
        if description or default_value:
            comment = f" {self.comment_prefix}"
        if len(enum_list) == 1:
            return f"{enum_list[0]}{comment}{title}{description}{default_value}{example}"
        else:
            return f"OPTIONS: {'| '.join(enum_list)}{comment}{title}{description}{default_value}{example}"  # noqa: E501

    def _is_root_schema(self, value: dict[str, Any]) -> bool:
        """Return True if value is the root schema (skip duplicating title/description)."""
        return value is self.schema

    def process_types(self, value: dict[str, Any], key: str | None = None) -> str | dict[str, Any]:
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
        options, format_, pattern = self._get_options_format_pattern(value)

        if "type" in value:
            if value["type"] == "string":
                type_name = "string"

                length_range = ""
                if "minLength" in value and "maxLength" in value:
                    length_range = f" ({value['minLength']}-{value['maxLength']} chars)"
                elif "minLength" in value:
                    length_range += f" (>= {value['minLength']} chars)"
                elif "maxLength" in value:
                    length_range += f" (<= {value['maxLength']} chars)"
                if title or description or options or default_value or example:
                    comment = f" {self.comment_prefix}"
                return f"{type_name}{pattern}{format_}{length_range}{comment}{title}{description}{options}{default_value}{example}"  # noqa: E501
            elif value["type"] in ["number", "integer"]:
                type_name = "float" if value["type"] == "number" else "int"
                value_range = ""
                if "minimum" in value and "maximum" in value:
                    value_range = f" ({value['minimum']} to {value['maximum']})"
                elif "minimum" in value:
                    value_range += f" (>= {value['minimum']})"
                elif "maximum" in value:
                    value_range += f" (<= {value['maximum']})"
                if title or description or options or default_value or example:
                    comment = f" {self.comment_prefix}"
                return f"{type_name}{format_}{pattern}{value_range}{comment}{title}{description}{options}{default_value}{example}"  # noqa: E501
            elif value["type"] == "boolean":
                type_name = "bool"
                if title or description or default_value or example:
                    comment = f" {self.comment_prefix}"
                return f"{type_name}{comment}{title}{description}{default_value}{example}"
            elif value["type"] == "array":
                unique_items = ""
                if ("_uniqueItems" in value and value["_uniqueItems"]) or (
                    "uniqueItems" in value and value["uniqueItems"]
                ):
                    unique_items = "UNIQUE"

                items_range = ""
                if "minItems" in value and "maxItems" in value:
                    items_range = f" ({value['minItems']}-{value['maxItems']} {unique_items} items)"
                elif "minItems" in value:
                    items_range += f" (>= {value['minItems']} {unique_items} items)"
                elif "maxItems" in value:
                    items_range += f" (<= {value['maxItems']} {unique_items} items)"
                elif unique_items:
                    items_range = f" ({unique_items} items)"

                items: dict[str, Any] | str | list[Any] = {}
                if "items" in value and value["items"]:
                    items = self._process_schema_recursive(value["items"])
                if items and isinstance(items, dict | list):
                    if items_range or title or description or default_value or example:
                        comment = f" {self.comment_prefix}"
                        if key is not None:
                            self.pending_postfix[key] = (
                                f"{comment}{title}{description}{items_range}{default_value}{example}"
                            )
                    return str([items]) if isinstance(items, dict) else str(items)
                elif items and isinstance(items, str | int | float | bool):
                    comment = f" {self.comment_prefix}"
                    return f"{items} []{items_range}{comment}{title}{description}{default_value}{example}"  # noqa: E501
                else:
                    comment = f" {self.comment_prefix}"
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
                    return f"{value['type'][0]} {format_}{pattern}{comment}{title}{description}{options}{default_value}{example}"  # noqa: E501
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
                                return str([array_items])
                            return str(array_items)
                        elif isinstance(array_items, str | int | float | bool):
                            return f"{array_items} []"
                    return f"{value['type'][0]} {format_}{pattern} or null {comment}{title}{description}{options}{default_value}{example}"  # noqa: E501
                else:
                    return f"{', '.join(value['type'])} {format_}{pattern}{comment}{title}{description}{options}{default_value}{example}"  # noqa: E501

        return ""

    def _process_schema_recursive(self, schema: dict[str, Any]) -> dict[str, Any] | str:
        """
        Recursively process schema structure (trial's implementation).

        Args:
            schema: Schema dictionary to process.

        Returns:
            Processed schema as dict or string.
        """
        output: dict[str, Any] = {}
        required = schema.get("required", [])

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
            # Set additionalProperties comment for empty object if present
            additional_props_comment = self.process_additional_properties(schema)
            if additional_props_comment:
                output["__additional_properties__"] = additional_props_comment
            return output

        if "properties" in schema and schema["properties"]:
            for prop_name, value in schema["properties"].items():
                comment = ""
                field_dependencies = self._get_fields_dependencies(schema, prop_name)

                processed_prop_name = prop_name
                if prop_name in required:
                    processed_prop_name = f"{prop_name}*"
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
                elif "type" in value:
                    result = self.process_types(value, processed_prop_name)
                    if isinstance(result, dict):
                        output[processed_prop_name] = result
                    elif isinstance(result, str):
                        output[processed_prop_name] = result
                    else:
                        output[processed_prop_name] = str(result)
                elif "enum" in value and value["enum"]:
                    output[processed_prop_name] = self.process_enum(value, processed_prop_name)
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
                        output[processed_prop_name] = str(value)

                if (field_dependencies) and processed_prop_name not in self.pending_postfix:
                    comment = f" {self.comment_prefix}"
                    self.pending_postfix[processed_prop_name] = f"{comment}{field_dependencies}"

            # Set additionalProperties comment for object with properties if present
            additional_props_comment = self.process_additional_properties(schema)
            if additional_props_comment:
                output["__additional_properties__"] = additional_props_comment

        elif "anyOf" in schema and schema["anyOf"]:
            return self.process_anyof(schema)
        elif "oneOf" in schema and schema["oneOf"]:
            return self.process_oneof(schema)
        elif "type" in schema and schema["type"]:
            result = self.process_types(schema)
            if isinstance(result, dict):
                return result
            return str(result) if not isinstance(result, str) else result
        elif "allOf" in schema and schema["allOf"]:
            return self.process_allof(schema)
        elif "enum" in schema and schema["enum"]:
            return self.process_enum(schema)
        elif "$ref" in schema and schema["$ref"]:
            return self.process_ref(schema)

        return output

    def get_required_fields_comment(self) -> str:
        """
        Get a comment explaining the required field notation.

        Returns:
            Comment string explaining asterisk notation for required fields.
        """
        if not self.schema.get("required", None):
            return ""
        return f"{self.comment_prefix} Fields marked with * are required\n"

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

        if "title" in schema and schema["title"]:
            comments.append(f"{self.comment_prefix}Title: {schema['title']}")

        if "description" in schema and schema["description"]:
            comments.append(f"{self.comment_prefix} {schema['description']}")

        if comments:
            return "\n".join(comments) + "\n"
        return ""

    def _jsonish_dump(
        self, obj: dict[str, Any] | list[Any] | Any, indent: int = 0, is_root: bool = False
    ) -> str:
        """
        Custom serializer for JSONish format that handles __additional_properties__.

        Args:
            obj: Object to serialize (dict, list, or primitive).
            indent: Current indentation level.
            is_root: True when serializing the top-level object (for root additionalProperties).

        Returns:
            Serialized JSONish string.
        """
        if isinstance(obj, dict):
            # Extract the reserved key if present
            additional_props_comment: str = str(obj.get("__additional_properties__", ""))
            if is_root and additional_props_comment:
                rest = additional_props_comment.strip()
                suffix = rest[2:].strip() if rest.startswith("//") else rest
                additional_props_comment = f" // Root: {suffix}"

            # Build the dict content (excluding the reserved key)
            parts = []
            for k, v in obj.items():
                if k == "__additional_properties__":
                    continue
                # Recursively serialize the value
                serialized_value = self._jsonish_dump(v, indent + 1, is_root=False)
                # For multi-line values, indent properly
                if "\n" in serialized_value:
                    parts.append(f"{k}: {serialized_value}")
                else:
                    parts.append(f"{k}: {serialized_value}")

            if not parts:
                # Empty dict
                if additional_props_comment:
                    return "{" + additional_props_comment + "\n" + " " * indent + "}"
                return "{}"

            # Build the result
            indent_str = " " * (indent + 1)
            content = (",\n" + indent_str).join(parts)
            result = "{\n" + indent_str + content + "\n" + " " * indent + "}"

            # Add the comment before the closing brace if present
            if additional_props_comment:
                # Insert comment before the last closing brace
                lines = result.rsplit("\n", 1)
                if len(lines) == 2:
                    result = lines[0] + additional_props_comment + "\n" + lines[1]

            return result
        elif isinstance(obj, list):
            # Handle list serialization
            if not obj:
                return "[]"
            serialized_items = [self._jsonish_dump(item, indent + 1) for item in obj]
            if any("\n" in item for item in serialized_items):
                # Multi-line items
                indent_str = " " * (indent + 1)
                content = (",\n" + indent_str).join(serialized_items)
                return "[\n" + indent_str + content + "\n" + " " * indent + "]"
            else:
                # Single-line items
                return "[" + ", ".join(serialized_items) + "]"
        elif isinstance(obj, str):
            # Don't add quotes (JSONish style)
            return obj
        elif obj is None:
            return "null"
        elif isinstance(obj, bool):
            return "true" if obj else "false"
        else:
            return str(obj)

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
                    # Check if the value is a dict or list on the same line or next lines
                    if "{" in line or "[" in line:
                        # Count braces/brackets to find the matching closing one
                        open_count = (
                            line.count("{") - line.count("}") + line.count("[") - line.count("]")
                        )
                        j = i + 1
                        processed_lines = [line]

                        # Find the line with the matching closing brace/bracket
                        while j < len(lines) and open_count > 0:
                            next_line = lines[j]
                            open_count += (
                                next_line.count("{")
                                - next_line.count("}")
                                + next_line.count("[")
                                - next_line.count("]")
                            )
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
                                    closing_line = closing_line[:-1].rstrip() + f" {postfix},"
                                else:
                                    closing_line = closing_line + f" {postfix}"
                                processed_lines[-1] = closing_line
                                # Skip this line in the next iteration
                                i = j
                                property_found = True
                                break
                            j += 1

                        # Add all processed lines to result
                        result_lines.extend(processed_lines)
                        if property_found:
                            break
                    else:
                        closing_line = line.rstrip()
                        if closing_line.endswith(","):
                            closing_line = closing_line[:-1].rstrip() + f" {postfix},"
                        else:
                            closing_line = closing_line + f" {postfix}"
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
            return self.simplified_schema
        output = self._process_schema_recursive(self.schema)
        output_string = ""
        if output and isinstance(output, dict):
            output_string = self._jsonish_dump(output, indent=0, is_root=True).replace('"', "")
        else:
            output_string = str(output)
        if self.schema.get("type") == "array":
            output_string = f"// Array of (items):\n{output_string}"
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

        output_string = self._apply_pending_postfix(output_string)
        self.simplified_schema = output_string.replace("  ", " ")
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
