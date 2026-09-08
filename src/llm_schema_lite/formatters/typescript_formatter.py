"""TypeScript interface formatter for transforming Pydantic schemas."""

import re
from io import StringIO
from typing import Any, Final

from .base import BaseFormatter, ContainerShape, classify_container, format_literal_value


class TypeScriptFormatter(BaseFormatter):
    """
    Transforms Pydantic schema into TypeScript interface format.

    This formatter follows Pattern A: it uses base class schema processing
    (via process_schema and related methods) and caches processed data in
    _processed_data within transform_schema for improved performance on
    subsequent calls.

    This formatter creates TypeScript-style interface definitions,
    optionally including metadata as inline comments.

    Example output:
        interface Schema {
          name: string;  // Product name, minLength: 1
          age: number;  // min: 0, max: 120
          is_active: boolean;  // (defaults to true)
        }
    """

    @property
    def TYPE_MAP(self) -> dict[str, str]:
        """Type mapping for TypeScript format."""
        return {
            "string": "string",
            "integer": "number",
            "number": "number",
            "boolean": "boolean",
            "array": "Array",
            "object": "object",
            "null": "null",
        }

    @property
    def comment_prefix(self) -> str:
        """Comment prefix for TypeScript format."""
        return "//"

    def add_metadata(self, representation: str, value: dict[str, Any]) -> str:
        """
        Add metadata comments to a field representation.

        Args:
            representation: The base field representation.
            value: The field definition containing metadata.

        Returns:
            Field representation with metadata comments. A multi-line description
            (a real ``"\\n"`` in a metadata part) becomes an inline first line plus one or
            more continuation comment lines at a fixed 2-space member indent.
        """
        if not self.include_metadata:
            return representation

        # The const value is already the rendered type token (process_const); METADATA_MAP
        # would otherwise restate it as "const: X" right beside it. Mirrors
        # YAMLFormatter.add_metadata's exclude for the same key. "(defaults to X)" is NOT
        # excluded -- a default beside a const is real information.
        metadata_parts = [p for p in self.format_metadata_parts(value, exclude=("const",)) if p]
        if not metadata_parts:
            return representation

        joined = ", ".join(metadata_parts)
        first, _, rest = joined.partition("\n")
        rendered = f"{representation}  // {first}"
        if not rest:
            return rendered
        # Every interface member is emitted at a fixed two-space indent (see the
        # f"  {name}: {type};" sites), so the continuation indent is a constant --
        # unlike YAML, TypeScript has no depth-varying member indent to derive.
        return "\n".join([rendered, *self.comment_lines(rest, "  ")])

    def process_additional_properties(
        self, schema: dict[str, Any], show_structure: bool = True
    ) -> str:
        """Process additionalProperties constraint for TypeScript."""
        additional_props = schema.get("additionalProperties")
        if additional_props is False:
            return " // no additional properties"
        elif isinstance(additional_props, dict) and additional_props:
            if not schema.get("properties"):
                # Pure mapping (classify_container rule 6/C1): the value type is rendered
                # structurally by the caller's mapping renderer, never as a comment.
                return ""
            if not show_structure:
                # Structure shown via placeholder key, just indicate it's allowed
                return " // any properties allowed"

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
                return f" // additional: {type_str} with {', '.join(prop_details)}"
            if required:
                return f" // additional: {type_str} with required {', '.join(required)}"
            return f" // additional: {type_str}"
        return ""

    def _is_complex_additional_props(self, schema: dict[str, Any]) -> bool:
        """True when ``schema`` classifies as a MAPPING (see `classify_container`)."""
        return classify_container(schema).kind == "mapping"

    def process_anyof(self, anyof: dict[str, Any]) -> str:
        """
        Process an anyOf field (union types) for TypeScript.

        Args:
            anyof: Dictionary containing anyOf definition.

        Returns:
            Formatted union type representation.
        """
        anyof_list = anyof.get("anyOf", [])
        if not anyof_list:
            return "string"

        item_types = []
        for item in anyof_list:
            if not isinstance(item, dict):
                continue
            if not item:
                item_types.append("any")
                continue

            # Check for properties BEFORE type, because objects with properties have both
            if "properties" in item:
                # Handle object schemas in anyOf - format as TypeScript object type
                processed_props = self.process_properties(item["properties"])
                props_str = ", ".join(f"{k}: {v}" for k, v in processed_props.items())
                item_types.append(f"{{ {props_str} }}")
            elif "enum" in item:
                item_types.append(self.process_enum(item))
            elif "const" in item:
                item_types.append(self.process_const(item))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "type" in item:
                item_types.append(self.process_type_value(item))

        # Limit the number of union types to prevent excessive expansion
        # Use same tiers as base formatter
        if self._global_expansion_count > 100:
            max_items = 2  # Very aggressive for deep recursion
        elif self._global_expansion_count > 30:
            max_items = 3  # Aggressive
        elif self._global_expansion_count > 10:
            max_items = 4  # Moderate
        else:
            max_items = 5  # Conservative start

        if len(item_types) > max_items:
            return f"anyOf: {len(item_types)} options"
        else:
            return self.config.union_separator.join(item_types) if item_types else "string"

    def process_enum(self, enum_value: dict[str, Any]) -> str:
        """
        Process an enum field for TypeScript.

        Args:
            enum_value: Dictionary containing enum definition.

        Returns:
            Formatted enum representation as TypeScript union literals.
            When x-enum-descriptions or x-enum-aliases exist, appends inline comment.
        """
        enum_list = enum_value.get("enum", [])
        if not enum_list:
            return "string"

        # Rendered through format_literal_value: bool before int (bool is an int
        # subclass), numbers/bools bare, strings JSON-quoted and escaped, None -> null.
        enum_literals = [format_literal_value(val) for val in enum_list]
        type_str = self.config.union_separator.join(enum_literals)
        descs, alias_map = self._extract_enum_metadata(enum_value)
        if not self._should_include_metadata("x-enum-descriptions"):
            descs = {}
        if not self._should_include_metadata("x-enum-aliases"):
            alias_map = {}
        if not descs and not alias_map:
            return type_str
        # Inline comment: OPTIONS with descriptions and per-value lines
        parts = ["OPTIONS with descriptions"]
        for val in enum_list:
            canonical = str(val) if not isinstance(val, bool) else ("true" if val else "false")
            if canonical in descs and descs[canonical]:
                part = f"{canonical}: {descs[canonical]}"
                if canonical in alias_map and alias_map[canonical]:
                    part += f" (aliases: {', '.join(alias_map[canonical])})"
            elif canonical in alias_map and alias_map[canonical]:
                part = f"{canonical} (aliases: {', '.join(alias_map[canonical])})"
            else:
                continue
            parts.append(part)
        if len(parts) > 1:
            type_str += "  // " + "; ".join(parts)
        return type_str

    def process_const(self, const_value: dict[str, Any]) -> str:
        """
        Process a const field (single literal value) for TypeScript.

        Args:
            const_value: Dictionary containing const definition.

        Returns:
            Formatted const representation as a TypeScript literal, rendered through
            ``format_literal_value`` -- JSON-quoted and escaped for a ``str``, bare for
            numbers/bools, ``"null"`` for ``None``.
        """
        return format_literal_value(const_value.get("const"))

    def process_type_value(self, type_value: dict[str, Any]) -> str:
        """
        Process a type field for TypeScript.

        Args:
            type_value: Dictionary containing type definition.

        Returns:
            Formatted type representation.
        """
        type_name = type_value.get("type", "any")

        # Handle array of types (union types like ["string", "null"])
        if isinstance(type_name, list):
            if len(type_name) == 1:
                type_name = type_name[0]
            elif "null" in type_name and len(type_name) == 2:
                # Handle nullable types like ["string", "null"] -> "string | null"
                non_null_type = next(t for t in type_name if t != "null")
                type_str = self.TYPE_MAP.get(non_null_type, non_null_type)
                return f"{type_str} | null"
            else:
                # Multiple non-null types - treat as union
                type_strs = [self.TYPE_MAP.get(t, t) for t in type_name if t != "null"]
                return self.config.union_separator.join(s for s in type_strs if s is not None)

        # Now type_name is guaranteed to be a string
        type_str = self.TYPE_MAP.get(type_name, type_name)

        shape = classify_container(type_value)
        if shape.kind == "mapping":
            return self.render_mapping(shape)
        if shape.kind == "tuple":
            return self.render_tuple_token(shape, type_value)

        # Validation constraints below are already gated by the producers themselves
        # (each bound is resolved independently — no `or` over the pair); this method
        # must not re-gate.
        if type_name == "string":
            length_range = self.length_range_token(type_value)
            if length_range:
                type_str = f"{type_str} ({length_range})"
        elif type_name in ["number", "integer"]:
            range_info = self.numeric_range_token(type_value)
            if range_info:
                type_str = f"{type_str} ({range_info})"

        # Handle array type (consolidate both "Array" and "array" cases)
        if type_str == "Array" or type_name == "array":
            items = type_value.get("items")
            if not items:
                array_type = "Array<any>"
            elif isinstance(items, bool):
                array_type = "Array<any>" if items else "Array<any>"
            elif isinstance(items, dict):
                # Handle object items with properties - expand inline
                if "properties" in items:
                    processed_properties = self.process_properties(items["properties"])
                    props_str = ", ".join(f"{k}: {v}" for k, v in processed_properties.items())
                    array_type = f"Array<{{ {props_str} }}>"
                # Handle type in items
                elif "type" in items:
                    items_type = self.process_type_value(items)
                    array_type = f"Array<{items_type}>"
                # Handle $ref in items
                elif "$ref" in items:
                    items_type = self.process_ref(items)
                    array_type = f"Array<{items_type}>"
                # Handle anyOf in items
                elif "anyOf" in items:
                    items_type = self.process_anyof(items)
                    array_type = f"Array<{items_type}>"
                # Handle allOf in items
                elif "allOf" in items:
                    items_type = self.process_allof(items)
                    array_type = f"Array<{items_type}>"
                # Handle oneOf in items
                elif "oneOf" in items:
                    items_type = self.process_oneof(items)
                    array_type = f"Array<{items_type}>"
                else:
                    array_type = "Array<any>"
            else:
                array_type = "Array<any>"

            if self.include_metadata:
                array_type += self.format_array_constraints(type_value)
                if "contains" in type_value:
                    array_type += self.process_contains(type_value)
                # process_unique_items is no longer called here (it is what commented out
                # the terminating ';').

            return array_type

        return str(type_str)

    def recursion_placeholder(self, type_name: str) -> str:
        """Block-comment form: a `//` inside an inline object literal swallows the line."""
        return f"object /* recursive: {type_name} */"

    @staticmethod
    def _member_line(label: str, token: str) -> str:
        """One interface member line, with ``;`` bound to the type, never after a ``//``.

        ``label`` is the fully-formed left side -- ``name*``/``name`` from
        ``format_field_name`` or the ``$defs`` loops, or the literal ``"[key: string]"``.
        ``token`` is whatever ``process_property``/``render_type_token`` produced, which
        ``add_metadata`` may already have suffixed with ``"  // ..."`` plus continuation
        comment lines. The ``;`` lands immediately after the type; the comment block is
        copied through verbatim, so continuation lines stay pure comments.
        """
        # ponytail: textual split on the "  // " add_metadata inserts, same convention as
        # _inline_comment. A string literal containing "  // " mis-splits; carry the comment
        # out of band (base's defer_comment machinery) if that ever appears in a fixture.
        type_part, sep, comment = token.partition("  // ")
        if not sep:
            return f"  {label}: {token};\n"
        return f"  {label}: {type_part};{sep}{comment}\n"

    _EMBEDDED_COMMENT_BREAK: Final[re.Pattern[str]] = re.compile(r"\s*\n\s*//\s*")

    @staticmethod
    def _inline_comment(value: Any) -> str:
        """Rewrite a trailing ``// ...`` comment as a block comment.

        A ``//`` comment inside a single-line inline object literal would swallow the
        remainder of the line, including the closing brace and every later field. When
        ``add_metadata`` emitted continuation lines for a multi-line description, those
        lines carry their own ``\\n  // `` breaks; folding this literal back into one
        physical line must strip those markers too, or a ``//`` leaks inside the
        ``/* ... */`` wrapper and reads as a nested comment.
        """
        text = str(value)
        head, sep, tail = text.partition("  // ")
        if not sep:
            return text
        # An inline literal is one physical line: fold the continuation comment lines
        # add_metadata produced back in, stripping their "//" so no marker leaks inside
        # the /* ... */ wrapper. Runs BEFORE the "*/" escape below.
        tail = TypeScriptFormatter._EMBEDDED_COMMENT_BREAK.sub(" ", tail)
        safe_tail = tail.replace("*/", "* /")
        return f"{head} /* {safe_tail} */"

    def render_mapping(self, shape: ContainerShape) -> str:
        """One-line ``Record<K, V>`` token for a MAPPING shape."""
        if shape.value_schema is not None:
            value_token = self.render_type_token(shape.value_schema)
        else:
            value_token = "any"
        value_token = self._inline_comment(value_token)
        return f"Record<{self.key_token(shape)}, {value_token}>"

    def render_tuple(self, shape: ContainerShape) -> str:
        """One-line ``[A, B]`` / ``[A, B, ...C[]]`` token for a TUPLE shape."""
        tokens = [self.render_type_token(s) for s in shape.prefix_schemas]
        if shape.rest_schema is not None:
            tokens.append(f"...{self.render_type_token(shape.rest_schema)}[]")
        return "[" + ", ".join(tokens) + "]"

    def dict_to_string(self, value: Any, indent: int = 1) -> str:
        """
        Convert a dictionary or list to a formatted string representation.

        This is used for processed properties dict or nested objects.

        Args:
            value: The value to convert (dict, list, or primitive).
            indent: Current indentation level.

        Returns:
            Formatted string representation.
        """
        if isinstance(value, dict):
            if not value:  # Empty dict
                return "{}"

            # Format as a TypeScript inline object literal, matching the convention already
            # used for anyOf object members and inline array items.
            pairs = [f"{k}: {self._inline_comment(v)}" for k, v in value.items()]
            return "{ " + ", ".join(pairs) + " }"
        elif isinstance(value, list):
            return "[" + ", ".join(str(v) for v in value) + "]"
        else:
            return str(value)

    def transform_schema(self) -> str:
        """
        Transform schema into TypeScript interface syntax.

        This method implements a two-branch structure:
        1. If no properties, handle schema-level-only cases
        2. Main flow: process properties and return

        Returns:
            TypeScript interface definition as a string.
        """
        self._reset_ref_state()
        # Root-level $ref: adopt the resolved def as the effective root and count it as
        # the first expansion of that type, so the depth knob means the same thing for
        # ``Node`` and for ``Root(root: Node)``.
        root_ref_key = self._adopt_root_ref()
        if root_ref_key is not None:
            self._root_ref_key = root_ref_key
        # First branch: no properties - handle schema-level-only cases
        if not self.properties:
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
                # For empty object type, check for complex additionalProperties first
                if self.schema.get("type") == "object":
                    if self._is_complex_additional_props(self.schema):
                        # Build interface with placeholder
                        result = "interface Schema {\n"
                        shape = classify_container(self.schema)
                        value_token = (
                            self.render_type_token(shape.value_schema)
                            if shape.value_schema is not None
                            else "any"
                        )
                        result += self._member_line("[key: string]", value_token)
                        result += "}"

                        # Add comment
                        additional_comment = self.process_additional_properties(
                            self.schema, show_structure=False
                        )
                        if additional_comment and self.include_metadata:
                            result += f"\n{additional_comment}"
                        return result
                    if not schema_level_features:
                        return "interface Schema {}"
                type_content = self.process_type_value(self.schema)
                result = f"type Schema = {type_content};"
                # Add schema-level features as comments if present
                if schema_level_features and (
                    self.include_metadata or self.emits_closed_world_marker(self.schema)
                ):
                    result = (
                        f"// Schema-level constraints: {schema_level_features.strip()}\n{result}"
                    )
                return result
            elif "oneOf" in self.schema:
                oneof_content = self.process_oneof(self.schema)
                result = f"type Schema = {oneof_content};"
                if schema_level_features and (
                    self.include_metadata or self.emits_closed_world_marker(self.schema)
                ):
                    result = (
                        f"// Schema-level constraints: {schema_level_features.strip()}\n{result}"
                    )
                return result
            elif "anyOf" in self.schema:
                anyof_content = self.process_anyof(self.schema)
                result = f"type Schema = {anyof_content};"
                if schema_level_features and (
                    self.include_metadata or self.emits_closed_world_marker(self.schema)
                ):
                    result = (
                        f"// Schema-level constraints: {schema_level_features.strip()}\n{result}"
                    )
                return result
            elif "allOf" in self.schema:
                allof_content = self.process_allof(self.schema)
                result = f"type Schema = {allof_content};"
                if schema_level_features and (
                    self.include_metadata or self.emits_closed_world_marker(self.schema)
                ):
                    result = (
                        f"// Schema-level constraints: {schema_level_features.strip()}\n{result}"
                    )
                return result
            else:
                # Return schema-level features as comments if present
                if schema_level_features and (
                    self.include_metadata or self.emits_closed_world_marker(self.schema)
                ):
                    return (
                        f"// Schema-level constraints: {schema_level_features.strip()}\n"
                        "interface Schema {}"
                    )
                return "interface Schema {}"

        # Process main interface
        main_output = StringIO()

        # Add schema info comment if present
        schema_info_comment = self.get_schema_info_comment()
        if schema_info_comment:
            main_output.write(f"{schema_info_comment}\n")

        # Add required fields comment if there are required fields
        required_comment = self.get_required_fields_comment()
        if required_comment:
            main_output.write(f"{required_comment}\n")

        main_output.write("interface Schema {\n")

        with self._expanding(self._root_ref_key):
            processed_properties = self.process_properties(self.properties)
        for name, prop_type in processed_properties.items():
            # process_properties() already includes metadata via process_property()
            # so we don't need to add it again
            main_output.write(self._member_line(name, prop_type))

        # Add placeholder for complex additionalProperties
        if self._is_complex_additional_props(self.schema):
            shape = classify_container(self.schema)
            value_token = (
                self.render_type_token(shape.value_schema)
                if shape.value_schema is not None
                else "any"
            )
            main_output.write(self._member_line("[key: string]", value_token))

        main_output.write("}")

        # Add additionalProperties comment if present and metadata is enabled
        if self.include_metadata or self.emits_closed_world_marker(self.schema):
            show_structure = not self._is_complex_additional_props(self.schema)
            additional_props_comment = self.process_additional_properties(
                self.schema, show_structure=show_structure
            )
            if additional_props_comment:
                main_output.write(f"\n{additional_props_comment}")

        # Written for introspection only; there is exactly one render path.
        self._processed_data = processed_properties

        return main_output.getvalue()
