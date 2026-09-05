"""YAML-style formatter for transforming Pydantic schemas.

This formatter creates a clean YAML-like representation with Python-style
type hints, optionally including metadata as inline comments.
Feature parity with JSONish formatter for metadata, enums, unions, types,
dependencies, and $ref default.
"""

from typing import Any

import yaml

from .base import BaseFormatter, ContainerShape, classify_container
from .config import FormatterConfig, with_format_default_separator


def _is_null_schema(member: Any) -> bool:
    """True for the ``{"type": "null"}`` half of a Pydantic ``X | None`` union.

    Also accepts the draft-7 spelling where ``type`` is a list containing ``"null"``.
    """
    if not isinstance(member, dict):
        return False
    type_name = member.get("type")
    return type_name == "null" or (isinstance(type_name, list) and "null" in type_name)


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

    def __init__(
        self,
        schema: dict[str, Any],
        config: "FormatterConfig | None" = None,
        include_metadata: bool | None = None,
    ):
        """
        Initialize the YAML formatter.

        Args:
            schema: JSON schema from Pydantic model_json_schema.
            config: FormatterConfig for customizing formatter behavior.
            include_metadata: Deprecated. Use config.include_metadata instead.
        """
        config = with_format_default_separator(config, " OR ")

        super().__init__(schema, config, include_metadata)

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

    @property
    def deferred_comment_gap(self) -> str:
        """Two spaces before a hoisted ``#`` comment.

        Matches ``add_metadata``'s existing ``f"  # {...}"`` suffix convention.
        """
        return "  "

    def _resolve_mapping_value(self, value_schema: dict[str, Any]) -> dict[str, Any]:
        """Resolve a ``$ref`` mapping value schema to its ``$defs`` entry, else return it."""
        ref = value_schema.get("$ref")
        if isinstance(ref, str):
            ref_match = self.REF_PATTERN.search(ref)
            if ref_match:
                ref_def = self.defs.get(ref_match.group(1))
                if isinstance(ref_def, dict):
                    return ref_def
        return value_schema

    def _mapping_value_is_structural(self, value_schema: dict[str, Any]) -> bool:
        """True when a MAPPING's value schema needs a nested YAML block, not a one-line token.

        True for: a schema with its own `properties`; a `$ref` whose resolved def has
        `properties`; or a schema carrying `anyOf`/`oneOf`/`allOf`/a nested (dict-valued or
        `True`) `additionalProperties`. False for a plain scalar/enum/`Any` value schema.
        """
        if not isinstance(value_schema, dict) or not value_schema:
            return False

        resolved = self._resolve_mapping_value(value_schema)
        if resolved.get("properties"):
            return True

        for key in ("anyOf", "oneOf", "allOf"):
            if value_schema.get(key):
                return True

        nested_additional = value_schema.get("additionalProperties")
        if isinstance(nested_additional, dict) and nested_additional:
            return True
        return nested_additional is True

    def _mapping_ref_key(self, value_schema: dict[str, Any]) -> str | None:
        """Def name a MAPPING value ``$ref``s, or None."""
        ref = value_schema.get("$ref")
        if isinstance(ref, str):
            ref_match = self.REF_PATTERN.search(ref)
            if ref_match and ref_match.group(1) in self.defs:
                return ref_match.group(1)
        return None

    def _mapping_value_pairs(self, value_schema: dict[str, Any]) -> dict[str, str] | None:
        """Marked ``name -> rendered type`` pairs of a structural mapping value, if it has any.

        This path resolves a ``$ref`` directly rather than through ``process_ref``, so it
        applies ``max_recursion_depth`` itself; returning None on re-entry hands the value
        back to ``process_property``, which emits the recursion placeholder.
        """
        ref_key = self._mapping_ref_key(value_schema)
        if ref_key is not None:
            reentries = self._ref_expansion_path.count(ref_key)
            if reentries >= 1 and reentries >= self.config.max_recursion_depth:
                self._truncation_epoch += 1
                return None

        resolved = self._resolve_mapping_value(value_schema)
        properties = resolved.get("properties")
        if not isinstance(properties, dict) or not properties:
            return None

        required = set(resolved.get("required", []) or [])
        pairs: dict[str, str] = {}
        with self._expanding(ref_key):
            for prop_name, prop_def in properties.items():
                if prop_name in required:
                    marked = f"{prop_name}{self.config.required_marker}"
                else:
                    marked = f"{prop_name}{self.config.optional_marker}"
                pairs[marked] = self.process_property(prop_def)
        return pairs

    def _build_mapping_block(self, value_schema: dict[str, Any]) -> dict[str, Any] | str:
        """Build the nested dict `yaml.dump` will render for a *structural* mapping value.

        Resolves a `$ref` value schema to its def's `properties` + `required` directly (rather
        than embedding `process_ref`'s multi-line `dict_to_string` scalar) so a model-valued
        mapping renders as a real nested block, not a quoted multi-line string.
        """
        pairs = self._mapping_value_pairs(value_schema)
        if pairs is not None:
            return dict(pairs)
        return self.process_property(value_schema)

    def _dump_yaml(self, data: dict[str, Any]) -> str:
        """
        Dump a dictionary to YAML format, then resolve deferred comment markers.

        The hoist MUST run *after* ``yaml.dump``: while a scalar still carries its opaque
        marker it contains no ``": "`` and no ``" #"``, so PyYAML emits it as a bare plain
        scalar on one physical line. Substituting the comment text before the dump would
        reintroduce both and PyYAML would re-quote (and possibly fold) the line.

        Args:
            data: Dictionary to serialize to YAML.

        Returns:
            YAML string representation with every deferred marker replaced by a real
            trailing ``#`` comment.
        """
        result = yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        return self.hoist_deferred_comments(str(result)).rstrip()

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
            if not item:
                item_types.append("any")
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

        return self.config.union_separator.join(item_types) if item_types else "string"

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
            if not item:
                item_types.append("any")
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
            return "ONE OF: " + self.config.union_separator.join(item_types)
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
                # No re-wrap: ``dict_to_string`` already returns a brace flow literal, so
                # wrapping it again would double-brace AND re-embed a newline.
                item_types.append(self.dict_to_string(processed_props, indent=2))
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
        # Check minLength/maxLength based on metadata_inclusion config
        include_min_length = self._should_include_metadata("minLength")
        include_max_length = self._should_include_metadata("maxLength")
        min_len = type_value.get("minLength")
        max_len = type_value.get("maxLength")
        if (
            include_min_length
            and include_max_length
            and min_len is not None
            and max_len is not None
        ):
            parts.append(f"({min_len}-{max_len} chars)")
        elif include_min_length and min_len is not None:
            parts.append(f"(>= {min_len} chars)")
        elif include_max_length and max_len is not None:
            parts.append(f"(<= {max_len} chars)")
        # Check pattern based on metadata_inclusion config
        if type_value.get("pattern") and self._should_include_metadata("pattern"):
            parts.append(f"(PATTERN: {type_value['pattern']})")
        # Check format based on metadata_inclusion config
        if self._should_include_metadata("format"):
            if type_value.get("format"):
                parts.append(f"(FORMAT: {type_value['format']})")
            elif type_value.get("_format"):
                parts.append(f"(FORMAT: {type_value['_format']})")
        return " ".join(parts)

    def _format_number_range_jsonish(self, type_value: dict[str, Any]) -> str:
        """Format number range like JSONish: (min to max), (>= min), (<= max)."""
        include_min = self._should_include_metadata("minimum")
        include_max = self._should_include_metadata("maximum")
        min_val = type_value.get("minimum")
        max_val = type_value.get("maximum")
        if include_min and include_max and min_val is not None and max_val is not None:
            return f"({min_val} to {max_val})"
        if include_min and min_val is not None:
            return f"(>= {min_val})"
        if include_max and max_val is not None:
            return f"(<= {max_val})"
        return ""

    def process_type_value(self, type_value: dict[str, Any]) -> str:
        """
        Process type with JSONish constraint phrasing: string (min-max chars), (PATTERN: ...),
        number (min to max), array UNIQUE / (min-max UNIQUE items), object {} or expanded.
        """
        type_name = type_value.get("type", "any")

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
                return self.config.union_separator.join(s for s in type_strs if s is not None)

        type_str = self.TYPE_MAP.get(type_name, type_name)

        shape = classify_container(type_value)
        if shape.kind == "mapping":
            return self.render_mapping(shape)
        if shape.kind == "tuple":
            tuple_str = self.render_tuple(shape)
            if not (
                type_value.get("minItems")
                == type_value.get("maxItems")
                == len(shape.prefix_schemas)
            ):
                tuple_str += self.format_array_constraints(type_value)
            return tuple_str

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
                type_str = "list[any]"
            elif isinstance(items, bool):
                type_str = "list[any]"
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
                type_str = "list[any]"

            if self.include_metadata:
                type_str += self.format_array_constraints(type_value)
                if "contains" in type_value:
                    type_str += self.process_contains(type_value)
            # ``str(...)`` mirrors the fallthrough return below: ``type_str`` starts life as
            # ``TYPE_MAP.get(type_name, type_name)`` with an ``Any``-typed key, and the
            # ``add_metadata`` call that used to launder it here is gone (D9).
            return str(type_str)
        elif type_name == "object":
            if (
                "properties" not in type_value
                and "patternProperties" not in type_value
                and not isinstance(type_value.get("additionalProperties"), dict)
            ):
                return "{}"
            return "object"

        return self.add_metadata(str(type_str), type_value)

    def render_mapping(self, shape: ContainerShape) -> str:
        """``"dict[K, V]"`` in every string context (correction C3); never a nested block."""
        value_token = (
            self.render_type_token(shape.value_schema) if shape.value_schema is not None else "any"
        )
        if shape.value_schema is not None and self._mapping_value_is_structural(shape.value_schema):
            value_token = self._flow_mapping_value(shape.value_schema)
        return f"dict[{self.key_token(shape)}, {value_token}]"

    def render_tuple(self, shape: ContainerShape) -> str:
        """``"tuple[int, string]"`` / ``"tuple[int, string, ...string]"``."""
        tokens = [self.render_type_token(s) for s in shape.prefix_schemas]
        if shape.rest_schema is not None:
            tokens.append(f"...{self.render_type_token(shape.rest_schema)}")
        return f"tuple[{', '.join(tokens)}]"

    def _flow_mapping_value(self, value_schema: dict[str, Any]) -> str:
        """``"{a: int, b: string}"`` -- a YAML flow-mapping rendering of a structural value
        schema, for use inside a larger one-line string value.
        """
        pairs = self._mapping_value_pairs(value_schema)
        if pairs is None:
            return self.process_property(value_schema)
        return "{" + ", ".join(f"{k}: {v}" for k, v in pairs.items()) + "}"

    # ------------------------------------------------------------------
    # Nested-block builder (lsl-2026-09-04-006, Axis 1)
    #
    # An object-shaped property becomes a real ``dict``/``list`` that ``yaml.dump``
    # renders as an indented mapping, instead of being stringified into a quoted
    # multi-line scalar and duplicated into a hoisted ``Class.field`` section.
    # ------------------------------------------------------------------

    def _object_ref_key(self, schema: Any) -> str | None:
        """The ``$defs`` key of ``schema["$ref"]`` iff that def has non-empty ``properties``.

        Returns ``None`` for anything else: not a dict, no ``$ref``, an unresolved ref, or a
        ``$defs`` entry that isn't an object-with-properties (enum, oneOf, scalar def, ...).
        Mirrors ``base.process_ref``'s own def-type dispatch (``if "properties" in ref_def
        and ref_def["properties"]``) so the two never disagree about which ``$ref`` targets
        are block-eligible.
        """
        if not isinstance(schema, dict):
            return None
        ref = schema.get("$ref")
        if not isinstance(ref, str):
            return None
        ref_match = self.REF_PATTERN.search(ref)
        if not ref_match:
            return None
        ref_key = ref_match.group(1)
        ref_def = self.defs.get(ref_key)
        if isinstance(ref_def, dict) and ref_def.get("properties"):
            return ref_key
        return None

    def _closed_world_note(self, schema: dict[str, Any]) -> str:
        """R1: the def's ``no additional properties`` / ``additional: <type>`` fragment, bare.

        The closed-world marker is STRUCTURAL, so ``emits_closed_world_marker`` bypasses the
        metadata gates -- this reproduces the gate the deleted ``$defs`` loops used, verbatim.
        Rendering is delegated to the untouched ``process_additional_properties``; only its
        leading comment prefix is stripped so the fragment can share a key's comment slot.
        """
        if not (self.include_metadata or self.emits_closed_world_marker(schema)):
            return ""
        text = self.process_additional_properties(schema)
        if not text:
            return ""
        text = text.strip()
        if text.startswith(self.comment_prefix):
            text = text[len(self.comment_prefix) :]
        return text.strip()

    def _properties_block(
        self, def_schema: dict[str, Any], ref_key: str | None = None
    ) -> dict[str, Any] | None:
        """The nested ``dict`` for an object node (a resolved ``$ref`` or an inline object).

        When ``ref_key`` is given (the ``$ref`` case) this enforces the full 014 recursion
        contract and returns ``None`` to hand the property back to the string path
        (``process_ref`` -> ``recursion_placeholder``) when truncation applies. When
        ``ref_key`` is ``None`` (an inline ``type: object`` + ``properties``, with no ``$ref``
        to re-enter) it renders unconditionally.

        Deliberately never reads or writes ``self._ref_cache``: caching a real ``dict`` would
        let PyYAML emit ``&id001``/``*id001`` aliases for a def used by two properties.
        """
        if ref_key is not None:
            # base.process_ref's own order: the unconditional global-budget guard, then the
            # same-type re-entry guard, and only then the budget increment -- so a truncated
            # re-entry never consumes a budget slot.
            if self._global_expansion_count >= self._global_expansion_budget:
                return None
            reentries = self._ref_expansion_path.count(ref_key)
            if reentries >= 1 and reentries >= self.config.max_recursion_depth:
                self._truncation_epoch += 1
                return None
            self._global_expansion_count += 1
        with self._expanding(ref_key):
            # The def owns its own ``required`` list; ``format_field_name`` reads the top of
            # this stack so nested markers come from the DEF, not from the ROOT (D3). The
            # ``finally`` is load-bearing: a leaked push inverts every later root-level marker.
            self._nested_required_stack.append(set(def_schema.get("required", [])))
            try:
                return self.process_properties(def_schema["properties"])
            finally:
                self._nested_required_stack.pop()

    def _structural_block(
        self, value: dict[str, Any]
    ) -> tuple[dict[str, Any] | list[Any] | None, list[str]]:
        """``(container, structural_notes)`` for the block-eligible shapes, else ``(None, [])``.

        ``structural_notes`` is pre-ordered ``OR null`` first, then the closed-world note; the
        caller (``_compose_key_note``) appends metadata after those. A shape not on the
        eligibility table below intentionally falls through to the pre-existing string path in
        ``process_properties``, which is unconditionally safe -- it is today's behaviour.
        """
        # Rule 1 -- a ``$ref`` to an object def.
        ref_key = self._object_ref_key(value)
        if ref_key is not None:
            block = self._properties_block(self.defs[ref_key], ref_key)
            if block is None:
                return None, []  # 014-truncated: the string path renders the placeholder
            note = self._closed_world_note(self.defs[ref_key])
            return block, [note] if note else []

        # Rule 2 -- a two-member ``X | None`` union whose non-null half is block-eligible.
        for composition_key in ("anyOf", "oneOf"):
            members = value.get(composition_key)
            if not isinstance(members, list) or len(members) != 2:
                continue
            others = [m for m in members if not _is_null_schema(m)]
            if len(others) != 1 or not isinstance(others[0], dict):
                return None, []
            inner, inner_notes = self._structural_block(others[0])
            if inner is None:
                return None, []
            null_note = f"{self.config.union_separator.strip()} null"
            return inner, [null_note, *inner_notes]

        # Rule 3 -- an array of block-eligible items. A nullable ITEM is declined: a YAML
        # sequence has no key to hang the item's own ``OR null`` note on.
        if value.get("type") == "array":
            items = value.get("items")
            if isinstance(items, dict):
                inner, inner_notes = self._structural_block(items)
                if inner is not None and not any(n.endswith(" null") for n in inner_notes):
                    return [inner], inner_notes
            return None, []

        # Rule 4 -- an inline ``type: object`` + ``properties``, with no ``$ref`` (raw JSON
        # Schema only; Pydantic always emits a ``$ref``). ``classify_container`` is used in
        # place of a hand-rolled check so composition keys and list/tuple shapes decline on
        # their own, and so the ``patternProperties``-as-object-marker subtleties stay in the
        # one place that already knows about them.
        if classify_container(value).kind == "object" and value.get("properties"):
            block = self._properties_block(value, ref_key=None)
            if block is None:  # pragma: no cover - unreachable without a ref_key
                return None, []
            note = self._closed_world_note(value)
            return block, [note] if note else []

        # Rule 5 -- not block-eligible.
        return None, []

    @staticmethod
    def _token_owned_metadata_keys(value: dict[str, Any]) -> tuple[str, ...]:
        """Metadata keys this formatter's TYPE TOKEN already states for ``value``.

        ``process_type_value`` embeds ``(PATTERN: ...)``/``(FORMAT: ...)`` into the token
        only on the ``type_name == "string"`` arm, so those keys must be excluded from the
        trailing comment for exactly that node shape -- and only that one. A ``pattern`` on
        a node without ``"type": "string"`` is never in the token, so it must still reach
        the comment or the constraint disappears from the prompt entirely.

        Length and range keys are NOT listed here: ``format_metadata_parts`` already skips
        those internally (base.py, the "integrated into the type description" arms), and
        that skip is correct for TypeScript too. ``pattern``/``format`` cannot move into
        that function because TypeScript's token does not embed them.

        Args:
            value: The field/node definition being rendered.

        Returns:
            ``("pattern", "format")`` when ``value["type"] == "string"``, else ``()``.
        """
        return ("pattern", "format") if value.get("type") == "string" else ()

    def _metadata_parts(self, value: dict[str, Any]) -> list[str]:
        """The comment fragments ``add_metadata`` would append for ``value``, in its order.

        The block-branch analogue of ``add_metadata``'s part-collection body, minus the
        ``representation``-specific bits (a block's "representation" is the ``dict``/``list``
        itself, which cannot carry an inline comment). Reuses the same single gate so the two
        can never drift apart.
        """
        if not self.include_metadata:
            return []

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

        # ``title``/``description``/``default`` are already supplied above; METADATA_MAP must
        # never re-supply them (D5: that is what restated ``(defaults to X)`` next to
        # ``(default=X)``).
        exclude = ("title", "description", "default") + self._token_owned_metadata_keys(value)
        available_metadata = self.get_available_metadata(value)
        if available_metadata:
            filtered_metadata = [m for m in available_metadata if m not in exclude]
            if filtered_metadata:
                parts.extend(self.format_metadata_parts(value, exclude=exclude))
        return parts

    def _compose_key_note(
        self,
        formatted_name: str,
        structural_notes: list[str],
        value: dict[str, Any],
        prop_name: str,
    ) -> str:
        """Join every fragment for one block-eligible property into ONE slot on its key.

        Order: structural notes (already ``OR null``-then-closed-world ordered), then the
        gated metadata parts, then the gated ``(DEPENDS ON: ...)``. Returns ``formatted_name``
        unchanged when there is nothing to say -- no empty ``defer_comment``, no stray marker.
        """
        fragments = list(structural_notes)
        fragments.extend(self._metadata_parts(value))

        dep = self._get_fields_dependencies(self.schema, prop_name)
        if dep and self.include_metadata:
            # ``_get_fields_dependencies`` already returns a fully parenthesised
            # "(DEPENDS ON: a, b)" -- append it verbatim or it double-wraps. Gated on the
            # master switch only, matching the plain branch in ``process_properties``.
            fragments.append(dep)

        if not fragments:
            return formatted_name
        return formatted_name + self.defer_comment("; ".join(fragments))

    def format_field_name(self, field_name: str) -> str:
        """Mark requiredness against the innermost ``$defs`` ``required``, not the ROOT one.

        The stack is pushed by ``_properties_block`` and by ``base.process_ref``; YAML simply
        never read it before, so a nested field inherited the root model's required set (D3).
        An empty stack means we are at the root, where ``super()`` is already correct.
        """
        if self._nested_required_stack:
            required = self._nested_required_stack[-1]
            if field_name in required:
                return f"{field_name}{self.config.required_marker}"
            return f"{field_name}{self.config.optional_marker}"
        return super().format_field_name(field_name)

    def recursion_placeholder(self, type_name: str) -> str:
        """Defer the ``recursive: X`` note so a later ``OR null``/metadata joins the same slot.

        Hoists back to base's literal ``"object  # recursive: X"`` byte for byte, but until
        the hoist the scalar carries no ``#``, so PyYAML leaves it unquoted and an enclosing
        nullable union or field default folds into the one slot instead of trailing a second
        comment behind a quoted string.
        """
        return f"object{self.defer_comment(f'recursive: {type_name}')}"

    def add_metadata(self, representation: str, value: dict[str, Any]) -> str:
        """
        Add metadata comments to a field representation (JSONish parity:
        title, description, id, $comment, default, example/examples).

        When ``representation`` carries a deferred-comment marker the metadata is routed
        *into* that marker's slot rather than appended as a ``"  # ..."`` suffix. That is
        the plain-scalar invariant: appending ``#`` (or ``": "``) to a marker-bearing
        scalar makes PyYAML quote the whole line.
        """
        if not self.include_metadata:
            return representation

        # Marker presence is the precise replacement for the old
        # ``"enum" in value and "#" in representation`` substring hack, which never fired
        # for a ``$ref``'d enum (the ``enum`` key lives in the ``$defs`` node, not in the
        # property schema) and so restated the default several times on one line.
        deferred = self.carries_deferred_comment(representation)

        title, description, default_value, example = self._get_title_description_default_value(
            value
        )
        parts = []
        if title and not deferred:
            # D4: never emit the enum's Pydantic class name alongside a hoisted comment.
            parts.append(title.strip())
        if description:
            parts.append(description.strip())
        if default_value:
            parts.append(default_value.strip())
        if example:
            parts.append(example.strip())

        # Base METADATA_MAP-style parts for pattern, format, etc. (when not in type).
        # ``title``/``description`` are already supplied above by
        # ``_get_title_description_default_value``; METADATA_MAP must never re-supply them.
        # ``const`` is excluded as well for a marker-bearing representation, whose slot body
        # already renders that value as ``one of: ...``.
        # ``default`` is excluded in BOTH arms (D5): ``default_value`` was already appended to
        # ``parts`` above, and METADATA_MAP would otherwise restate it as ``(defaults to X)``
        # right beside the ``(default=X)`` that is already there. ``(default=X)`` survives.
        exclude = (
            ("title", "description", "default", "const")
            if deferred
            else ("title", "description", "default")
        ) + self._token_owned_metadata_keys(value)
        available_metadata = self.get_available_metadata(value)
        if available_metadata:
            filtered_metadata = [m for m in available_metadata if m not in exclude]
            if filtered_metadata:
                parts.extend(self.format_metadata_parts(value, exclude=exclude))

        if not parts:
            return representation

        if deferred:
            # Drop anything the slot body (or the representation itself) already states,
            # then fold the survivors into the slot instead of appending a comment.
            # QUERY, not render: the hoist rewrites a multi-line body into several comment
            # lines, so a rendered resolve would no longer contain a multi-line part
            # verbatim and the description would be folded in twice.
            resolved = self.resolved_deferred_text(representation)
            survivors = [part for part in parts if part not in resolved]
            if not survivors:
                return representation
            return self.append_deferred_comment(str(representation), "; ".join(survivors))

        suffix = f"  # {', '.join(parts)}"
        if isinstance(representation, str):
            # Half A: kept deliberately. It is provably unreachable now that the branch below
            # defers (``defer_comment`` always returns a fresh opaque marker token, never the
            # literal suffix text), but it is two lines of idempotence insurance that a future
            # cleanup pass should not remove without re-reading this note.
            if representation.endswith(suffix):
                return representation
            # A plain scalar must never gain a literal ``#`` here: PyYAML would quote the whole
            # line, and a later structural note on the same key could not join it. Defer.
            return representation + self.defer_comment(", ".join(parts))
        # Half B: a non-str representation (the ``const`` case passes a raw ``int``) cannot
        # carry a marker token, so it keeps the literal suffix.
        return f"{representation}{suffix}"

    def process_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """
        Process properties and append per-field (DEPENDS ON: ...) when present (JSONish parity).
        For object properties with complex additionalProperties and no fixed properties,
        emit a dict with <key> placeholder.
        """
        # D7: this method is now RE-ENTERED recursively (via ``_properties_block``), so the
        # reset below would clobber whatever the outer call had accumulated. Save the outer
        # count and ADD it back at the end -- an accumulating save/restore, not a plain one,
        # because the trailer has to explain every ``<...>`` placeholder in the whole document,
        # including the ones that occur inside a nested block.
        outer_placeholder_count = getattr(self, "_nested_placeholder_count", 0)
        self._nested_placeholder_count = 0
        processed_properties: dict[str, Any] = {}
        for prop_name, value in properties.items():
            formatted_name = self.format_field_name(prop_name)
            if isinstance(value, dict):
                # Axis 1: an object-shaped property becomes a real dict/list that yaml.dump
                # indents, with every structural and metadata note collapsed into ONE slot on
                # its key. Disjoint from the mapping branch below by construction (a schema
                # declaring ``properties`` classifies as "object", never "mapping"), so trying
                # it first cannot steal a case the mapping branch would have handled.
                block, structural_notes = self._structural_block(value)
                if block is not None:
                    key = self._compose_key_note(formatted_name, structural_notes, value, prop_name)
                    processed_properties[key] = block
                    continue
                shape = classify_container(value)
                if shape.kind == "mapping":
                    key = f"<{self.key_token(shape)}>"
                    if shape.value_schema is not None and self._mapping_value_is_structural(
                        shape.value_schema
                    ):
                        processed_properties[formatted_name] = {
                            key: self._build_mapping_block(shape.value_schema)
                        }
                        self._nested_placeholder_count += 1
                    else:
                        processed_properties[formatted_name] = self.render_mapping(shape)
                    continue
            prop_str = self.process_property(value)
            dep = self._get_fields_dependencies(self.schema, prop_name)
            if dep and self.include_metadata:
                if self.carries_deferred_comment(prop_str):
                    prop_str = self.append_deferred_comment(prop_str, dep)
                else:
                    prop_str = f"{prop_str}  # {dep}"
            processed_properties[formatted_name] = prop_str
        self._nested_placeholder_count += outer_placeholder_count
        return processed_properties

    def process_ref(self, ref: dict[str, Any]) -> str:
        """
        Process $ref and append (default=...) when ref has default (JSONish parity).
        """
        result = super().process_ref(ref)
        if "default" in ref and result and self._should_include_metadata("default"):
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
        if (
            "title" in self.schema
            and self.schema["title"]
            and self._should_include_metadata("title")
        ):
            comments.append(f"Title: {self.schema['title']}")
        if (
            "description" in self.schema
            and self.schema["description"]
            and self._should_include_metadata("description")
        ):
            comments.append(f"Description: {self.schema['description']}")
        if comments:
            # D8: a multi-line ``description`` puts real newlines inside ``body``. Prefixing
            # only the joined string leaves every line after the first as a bare, uncommented
            # document line, which is a ``yaml.safe_load`` ScannerError. Prefix every line;
            # a blank one becomes a bare ``#`` with no dangling space.
            body = ", ".join(comments)
            return "\n".join(self.comment_lines(body)) + "\n"
        return ""

    def get_required_fields_comment(self) -> str:
        """Required fields comment with configurable marker."""
        if not self.include_metadata:
            return ""
        if not self.schema.get("required", None):
            return ""
        marker = self.config.required_marker
        return f"{self.comment_prefix} Fields marked with {marker} are required\n"

    def process_additional_properties(
        self, schema: dict[str, Any], show_structure: bool = True
    ) -> str:
        """Emit additionalProperties with # prefix for YAML (JSONish semantics)."""
        additional_props = schema.get("additionalProperties")
        if additional_props is False:
            return f" {self.comment_prefix} no additional properties"
        if isinstance(additional_props, dict) and additional_props:
            if not schema.get("properties"):
                # Pure mapping (classify_container rule 6/C1): the value type is rendered
                # structurally by the caller's mapping renderer, never as a comment.
                return ""
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
            indent: Recursion depth, threaded through the list branch's nested-dict
                recursion. The dict branch is a single-line flow literal and no longer
                indentation-sensitive.

        Returns:
            Formatted string representation as YAML key: value lines.
        """
        if isinstance(value, dict):
            if not value:  # Empty dict
                return "{}"

            # Format as a YAML flow-mapping literal: the caller embeds this string inside a
            # larger scalar (a 3-way union member, an allOf/oneOf branch, or a $ref whose
            # target is not block-eligible), so it must never introduce a literal newline --
            # PyYAML has no way to fold a multi-line single-quoted scalar back out.
            # ``TypeScriptFormatter.dict_to_string`` already does exactly this.
            return "{" + ", ".join(f"{k}: {v}" for k, v in value.items()) + "}"
        elif isinstance(value, list):
            # For arrays, show the item type
            if value and isinstance(value[0], dict):
                return f"list[{self.dict_to_string(value[0], indent + 1)}]"
            else:
                return "list"
        else:
            return str(value)

    def transform_schema(self) -> str:
        """Public entry point: run ``_transform_schema_impl``, then a final hoist sweep.

        Covers the four scalar return paths that never call ``_dump_yaml``. Safe to run
        over already-hoisted text: a completed pass leaves no token behind, and a token
        cannot be forged from schema text (the per-instance nonce).

        Returns:
            YAML-style schema definition as a string, with no deferred marker left.
        """
        return self.hoist_deferred_comments(self._transform_schema_impl())

    def _transform_schema_impl(self) -> str:
        """
        Transform schema into YAML-style format.

        Returns:
            YAML-style schema definition as a string.
        """
        self._reset_ref_state()
        # Root-level $ref: adopt the resolved def as the effective root and count it as
        # the first expansion of that type, so the depth knob means the same thing for
        # ``Node`` and for ``Root(root: Node)``.
        root_ref_key = self._adopt_root_ref()
        if root_ref_key is not None:
            self._root_ref_key = root_ref_key
        # First branch: if _processed_data is set, build from cache
        if hasattr(self, "_processed_data") and self._processed_data:
            # No $defs section loop: every def with ``properties`` is now rendered INLINE, as
            # a real nested block on the property that references it. The hoisted
            # ``Class.field`` sections this used to emit were a duplicate of that data.
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

            # Add additionalProperties comment (short if structure already in cached data).
            # The placeholder key is now the rendered key *type* (e.g. "<string>"), not the
            # literal "<key>", so match its shape rather than the old sentinel.
            has_placeholder_key = any(
                isinstance(k, str) and k.startswith("<") and k.endswith(">")
                for k in self._processed_data
            )
            if has_placeholder_key:
                additional_props_comment = self.process_additional_properties(
                    self.schema, show_structure=False
                )
            elif self.include_metadata or self.emits_closed_world_marker(self.schema):
                additional_props_comment = self.process_additional_properties(self.schema)
            else:
                additional_props_comment = ""
            if additional_props_comment:
                main_parts.append(additional_props_comment)

            return "\n".join(main_parts)

        # Second branch: no properties - handle schema-level-only cases
        if not self.properties:
            # Check for complex additionalProperties in empty object schemas
            if self.schema.get("type") == "object":
                shape = classify_container(self.schema)
                if shape.kind == "mapping":
                    key = f"<{self.key_token(shape)}>"
                    output_dict: dict[str, Any]
                    if shape.value_schema is not None and self._mapping_value_is_structural(
                        shape.value_schema
                    ):
                        output_dict = {key: self._build_mapping_block(shape.value_schema)}
                    else:
                        value_token = (
                            self.render_type_token(shape.value_schema)
                            if shape.value_schema is not None
                            else "any"
                        )
                        output_dict = {key: value_token}

                    result = self._dump_yaml(output_dict)
                    # "any properties allowed" is dropped for mappings (design 4.1 / A9) --
                    # no process_additional_properties(show_structure=False) call here.
                    return self._add_prefix(result)

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
            additional_props_feature = self.process_additional_properties(self.schema)
            if additional_props_feature:
                schema_level_features += additional_props_feature

            # Handle schema with type but no properties
            if "type" in self.schema:
                type_content = self.process_type_value(self.schema)
                # For object type with no properties, return {} instead of "object"/"dict"
                if type_content in ("object", "dict") and not schema_level_features:
                    return self._add_prefix("{}")
                # Add schema-level features as comments (e.g. additionalProperties)
                if schema_level_features:
                    return self._add_prefix(
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{type_content}"
                    )
                return self._add_prefix(type_content)
            elif "oneOf" in self.schema:
                oneof_content = self.process_oneof(self.schema)
                if schema_level_features and self.include_metadata:
                    return self._add_prefix(
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{oneof_content}"
                    )
                else:
                    return self._add_prefix(oneof_content)
            elif "anyOf" in self.schema:
                anyof_content = self.process_anyof(self.schema)
                if schema_level_features and self.include_metadata:
                    return self._add_prefix(
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{anyof_content}"
                    )
                else:
                    return self._add_prefix(anyof_content)
            elif "allOf" in self.schema:
                allof_content = self.process_allof(self.schema)
                if schema_level_features and self.include_metadata:
                    return self._add_prefix(
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{allof_content}"
                    )
                else:
                    return self._add_prefix(allof_content)
            if schema_level_features and self.include_metadata:
                constraints_comment = f"# Schema-level constraints: {schema_level_features.strip()}"
                return self._add_prefix(constraints_comment)
            else:
                return self._add_prefix("{}")

        # Third branch: main flow with properties.
        # No $defs section loop here either -- see the cached branch above.
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
        with self._expanding(self._root_ref_key):
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
            key_shape = ContainerShape("mapping", key_schema=self.schema.get("propertyNames"))
            key = f"<{self.key_token(key_shape)}>"
            if "properties" in additional_props and additional_props["properties"]:
                inner_required = set(additional_props.get("required", []))
                inner: dict[str, Any] = {}
                for prop_name, prop_def in additional_props["properties"].items():
                    # Use config markers for formatting
                    if prop_name in inner_required:
                        formatted_name = f"{prop_name}{self.config.required_marker}"
                    else:
                        formatted_name = f"{prop_name}{self.config.optional_marker}"
                    inner[formatted_name] = self.process_property(prop_def)
                processed_properties[key] = inner
            else:
                processed_properties[key] = self.process_property(additional_props)

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
        elif self.include_metadata or self.emits_closed_world_marker(self.schema):
            additional_props_comment = self.process_additional_properties(self.schema)
        else:
            additional_props_comment = ""
        if additional_props_comment:
            main_parts.append(additional_props_comment)

        # Set _processed_data for future calls (caching)
        self._processed_data = processed_properties

        result = "\n".join(main_parts)

        # Add prefix if configured
        if self.config.prefix:
            result = self.config.prefix + result

        return result

    def _add_prefix(self, output_string: str) -> str:
        """Add prefix to output if configured."""
        if self.config.prefix:
            return self.config.prefix + output_string
        return output_string
