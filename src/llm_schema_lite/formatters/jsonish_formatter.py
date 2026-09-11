"""JSONish formatter for transforming Pydantic schemas into BAML-like format."""

import copy
import json
import re
from typing import Any

from .base import DEFERRED_CLOSE, DEFERRED_OPEN, IDENTITY_TAG, BaseFormatter, classify_container
from .config import FormatterConfig, with_format_default_separator


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
        config = with_format_default_separator(config, " OR ")

        super().__init__(schema, config, include_metadata)
        # Trial-specific state. NOTE: the keys of pending_postfix / pending_recursion /
        # pending_prefix are "<marked property name><identity token>", NOT a bare property
        # name — do not match against these keys without the token (see _mint_identity).
        self.processed_ref_cache: dict[str, dict[str, Any] | str | list[Any]] = {}
        self.pending_postfix: dict[str, str] = {}
        self.pending_recursion: dict[str, str] = {}
        self.pending_prefix: dict[str, str] = {}
        self.pending_root_postfix: str = ""
        """Postfix owed by the ROOT itself, which has no property key to ride on.

        ``pending_postfix`` is a *keyed* side-channel resolved by ``_apply_pending_postfix``
        against a ``key:`` line. A root has no key line, so its postfix needs its own slot,
        appended directly to the serialized body in ``transform_schema``.
        """
        self.simplified_schema: str | None = None
        # Occurrence-identity tokens: disambiguate same-named properties at different
        # depths so the three pending_* maps above are exact-match and per-occurrence.
        self._identity_counter: int = 0  # monotonic per instance; deliberately never reset
        self._identity_pattern: re.Pattern[str] = re.compile(
            f"{DEFERRED_OPEN}{IDENTITY_TAG}{self._deferred_nonce}\\.\\d+{DEFERRED_CLOSE}"
        )

    TYPE_MAP = {"number": "float", "integer": "int", "boolean": "bool"}
    comment_prefix = "//"

    def sanitize_comment_text(self, text: str) -> str:
        """Collapse ``text`` to a single line for JSONish's end-of-line ``//`` comments.

        Every run of real whitespace -- spaces, tabs, ``\\n``, ``\\r`` -- becomes one space,
        and the ends are stripped. Without this, a real newline in a description either
        escapes into a literal two-character ``\\n`` (the dict-value path, pre-``json.dumps``)
        or breaks out of the comment as a bare, uncommented document line (the
        ``pending_postfix`` path, post-``json.dumps``). Per the ticket's 2026-09-05 decision
        a multi-line description collapses to ONE line joined by single spaces; this
        deliberately diverges from YAML/TypeScript (which split into one comment line per
        physical line) and matches TypeScript's inline-object-literal behaviour.

        Nothing is decoded: an authored two-character backslash-n stays two characters, an
        authored ``"`` stays one character, an authored ``//`` stays verbatim. Only real
        whitespace runs are touched.

        Args:
            text: Raw value read from the schema, before any comment prefix or
                concatenation.

        Returns:
            ``text`` with internal whitespace runs collapsed to one space and the ends
            stripped.
        """
        return " ".join(str(text).split())

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
            return f" {self.comment_prefix} {self.sanitize_comment_text(schema['description'])}"
        return ""

    def _defer_comment_body(self, head: str, tail: str, has_description: bool) -> str:
        """Route the trailing comment body of a scalar representation into the deferred slot.

        Called only on the string-returning branches of ``process_types`` /
        ``process_ref`` / ``process_anyof`` / ``process_oneof`` / ``process_allof`` whose
        result becomes a **dict value** -- and is therefore about to meet ``json.dumps``,
        the single place JSONish introduces escape sequences. A deferred marker contains
        no ``"`` and no ``\\``, so ``json.dumps`` has nothing to escape,
        ``_scan_remove_string_delimiters`` copies it through untouched, and
        ``hoist_deferred_comments`` re-emits the body verbatim as a trailing comment
        before any trailing comma.

        The WHOLE body from the comment marker onward is deferred (title + description +
        default + example), not just the description fragment: the hoist appends its
        comment at end of line, so deferring only the description would strand
        ``(default=...)`` in a first, separate ``//`` run and produce two comment runs on
        one line. ``pattern``/``format``/``length_range``/``items_range`` are interpolated
        BEFORE the comment marker in every f-string, so they are excluded by construction
        and keep their current escaped spelling.

        The split is searched for in ``tail`` only. ``head`` may already hold rendered
        text with comments of its own -- a union's members, a pattern containing
        ``" // "`` -- and splitting there would turn the rest of it into comment lines.

        Args:
            head: The representation up to this node's own comment, e.g. ``'string'``
                or a union's joined members.
            tail: This node's own comment run, e.g. ``' // say "hi" now'``.
            has_description: Whether a non-empty description fragment was actually
                produced for this node. Only then is anything routed, so every
                description-free field stays on the literal path with its golden intact.

        Returns:
            ``head + tail`` unchanged when ``has_description`` is false, when ``tail``
            carries no ``f" {self.comment_prefix} "`` occurrence, or when either already
            carries a deferred marker. Otherwise ``tail`` split at its FIRST such
            occurrence into ``before`` and ``body``, returned as
            ``head + before + self.defer_comment(body)``.
        """
        representation = head + tail
        if not has_description or self.carries_deferred_comment(representation):
            return representation
        before, sep, body = tail.partition(f" {self.comment_prefix} ")
        if not sep:
            return representation
        return head + before + self.defer_comment(body)

    def _get_options_format_pattern(self, value: dict[str, Any]) -> tuple[str, str]:
        """Extract format and pattern from schema value.

        Note: this previously also returned an `options` element derived from an
        `OPTIONS: ...` string built from `value["enum"]`. That branch was proven
        unreachable (every caller of `process_types` checks `enum` before reaching
        this code) and was removed.

        Delegates to `BaseFormatter.pattern_token` / `format_token`; output is
        byte-identical to the previous inline implementation.
        """
        pattern_frag = self.pattern_token(value)
        pattern = f" ({pattern_frag})" if pattern_frag else ""

        format_frag = self.format_token(value)
        format_ = f" ({format_frag})" if format_frag else ""

        return format_, pattern

    def process_ref(  # type: ignore[override]
        self, value: dict[str, Any], key: str | None = None
    ) -> str | dict[str, Any] | list[Any]:
        """
        Process a $ref reference using trial's logic.

        Reads both the property node's own ``description`` (``value``) and the resolved
        definition's ``description`` (``_def``). BOTH are emitted when present, and
        de-duplicated when their sanitized texts are equal. YAML and TypeScript carry no
        "the property wins" precedent -- their ``$ref`` renderers never read the
        definition's description at all -- so this is additive, not a reconciliation of an
        existing precedence. The definition's description keeps its ``pending_prefix``
        slot on a block's opening line; the property's own description becomes the field's
        trailing comment via ``pending_postfix`` (which renders on the block's CLOSING
        line).

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

        if self._reentry_truncated(_ref):
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
        if "default" in value and self.config.includes("default"):
            if isinstance(output, str):
                output = output + f" (default='{value['default']}')"

        # Include description from resolved definition if available
        if (
            isinstance(_def, dict)
            and "description" in _def
            and _def["description"]
            and self.config.includes("description")
        ):
            def_description = (
                f" {self.comment_prefix} {self.sanitize_comment_text(_def['description'])}"  # noqa: E501
            )
            if self.carries_deferred_comment(output):
                # Plain-scalar invariant: never append a bare `//` comment to a
                # marker-bearing representation. Suffixes appended later (` []` from a
                # container, ` OR null` from an anyOf) would land *after* that comment
                # and be swallowed into it by the hoist. Route the text into the
                # marker's slot instead -- and only when `process_enum` has not already
                # folded this same `$defs` description in.
                resolved = self.resolved_deferred_text(output)
                sanitized_def = self.sanitize_comment_text(_def["description"])
                if sanitized_def not in resolved:
                    output = self.append_deferred_comment(str(output), sanitized_def)
            elif isinstance(output, str) and def_description not in output:
                # The result of this arm becomes a dict value and therefore meets
                # `json.dumps`; route the comment body through the deferred channel so a
                # quoted or multi-line definition description is not re-escaped.
                output = self._defer_comment_body(str(output), def_description, True)
            elif isinstance(output, dict | list) and key is not None:
                self.pending_prefix[key] = def_description.strip()

        # The property node's own description (defect 2, lsl-2026-09-05-005). Read
        # INDEPENDENTLY of the definition's -- both are emitted when present -- gated by
        # the same metadata gate and de-duplicated against whatever text the node already
        # carries. Without the gate, the `include_metadata=False` and
        # `include_descriptions=False` arms of the metadata matrix would start leaking
        # description text through the `$ref` path.
        if "description" in value and value["description"] and self.config.includes("description"):
            prop_description = self.sanitize_comment_text(value["description"])
            if self.carries_deferred_comment(output):
                # Plain-scalar invariant, as above: fold into the marker's slot rather
                # than appending a second literal `//` run.
                if prop_description not in self.resolved_deferred_text(output):
                    output = self.append_deferred_comment(str(output), prop_description)
            elif isinstance(output, str):
                fragment = f" {self.comment_prefix} {prop_description}"
                if fragment not in output:
                    output = self._defer_comment_body(str(output), fragment, True)
            elif isinstance(output, dict | list) and key is not None:
                # MERGE, never overwrite: `pending_postfix[key]` may already hold an
                # `OR null ...` fragment from `process_anyof` or a `default` fragment, and
                # clobbering it would silently delete that information.
                fragment = f"{self.comment_prefix} {prop_description}"
                existing = self.pending_postfix.get(key, "")
                if prop_description not in existing:
                    self.pending_postfix[key] = (
                        f"{existing.rstrip()} {fragment}" if existing else fragment
                    )

        if isinstance(output, dict | list):
            return copy.deepcopy(output)
        return str(output) if not isinstance(output, str) else output

    def _process_composition(
        self,
        value: dict[str, Any],
        key: str | None,
        keyword: str,
        null_label: str,
        joiner: str,
        label: str = "",
    ) -> str | dict[str, Any] | list[Any]:
        """Render an anyOf / oneOf / allOf node; the three differ only in the arguments.

        Args:
            value: Dictionary containing the ``keyword`` member list.
            key: Optional property key for postfix tracking.
            keyword: ``"anyOf"``, ``"oneOf"`` or ``"allOf"``. allOf alone drops the node's
                own description and merges members that are all dicts.
            null_label: Postfix head for a two-member ``[container, null]``.
            joiner: Separator between rendered members.
            label: Text before the joined members when there is more than one.

        Returns:
            Formatted representation (string, or dict/list for nested serialization).
        """
        comment = ""
        title, description, default_value, example = self._get_title_description_default_value(
            value
        )
        if self._is_root_schema(value):
            title, description = "", ""
        if keyword == "allOf" and value.get("description"):
            # If this allOf node has its own description (e.g. same as parent
            # items.description), skip it: it is already shown in the array header.
            description = ""
        items: list[dict[str, Any] | str | list[Any]] = []
        for item in value.get(keyword, []):
            # Include description from individual members inline
            item_desc = self._extract_description(item)
            if isinstance(item, dict) and item.get("$ref"):
                processed_item = self.process_ref(item, key)
            else:
                processed_item = self._process_schema_recursive(item)
            if item_desc:
                if isinstance(processed_item, dict | list):
                    processed_item = self._jsonish_dump(processed_item) + item_desc
                else:
                    processed_item = str(processed_item) + item_desc
            items.append(processed_item)

        if description or default_value:
            comment = f" {self.comment_prefix}"
        tail = f"{comment}{title}{description}{default_value}{example}"
        if len(items) == 2 and isinstance(items[0], dict | list) and items[1] == "null":
            postfix = f"{null_label} {tail}".rstrip()
            if key is not None:
                self.pending_postfix[key] = postfix
            elif self._is_root_schema(value):
                self.pending_root_postfix = postfix
            return items[0]
        if keyword == "allOf":
            merged = self._merge_allof_objects(items)
            if merged is not None:
                return merged
        str_items = [
            self._jsonish_dump(item) if isinstance(item, dict | list) else str(item)
            for item in items
        ]
        output = label + joiner.join(str_items) if len(str_items) > 1 else str_items[0]
        return self._defer_comment_body(output, tail, bool(description))

    def process_anyof(  # type: ignore[override]
        self, value: dict[str, Any], key: str | None = None
    ) -> str | dict[str, Any] | list[Any]:
        """Process anyOf union types (see ``_process_composition``)."""
        return self._process_composition(
            value, key, "anyOf", "OR null", self.config.union_separator
        )

    def process_oneof(  # type: ignore[override]
        self, value: dict[str, Any], key: str | None = None
    ) -> str | dict[str, Any] | list[Any]:
        """Process oneOf exclusive choice types (see ``_process_composition``)."""
        return self._process_composition(
            value, key, "oneOf", "ONE OF:", self.config.union_separator, "ONE OF: "
        )

    def _merge_allof_objects(
        self, items: list[dict[str, Any] | str | list[Any]]
    ) -> dict[str, Any] | None:
        """If all items are dicts, merge them by key (shallow merge). Otherwise return None."""
        dicts = [x for x in items if isinstance(x, dict)]
        if len(dicts) != len(items):
            return None
        return {k: v for d in dicts for k, v in d.items()}

    def process_allof(  # type: ignore[override]
        self, value: dict[str, Any], key: str | None = None
    ) -> str | dict[str, Any] | list[Any]:
        """Process allOf intersection types (see ``_process_composition``)."""
        return self._process_composition(value, key, "allOf", "AND null", " AND ")

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

                length_tok = self.length_range_token(value)
                length_range = f" ({length_tok})" if length_tok else ""
                if title or description or default_value or example:
                    comment = f" {self.comment_prefix}"
                return self._defer_comment_body(
                    f"{type_name}{pattern}{format_}{length_range}",
                    f"{comment}{title}{description}{default_value}{example}",
                    bool(description),
                )
            elif value["type"] in ["number", "integer"]:
                type_name = "float" if value["type"] == "number" else "int"
                range_tok = self.numeric_range_token(value)
                value_range = f" ({range_tok})" if range_tok else ""
                if title or description or default_value or example:
                    comment = f" {self.comment_prefix}"
                return self._defer_comment_body(
                    f"{type_name}{format_}{pattern}{value_range}",
                    f"{comment}{title}{description}{default_value}{example}",
                    bool(description),
                )
            elif value["type"] == "boolean":
                type_name = "bool"
                if title or description or default_value or example:
                    comment = f" {self.comment_prefix}"
                return self._defer_comment_body(
                    type_name,
                    f"{comment}{title}{description}{default_value}{example}",
                    bool(description),
                )
            elif value["type"] == "array":
                shape = classify_container(value)
                if shape.kind == "tuple":
                    tuple_str = self.render_tuple_token(shape, value)
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
                    return self._defer_comment_body(
                        f"{items} []{items_range}",
                        f"{comment}{title}{description}{default_value}{example}",
                        bool(description),
                    )
                else:
                    comment = (
                        f" {self.comment_prefix}"
                        if (title or description or default_value or example)
                        else ""
                    )
                    return self._defer_comment_body(
                        f"[]{items_range}",
                        f"{comment}{title}{description}{default_value}{example}",
                        bool(description),
                    )
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
                    return self._defer_comment_body(
                        f"{value['type'][0]} {format_}{pattern}",
                        f"{comment}{title}{description}{default_value}{example}",
                        bool(description),
                    )
                elif len(value["type"]) == 2 and "null" in value["type"]:
                    if "array" in value["type"]:
                        array_items: dict[str, Any] | str | list[Any] = {}
                        if title or description or default_value or example:
                            comment = f" {self.comment_prefix}"
                            if key is not None:
                                fragment = (
                                    f"or null {comment}{title}{description}{default_value}{example}"
                                )
                                self.pending_postfix[key] = fragment.rstrip()
                        if "items" in value and value["items"]:
                            array_items = self._process_schema_recursive(value["items"])
                        if isinstance(array_items, dict | list):
                            if isinstance(array_items, dict):
                                return [array_items]
                            return array_items
                        elif isinstance(array_items, str | int | float | bool):
                            return f"{array_items} []"
                    return self._defer_comment_body(
                        f"{value['type'][0]} {format_}{pattern} or null ",
                        f"{comment}{title}{description}{default_value}{example}",
                        bool(description),
                    )
                else:
                    return self._defer_comment_body(
                        f"{', '.join(value['type'])} {format_}{pattern}",
                        f"{comment}{title}{description}{default_value}{example}",
                        bool(description),
                    )

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

    def _mint_identity(self, processed_prop_name: str) -> str:
        """Append a fresh per-occurrence identity token to a marked property name.

        Called exactly once per property occurrence, at the single mint site in the
        property loop. The returned string becomes both the ``output`` dict key and the
        ``key`` argument handed to every ``process_*`` writer for this occurrence, which
        is what makes ``_apply_pending_prefix``/``_apply_pending_postfix`` exact matches
        instead of name-substring matches.

        Contract on ``key``: it is only ever used as a pending-map key or passed through
        to a nested ``process_ref``. It is NEVER embedded in returned output text and
        NEVER compared against anything. Any new ``process_*`` writer must preserve this,
        or the token will leak past ``_strip_identity_tokens``.

        Args:
            processed_prop_name: The property name with its required/optional marker
                already appended (e.g. ``"kids*"`` or ``"kids"``).

        Returns:
            ``processed_prop_name`` with ``⟪lslid<nonce>.<n>⟫`` appended, where ``<n>`` is
            this instance's next monotonic counter value.
        """
        token = "".join(
            (
                DEFERRED_OPEN,
                IDENTITY_TAG,
                self._deferred_nonce,
                ".",
                str(self._identity_counter),
                DEFERRED_CLOSE,
            )
        )
        self._identity_counter += 1
        return f"{processed_prop_name}{token}"

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
                processed_prop_name = self._mint_identity(processed_prop_name)
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
                    output[processed_prop_name] = self.process_const(value)
                elif "enum" in value and value["enum"]:
                    # Priority: enum before type (multiple literal values)
                    output[processed_prop_name] = self.process_enum(value)
                elif "type" in value:
                    output[processed_prop_name] = self.process_types(value, processed_prop_name)
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
            return self.process_types(schema)
        elif "allOf" in schema and schema["allOf"]:
            return self.process_allof(schema)
        elif "$ref" in schema and schema["$ref"]:
            return self.process_ref(schema)

        return output

    def get_required_fields_comment(self) -> str:
        """The base legend plus the trailing newline JSONish's assembly expects."""
        legend = super().get_required_fields_comment()
        return f"{legend}\n" if legend else ""

    def get_schema_info_comment(self) -> str:
        """
        Get a comment containing schema title and description if present.

        Returns:
            Comment string with schema title and description, or empty string if neither present.
        """
        return self.get_info_comment(self.effective_root_schema())

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

        if "title" in schema and schema["title"] and self.config.includes("title"):
            comments.append(f"{self.comment_prefix}Title: {schema['title']}")

        if (
            "description" in schema
            and schema["description"]
            and self.config.includes("description")
        ):
            comments.append(
                f"{self.comment_prefix} {self.sanitize_comment_text(schema['description'])}"
            )

        if comments:
            return "\n".join(comments) + "\n"
        return ""

    def _jsonish_dump(self, obj: dict[str, Any] | list[Any] | Any) -> str:
        """
        Serialize object to JSONish format using json.dumps with postprocessing.

        Args:
            obj: Object to serialize (dict, list, or primitive).

        Returns:
            Serialized JSONish string.
        """
        # Step 1: Use standard json.dumps with indentation
        json_output = json.dumps(obj, indent=2, ensure_ascii=False)

        # Step 2: Process __additional_properties__ and convert to comments
        json_output = self._process_additional_properties(json_output)

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

    def _process_additional_properties(self, json_string: str) -> str:
        """
        Convert __additional_properties__ to inline comments.

        Args:
            json_string: JSON string potentially containing __additional_properties__.

        Returns:
            JSON string with __additional_properties__ converted to comments.
        """
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

                    # The ``// Root:`` prefix is driven by which sentinel key matched, so a
                    # nested model never inherits it.
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
            content = re.sub(" {2,}", " ", content)

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

    def _strip_identity_tokens(self, text: str) -> str:
        """Remove every identity token minted by this instance from finished text.

        Idempotent and safe on text with no tokens. Must run AFTER both
        ``_apply_pending_postfix`` and ``_apply_pending_prefix`` have consumed the tokens
        for matching, and BEFORE ``hoist_deferred_comments`` and before
        ``self.simplified_schema`` is assigned.

        Args:
            text: Rendered JSONish text that may still contain identity tokens.

        Returns:
            ``text`` with all ``⟪lslid<this instance's nonce>.<n>⟫`` substrings removed.
        """
        if DEFERRED_OPEN not in text:  # fast path, mirrors hoist_deferred_comments
            return text
        return self._identity_pattern.sub("", text)

    def _apply_pending_prefix(self, output_string: str) -> str:
        """Append opening-line comments (e.g. a ``$defs`` docstring) to block openers."""
        if not self.pending_prefix:
            return output_string
        lines = output_string.split("\n")
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            for key, prefix in self.pending_prefix.items():
                # Exact match: `key` carries a per-occurrence identity token.
                if stripped.startswith(f"{key}:"):
                    if line.rstrip().endswith("{"):
                        lines[idx] = f"{line.rstrip()} {prefix}"
                    break
        return "\n".join(lines)

    def _join_postfix(self, line: str, postfix: str) -> str:
        """Append ``postfix`` to ``line``, keeping at most one comment marker on the line.

        When both the line and the postfix already carry ``comment_prefix``, the postfix's
        comment body is folded into the line's existing comment as a comma-separated
        continuation, and any segment already present on the line is dropped.

        Since a described comment body is routed through the deferred channel, ``line``
        may itself carry a deferred marker (not yet a literal ``//``) at the moment this
        runs. When ``line`` carries a deferred marker AND ``postfix`` contains
        ``comment_prefix``, ``postfix``'s comment body is folded into the marker's slot
        via ``append_deferred_comment`` rather than appended as a second literal ``//``
        run -- otherwise the two comments render in the wrong order with two markers on
        one line. ``append_deferred_comment`` joins with ``"; "``, not this method's
        ``", "``.
        """
        marker = self.comment_prefix
        if self.carries_deferred_comment(line) and marker in postfix:
            _, _, body = postfix.partition(marker)
            return self.append_deferred_comment(line, body.strip())
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
                # Exact match: `key` carries a per-occurrence identity token, so no
                # separate "with/without asterisk" branch is needed or correct.
                stripped = line.lstrip()
                if stripped.startswith(f"{key}:"):
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
        self.pending_postfix.clear()
        self.pending_prefix.clear()
        self.pending_recursion.clear()
        self.pending_root_postfix = ""
        output = self._process_schema_recursive(self.schema)
        output_string = ""
        if output and isinstance(output, dict | list):
            output_string = self._jsonish_dump(output)
            output_string = self._collapse_array_object_brackets(output_string)
        else:
            output_string = str(output)
        if self.pending_root_postfix:
            output_string = f"{output_string} {self.pending_root_postfix}"
        info, legend = self.root_decorations()
        output_string = f"{info}{legend}{output_string}"
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
        output_string = self._strip_identity_tokens(output_string)
        # ORDERING INVARIANT: any future pass that matches on property names goes ABOVE
        # this line; any pass producing final user-facing text goes BELOW it.
        output_string = self.hoist_deferred_comments(output_string)
        output_string = self._rstrip_lines(output_string)
        self.simplified_schema = output_string
        return self._add_prefix(output_string)

    def _rstrip_lines(self, text: str) -> str:
        """Guarantee that no emitted line ends in whitespace (ticket AC-3).

        Runs as the LAST transformation in ``transform_schema``, below the ordering
        invariant comment -- this pass produces final user-facing text and must never run
        before a pass that matches on property names (``_apply_pending_postfix`` /
        ``_apply_pending_prefix``, which both already run above it).

        This is a GLOBAL, structural backstop, not a substitute for the local
        ``.rstrip()`` fixes on the keyed ``pending_postfix`` f-strings: those keep the
        stored postfix values clean for ``_join_postfix``'s and
        ``_merge_pending_recursion``'s own ``.rstrip()``/``in``-tests, while this pass is
        what makes AC-3 true structurally, including for any site future routing
        introduces.

        Args:
            text: Fully assembled JSONish output, immediately before ``_add_prefix``.

        Returns:
            ``text`` with every physical line right-stripped. A blank line rstrips to
            ``""`` and stays blank; leading indentation is untouched.
        """
        return "\n".join(line.rstrip() for line in text.split("\n"))

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """
        Estimate token count for the simplified schema.

        Args:
            encoding: Tokenizer encoding to use (default: cl100k_base for GPT-4).

        Returns:
            Estimated token count.
        """
        from ..core import _count_tokens

        return _count_tokens(self.transform_schema(), encoding)

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
        from ..core import _compare_tokens

        return _compare_tokens(
            original_schema or self.schema, simplified_schema or self.transform_schema(), encoding
        )
