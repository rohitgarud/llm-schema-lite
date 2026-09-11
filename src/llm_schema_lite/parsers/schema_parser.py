"""Schema parser for parsing with schema validation and partial extraction."""

import json
import re
from collections.abc import Iterable
from typing import Any, cast

from jsonschema import Draft202012Validator, FormatChecker
from pydantic import BaseModel

from ..coercion import ParseConfig, coerce_to_schema
from ..exceptions import ConversionError
from ..schema_enrichment import enrich_schema_with_enum_metadata
from .base import BaseParser
from .json_parser import JSONParser

# =============================================================================
# Module-level helper functions (for internal use and re-export)
# =============================================================================


def _get_json_schema(schema: "type[BaseModel] | dict[str, Any] | str") -> dict[str, Any]:
    """Convert schema input to JSON schema dict."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        json_schema = schema.model_json_schema()
        enrich_schema_with_enum_metadata(schema, json_schema)
        return json_schema
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, str):
        try:
            return cast(dict[str, Any], json.loads(schema))
        except json.JSONDecodeError as e:
            raise ConversionError(f"Invalid JSON schema string: {e}") from e
    raise ConversionError(
        f"Unsupported schema type: {type(schema)}. Expected Pydantic BaseModel, dict, or str."
    )


def _validate_field(value: Any, field_schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single field against its schema."""
    try:
        format_checker = FormatChecker()
        validator = Draft202012Validator(field_schema, format_checker=format_checker)
        errors = list(validator.iter_errors(value))
        if not errors:
            return True, []
        error_messages = [err.message for err in errors]
        return False, error_messages
    except Exception:
        return True, []


MARKER_WALK_MAX_DEPTH = 8


def _marker_object_nodes(
    node: Any, defs: dict[str, Any], _seen: frozenset[str] = frozenset()
) -> list[dict[str, Any]]:
    """Expand a schema node to the concrete object nodes it can denote.

    Resolves $ref (cycle-guarded by _seen) and flattens anyOf / oneOf / allOf into the
    list of non-`null` branches, the node itself first. Returns [] when nothing concrete
    is reachable (a non-dict node, an unresolvable $ref, a $ref already on this path).

    Uses a FUNCTION-LOCAL import of _resolve_ref: a module-level
    `from ..validators.enum_aliases import _resolve_ref` is a measured circular import
    (parsers/__init__ -> schema_parser -> validators/__init__ -> yaml_validators ->
    parsers). This mirrors the file's own convention -- `from ..validators import
    JSONValidator` is already function-local in SchemaParser.parse, for the same
    circular-import reason.

    Args:
        node: A JSON-schema node -- typically a `properties` value, an `items` value, or
            a `$defs` entry. May be a $ref, a plain object node, or a union construct.
        defs: The root schema's $defs (or definitions) dict, for $ref resolution.
        _seen: $ref strings already resolved on this path (cycle guard). Callers never
            pass this; it is threaded internally across recursive calls.

    Returns:
        A list of schema nodes that `node` can concretely denote, most specific first.
    """
    # Local import to avoid circular import
    from ..validators.enum_aliases import _resolve_ref

    if not isinstance(node, dict):
        return []

    ref = node.get("$ref")
    if isinstance(ref, str):
        if ref in _seen:
            return []
        resolved = _resolve_ref(ref, defs)
        if resolved is None:
            return []
        return _marker_object_nodes(resolved, defs, _seen | {ref})

    nodes: list[dict[str, Any]] = [node]
    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = node.get(keyword)
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if isinstance(branch, dict) and branch.get("type") == "null":
                continue
            nodes.extend(_marker_object_nodes(branch, defs, _seen))
    return nodes


def normalize_marker_keys_recursive(
    data: Any,
    schema: dict[str, Any],
    marker: str,
    defs: dict[str, Any] | None = None,
    _depth: int = 0,
) -> Any:
    """Walk data and schema in parallel, remapping marked keys at every object level.

    Container contract:
      - properties (nested BaseModel): remap keys at this level via normalize_marker_keys
        using the MERGED properties of _marker_object_nodes(schema) as known_keys, then
        descend each value with its property subschema.
      - $ref: resolved via _marker_object_nodes before any remap or descent.
      - items (list[Model], list[list[Model]]): descend every element with the same
        subschema.
      - prefixItems (tuple[Model, int]): descend positionally; elements past the tuple
        length are left alone.
      - additionalProperties (dict[str, Model]): descend VALUES ONLY -- never remap the
        outer mapping's own keys as if they were declared properties.
      - anyOf / oneOf / allOf: merge the properties of all non-null branches for the key
        remap; descend a value with the FIRST branch that declares that key.

    Never touches the keys of an open-ended mapping: a dict[str, X] node has no
    `properties`, so the remap step never runs there and only values are descended. A
    field genuinely named e.g. "name*" is protected by normalize_marker_keys' rule 1.

    Depth guard: MARKER_WALK_MAX_DEPTH (8), counted in DATA depth, not schema depth.
    Beyond the cap, remapping stops and data is returned as-is for that subtree -- a
    still-marked key then fails jsonschema/model_validate loudly (ConversionError), never
    silently. This constant is deliberately NOT linked to
    FormatterConfig.max_recursion_depth, which is a render-side cap on a different walk.

    Args:
        data: The parsed reply data at this level (dict, list, or scalar).
        schema: The JSON-schema node describing `data` at this level (the root schema on
            the initial call).
        marker: The trailing marker to strip; falsy is a global no-op.
        defs: The root schema's $defs/definitions. Resolved from `schema` on the initial
            call (_depth == 0) when not supplied; threaded unchanged on recursive calls.
        _depth: Data-depth counter. Callers never pass this.

    Returns:
        A new structure with the same shape as `data`, with marker keys remapped at every
        reachable object level; `data` unchanged past MARKER_WALK_MAX_DEPTH or where the
        schema resolves to nothing concrete.
    """
    if not marker or _depth > MARKER_WALK_MAX_DEPTH:
        return data

    if defs is None:
        raw_defs = schema.get("$defs") or schema.get("definitions") or {}
        defs = raw_defs if isinstance(raw_defs, dict) else {}

    nodes = _marker_object_nodes(schema, defs)
    if not nodes:
        return data

    if isinstance(data, dict):
        merged: dict[str, Any] = {}
        additional: Any = None
        for node in nodes:
            props = node.get("properties")
            if isinstance(props, dict):
                for name, subschema in props.items():
                    merged.setdefault(name, subschema)
            extra = node.get("additionalProperties")
            if additional is None and isinstance(extra, dict):
                additional = extra

        if merged:
            data = normalize_marker_keys(data, merged.keys(), marker)

        result: dict[str, Any] = {}
        for key, value in data.items():
            child_schema = merged.get(key, additional)
            if isinstance(child_schema, dict):
                result[key] = normalize_marker_keys_recursive(
                    value, child_schema, marker, defs, _depth + 1
                )
            else:
                result[key] = value
        return result

    if isinstance(data, list):
        prefix_items: list[Any] = []
        items_schema: Any = None
        for node in nodes:
            candidate_prefix = node.get("prefixItems")
            if not prefix_items and isinstance(candidate_prefix, list):
                prefix_items = candidate_prefix
            candidate_items = node.get("items")
            if items_schema is None and isinstance(candidate_items, dict):
                items_schema = candidate_items

        walked: list[Any] = []
        for index, element in enumerate(data):
            element_schema: Any = None
            if index < len(prefix_items):
                element_schema = prefix_items[index]
            elif items_schema is not None:
                element_schema = items_schema
            if isinstance(element_schema, dict):
                walked.append(
                    normalize_marker_keys_recursive(
                        element, element_schema, marker, defs, _depth + 1
                    )
                )
            else:
                walked.append(element)
        return walked

    return data


def normalize_marker_keys(
    data: dict[str, Any], known_keys: Iterable[str], marker: str
) -> dict[str, Any]:
    """Map trailing-marker keys in ONE dict onto the known names they denote.

    Pure, single-marker, single-object-level. This is the extraction of the three-rule
    algorithm that used to live only inside SchemaParser, with
    `properties` generalised to `known_keys` so both SchemaParser and
    StructuredOutputAdapter can call it with their own key universe (JSON-schema
    properties vs. signature.output_fields).

    Rule order -- verbatim always wins:
      1. key in known_keys -> kept as-is (checked FIRST), so a schema that legitimately
         declares a property literally named "name*" is never disturbed.
      2. key.endswith(marker) and the stripped name is non-empty, IS in known_keys, and
         is NOT already a key in `data` -> remapped to the stripped name.
      3. otherwise -> kept unchanged, left for downstream to drop as the unknown key it
         is. Stripping must never invent a name that was not already known.

    Degenerate inputs (`marker` falsy, `data` empty, `known_keys` empty) are a no-op:
    the identity of `data` is returned unchanged, never a copy.

    Args:
        data: The dict to normalize keys of. Never mutated.
        known_keys: The declared/accepted key universe for this call (schema
            properties, or signature.output_fields.keys()).
        marker: The trailing marker to strip, e.g. "*". Falsy disables stripping.

    Returns:
        A new dict (or `data` itself, unchanged, for the no-op cases) with the same
        values and remapped keys.
    """
    if not marker or not data:
        return data
    known = set(known_keys)
    if not known:
        return data

    result: dict[str, Any] = {}
    for key, value in data.items():
        if key in known:
            result[key] = value
            continue
        if key.endswith(marker):
            stripped = key[: -len(marker)]
            if stripped and stripped in known and stripped not in data:
                result[stripped] = value
                continue
        result[key] = value
    return result


def _build_result(
    data: dict[str, Any],
    schema: "type[BaseModel] | dict[str, Any] | str",
    *,
    validate: bool = False,
) -> "BaseModel | dict[str, Any]":
    """Build result as Pydantic model if schema is a BaseModel, otherwise return dict.

    When `schema` is a BaseModel subclass:
      - `validate=False` (default, unchanged): `schema.model_construct(**data)` --
        bypasses validation, tolerates partial/None fields. This is what every existing
        caller gets, including the direct-call tests in tests/test_partial_extraction.py.
      - `validate=True`: `schema.model_validate(data)`, with `pydantic.ValidationError`
        caught and re-raised as `ConversionError(f"Validation failed: {exc}") from exc` --
        the same "Validation failed: ..." prefix the non-partial route already produces,
        so the public error contract is unchanged in shape. This is what builds real
        nested model instances rather than leaving nested data as raw dicts.
    When `schema` is not a BaseModel (dict or str), `validate` has no effect; `data` is
    returned unchanged, as today.

    Args:
        data: The parsed data dictionary.
        schema: Pydantic BaseModel subclass, JSON-schema dict, or JSON-schema string.
        validate: Keyword-only. See the contract above. Default False keeps every
            pre-existing caller behaviorally identical.

    Returns:
        Pydantic model instance or dict.

    Raises:
        ConversionError: only when `validate` is True and `schema.model_validate` raises
            pydantic.ValidationError.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        if validate:
            from pydantic import ValidationError

            try:
                return schema.model_validate(data)
            except ValidationError as exc:
                raise ConversionError(f"Validation failed: {exc}") from exc
        # Use model_construct to bypass validation - allows partial fields with None
        return schema.model_construct(**data)
    return data


# =============================================================================
# Internal function for schema-aware parsing (called by core.loads)
# =============================================================================


def parse_with_schema(
    text: str,
    schema: "type[BaseModel] | dict[str, Any] | str",
    parse_config: ParseConfig,
) -> tuple["BaseModel | dict[str, Any]", dict[str, Any]]:
    """
    Parse LLM output with schema validation and optional partial extraction.

    This is an internal function called by core.loads() when a schema is provided.
    It implements the full parsing logic that was previously in parse_llm_output().

    Args:
        text: The text content to parse
        schema: Pydantic BaseModel, JSON schema dict, or JSON schema string
        parse_config: Configuration for parsing/coercion behavior (includes partial flag)

    Returns:
        Tuple of (parsed_result, metadata)
        - parsed_result: Model instance or dict with extracted fields
        - metadata: Dict with failed_fields info (empty dict if all fields valid)

    Raises:
        ConversionError: If partial=False and parsing fails, or if required fields fail
    """
    # Create SchemaParser with the schema
    schema_parser = SchemaParser(schema=schema, parse_config=parse_config)

    if parse_config.partial:
        # Partial mode: use field-by-field extraction
        result_dict, failed_fields = schema_parser._parse_partial(text, repair=True)
        final_result = schema_parser.build_result(result_dict)
        return final_result, {"failed_fields": failed_fields}
    else:
        # Full validation mode: SchemaParser.parse already validated against the schema
        result_dict = schema_parser.parse(text, repair=True)
        final_result = schema_parser.build_result(result_dict, validate=True)
        return final_result, {}


class SchemaParser(BaseParser):
    """
    Schema-aware parser with support for partial extraction.

    Parses text content and validates/coerces against a provided schema.
    Supports partial extraction mode to extract valid fields even when
    some fields fail validation.
    """

    def __init__(
        self,
        schema: "type[BaseModel] | dict[str, Any] | str",
        parse_config: ParseConfig | None = None,
    ):
        """
        Initialize SchemaParser with schema.

        Args:
            schema: Pydantic BaseModel, JSON schema dict, or JSON schema string
            parse_config: Configuration for parsing/coercion behavior
        """
        self._schema = schema
        self._parse_config = parse_config or ParseConfig()
        self._json_parser = JSONParser()
        # Use module-level function
        self._json_schema = _get_json_schema(schema)

    def _parse_to_dict(self, text: str, repair: bool, *, rescue_embedded: bool) -> Any:
        """Sole text->dict entry point for both SchemaParser routes.

        Parses via self._json_parser, optionally rescues embedded JSON on failure, then
        normalizes required-marker keys via normalize_marker_keys_recursive exactly once,
        walking self._json_schema with self._parse_config.strip_required_marker (an empty
        marker is a no-op). That single call is a RECURSIVE walk of data and schema in
        parallel, not a flat top-level remap: it strips markers at every reachable nested
        object level.

        `rescue_embedded` PRESERVES an existing asymmetry between SchemaParser.parse's
        non-partial branch and _parse_partial rather than introducing a new one: the
        non-partial route lets a ConversionError from self._json_parser.parse
        propagate; the partial route catches it and falls back to
        self._extract_json_from_text. Folding both into one unconditional behaviour
        would silently change the non-partial route's failure mode for garbage input
        from a parse error into a validation error, which is out of scope for
        lsl-2026-09-04-008. _extract_json_from_text itself is unchanged.

        Args:
            text: The text content to parse.
            repair: Whether to attempt repair for malformed content, forwarded to
                self._json_parser.parse.
            rescue_embedded: True for the _parse_partial route (catches ConversionError
                and retries via self._extract_json_from_text); False for the
                SchemaParser.parse non-partial route (lets ConversionError propagate).

        Returns:
            Whatever self._json_parser.parse (or the embedded-JSON rescue) produced, run
            through normalize_marker_keys_recursive if it is a dict. Non-dict results are
            returned unchanged -- both callers have their own non-dict handling.

        Raises:
            ConversionError: propagates from self._json_parser.parse when
                rescue_embedded is False, or from the embedded-JSON rescue when
                rescue_embedded is True and the rescue also fails.
        """
        if rescue_embedded:
            try:
                parsed = self._json_parser.parse(text, repair)
            except ConversionError:
                parsed = self._extract_json_from_text(text)
        else:
            parsed = self._json_parser.parse(text, repair)

        if isinstance(parsed, dict):
            marker = self._parse_config.strip_required_marker
            return normalize_marker_keys_recursive(parsed, self._json_schema, marker)
        return parsed

    def parse(self, text: str, repair: bool = True) -> dict[str, Any]:
        """
        Parse text content with schema validation.

        When partial mode is enabled in config, extracts valid fields even
        when some fields fail validation.

        Args:
            text: The text content to parse
            repair: Whether to attempt repair for malformed content

        Returns:
            Tuple of (parsed_result, metadata)
            - parsed_result: Dict with extracted fields
            - metadata: Dict with failed_fields info (empty if all fields valid)

        Raises:
            ConversionError: If parsing fails and partial mode is disabled,
                            or if required fields fail in partial mode
        """
        from ..validators import JSONValidator

        if self._parse_config.partial:
            # For partial mode, return just the dict (callers should use _parse_partial directly)
            return self._parse_partial(text, repair)[0]
        else:
            # Full validation mode - parse then validate
            parsed: dict[str, Any] = self._parse_to_dict(text, repair, rescue_embedded=False)

            # Validate against full schema
            is_valid, errors = JSONValidator(self._schema).validate(parsed, return_all_errors=True)
            if not is_valid:
                error_list = errors if errors else []
                raise ConversionError(f"Validation failed: {'; '.join(error_list)}")

            return parsed

    def _parse_partial(self, text: str, repair: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Parse with partial extraction - extract valid fields even if some fail.

        Args:
            text: The text content to parse
            repair: Whether to attempt repair for malformed content

        Returns:
            Tuple of (result_dict, failed_fields_dict)

        Raises:
            ConversionError: If required fields are missing or fail validation
        """
        # First, parse the text to dict (marker-key normalization happens inside)
        data: dict[str, Any] = self._parse_to_dict(text, repair, rescue_embedded=True)

        if not isinstance(data, dict):
            data = {}

        # Get required fields
        required_fields = set(self._json_schema.get("required", []))
        properties = self._json_schema.get("properties", {})

        # Track results
        result_dict: dict[str, Any] = {}
        failed_fields: dict[str, Any] = {}

        # Process each field in schema
        for field_name, field_schema in properties.items():
            # Check if field is present in data
            has_field = field_name in data

            # If field not present in data
            if not has_field:
                if field_name in required_fields:
                    raise ConversionError(f"Required field '{field_name}' is missing")
                # Optional field not provided - will default to None
                continue

            value = data[field_name]

            # Try to coerce and validate the field
            try:
                # Coerce the value - need to wrap in proper object schema
                if self._parse_config.allow_coercion:
                    field_object_schema = {
                        "type": "object",
                        "properties": {field_name: field_schema},
                    }
                    coerced_value, _ = coerce_to_schema(
                        {field_name: value}, field_object_schema, self._parse_config
                    )
                    field_value = coerced_value.get(field_name, value)
                else:
                    field_value = value

                # Validate the field
                is_valid, errors = _validate_field(field_value, field_schema)
                if is_valid:
                    result_dict[field_name] = field_value
                else:
                    # Field failed validation
                    if field_name in required_fields:
                        raise ConversionError(
                            f"Required field '{field_name}' failed validation: "
                            f"{errors[0] if errors else 'unknown error'}"
                        )
                    # Optional field - set to None and track failure
                    result_dict[field_name] = None
                    failed_fields[field_name] = value
            except ConversionError:
                # Re-raise ConversionError (for required fields)
                raise
            except Exception as err:
                # Any other exception means validation/coercion failed
                if field_name in required_fields:
                    raise ConversionError(
                        f"Required field '{field_name}' failed validation"
                    ) from err
                result_dict[field_name] = None
                failed_fields[field_name] = value

        return result_dict, failed_fields

    def _extract_json_from_text(self, text: str) -> dict[str, Any]:
        """Extract JSON object from text that may contain extra content."""
        # Try to find JSON object in the text
        json_match = re.search(r"\{[^{}]*\}", text)
        if json_match:
            try:
                parsed: dict[str, Any] = JSONParser().parse(json_match.group(), repair=True)
                return parsed
            except ConversionError:
                pass

        # If no JSON found, return empty dict
        return {}

    def build_result(
        self, data: dict[str, Any], *, validate: bool = False
    ) -> "BaseModel | dict[str, Any]":
        """
        Build result as Pydantic model if schema is a BaseModel, otherwise return dict.

        Args:
            data: The parsed data dictionary
            validate: Forwarded to the module-level _build_result. False (default)
                keeps model_construct; True switches to model_validate with
                ValidationError translated to ConversionError. See _build_result's
                docstring for the full contract.

        Returns:
            Pydantic model instance or dict
        """
        # Delegate to module-level function
        return _build_result(data, self._schema, validate=validate)
