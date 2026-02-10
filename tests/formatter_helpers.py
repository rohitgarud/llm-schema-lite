"""Test helpers for formatter outputs.

Currently focused on the JSONish formatter output shape.
"""

from __future__ import annotations

import re
from typing import Any

_FIELD_LINE_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)(\*)?\s*:")


def parse_jsonish_root_fields(text: str) -> list[tuple[str, bool]]:
    """Parse root-level JSONish field lines.

    Extracts `(field_name, is_required)` from lines like:

      name*: string
      email: string

    Notes:
    - Only matches fields at the *root object* level (depth==1), so nested objects
      don't affect required/optional assertions for the root schema.
    - Ignores anything after '//' on a line (comments).
    """

    depth = 0
    fields: list[tuple[str, bool]] = []

    for raw_line in text.splitlines():
        # Strip inline comments so braces inside comments don't affect depth.
        line = raw_line.split("//", 1)[0]

        depth_before = depth
        if depth_before == 1:
            m = _FIELD_LINE_RE.match(line)
            if m:
                fields.append((m.group(1), m.group(2) == "*"))

        # Update depth after processing this line, based on braces/brackets.
        # This is intentionally simple: JSONish output removes quotes, so we don't
        # try to handle quoted braces.
        depth += line.count("{") - line.count("}")

    return fields


def parse_typescript_interface_fields(
    text: str, interface_name: str = "Schema"
) -> list[tuple[str, bool]]:
    """Parse fields from a specific TypeScript interface block.

    Extracts `(field_name, is_required)` from lines like:

      name*: string;
      email: string | null;

    Notes:
    - Only parses the requested `interface <name> { ... }` body.
    - Ignores comment-only lines that start with `//`.
    """

    start_pat = re.compile(rf"^\s*interface\s+{re.escape(interface_name)}\s*\{{\s*$")
    in_block = False
    depth = 0
    fields: list[tuple[str, bool]] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not in_block:
            if start_pat.match(line):
                in_block = True
                depth = 1
            continue

        # Track brace depth to know when the interface ends.
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            break

        stripped = line.lstrip()
        if not stripped or stripped.startswith("//"):
            continue

        m = _FIELD_LINE_RE.match(line)
        if m:
            fields.append((m.group(1), m.group(2) == "*"))

    return fields


def parse_yaml_root_fields(text: str) -> list[tuple[str, bool]]:
    """Parse root-level fields from YAML formatter output.

    YAMLFormatter encodes requiredness by appending '*' to the key name, e.g.:
      name*: str
      email: str
    """

    # Local import: tests already depend on PyYAML.
    import yaml  # type: ignore[import-not-found]

    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        return []

    fields: list[tuple[str, bool]] = []
    for k in parsed.keys():
        if not isinstance(k, str):
            continue
        if k.endswith("*"):
            fields.append((k[:-1], True))
        else:
            fields.append((k, False))
    return fields


def assert_required_optional_fields_match_schema(
    fields: list[tuple[str, bool]], schema: dict[str, Any]
) -> None:
    """Assert parsed `(name, required)` fields match schema required/properties."""

    required_in_schema = set(schema.get("required", []) or [])
    properties_in_schema = set((schema.get("properties", {}) or {}).keys())

    field_names = {name for name, _ in fields}
    required_in_output = {name for name, is_req in fields if is_req}

    missing_props = properties_in_schema - field_names
    assert not missing_props, f"Missing properties in output: {sorted(missing_props)}"

    assert required_in_output == required_in_schema, (
        f"Required fields in output should match schema. "
        f"schema={sorted(required_in_schema)} output={sorted(required_in_output)}"
    )


def assert_schema_title_comment_consistent(
    result: str, schema: dict[str, Any], include_metadata: bool, comment_prefix: str
) -> None:
    """Assert presence/absence of schema title comment based on schema + flag."""

    # JSONish uses `//Title:`, base formatters use `// Title:` / `# Title:`.
    has_title_comment = f"{comment_prefix}Title:" in result or f"{comment_prefix} Title:" in result
    expected = bool(include_metadata and schema.get("title"))
    assert has_title_comment == expected, (
        f"Title comment presence mismatch: expected={expected} actual={has_title_comment}. "
        f"Snippet: {result[:200]!r}"
    )


def assert_required_optional_consistent(result: str, schema: dict[str, Any]) -> None:
    """Assert that required fields are marked with '*' and optional fields are not."""

    required = set(schema.get("required", []) or [])
    properties = set((schema.get("properties", {}) or {}).keys())

    # If schema has no properties, there's nothing to check.
    if not properties:
        return

    for field in required:
        assert f"{field}*:" in result, (
            f"Required field '{field}' should be marked with '*'. "
            f"Schema required={sorted(required)}. Output snippet: {result[:200]!r}"
        )

    for field in properties - required:
        assert f"{field}*:" not in result, (
            f"Optional field '{field}' must not be marked with '*'. "
            f"Schema required={sorted(required)}. Output snippet: {result[:200]!r}"
        )


def assert_schema_info_comment_presence(result: str, include_metadata: bool) -> None:
    """Assert presence/absence of schema-level info comments based on flag.

    JSONishFormatter currently includes the "Fields marked with * are required"
    comment whenever the schema has required fields, even when include_metadata=False.
    So this helper only checks schema *info* (e.g. `//Title:`), not required-field comments.
    """

    has_title_comment = "//Title:" in result
    if include_metadata:
        assert has_title_comment, (
            "Expected schema title comment when include_metadata=True. "
            f"Output snippet: {result[:200]!r}"
        )
    else:
        assert not has_title_comment, (
            "Did not expect schema title comment when include_metadata=False. "
            f"Output snippet: {result[:200]!r}"
        )
