"""Test helpers for formatter outputs.

Currently focused on the JSONish formatter output shape.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from pydantic import BaseModel

from llm_schema_lite import FormatterConfig, simplify_schema
from llm_schema_lite.schema_normalization import normalize_schema_titles

_FIELD_LINE_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)(\*)?\s*:")
_BLOCK_COMMENT_RE = re.compile(r"/\*(.*?)\*/")
_FORMAT_TYPES = ("jsonish", "yaml", "typescript")


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
    normalized_schema = normalize_schema_titles(schema)
    expected = bool(include_metadata and normalized_schema.get("title"))
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


def assert_schema_info_comment_presence(
    result: str, include_metadata: bool, schema: dict[str, Any] | None = None
) -> None:
    """Assert presence/absence of schema-level info comments based on flag.

    JSONishFormatter currently includes the "Fields marked with * are required"
    comment whenever the schema has required fields, even when include_metadata=False.
    So this helper only checks schema *info* (e.g. `//Title:`), not required-field comments.

    When `schema` is provided, the expected title-comment presence is computed from
    its *normalized* title (auto-generated titles stripped) combined with
    `include_metadata`, matching `assert_schema_title_comment_consistent`. When
    `schema` is omitted, behaviour is byte-identical to before this parameter existed.
    """

    has_title_comment = "//Title:" in result
    if schema is not None:
        normalized_schema = normalize_schema_titles(schema)
        expected = bool(include_metadata and normalized_schema.get("title"))
        assert has_title_comment == expected, (
            f"Title comment presence mismatch: expected={expected} actual={has_title_comment}. "
            f"Snippet: {result[:200]!r}"
        )
        return

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


def render_all_formatters(
    model: type[BaseModel], config: FormatterConfig | None = None
) -> dict[str, str]:
    """Render `model` through JSONish, YAML, and TypeScript with the same config.

    A fresh `dataclasses.replace(config)` copy is forwarded to each of the three
    `simplify_schema` calls (when `config` is given) rather than the same instance.
    This is render isolation, not a workaround: no formatter constructor mutates its
    caller's `FormatterConfig` (JSONish and YAML both apply their default union
    separator through `with_format_default_separator`, which copies rather than
    writes through). The per-call copy exists so the three renders stay independent
    of each other regardless of what a future formatter does, and so a caller can
    inspect its own `config` unchanged after calling this helper -- not because any
    current formatter would otherwise corrupt it.

    Args:
        model: Pydantic model (or anything else `simplify_schema` accepts) to render.
        config: FormatterConfig forwarded, unmutated, to each of the three renders.

    Returns:
        Mapping of format name ("jsonish", "yaml", "typescript") to rendered string.
    """

    return {
        fmt: simplify_schema(
            model,
            config=dataclasses.replace(config) if config is not None else None,
            format_type=fmt,
        ).to_string()
        for fmt in _FORMAT_TYPES
    }


def _line_comment_slots(line: str, marker: str) -> list[str]:
    """Extract every `marker`-delimited comment body found on one physical line.

    A `marker` occurrence only starts a comment slot when it sits at the very start of
    the line or is immediately preceded by whitespace. Every formatter here always emits
    its trailing comment after such a whitespace gap (see `base.py`'s
    `deferred_comment_gap` / `_hoist_deferred_line`, and the schema-level `f"{prefix} ..."`
    comments), while occurrences of the same characters *inside a value* never are: a
    `//` inside `https://...` is glued directly to the scheme letters, and a literal `#`
    inside a regex pattern is glued to whatever character precedes it. This one rule
    disambiguates both cases without special-casing URLs or patterns individually.

    Multiple qualifying markers on one line each start their own slot, running up to the
    next qualifying marker (or end of line) -- this matters for YAML, whose folded
    scalars sometimes pack two independent `# ...` bodies onto a single physical line.

    Args:
        line: One physical line of rendered formatter output.
        marker: The comment marker to look for ("//" or "#").

    Returns:
        The stripped text of every qualifying comment slot on `line`, in left-to-right
        order; empty when `line` carries no qualifying marker.
    """

    positions: list[int] = []
    search_from = 0
    while True:
        found = line.find(marker, search_from)
        if found == -1:
            break
        if found == 0 or line[found - 1].isspace():
            positions.append(found)
        search_from = found + len(marker)

    slots: list[str] = []
    for i, pos in enumerate(positions):
        start = pos + len(marker)
        end = positions[i + 1] if i + 1 < len(positions) else len(line)
        body = line[start:end].strip()
        # The structural separator that follows a hoisted comment (a trailing "," for
        # JSONish, ";" for TypeScript) is not part of the comment text.
        body = body.rstrip(",;").strip()
        if body:
            slots.append(body)
    return slots


def extract_comment_slots(text: str, fmt: str) -> list[str]:
    """Extract the text of every comment slot in `text` for formatter `fmt`.

    A naive `"//" not in output` (or `"#" not in output`) is not soundly testable: field
    descriptions may themselves contain `//`, rendered defaults may contain `https://`,
    and patterns may contain a literal `#`. None of those are comments, so this function
    recognises only the marker shapes each formatter actually emits comments with --
    `//` line comments for JSONish, `#` line comments for YAML, and *both* `//` line
    comments and `/* ... */` block comments for TypeScript -- anchored on the
    whitespace-gap convention documented on `_line_comment_slots`.

    Args:
        text: Rendered formatter output.
        fmt: One of "jsonish", "yaml", "typescript".

    Returns:
        Line-level `//` bodies for jsonish, `#` bodies for yaml, plus `/* ... */`
        bodies for typescript, in the order they appear in `text`.

    Raises:
        ValueError: If `fmt` is not one of "jsonish", "yaml", "typescript".
    """

    if fmt not in _FORMAT_TYPES:
        raise ValueError(f"Unknown formatter fmt: {fmt!r}; expected one of {_FORMAT_TYPES}")

    marker = "#" if fmt == "yaml" else "//"
    slots: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line
        if fmt == "typescript":
            for block_match in _BLOCK_COMMENT_RE.finditer(line):
                body = block_match.group(1).strip()
                if body:
                    slots.append(body)
            line = _BLOCK_COMMENT_RE.sub("", line)
        slots.extend(_line_comment_slots(line, marker))

    return slots
