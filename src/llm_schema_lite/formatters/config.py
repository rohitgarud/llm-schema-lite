"""Configuration dataclass for formatters."""

from dataclasses import dataclass, replace

DESCRIPTION_KEYWORDS: frozenset[str] = frozenset(
    {"title", "description", "id", "$comment", "x-enum-descriptions"}
)

STRUCTURAL_KEYWORDS: frozenset[str] = frozenset({"enum"})
"""Keywords whose value is structural, not metadata: emitted regardless of `include_metadata`,
`include_constraints`/`include_descriptions`, and `metadata_inclusion`. See `FormatterConfig`'s
`Precedence:` docstring paragraph. Not exported (matches `DESCRIPTION_KEYWORDS`)."""

# Package-wide default union separator; formatters compare against this value (not a
# sentinel) to decide whether the caller already chose their own separator.
DEFAULT_UNION_SEPARATOR: str = " | "

# Default metadata inclusion configuration
# Includes all constraint metadata by default, excludes examples for token savings
DEFAULT_METADATA_INCLUSION: dict[str, bool] = {
    # Constraints - included by default (True)
    "pattern": True,
    "format": True,
    "minimum": True,
    "maximum": True,
    "minLength": True,
    "maxLength": True,
    "minItems": True,
    "maxItems": True,
    "uniqueItems": True,
    "const": True,
    "default": True,
    "title": True,
    # Examples - excluded by default (False) for token savings
    "examples": False,
}


@dataclass
class FormatterConfig:
    """
    Configuration class for formatter behavior.

    This class allows customization of formatter output including:
    - Instruction prefix
    - Union separators and markers
    - Indentation
    - Optional/required field markers
    - Description/constraint inclusion toggles
    - Metadata inclusion per-keyword control

    Attributes:
        prefix: Instruction prefix to prepend to output (default: None)
        union_separator: Separator for union types (default: " | ")
        indent: Number of spaces for indentation (default: 2)
        optional_marker: Marker for optional fields (default: "" - none)
        required_marker: Marker for required fields (default: "*")
        include_descriptions: Category flag for the description-family keywords listed in
            DESCRIPTION_KEYWORDS (default: True). Narrows include_metadata; cannot restore
            what include_metadata removed.
        include_constraints: Category flag for every metadata keyword not in
            DESCRIPTION_KEYWORDS (default: True). Narrows include_metadata; cannot restore
            what include_metadata removed.
        include_metadata: Master switch for all metadata (default: True). When False, no
            metadata keyword is emitted regardless of the category flags or
            metadata_inclusion.
        max_recursion_depth: How many times a recursive $ref's body is rendered on any one
            path before it is replaced by a placeholder. Default 2. A value of 0 behaves as
            1 (the first expansion of any $ref is always unconditional). Negative values
            raise ValueError.
        max_ref_expansions: Total $ref expansions allowed across the whole render, an
            unconditional safety net over the per-path max_recursion_depth. Default 150.
            Once exhausted every further $ref renders as `object // budget exhausted: Name`.
            Raise it for a large definition graph that is legitimately wide rather than
            deep; it also tiers the anyOf/oneOf member caps. Must be >= 1.
        backreference_min_chars: Smallest rendered body worth replacing with a named
            back-reference (`object // defined above: Address`) on a REPEAT occurrence.
            Default 200. Below it a definition is simply inlined again, which is what two
            sibling fields sharing a $ref should look like. The two populations separate
            cleanly at the default: the largest body the sibling-inline tests rely on is
            158 chars, the smallest definition in the 244-reference corpus schema is 409.
            0 names every repeat; a very large value inlines every repeat. Must be >= 0.
        metadata_inclusion: Dictionary to control which metadata keywords are included
            in output. Keys are metadata keyword names, values are booleans.
            If not provided, defaults to DEFAULT_METADATA_INCLUSION.
            Example: {"pattern": True, "format": True, "examples": False}

    Precedence:
        A metadata keyword is emitted only if **all** of `include_metadata`, its category flag
        (`include_descriptions` for the `title`/`description`-family keywords listed in
        `DESCRIPTION_KEYWORDS`, `include_constraints` for every other keyword), and
        `metadata_inclusion.get(keyword, True)` are true; a narrower gate can only remove
        metadata, never restore what a wider gate removed. Structural information —
        required/optional markers, container tokens, the `additionalProperties: false`
        closed-world marker, and any keyword in `STRUCTURAL_KEYWORDS` (currently just
        `"enum"`) — is not metadata and is emitted regardless of all three flags.
    """

    prefix: str | None = None
    union_separator: str = DEFAULT_UNION_SEPARATOR
    indent: int = 2
    optional_marker: str = ""
    required_marker: str = "*"
    include_descriptions: bool = True
    include_constraints: bool = True
    include_metadata: bool = True
    max_recursion_depth: int = 2
    max_ref_expansions: int = 150
    backreference_min_chars: int = 200
    metadata_inclusion: dict[str, bool] = None  # type: ignore[assignment]

    def includes(self, key: str) -> bool:
        """Return True iff `key` is structural, or survives all three narrowing gates."""
        if key in STRUCTURAL_KEYWORDS:
            return True
        if not self.include_metadata:
            return False
        category_flag = (
            self.include_descriptions if key in DESCRIPTION_KEYWORDS else self.include_constraints
        )
        if not category_flag:
            return False
        return self.metadata_inclusion.get(key, True)

    def __post_init__(self) -> None:
        """Post-initialization to handle metadata_inclusion defaults."""
        if self.max_recursion_depth < 0:
            raise ValueError(f"max_recursion_depth must be >= 0, got {self.max_recursion_depth}")
        if self.max_ref_expansions < 1:
            raise ValueError(f"max_ref_expansions must be >= 1, got {self.max_ref_expansions}")
        if self.backreference_min_chars < 0:
            raise ValueError(
                f"backreference_min_chars must be >= 0, got {self.backreference_min_chars}"
            )
        # Merge any user-provided dict over the defaults (user values win).
        merged = DEFAULT_METADATA_INCLUSION.copy()
        if self.metadata_inclusion is not None:
            merged.update(self.metadata_inclusion)
        self.metadata_inclusion = merged


def with_format_default_separator(
    config: FormatterConfig | None, separator: str
) -> FormatterConfig:
    """Return a config that renders unions with ``separator`` unless the caller chose one.

    A formatter calls this in ``__init__`` to apply its own default union separator
    without ever writing through the object it was handed: the caller's
    ``FormatterConfig`` must survive the call unchanged so it can be reused across
    formatters (and by the DSPy adapter).

    ``None`` yields a fresh config carrying ``separator``. A config still holding the
    package default (``DEFAULT_UNION_SEPARATOR``) yields a ``dataclasses.replace`` copy
    carrying ``separator`` -- ``config`` itself is left untouched. Any other config
    (the caller set their own separator) is returned unchanged, by identity, so an
    explicit ``union_separator`` always wins and no copy is made in that case.

    Args:
        config: The caller-supplied config, or ``None``.
        separator: This format's default union separator (e.g. ``" OR "``).

    Returns:
        A ``FormatterConfig`` carrying `separator` when the input was ``None`` or still
        at the package default; otherwise the same object that was passed in.
    """
    if config is None:
        return FormatterConfig(union_separator=separator)
    if config.union_separator == DEFAULT_UNION_SEPARATOR:
        return replace(config, union_separator=separator)
    return config
