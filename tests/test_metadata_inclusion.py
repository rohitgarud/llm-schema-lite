"""Tests for metadata_inclusion configuration feature."""

from pydantic import BaseModel, Field

from llm_schema_lite import FormatterConfig, simplify_schema


class ModelWithMetadata(BaseModel):
    """Model with various metadata for testing."""

    name: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z]+$")
    age: int = Field(..., ge=0, le=120)
    email: str = Field(..., json_schema_extra={"format": "email"})
    tags: list[str] = Field(default_factory=list, min_length=1, max_length=10)
    status: str = "active"  # has default


class TestMetadataInclusionDefaults:
    """Test suite for default metadata_inclusion behavior."""

    def test_default_includes_constraints(self):
        """Default config should include constraint metadata."""
        config = FormatterConfig()
        result = simplify_schema(ModelWithMetadata, config=config, format_type="jsonish")
        output = result.to_string()

        # Should include minLength/maxLength
        assert "minLength" in output or "1" in output  # minLength: 1
        assert "maxLength" in output or "100" in output

        # Should include pattern
        assert "PATTERN" in output or "pattern" in output.lower()

        # Should include format
        assert "FORMAT" in output or "email" in output

    def test_default_excludes_examples(self):
        """Default config should NOT include examples."""

        # First, create a model with examples
        class ModelWithExamples(BaseModel):
            code: str = Field(default="ABC", json_schema_extra={"examples": ["XYZ", "123"]})

        # First, let's verify examples are NOT included by default
        config_default = FormatterConfig()
        result = simplify_schema(ModelWithExamples, config=config_default, format_type="jsonish")
        output = result.to_string()

        # Should NOT have EXAMPLES by default
        assert "EXAMPLE" not in output


class TestMetadataInclusionExcludeAll:
    """Test suite for exclude all metadata."""

    def test_include_metadata_false_disables_all(self):
        """Setting include_metadata=False should disable all metadata."""
        config = FormatterConfig(include_metadata=False)
        result = simplify_schema(ModelWithMetadata, config=config)
        output = result.to_string()

        # No constraint metadata should appear
        assert "PATTERN" not in output
        assert "FORMAT" not in output
        assert "minLength" not in output
        assert "maxLength" not in output
        assert "1-100" not in output


class TestMetadataInclusionCustom:
    """Test suite for custom metadata_inclusion settings."""

    def test_pattern_false(self):
        """Test disabling pattern specifically."""
        config = FormatterConfig(metadata_inclusion={"pattern": False})
        result = simplify_schema(ModelWithMetadata, config=config, format_type="jsonish")
        output = result.to_string()

        # Should NOT have pattern
        assert "PATTERN" not in output
        assert "^[a-z]+$" not in output
        # Other constraints should still be present
        assert "minLength" in output or "1" in output

    def test_format_false(self):
        """Test disabling format specifically."""
        config = FormatterConfig(metadata_inclusion={"format": False})
        result = simplify_schema(ModelWithMetadata, config=config, format_type="jsonish")
        output = result.to_string()

        # Should NOT have format metadata
        assert "FORMAT:" not in output
        # Note: The word "email" may still appear in description comments, which is fine

    def test_minmax_length_false(self):
        """Test disabling minLength/maxLength specifically."""
        config = FormatterConfig(metadata_inclusion={"minLength": False, "maxLength": False})
        result = simplify_schema(ModelWithMetadata, config=config, format_type="jsonish")
        output = result.to_string()

        # Should NOT have length constraints
        assert "chars" not in output
        # But pattern should still be there
        assert "PATTERN" in output or "^[a-z]+$" in output

    def test_minmax_value_false(self):
        """Test disabling minimum/maximum specifically."""

        class RangeModel(BaseModel):
            value: int = Field(..., ge=0, le=100)

        config = FormatterConfig(metadata_inclusion={"minimum": False, "maximum": False})
        result = simplify_schema(RangeModel, config=config, format_type="jsonish")
        output = result.to_string()

        # Should NOT have min/max values
        assert "0 to 100" not in output
        assert ">= 0" not in output
        assert "<= 100" not in output

    def test_examples_true(self):
        """Test enabling examples explicitly."""

        # Create model with examples in json_schema_extra
        class ModelWithExamples(BaseModel):
            code: str = Field(default="ABC", json_schema_extra={"examples": ["XYZ", "123"]})

        config = FormatterConfig(metadata_inclusion={"examples": True})
        result = simplify_schema(ModelWithExamples, config=config, format_type="jsonish")
        output = result.to_string()

        # Examples should now appear
        assert "EXAMPLE" in output or "XYZ" in output

    def test_default_false(self):
        """Test disabling default values."""

        class ModelWithDefault(BaseModel):
            status: str = "active"

        config = FormatterConfig(metadata_inclusion={"default": False})
        result = simplify_schema(ModelWithDefault, config=config, format_type="jsonish")
        output = result.to_string()

        # Should NOT have default value
        assert "default" not in output.lower() or "default=" not in output


class TestMetadataInclusionTypeScript:
    """Test suite for metadata_inclusion with TypeScript formatter."""

    def test_typescript_format_respects_inclusion(self):
        """Test TypeScript formatter respects metadata_inclusion."""
        config = FormatterConfig(metadata_inclusion={"format": False, "pattern": False})
        result = simplify_schema(ModelWithMetadata, config=config, format_type="typescript")
        output = result.to_string()

        # Should NOT have format in metadata comment
        assert "format:" not in output.lower()
        # Should NOT have pattern in metadata comment
        assert "pattern:" not in output.lower()

    def test_typescript_length_constraints(self):
        """Test TypeScript formatter respects minLength/maxLength."""
        config = FormatterConfig(metadata_inclusion={"minLength": False, "maxLength": False})
        result = simplify_schema(ModelWithMetadata, config=config, format_type="typescript")
        output = result.to_string()

        # Should NOT have length range in type
        assert "chars" not in output


class TestMetadataInclusionYAML:
    """Test suite for metadata_inclusion with YAML formatter."""

    def test_yaml_format_respects_inclusion(self):
        """Test YAML formatter respects metadata_inclusion."""
        config = FormatterConfig(metadata_inclusion={"format": False})
        result = simplify_schema(ModelWithMetadata, config=config, format_type="yaml")
        output = result.to_string()

        # Should NOT have FORMAT metadata
        assert "FORMAT:" not in output
        # Note: The word "email" may still appear in description comments, which is fine


class TestMetadataInclusionPartial:
    """Test suite for partial metadata_inclusion dictionaries."""

    def test_partial_dict_merges_with_defaults(self):
        """Partial dict should merge with defaults."""
        config = FormatterConfig(metadata_inclusion={"examples": True})

        # examples should be True (overridden)
        assert config.metadata_inclusion["examples"] is True
        # Other keys should use defaults (True for constraints)
        assert config.metadata_inclusion["pattern"] is True
        assert config.metadata_inclusion["format"] is True

    def test_partial_dict_only_overrides_specified(self):
        """Only specified keys should be overridden, rest use defaults."""
        config = FormatterConfig(metadata_inclusion={"pattern": False})

        # Only pattern should be False
        assert config.metadata_inclusion["pattern"] is False
        # format should still be True (default)
        assert config.metadata_inclusion["format"] is True
        # minimum/maximum should still be True (default)
        assert config.metadata_inclusion["minimum"] is True
        assert config.metadata_inclusion["maximum"] is True


class TestMetadataInclusionArrayConstraints:
    """Test suite for array constraint metadata."""

    def test_unique_items_respected(self):
        """Test uniqueItems can be controlled via metadata_inclusion."""

        class ArrayModel(BaseModel):
            items: list[str] = Field(default_factory=list)

        config = FormatterConfig(metadata_inclusion={"uniqueItems": False})
        result = simplify_schema(ArrayModel, config=config, format_type="jsonish")
        output = result.to_string()

        # uniqueItems should not appear in output when disabled
        assert "UNIQUE" not in output
        assert "unique" not in output.lower()

    def test_min_items_max_items_respected(self):
        """Test minItems/maxItems can be controlled via metadata_inclusion."""

        class ItemsModel(BaseModel):
            tags: list[str] = Field(..., min_length=1, max_length=10)

        config = FormatterConfig(metadata_inclusion={"minItems": False, "maxItems": False})
        result = simplify_schema(ItemsModel, config=config, format_type="jsonish")
        output = result.to_string()

        # minItems/maxItems should not appear when disabled
        # (may still show as length range depending on formatter)
        assert "minItems" not in output
        assert "maxItems" not in output

    def test_container_type_constraint_words_agree_across_formatters(self):
        """The three formatters use the same constraint vocabulary for dict/tuple/set fields.

        Regression for lsl-2026-09-04-015: pins that "unique" / "2-2 items" phrasing is now
        identical (modulo the tuple[] / Record<> / {} wrapper) across jsonish/typescript/yaml.
        """

        class ContainerModel(BaseModel):
            tags: set[str]
            bounded: set[str] = Field(..., min_length=2, max_length=2)
            mapping: dict[str, int]

        outputs = {
            fmt: simplify_schema(ContainerModel, format_type=fmt).to_string()
            for fmt in ("jsonish", "typescript", "yaml")
        }

        for fmt, output in outputs.items():
            # uniqueItems is on for a `set` field -- every formatter says the same word.
            assert "unique" in output, f"{fmt} lost the uniqueItems word: {output!r}"
            assert output.count("unique") == 2, f"{fmt} has the wrong unique count: {output!r}"
            # The shared min/max phrasing.
            assert "2-2 items" in output, f"{fmt} lost the shared items-range phrasing: {output!r}"
            # The old, per-formatter spellings are gone.
            assert "UNIQUE" not in output, f"{fmt} still uses YAML's uppercase spelling: {output!r}"
            assert "length:" not in output, f"{fmt} still uses TS's `length:` spelling: {output!r}"
            assert "≥" not in output, f"{fmt} still uses TS's >= glyph: {output!r}"
            assert "≤" not in output, f"{fmt} still uses TS's <= glyph: {output!r}"

        # The mapping field renders structurally in all three, never as `additional:`.
        for fmt, output in outputs.items():
            assert "additional:" not in output, f"{fmt} still leaks `additional:`: {output!r}"


class TestMetadataInclusionBackwardCompatibility:
    """Test suite for backward compatibility."""

    def test_old_api_still_works(self):
        """Test that old API (include_metadata=False) still works."""
        # Using the old API
        result = simplify_schema(ModelWithMetadata, include_metadata=False)
        output = result.to_string()

        # Should have no metadata
        assert "PATTERN" not in output
        assert "FORMAT" not in output

    def test_default_config_backward_compatible(self):
        """Test that default config maintains backward compatibility."""
        config = FormatterConfig()

        # Should have default behavior (include constraints, exclude examples)
        result = simplify_schema(ModelWithMetadata, config=config, format_type="jsonish")
        output = result.to_string()

        # Constraints should be included
        assert "PATTERN" in output or "pattern" in output.lower()
        assert "FORMAT" in output or "email" in output

    def test_legacy_include_metadata_does_not_mutate_caller_config(self, patient_model):
        """The deprecated `include_metadata=` kwarg must not mutate the caller's config.

        Regression for lsl-2026-09-04-005: `BaseFormatter.__init__` now takes a
        `dataclasses.replace(config)` copy before applying the legacy kwarg override, so the
        caller's own `FormatterConfig` instance is left untouched.
        """
        cfg = FormatterConfig(include_metadata=True)
        simplify_schema(patient_model, config=cfg, include_metadata=False)
        assert cfg.include_metadata is True

    def test_metadata_inclusion_description_false_suppresses_descriptions(self, patient_model):
        """`metadata_inclusion={"description": False}` must suppress descriptions.

        Regression for lsl-2026-09-04-005: before this ticket, the "description" key in
        `metadata_inclusion` was a documented no-op in both JSONish and YAML because
        description text was emitted through a separate, ungated extractor. Both formatters
        now route through `BaseFormatter._should_include_metadata`, so this is a genuine new
        behaviour guarantee.
        """
        config = FormatterConfig(metadata_inclusion={"description": False})

        for fmt in ("jsonish", "yaml"):
            output = simplify_schema(patient_model, config=config, format_type=fmt).to_string()
            assert "Full name" not in output, f"{fmt} still leaked the 'name' description"
            assert "A patient record." not in output, f"{fmt} still leaked the schema description"
