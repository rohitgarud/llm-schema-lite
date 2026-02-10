"""Tests for YAML formatter using PyYAML."""

import yaml

from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter

# Import models from conftest
from tests.conftest import (
    OrderedFieldsModel,
    PersonWithAddress,
    SimpleFormatterModel,
)

# Alias for backwards compatibility in tests
SimpleModel = SimpleFormatterModel


def test_yaml_formatter_produces_valid_yaml():
    """Test that YAML formatter produces valid YAML that can be parsed."""
    schema = SimpleModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Verify the output is valid YAML
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)
    assert "name*" in parsed  # Required field
    assert "age*" in parsed  # Required field
    assert "email" in parsed  # Optional field


def test_yaml_formatter_without_metadata():
    """Test YAML formatter without metadata."""
    schema = SimpleModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Verify the output is valid YAML
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)
    assert "name*" in parsed
    assert "age*" in parsed
    assert "email" in parsed


def test_yaml_formatter_with_nested_defs():
    """Test YAML formatter with nested $defs."""
    schema = PersonWithAddress.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain both Address section and main Person properties
    assert "# Address" in result
    assert "Address.street*:" in result
    assert "Address.city*:" in result
    assert "name*:" in result or "name*: str" in result


def test_yaml_formatter_key_order_preserved():
    """Test that YAML formatter preserves key order (dict order)."""
    schema = OrderedFieldsModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    lines = result.strip().split("\n")
    # Filter out comment lines
    field_lines = [line for line in lines if not line.strip().startswith("#")]

    # Check that fields appear in order
    first_idx = next(i for i, line in enumerate(field_lines) if "first*:" in line)
    second_idx = next(i for i, line in enumerate(field_lines) if "second*:" in line)
    third_idx = next(i for i, line in enumerate(field_lines) if "third*:" in line)

    assert first_idx < second_idx < third_idx


def test_yaml_formatter_caching():
    """Test that formatter caching works correctly."""
    schema = SimpleModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)

    # First call
    result1 = formatter.transform_schema()
    # Second call should use cache
    result2 = formatter.transform_schema()

    assert result1 == result2
    assert hasattr(formatter, "_processed_data")
    assert formatter._processed_data is not None
