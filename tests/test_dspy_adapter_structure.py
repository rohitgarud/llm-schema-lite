"""Tests for DSPy adapter structure that don't require DSPy to be installed."""

import pytest


class TestAdapterImports:
    """Test that adapter can be imported and has correct structure."""

    def test_output_mode_enum_exists(self):
        """Test that OutputMode enum can be imported."""
        try:
            from llm_schema_lite.dspy_integration.adapters.structured_output_adapter import (
                OutputMode,
            )

            # Verify enum values
            assert hasattr(OutputMode, "JSON")
            assert hasattr(OutputMode, "JSONISH")
            assert hasattr(OutputMode, "YAML")

            # Verify enum values
            assert OutputMode.JSON.value == "json"
            assert OutputMode.JSONISH.value == "jsonish"
            assert OutputMode.YAML.value == "yaml"
        except ImportError as e:
            if "dspy" in str(e).lower():
                pytest.skip("DSPy not installed")
            else:
                raise

    def test_adapter_class_exists(self):
        """Test that StructuredOutputAdapter class can be imported."""
        try:
            from llm_schema_lite.dspy_integration.adapters.structured_output_adapter import (
                StructuredOutputAdapter,
            )

            # Verify class exists
            assert StructuredOutputAdapter is not None

            # Verify it's a class
            assert isinstance(StructuredOutputAdapter, type)
        except ImportError as e:
            if "dspy" in str(e).lower():
                pytest.skip("DSPy not installed")
            else:
                raise

    def test_adapter_has_required_methods(self):
        """Test that adapter has all required methods."""
        try:
            from llm_schema_lite.dspy_integration.adapters.structured_output_adapter import (
                StructuredOutputAdapter,
            )

            # Check for required methods
            required_methods = [
                "__init__",
                "__call__",
                "acall",
                "format_field_structure",
                "user_message_output_requirements",
                "format_field_with_value",
                "format_assistant_message_content",
                "parse",
                "_parse_json",
                "_parse_yaml",
                "_translate_field_type",
                "_get_complex_type_description",
                "_format_yaml_output",
            ]

            for method_name in required_methods:
                assert hasattr(
                    StructuredOutputAdapter, method_name
                ), f"StructuredOutputAdapter missing method: {method_name}"
        except ImportError as e:
            if "dspy" in str(e).lower():
                pytest.skip("DSPy not installed")
            else:
                raise

    def test_module_exports(self):
        """Test that module exports are correct."""
        try:
            from llm_schema_lite.dspy_integration.adapters import (
                OutputMode,
                StructuredOutputAdapter,
            )

            assert OutputMode is not None
            assert StructuredOutputAdapter is not None
        except ImportError as e:
            if "dspy" in str(e).lower():
                pytest.skip("DSPy not installed")
            else:
                raise

    def test_top_level_exports(self):
        """Test that top-level module exports are correct."""
        try:
            from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter

            assert OutputMode is not None
            assert StructuredOutputAdapter is not None
        except ImportError as e:
            if "dspy" in str(e).lower():
                pytest.skip("DSPy not installed")
            else:
                raise


class TestAdapterDocumentation:
    """Test that adapter has proper documentation."""

    def test_adapter_has_docstring(self):
        """Test that adapter class has docstring."""
        try:
            from llm_schema_lite.dspy_integration.adapters.structured_output_adapter import (
                StructuredOutputAdapter,
            )

            assert StructuredOutputAdapter.__doc__ is not None
            assert len(StructuredOutputAdapter.__doc__) > 50

            # Check for key documentation elements
            doc = StructuredOutputAdapter.__doc__.lower()
            assert "adapter" in doc or "structured" in doc
        except ImportError as e:
            if "dspy" in str(e).lower():
                pytest.skip("DSPy not installed")
            else:
                raise

    def test_output_mode_has_docstring(self):
        """Test that OutputMode enum has docstring."""
        try:
            from llm_schema_lite.dspy_integration.adapters.structured_output_adapter import (
                OutputMode,
            )

            assert OutputMode.__doc__ is not None
            assert "json" in OutputMode.__doc__.lower()
        except ImportError as e:
            if "dspy" in str(e).lower():
                pytest.skip("DSPy not installed")
            else:
                raise


class TestHelperFunctions:
    """Test helper functions in the adapter module."""

    def test_helper_function_exists(self):
        """Test that helper function exists."""
        try:
            from llm_schema_lite.dspy_integration.adapters.structured_output_adapter import (
                _get_structured_outputs_response_format,
            )

            assert _get_structured_outputs_response_format is not None
            assert callable(_get_structured_outputs_response_format)
        except ImportError as e:
            if "dspy" in str(e).lower():
                pytest.skip("DSPy not installed")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
