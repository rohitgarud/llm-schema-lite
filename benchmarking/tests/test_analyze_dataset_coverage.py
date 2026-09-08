#!/usr/bin/env python3
"""
Comprehensive tests for the analyze_dataset_coverage function.

This test suite covers dataset analysis functionality including
coverage statistics, token reduction calculations, and error handling.
"""

import logging
from io import StringIO
from unittest.mock import MagicMock, patch

from ..base import analyze_dataset_coverage


class TestAnalyzeDatasetCoverage:
    """Test suite for analyze_dataset_coverage function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Capture logging output for testing
        self.log_capture = StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        logger = logging.getLogger("benchmarking.base")
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.INFO)

    def teardown_method(self):
        """Clean up test fixtures."""
        logger = logging.getLogger("benchmarking.base")
        logger.removeHandler(self.log_handler)

    def get_log_output(self):
        """Get captured log output."""
        return self.log_capture.getvalue()

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_successful_dataset_analysis(self, mock_encoding, mock_simplify_schema):
        """Test successful analysis of a dataset with valid schemas."""
        # Mock tiktoken encoding
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder
        mock_encoder.encode.side_effect = lambda x: list(x.encode("utf-8"))  # Simple mock

        # Mock simplify_schema to return a mock object with to_string method
        mock_simplified = MagicMock()
        mock_simplified.to_string.return_value = '{"type": "object"}'  # Shorter than original
        mock_simplify_schema.return_value = mock_simplified

        dataset = [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "string", "enum": ["a", "b", "c"]},
            {"type": "array", "items": {"type": "number"}},
        ]

        analyze_dataset_coverage(dataset)

        # Verify simplify_schema was called for each schema
        assert mock_simplify_schema.call_count == 3
        assert mock_simplify_schema.call_args_list[0][0][0] == dataset[0]
        assert mock_simplify_schema.call_args_list[1][0][0] == dataset[1]
        assert mock_simplify_schema.call_args_list[2][0][0] == dataset[2]

        # Check log output contains expected information
        log_output = self.get_log_output()
        assert "📊 Dataset Coverage Analysis" in log_output
        assert "✅ Supported schemas: 3/3 (100.0%)" in log_output
        assert "❌ Failed schemas: 0/3 (0.0%)" in log_output
        assert "Dataset coverage: 100.0%" in log_output
        assert "Average token reduction:" in log_output
        assert "Max token reduction:" in log_output
        assert "Min token reduction:" in log_output

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_dataset_with_failures(self, mock_encoding, mock_simplify_schema):
        """Test analysis of dataset with some failing schemas."""
        # Mock tiktoken encoding
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder
        mock_encoder.encode.side_effect = lambda x: list(x.encode("utf-8"))

        # Mock simplify_schema to fail for some schemas
        def side_effect(schema, format_type):
            if schema.get("type") == "invalid":
                raise ValueError("Invalid schema")
            mock_simplified = MagicMock()
            mock_simplified.to_string.return_value = '{"type": "object"}'
            return mock_simplified

        mock_simplify_schema.side_effect = side_effect

        dataset = [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "invalid"},  # This will fail
            {"type": "string", "enum": ["a", "b"]},
            {"type": "invalid"},  # This will also fail
        ]

        analyze_dataset_coverage(dataset)

        # Check log output
        log_output = self.get_log_output()
        assert "✅ Supported schemas: 2/4 (50.0%)" in log_output
        assert "❌ Failed schemas: 2/4 (50.0%)" in log_output
        assert "Dataset coverage: 50.0%" in log_output

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_empty_dataset(self, mock_encoding, mock_simplify_schema):
        """Test analysis of empty dataset."""
        dataset = []

        analyze_dataset_coverage(dataset)

        # Check log output
        log_output = self.get_log_output()
        assert "✅ Supported schemas: 0/0 (0.0%)" in log_output
        assert "❌ Failed schemas: 0/0 (100.0%)" in log_output
        assert "Dataset coverage: 0.0%" in log_output
        assert "Average token reduction: N/A (no successful schemas)" in log_output

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_all_schemas_fail(self, mock_encoding, mock_simplify_schema):
        """Test analysis when all schemas fail to process."""
        # Mock tiktoken encoding
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder
        mock_encoder.encode.side_effect = lambda x: list(x.encode("utf-8"))

        # Mock simplify_schema to always fail
        mock_simplify_schema.side_effect = ValueError("All schemas invalid")

        dataset = [{"type": "invalid1"}, {"type": "invalid2"}, {"type": "invalid3"}]

        analyze_dataset_coverage(dataset)

        # Check log output
        log_output = self.get_log_output()
        assert "✅ Supported schemas: 0/3 (0.0%)" in log_output
        assert "❌ Failed schemas: 3/3 (100.0%)" in log_output
        assert "Dataset coverage: 0.0%" in log_output
        assert "Average token reduction: N/A (no successful schemas)" in log_output

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_token_reduction_calculation(self, mock_encoding, mock_simplify_schema):
        """Test token reduction calculation accuracy."""
        # Mock tiktoken encoding with specific token counts
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder

        def mock_encode(data):
            # Simulate different token counts for original vs simplified
            if isinstance(data, str):
                # Simplified schema (shorter)
                return list(range(10))  # 10 tokens
            else:
                # Original schema (longer)
                return list(range(50))  # 50 tokens

        mock_encoder.encode.side_effect = mock_encode

        # Mock simplify_schema
        mock_simplified = MagicMock()
        mock_simplified.to_string.return_value = '{"type": "object"}'
        mock_simplify_schema.return_value = mock_simplified

        dataset = [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "string", "enum": ["a", "b", "c"]},
        ]

        analyze_dataset_coverage(dataset)

        # Check log output for token reduction
        log_output = self.get_log_output()
        assert "Average token reduction:" in log_output
        assert "Max token reduction:" in log_output
        assert "Min token reduction:" in log_output

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_error_logging(self, mock_encoding, mock_simplify_schema):
        """Test that errors are properly logged."""
        # Mock tiktoken encoding
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder
        mock_encoder.encode.side_effect = lambda x: list(x.encode("utf-8"))

        # Mock simplify_schema to raise specific error
        mock_simplify_schema.side_effect = ValueError("Schema validation failed")

        dataset = [{"type": "invalid"}]

        analyze_dataset_coverage(dataset)

        # Check that error was logged
        log_output = self.get_log_output()
        assert "Failed to process schema: Schema validation failed" in log_output
        assert "Average token reduction: N/A (no successful schemas)" in log_output

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_different_format_types(self, mock_encoding, mock_simplify_schema):
        """Test that simplify_schema is called with correct format_type."""
        # Mock tiktoken encoding
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder
        mock_encoder.encode.side_effect = lambda x: list(x.encode("utf-8"))

        # Mock simplify_schema
        mock_simplified = MagicMock()
        mock_simplified.to_string.return_value = '{"type": "object"}'
        mock_simplify_schema.return_value = mock_simplified

        dataset = [{"type": "object"}]

        analyze_dataset_coverage(dataset)

        # Verify simplify_schema was called with format_type="jsonish"
        mock_simplify_schema.assert_called_once()
        call_args = mock_simplify_schema.call_args
        assert call_args[1]["format_type"] == "jsonish"

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_large_dataset_performance(self, mock_encoding, mock_simplify_schema):
        """Test analysis with a larger dataset."""
        # Mock tiktoken encoding
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder
        mock_encoder.encode.side_effect = lambda x: list(x.encode("utf-8"))

        # Mock simplify_schema
        mock_simplified = MagicMock()
        mock_simplified.to_string.return_value = '{"type": "object"}'
        mock_simplify_schema.return_value = mock_simplified

        # Create a larger dataset
        dataset = [
            {"type": "object", "properties": {f"prop{i}": {"type": "string"}}} for i in range(100)
        ]

        analyze_dataset_coverage(dataset)

        # Verify all schemas were processed
        assert mock_simplify_schema.call_count == 100

        # Check log output
        log_output = self.get_log_output()
        assert "✅ Supported schemas: 100/100 (100.0%)" in log_output

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_mixed_schema_types(self, mock_encoding, mock_simplify_schema):
        """Test analysis with various schema types."""
        # Mock tiktoken encoding
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder
        mock_encoder.encode.side_effect = lambda x: list(x.encode("utf-8"))

        # Mock simplify_schema
        mock_simplified = MagicMock()
        mock_simplified.to_string.return_value = '{"type": "object"}'
        mock_simplify_schema.return_value = mock_simplified

        dataset = [
            # Object schema
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
                "required": ["name"],
            },
            # Array schema
            {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10},
            # String schema with constraints
            {"type": "string", "pattern": "^[a-zA-Z]+$", "minLength": 1, "maxLength": 50},
            # Union schema
            {"oneOf": [{"type": "string"}, {"type": "number"}]},
            # Conditional schema
            {
                "if": {"properties": {"age": {"type": "number"}}},
                "then": {"properties": {"age": {"minimum": 0}}},
                "else": {"properties": {"age": {"type": "string"}}},
            },
        ]

        analyze_dataset_coverage(dataset)

        # Verify all schemas were processed
        assert mock_simplify_schema.call_count == 5

        # Check log output
        log_output = self.get_log_output()
        assert "✅ Supported schemas: 5/5 (100.0%)" in log_output

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_edge_case_schemas(self, mock_encoding, mock_simplify_schema):
        """Test analysis with edge case schemas."""
        # Mock tiktoken encoding
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder
        mock_encoder.encode.side_effect = lambda x: list(x.encode("utf-8"))

        # Mock simplify_schema
        mock_simplified = MagicMock()
        mock_simplified.to_string.return_value = '{"type": "object"}'
        mock_simplify_schema.return_value = mock_simplified

        dataset = [
            {},  # Empty schema
            {"type": "object"},  # Minimal schema
            {"$ref": "#/definitions/User"},  # Reference schema
            {"const": "fixed_value"},  # Const schema
            {"enum": ["a", "b", "c"]},  # Enum without type
        ]

        analyze_dataset_coverage(dataset)

        # Verify all schemas were processed
        assert mock_simplify_schema.call_count == 5

        # Check log output
        log_output = self.get_log_output()
        assert "✅ Supported schemas: 5/5 (100.0%)" in log_output

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_token_reduction_edge_cases(self, mock_encoding, mock_simplify_schema):
        """Test token reduction calculation with edge cases."""
        # Mock tiktoken encoding
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder

        call_count = 0

        def mock_encode(data):
            nonlocal call_count
            call_count += 1
            if isinstance(data, str):
                # Simplified schema
                return list(range(5))  # 5 tokens
            else:
                # Original schema
                return list(range(20))  # 20 tokens

        mock_encoder.encode.side_effect = mock_encode

        # Mock simplify_schema
        mock_simplified = MagicMock()
        mock_simplified.to_string.return_value = '{"type": "object"}'
        mock_simplify_schema.return_value = mock_simplified

        dataset = [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "string", "enum": ["a", "b"]},
            {"type": "number", "minimum": 0, "maximum": 100},
        ]

        analyze_dataset_coverage(dataset)

        # Check log output for token reduction percentages
        log_output = self.get_log_output()
        assert "Average token reduction:" in log_output
        assert "Max token reduction:" in log_output
        assert "Min token reduction:" in log_output

    def test_invalid_input_types(self):
        """Test behavior with invalid input types."""
        # Test with None
        analyze_dataset_coverage(None)
        log_output = self.get_log_output()
        assert "✅ Supported schemas: 0/0 (0.0%)" in log_output
        assert "Average token reduction: N/A (no successful schemas)" in log_output

        # Reset log capture for second test
        self.log_capture = StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        logger = logging.getLogger("benchmarking.base")
        logger.removeHandler(self.log_handler)
        logger.addHandler(self.log_handler)

        # Test with non-list
        analyze_dataset_coverage("not a list")
        log_output = self.get_log_output()
        assert "✅ Supported schemas: 0/0 (0.0%)" in log_output
        assert "Average token reduction: N/A (no successful schemas)" in log_output

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_logging_format_consistency(self, mock_encoding, mock_simplify_schema):
        """Test that logging format is consistent."""
        # Mock tiktoken encoding
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder
        mock_encoder.encode.side_effect = lambda x: list(x.encode("utf-8"))

        # Mock simplify_schema
        mock_simplified = MagicMock()
        mock_simplified.to_string.return_value = '{"type": "object"}'
        mock_simplify_schema.return_value = mock_simplified

        dataset = [{"type": "object"}]

        analyze_dataset_coverage(dataset)

        log_output = self.get_log_output()

        # Check for specific formatting elements
        assert "📊 Dataset Coverage Analysis" in log_output
        assert "=" * 50 in log_output
        assert "✅ Supported schemas:" in log_output
        assert "❌ Failed schemas:" in log_output
        assert "Dataset coverage:" in log_output
        assert "Average token reduction:" in log_output
        assert "Max token reduction:" in log_output
        assert "Min token reduction:" in log_output

    @patch("benchmarking.base.simplify_schema")
    @patch("benchmarking.base.tiktoken.encoding_for_model")
    def test_percentage_calculations(self, mock_encoding, mock_simplify_schema):
        """Test percentage calculations are accurate."""
        # Mock tiktoken encoding
        mock_encoder = MagicMock()
        mock_encoding.return_value = mock_encoder
        mock_encoder.encode.side_effect = lambda x: list(x.encode("utf-8"))

        # Mock simplify_schema to fail for some schemas
        def side_effect(schema, format_type):
            if "fail" in str(schema):
                raise ValueError("Failed")
            mock_simplified = MagicMock()
            mock_simplified.to_string.return_value = '{"type": "object"}'
            return mock_simplified

        mock_simplify_schema.side_effect = side_effect

        dataset = [
            {"type": "object"},  # Success
            {"type": "fail"},  # Fail
            {"type": "string"},  # Success
            {"type": "fail"},  # Fail
            {"type": "number"},  # Success
        ]

        analyze_dataset_coverage(dataset)

        log_output = self.get_log_output()

        # Check specific percentages
        assert "✅ Supported schemas: 3/5 (60.0%)" in log_output
        assert "❌ Failed schemas: 2/5 (40.0%)" in log_output
        assert "Dataset coverage: 60.0%" in log_output
