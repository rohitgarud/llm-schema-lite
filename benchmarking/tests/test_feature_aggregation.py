#!/usr/bin/env python3
"""
Comprehensive tests for the feature aggregation functions.

This test suite covers feature counting, statistics generation,
and aggregation across datasets.
"""

import json
import logging
from io import StringIO

import pytest

from ..base import analyze_schema_feature_coverage, get_feature_statistics


class TestFeatureAggregation:
    """Test suite for feature aggregation functions."""

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

    def test_analyze_schema_feature_coverage_basic(self):
        """Test basic feature coverage analysis."""
        dataset = [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "string", "enum": ["a", "b", "c"]},
            {"type": "array", "items": {"type": "number"}},
        ]

        analyze_schema_feature_coverage(dataset)

        log_output = self.get_log_output()

        # Check that analysis was logged
        assert "📊 Schema Feature Coverage Analysis" in log_output
        assert "Total schemas processed: 3/3" in log_output
        assert "Success rate: 100.0%" in log_output
        assert "🔍 Feature Usage Statistics:" in log_output

        # Check that features were counted
        assert "type" in log_output
        assert "properties" in log_output
        assert "enum" in log_output
        assert "items" in log_output

    def test_analyze_schema_feature_coverage_with_failures(self):
        """Test feature coverage analysis with some failing schemas."""
        dataset = [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"invalid": "schema"},  # This might fail
            {"type": "string", "enum": ["a", "b"]},
        ]

        analyze_schema_feature_coverage(dataset)

        log_output = self.get_log_output()

        # Should still process successfully
        assert "📊 Schema Feature Coverage Analysis" in log_output
        assert "Total schemas processed:" in log_output

    def test_analyze_schema_feature_coverage_empty_dataset(self):
        """Test feature coverage analysis with empty dataset."""
        dataset = []

        analyze_schema_feature_coverage(dataset)

        log_output = self.get_log_output()

        assert "Total schemas processed: 0/0" in log_output
        assert "No features found in any schemas." in log_output

    def test_get_feature_statistics_basic(self):
        """Test basic feature statistics generation."""
        dataset = [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "string", "enum": ["a", "b", "c"]},
            {"type": "array", "items": {"type": "number"}},
        ]

        stats = get_feature_statistics(dataset)

        # Check basic stats
        assert stats["total_schemas"] == 3
        assert stats["processed_schemas"] == 3
        assert stats["success_rate"] == 100.0
        assert stats["total_unique_features"] > 0

        # Check feature counts
        assert "type" in stats["feature_counts"]
        assert "properties" in stats["feature_counts"]
        assert "enum" in stats["feature_counts"]
        assert "items" in stats["feature_counts"]

        # Check percentages
        assert "type" in stats["feature_percentages"]
        assert stats["feature_percentages"]["type"] == 100.0  # All schemas have type

        # Check sorted lists
        assert len(stats["features_by_frequency"]) > 0
        assert len(stats["features_by_name"]) > 0

    def test_get_feature_statistics_with_failures(self):
        """Test feature statistics with some failing schemas."""
        dataset = [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"invalid": "schema"},  # This might fail
            {"type": "string", "enum": ["a", "b"]},
        ]

        stats = get_feature_statistics(dataset)

        # Should handle failures gracefully
        assert stats["total_schemas"] == 3
        assert stats["processed_schemas"] >= 2  # At least 2 should succeed
        assert stats["success_rate"] >= 66.0

    def test_get_feature_statistics_empty_dataset(self):
        """Test feature statistics with empty dataset."""
        dataset = []

        stats = get_feature_statistics(dataset)

        assert stats["total_schemas"] == 0
        assert stats["processed_schemas"] == 0
        assert stats["success_rate"] == 0
        assert stats["total_unique_features"] == 0
        assert len(stats["feature_counts"]) == 0
        assert len(stats["feature_percentages"]) == 0

    def test_feature_frequency_sorting(self):
        """Test that features are sorted by frequency correctly."""
        dataset = [
            {"type": "object"},  # type appears 4 times (including nested)
            {"type": "string"},  # type appears 4 times (including nested)
            {"type": "array"},  # type appears 4 times (including nested)
            {"properties": {"name": {"type": "string"}}},  # properties: 1, type: 4
        ]

        stats = get_feature_statistics(dataset)

        # Check that features are sorted by frequency (descending)
        features_by_freq = stats["features_by_frequency"]
        assert len(features_by_freq) >= 2

        # Most frequent feature should be first
        most_frequent = features_by_freq[0]
        assert most_frequent[0] == "type"  # Should be "type"
        assert most_frequent[1] == 4  # Should appear 4 times (including nested)

    def test_schema_feature_mapping(self):
        """Test that schema-feature mapping is correct."""
        dataset = [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "string", "enum": ["a", "b"]},
            {"type": "array", "items": {"type": "number"}},
        ]

        stats = get_feature_statistics(dataset)

        # Check that each schema's features are tracked
        assert len(stats["schema_feature_map"]) == 3

        # Check specific schemas
        schema_0_features = stats["schema_feature_map"][0]
        assert "type" in schema_0_features
        assert "properties" in schema_0_features

        schema_1_features = stats["schema_feature_map"][1]
        assert "type" in schema_1_features
        assert "enum" in schema_1_features

        schema_2_features = stats["schema_feature_map"][2]
        assert "type" in schema_2_features
        assert "items" in schema_2_features

    def test_feature_percentages_calculation(self):
        """Test that feature percentages are calculated correctly."""
        dataset = [
            {"type": "object"},  # type: 100%
            {"type": "string"},  # type: 100%
            {"properties": {"name": {"type": "string"}}},  # properties: 33.3%
        ]

        stats = get_feature_statistics(dataset)

        # Check percentages
        assert stats["feature_percentages"]["type"] == 100.0
        assert stats["feature_percentages"]["properties"] == pytest.approx(33.33, rel=1e-2)

    def test_complex_schema_features(self):
        """Test feature aggregation with complex schemas."""
        dataset = [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "age": {"type": "number", "minimum": 0},
                },
                "required": ["name"],
                "if": {"properties": {"age": {"type": "number"}}},
                "then": {"properties": {"age": {"minimum": 0}}},
            },
            {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 10,
                "uniqueItems": True,
            },
        ]

        stats = get_feature_statistics(dataset)

        # Check that complex features are detected
        expected_features = {
            "type",
            "properties",
            "required",
            "minLength",
            "minimum",
            "if",
            "then",
            "items",
            "minItems",
            "maxItems",
            "uniqueItems",
        }

        detected_features = set(stats["feature_counts"].keys())
        assert expected_features.issubset(detected_features)

    def test_feature_statistics_metadata(self):
        """Test that all metadata fields are present in statistics."""
        dataset = [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "string", "enum": ["a", "b"]},
        ]

        stats = get_feature_statistics(dataset)

        # Check all required fields are present
        required_fields = [
            "total_schemas",
            "processed_schemas",
            "success_rate",
            "feature_counts",
            "feature_percentages",
            "schema_feature_map",
            "unique_features",
            "total_unique_features",
            "features_by_frequency",
            "features_by_name",
        ]

        for field in required_fields:
            assert field in stats

    def test_feature_statistics_data_types(self):
        """Test that statistics contain correct data types."""
        dataset = [{"type": "object", "properties": {"name": {"type": "string"}}}]

        stats = get_feature_statistics(dataset)

        # Check data types
        assert isinstance(stats["total_schemas"], int)
        assert isinstance(stats["processed_schemas"], int)
        assert isinstance(stats["success_rate"], float)
        assert isinstance(stats["feature_counts"], dict)
        assert isinstance(stats["feature_percentages"], dict)
        assert isinstance(stats["schema_feature_map"], dict)
        assert isinstance(stats["unique_features"], set)
        assert isinstance(stats["total_unique_features"], int)
        assert isinstance(stats["features_by_frequency"], list)
        assert isinstance(stats["features_by_name"], list)

    def test_large_dataset_performance(self):
        """Test performance with a larger dataset."""
        # Create a larger dataset
        dataset = [
            {"type": "object", "properties": {f"prop{i}": {"type": "string"}}} for i in range(50)
        ]

        stats = get_feature_statistics(dataset)

        # Should handle large dataset efficiently
        assert stats["total_schemas"] == 50
        assert stats["processed_schemas"] == 50
        assert stats["success_rate"] == 100.0

    def test_feature_statistics_json_serializable(self):
        """Test that statistics can be JSON serialized."""
        dataset = [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "string", "enum": ["a", "b"]},
        ]

        stats = get_feature_statistics(dataset)

        # Should be JSON serializable (except sets)
        json_stats = {
            "total_schemas": stats["total_schemas"],
            "processed_schemas": stats["processed_schemas"],
            "success_rate": stats["success_rate"],
            "feature_counts": stats["feature_counts"],
            "feature_percentages": stats["feature_percentages"],
            "schema_feature_map": stats["schema_feature_map"],
            "unique_features": list(stats["unique_features"]),  # Convert set to list
            "total_unique_features": stats["total_unique_features"],
            "features_by_frequency": stats["features_by_frequency"],
            "features_by_name": stats["features_by_name"],
        }

        # Should be able to serialize to JSON
        json_str = json.dumps(json_stats)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
