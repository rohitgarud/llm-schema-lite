"""Performance and memory usage tests for llm-schema-lite."""

import os
import time
from typing import Any

import psutil
import pytest

from llm_schema_lite import simplify_schema


class TestPerformance:
    """Performance tests for schema processing."""

    def create_large_schema(self, size: int = 100) -> dict[str, Any]:
        """Create a large schema for performance testing."""
        properties = {}
        for i in range(size):
            properties[f"field_{i}"] = {
                "type": "string",
                "description": f"Field {i} description",
                "minLength": 1,
                "maxLength": 100,
                "pattern": f"^field_{i}_pattern$",
            }

        return {"type": "object", "properties": properties, "required": list(properties.keys())}

    def create_deeply_nested_schema(self, depth: int = 10) -> dict[str, Any]:
        """Create a deeply nested schema for performance testing."""
        schema = {"type": "object", "properties": {}}
        current = schema["properties"]

        for i in range(depth):
            current[f"level_{i}"] = {"type": "object", "properties": {}}
            current = current[f"level_{i}"]["properties"]

        # Add a leaf property
        current["leaf"] = {"type": "string"}

        return schema

    def create_complex_union_schema(self, union_count: int = 50) -> dict[str, Any]:
        """Create a schema with many union types for performance testing."""
        anyof_options = []
        for i in range(union_count):
            anyof_options.append({"type": "string", "enum": [f"option_{i}_{j}" for j in range(5)]})

        return {"type": "object", "properties": {"complex_union": {"anyOf": anyof_options}}}

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_large_schema_performance(self, format_type):
        """Test performance with large schemas."""
        large_schema = self.create_large_schema(50)  # 50 fields

        start_time = time.time()
        schema = simplify_schema(large_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()
        end_time = time.time()

        processing_time = end_time - start_time

        # Should process within reasonable time (less than 1 second)
        assert processing_time < 1.0, f"Processing took too long: {processing_time:.3f}s"

        # Should produce substantial output
        assert len(output) > 1000, f"Output too short: {len(output)} characters"

        # Should include all fields
        for i in range(50):
            assert f"field_{i}" in output, f"Missing field_{i} in output"

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_deeply_nested_schema_performance(self, format_type):
        """Test performance with deeply nested schemas."""
        nested_schema = self.create_deeply_nested_schema(10)  # 10 levels deep

        start_time = time.time()
        schema = simplify_schema(nested_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()
        end_time = time.time()

        processing_time = end_time - start_time

        # Should process within reasonable time
        assert processing_time < 1.0, f"Processing took too long: {processing_time:.3f}s"

        # Should include the top level (formatter simplifies nested objects to 'object')
        assert "level_0" in output, "Missing level_0 in output"
        # The formatter simplifies nested objects, so we don't expect all levels to be expanded
        # This is the intended behavior to reduce token usage

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_complex_union_performance(self, format_type):
        """Test performance with complex union types."""
        union_schema = self.create_complex_union_schema(20)  # 20 union options

        start_time = time.time()
        schema = simplify_schema(union_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()
        end_time = time.time()

        processing_time = end_time - start_time

        # Should process within reasonable time
        assert processing_time < 1.0, f"Processing took too long: {processing_time:.3f}s"

        # Should include the union field
        assert "complex_union" in output

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_repeated_processing_performance(self, format_type):
        """Test performance with repeated processing of the same schema."""
        test_schema = self.create_large_schema(20)

        # Process the same schema multiple times
        times = []
        for _ in range(10):
            start_time = time.time()
            schema = simplify_schema(test_schema, format_type=format_type, include_metadata=True)
            schema.to_string()  # Process the schema
            end_time = time.time()
            times.append(end_time - start_time)

        # Average processing time should be reasonable
        avg_time = sum(times) / len(times)
        assert avg_time < 0.5, f"Average processing time too high: {avg_time:.3f}s"

        # Processing time should be consistent (not vary too much)
        max_time = max(times)
        min_time = min(times)
        assert (
            max_time - min_time < 0.2
        ), f"Processing time too inconsistent: {max_time:.3f}s - {min_time:.3f}s"

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_memory_usage(self, format_type):
        """Test memory usage during schema processing."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process a large schema
        large_schema = self.create_large_schema(100)  # 100 fields
        schema = simplify_schema(large_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50, f"Memory usage too high: {memory_increase:.1f}MB increase"

        # Should produce substantial output
        assert len(output) > 2000, f"Output too short: {len(output)} characters"

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_memory_cleanup(self, format_type):
        """Test that memory is properly cleaned up after processing."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process multiple large schemas
        for _i in range(5):
            large_schema = self.create_large_schema(50)
            schema = simplify_schema(large_schema, format_type=format_type, include_metadata=True)
            output = schema.to_string()
            del schema  # Explicit cleanup
            del output

        # Force garbage collection
        import gc

        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory should be cleaned up reasonably well
        assert (
            memory_increase < 20
        ), f"Memory not cleaned up properly: {memory_increase:.1f}MB increase"

    def test_token_reduction_efficiency(self):
        """Test that token reduction is efficient."""
        # Create a schema with lots of metadata
        verbose_schema = {
            "type": "object",
            "properties": {
                "field1": {
                    "type": "string",
                    "description": "This is a very long description that contains lots of words",
                    "minLength": 1,
                    "maxLength": 100,
                    "pattern": "^[a-zA-Z0-9_]+$",
                    "format": "string",
                    "title": "Field 1 Title",
                    "examples": ["example1", "example2", "example3"],
                },
                "field2": {
                    "type": "integer",
                    "description": "Another very long description with lots of unnecessary words",
                    "minimum": 0,
                    "maximum": 1000,
                    "multipleOf": 5,
                    "exclusiveMinimum": False,
                    "exclusiveMaximum": False,
                },
            },
        }

        # Process with and without metadata
        schema_with_metadata = simplify_schema(
            verbose_schema, format_type="jsonish", include_metadata=True
        )
        output_with_metadata = schema_with_metadata.to_string()

        schema_without_metadata = simplify_schema(
            verbose_schema, format_type="jsonish", include_metadata=False
        )
        output_without_metadata = schema_without_metadata.to_string()

        # Output without metadata should be significantly shorter
        reduction_ratio = len(output_without_metadata) / len(output_with_metadata)
        assert reduction_ratio < 0.5, f"Token reduction not effective: {reduction_ratio:.2f} ratio"

        # Both should include the field names
        assert "field1" in output_with_metadata
        assert "field1" in output_without_metadata
        assert "field2" in output_with_metadata
        assert "field2" in output_without_metadata


class TestStressTests:
    """Stress tests for extreme scenarios."""

    def test_extremely_large_schema(self):
        """Test with extremely large schema."""
        # Create a very large schema (500 fields)
        properties = {}
        for i in range(500):
            properties[f"field_{i}"] = {
                "type": "string",
                "description": f"Field {i} with a long description that contains many words",
                "minLength": 1,
                "maxLength": 100,
            }

        large_schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
        }

        start_time = time.time()
        schema = simplify_schema(large_schema, format_type="jsonish", include_metadata=True)
        output = schema.to_string()
        end_time = time.time()

        processing_time = end_time - start_time

        # Should still process within reasonable time (less than 2 seconds)
        assert processing_time < 2.0, f"Processing took too long: {processing_time:.3f}s"

        # Should produce very large output
        assert len(output) > 10000, f"Output too short: {len(output)} characters"

        # Should include many fields
        field_count = sum(1 for i in range(500) if f"field_{i}" in output)
        assert field_count > 400, f"Too few fields included: {field_count}/500"

    def test_extremely_deep_nesting(self):
        """Test with extremely deep nesting."""
        # Create a very deeply nested schema (50 levels)
        schema = {"type": "object", "properties": {}}
        current = schema["properties"]

        for i in range(50):
            current[f"level_{i}"] = {"type": "object", "properties": {}}
            current = current[f"level_{i}"]["properties"]

        # Add a leaf property
        current["leaf"] = {"type": "string"}

        start_time = time.time()
        processed_schema = simplify_schema(schema, format_type="jsonish", include_metadata=True)
        output = processed_schema.to_string()
        end_time = time.time()

        processing_time = end_time - start_time

        # Should process within reasonable time
        assert processing_time < 1.0, f"Processing took too long: {processing_time:.3f}s"

        # Should include the top level (formatter simplifies nested objects to 'object')
        assert "level_0" in output, "Missing level_0 in output"
        # The formatter simplifies deeply nested objects, so we don't expect the leaf to be visible
        # This is the intended behavior to reduce token usage

    def test_many_union_options(self):
        """Test with many union options."""
        # Create a schema with many union options (100 options)
        anyof_options = []
        for i in range(100):
            anyof_options.append({"type": "string", "enum": [f"option_{i}_{j}" for j in range(3)]})

        union_schema = {"type": "object", "properties": {"massive_union": {"anyOf": anyof_options}}}

        start_time = time.time()
        schema = simplify_schema(union_schema, format_type="jsonish", include_metadata=True)
        output = schema.to_string()
        end_time = time.time()

        processing_time = end_time - start_time

        # Should process within reasonable time
        assert processing_time < 1.0, f"Processing took too long: {processing_time:.3f}s"

        # Should include the union field
        assert "massive_union" in output

    def test_concurrent_processing_simulation(self):
        """Simulate concurrent processing by processing multiple schemas rapidly."""
        schemas = []
        for _i in range(10):
            schema = {
                "type": "object",
                "properties": {
                    f"field_{j}": {"type": "string", "description": f"Field {j} description"}
                    for j in range(20)
                },
            }
            schemas.append(schema)

        start_time = time.time()
        outputs = []
        for schema in schemas:
            processed = simplify_schema(schema, format_type="jsonish", include_metadata=True)
            outputs.append(processed.to_string())
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / len(schemas)

        # Average processing time should be reasonable
        assert avg_time < 0.1, f"Average processing time too high: {avg_time:.3f}s"

        # All outputs should be generated
        assert len(outputs) == 10
        for output in outputs:
            assert len(output) > 100, "Output too short"


class TestBenchmarking:
    """Benchmarking tests for performance comparison."""

    def benchmark_formatter_performance(self):
        """Benchmark performance across all formatters."""
        test_schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {
                    "type": "string",
                    "description": f"Field {i} description",
                    "minLength": 1,
                    "maxLength": 100,
                }
                for i in range(50)
            },
        }

        results = {}
        for format_type in ["jsonish", "typescript", "yaml"]:
            start_time = time.time()
            schema = simplify_schema(test_schema, format_type=format_type, include_metadata=True)
            output = schema.to_string()
            end_time = time.time()

            results[format_type] = {"time": end_time - start_time, "output_size": len(output)}

        # All formatters should perform reasonably
        for format_type, result in results.items():
            assert result["time"] < 0.5, f"{format_type} too slow: {result['time']:.3f}s"
            assert (
                result["output_size"] > 1000
            ), f"{format_type} output too small: {result['output_size']} chars"

        return results

    def benchmark_metadata_impact(self):
        """Benchmark the impact of metadata on performance."""
        test_schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {
                    "type": "string",
                    "description": f"Field {i} with a long description",
                    "minLength": 1,
                    "maxLength": 100,
                    "pattern": f"^field_{i}_pattern$",
                }
                for i in range(30)
            },
        }

        # Test with metadata
        start_time = time.time()
        schema_with_metadata = simplify_schema(
            test_schema, format_type="jsonish", include_metadata=True
        )
        output_with_metadata = schema_with_metadata.to_string()
        time_with_metadata = time.time() - start_time

        # Test without metadata
        start_time = time.time()
        schema_without_metadata = simplify_schema(
            test_schema, format_type="jsonish", include_metadata=False
        )
        output_without_metadata = schema_without_metadata.to_string()
        time_without_metadata = time.time() - start_time

        # Processing time should be similar (metadata doesn't significantly impact performance)
        time_ratio = time_with_metadata / time_without_metadata
        assert time_ratio < 2.0, f"Metadata processing too slow: {time_ratio:.2f}x slower"

        # Output size should be significantly different
        size_ratio = len(output_with_metadata) / len(output_without_metadata)
        assert size_ratio > 2.0, f"Metadata doesn't increase output size enough: {size_ratio:.2f}x"

        return {
            "time_with_metadata": time_with_metadata,
            "time_without_metadata": time_without_metadata,
            "size_with_metadata": len(output_with_metadata),
            "size_without_metadata": len(output_without_metadata),
        }
