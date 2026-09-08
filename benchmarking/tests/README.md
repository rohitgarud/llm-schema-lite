# Benchmarking Tests

This directory contains tests for the benchmarking and analysis tools in the `benchmarking` module.

## Test Files

- `test_analyze_schema_features.py` - Comprehensive tests for the `analyze_schema_features` function
- `test_analyze_dataset_coverage.py` - Comprehensive tests for the `analyze_dataset_coverage` function
- `test_feature_aggregation.py` - Comprehensive tests for feature aggregation and statistics functions

## Running Tests

To run the benchmarking tests:

```bash
# Run all benchmarking tests
python -m pytest benchmarking/tests/ -v

# Run specific test file
python -m pytest benchmarking/tests/test_analyze_schema_features.py -v

# Run from project root
cd /path/to/llm-schema-lite
python -m pytest benchmarking/tests/ -v
```
