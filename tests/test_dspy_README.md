# DSPy Integration Tests

This directory contains tests for the DSPy integration with llm-schema-lite.

## Test Files

- `test_dspy_adapter.py` - Unit tests for the StructuredOutputAdapter
- `test_dspy_integration.py` - Integration tests with DSPy workflows

## Requirements

To run the DSPy tests, you need to install the optional DSPy dependencies:

```bash
# Install DSPy dependencies
pip install -e ".[dspy]"

# Or using uv
uv pip install -e ".[dspy]"
```

## Running Tests

### Run all DSPy tests
```bash
pytest tests/test_dspy_adapter.py tests/test_dspy_integration.py -v
```

### Run only unit tests
```bash
pytest tests/test_dspy_adapter.py -v
```

### Run only integration tests
```bash
pytest tests/test_dspy_integration.py -v
```

### Run with coverage
```bash
pytest tests/test_dspy_adapter.py tests/test_dspy_integration.py --cov=dspy_integration --cov-report=html
```

## Test Structure

### Unit Tests (`test_dspy_adapter.py`)

Tests individual components of the StructuredOutputAdapter:

- **Adapter Initialization**: Test different configuration options
- **Field Translation**: Test schema generation for different field types
- **Format Field Structure**: Test prompt formatting
- **Output Requirements**: Test output format specifications
- **Field Value Formatting**: Test input/output formatting
- **JSON Parsing**: Test JSON parsing with various edge cases
- **YAML Parsing**: Test YAML parsing and fallback mechanisms
- **Mode-Specific Parsing**: Test routing to correct parsers
- **Complex Scenarios**: Test with nested Pydantic models
- **Error Handling**: Test error recovery and edge cases
- **Performance**: Test token efficiency

### Integration Tests (`test_dspy_integration.py`)

Tests the adapter with actual DSPy workflows:

- **Basic Integration**: Test with simple Predict modules
- **Chain of Thought**: Test with ChainOfThought modules
- **Complex Models**: Test with nested Pydantic models
- **Demos and Few-Shot**: Test with demonstrations
- **Error Recovery**: Test fallback mechanisms
- **Async Support**: Test async/await functionality
- **Token Efficiency**: Compare token usage across modes
- **Real-World Scenarios**: Test complete workflows

## Test Coverage

The tests cover:

- ✅ All three output modes (JSON, JSONish, YAML)
- ✅ Input schema simplification
- ✅ Output parsing and validation
- ✅ Error handling and fallbacks
- ✅ Integration with DSPy modules
- ✅ Token efficiency verification
- ✅ Edge cases and malformed inputs

## Notes

- Tests requiring PyYAML will be skipped if it's not installed
- Tests requiring DSPy will be skipped if it's not installed
- Mock LMs are used for integration tests to avoid API calls
- Some tests verify token efficiency by comparing prompt lengths

## Troubleshooting

### DSPy Not Found

If you see "DSPy not installed" errors:
```bash
pip install "dspy>=3.0.3"
```

### PyYAML Not Found

If YAML tests are skipped:
```bash
pip install pyyaml
```

### Import Errors

If you see import errors for `dspy_integration`:
```bash
# Make sure you're in the project root
cd /path/to/llm-schema-lite

# Install in editable mode
pip install -e .
```

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Use descriptive test names
3. Add docstrings explaining what's being tested
4. Mock external dependencies (LMs, APIs)
5. Test both success and failure cases
6. Update this README if adding new test categories
