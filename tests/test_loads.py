"""Comprehensive tests for the loads functionality."""

import pytest

from llm_schema_lite import loads
from llm_schema_lite.exceptions import ConversionError


class TestLoadsFunction:
    """Test suite for the loads function."""

    def test_json_basic_parsing(self):
        """Test basic JSON parsing."""
        data = loads('{"name": "John", "age": 30}', mode="json")
        assert data == {"name": "John", "age": 30}

    def test_json_nested_objects(self):
        """Test JSON parsing with nested objects."""
        json_str = '{"user": {"name": "John", "details": {"age": 30, "city": "NYC"}}}'
        data = loads(json_str, mode="json")
        expected = {"user": {"name": "John", "details": {"age": 30, "city": "NYC"}}}
        assert data == expected

    def test_json_arrays(self):
        """Test JSON parsing with arrays."""
        json_str = '{"items": ["apple", "banana", "cherry"], "counts": [1, 2, 3]}'
        data = loads(json_str, mode="json")
        expected = {"items": ["apple", "banana", "cherry"], "counts": [1, 2, 3]}
        assert data == expected

    def test_json_markdown_extraction(self):
        """Test JSON extraction from markdown code blocks."""
        markdown = '```json\n{"name": "Jane", "age": 25}\n```'
        data = loads(markdown, mode="json")
        assert data == {"name": "Jane", "age": 25}

    def test_json_markdown_with_extra_content(self):
        """Test JSON extraction from markdown with extra content."""
        markdown = 'Here is the data:\n```json\n{"name": "Bob", "age": 35}\n```\nEnd of data'
        data = loads(markdown, mode="json")
        assert data == {"name": "Bob", "age": 35}

    def test_json_repair_trailing_comma(self):
        """Test JSON repair for trailing comma."""
        malformed = '{"name": "Alice", "age": 28,}'
        data = loads(malformed, mode="json")
        assert data == {"name": "Alice", "age": 28}

    def test_json_repair_missing_quotes(self):
        """Test JSON repair for missing quotes."""
        malformed = '{name: "Charlie", age: 32}'
        data = loads(malformed, mode="json")
        assert data == {"name": "Charlie", "age": 32}

    def test_json_repair_disabled(self):
        """Test JSON parsing with repair disabled."""
        malformed = '{"name": "David", "age": 40,}'
        with pytest.raises(ConversionError):
            loads(malformed, mode="json", repair=False)

    def test_json_markdown_extraction_disabled(self):
        """Test JSON parsing with markdown extraction disabled."""
        markdown = '```json\n{"name": "Eve", "age": 22}\n```'
        with pytest.raises(ConversionError):
            loads(markdown, mode="json", extract_from_markdown=False, repair=False)

    def test_yaml_basic_parsing(self):
        """Test basic YAML parsing."""
        try:
            yaml_str = "name: John\nage: 30"
            data = loads(yaml_str, mode="yaml")
            assert data == {"name": "John", "age": 30}
        except ConversionError as e:
            if "PyYAML not available" in str(e):
                pytest.skip("PyYAML not available")
            else:
                raise

    def test_yaml_nested_objects(self):
        """Test YAML parsing with nested objects."""
        try:
            yaml_str = """user:
  name: John
  details:
    age: 30
    city: NYC"""
            data = loads(yaml_str, mode="yaml")
            expected = {"user": {"name": "John", "details": {"age": 30, "city": "NYC"}}}
            assert data == expected
        except ConversionError as e:
            if "PyYAML not available" in str(e):
                pytest.skip("PyYAML not available")
            else:
                raise

    def test_yaml_arrays(self):
        """Test YAML parsing with arrays."""
        try:
            yaml_str = """items:
  - apple
  - banana
  - cherry
counts: [1, 2, 3]"""
            data = loads(yaml_str, mode="yaml")
            expected = {"items": ["apple", "banana", "cherry"], "counts": [1, 2, 3]}
            assert data == expected
        except ConversionError as e:
            if "PyYAML not available" in str(e):
                pytest.skip("PyYAML not available")
            else:
                raise

    def test_yaml_markdown_extraction(self):
        """Test YAML extraction from markdown code blocks."""
        try:
            markdown = "```yaml\nname: Jane\nage: 25\n```"
            data = loads(markdown, mode="yaml")
            assert data == {"name": "Jane", "age": 25}
        except ConversionError as e:
            if "PyYAML not available" in str(e):
                pytest.skip("PyYAML not available")
            else:
                raise

    def test_yaml_yml_markdown_extraction(self):
        """Test YAML extraction from yml markdown code blocks."""
        try:
            markdown = "```yml\nname: Bob\nage: 35\n```"
            data = loads(markdown, mode="yaml")
            assert data == {"name": "Bob", "age": 35}
        except ConversionError as e:
            if "PyYAML not available" in str(e):
                pytest.skip("PyYAML not available")
            else:
                raise

    def test_yaml_repair_simple_key_value(self):
        """Test YAML repair for simple key-value pairs."""
        try:
            yaml_str = "name: Alice\nage: 28"
            data = loads(yaml_str, mode="yaml")
            assert data == {"name": "Alice", "age": 28}
        except ConversionError as e:
            if "PyYAML not available" in str(e):
                pytest.skip("PyYAML not available")
            else:
                raise

    def test_yaml_repair_disabled(self):
        """Test YAML parsing with repair disabled."""
        try:
            malformed = "name: Charlie\nage: 32\ninvalid: yaml: content"
            with pytest.raises(ConversionError):
                loads(malformed, mode="yaml", repair=False)
        except ConversionError as e:
            if "PyYAML not available" in str(e):
                pytest.skip("PyYAML not available")
            else:
                raise

    def test_invalid_mode(self):
        """Test loads with invalid mode."""
        with pytest.raises(ConversionError, match="Unsupported mode"):
            loads('{"name": "test"}', mode="invalid")

    def test_empty_text(self):
        """Test loads with empty text."""
        with pytest.raises(ConversionError, match="Empty or whitespace-only"):
            loads("", mode="json")

    def test_whitespace_only_text(self):
        """Test loads with whitespace-only text."""
        with pytest.raises(ConversionError, match="Empty or whitespace-only"):
            loads("   \n\t  ", mode="json")

    def test_json_invalid_syntax(self):
        """Test JSON parsing with invalid syntax that can't be repaired."""
        invalid_json = '{"name": "test" invalid syntax}'
        with pytest.raises(ConversionError):
            loads(invalid_json, mode="json", repair=False)

    def test_yaml_fallback_to_json(self):
        """Test YAML parsing fallback to JSON when YAML fails."""
        try:
            # This should fallback to JSON parsing
            data = loads('{"name": "fallback", "age": 42}', mode="yaml")
            assert data == {"name": "fallback", "age": 42}
        except ConversionError as e:
            if "PyYAML not available" in str(e):
                pytest.skip("PyYAML not available")
            else:
                raise

    def test_complex_json_with_special_characters(self):
        """Test JSON parsing with special characters."""
        json_str = (
            '{"message": "Hello, world! 🌍", "unicode": "测试", '
            '"special": "quotes \\" and \\n newlines"}'
        )
        data = loads(json_str, mode="json")
        expected = {
            "message": "Hello, world! 🌍",
            "unicode": "测试",
            "special": 'quotes " and \n newlines',
        }
        assert data == expected

    def test_yaml_with_comments(self):
        """Test YAML parsing with comments."""
        try:
            yaml_str = """# This is a comment
name: John  # Inline comment
age: 30
# Another comment
city: NYC"""
            data = loads(yaml_str, mode="yaml")
            expected = {"name": "John", "age": 30, "city": "NYC"}
            assert data == expected
        except ConversionError as e:
            if "PyYAML not available" in str(e):
                pytest.skip("PyYAML not available")
            else:
                raise

    def test_json_with_boolean_and_null(self):
        """Test JSON parsing with boolean and null values."""
        json_str = '{"active": true, "inactive": false, "data": null}'
        data = loads(json_str, mode="json")
        expected = {"active": True, "inactive": False, "data": None}
        assert data == expected

    def test_yaml_with_boolean_and_null(self):
        """Test YAML parsing with boolean and null values."""
        try:
            yaml_str = """active: true
inactive: false
data: null"""
            data = loads(yaml_str, mode="yaml")
            expected = {"active": True, "inactive": False, "data": None}
            assert data == expected
        except ConversionError as e:
            if "PyYAML not available" in str(e):
                pytest.skip("PyYAML not available")
            else:
                raise

    def test_json_numbers(self):
        """Test JSON parsing with various number types."""
        json_str = '{"int": 42, "float": 3.14, "negative": -10, "scientific": 1e5}'
        data = loads(json_str, mode="json")
        expected = {"int": 42, "float": 3.14, "negative": -10, "scientific": 100000.0}
        assert data == expected

    def test_yaml_numbers(self):
        """Test YAML parsing with various number types."""
        try:
            yaml_str = """int: 42
float: 3.14
negative: -10
scientific: 1e5"""
            data = loads(yaml_str, mode="yaml")
            # YAML treats 1e5 as a string, not a number, which is correct behavior
            expected = {"int": 42, "float": 3.14, "negative": -10, "scientific": "1e5"}
            assert data == expected
        except ConversionError as e:
            if "PyYAML not available" in str(e):
                pytest.skip("PyYAML not available")
            else:
                raise

    def test_markdown_extraction_edge_cases(self):
        """Test markdown extraction with edge cases."""
        # Multiple code blocks - should extract the first one
        markdown = '```json\n{"first": "block"}\n```\n```json\n{"second": "block"}\n```'
        data = loads(markdown, mode="json")
        assert data == {"first": "block"}

        # Code block with language specification
        markdown = '```json:pretty\n{"formatted": "json"}\n```'
        data = loads(markdown, mode="json")
        assert data == {"formatted": "json"}

    def test_repair_parameter_combinations(self):
        """Test different combinations of repair and extract_from_markdown parameters."""
        # Test with both repair and markdown extraction enabled (default)
        malformed_markdown = '```json\n{"name": "test",}\n```'
        data = loads(malformed_markdown, mode="json", repair=True, extract_from_markdown=True)
        assert data == {"name": "test"}

        # Test with repair enabled but markdown extraction disabled
        malformed_json = '{"name": "test",}'
        data = loads(malformed_json, mode="json", repair=True, extract_from_markdown=False)
        assert data == {"name": "test"}

        # Test with both disabled
        valid_json = '{"name": "test"}'
        data = loads(valid_json, mode="json", repair=False, extract_from_markdown=False)
        assert data == {"name": "test"}

    def test_json_object_extraction(self):
        """Test JSON object extraction from embedded text."""
        # Test JSON embedded in other text (disable markdown extraction to use JSON extraction)
        text_with_json = 'Here is the result: {"name": "Charlie", "age": 35} and some other text'
        data = loads(text_with_json, mode="json", extract_from_markdown=False)
        assert data == {"name": "Charlie", "age": 35}

        # Test multiple JSON objects - should extract the first one
        multiple_json = 'First: {"a": 1} Second: {"b": 2}'
        data = loads(multiple_json, mode="json", extract_from_markdown=False)
        assert data == {"a": 1}


class TestLoadsExtractFromMarkdown:
    """Test markdown extraction functionality."""

    def test_extract_json_from_markdown_codeblock(self):
        """Test extracting JSON from markdown code block."""
        markdown = """# API Response

Here's the response:

```json
{
  "id": 123,
  "name": "test"
}
```

That's the result."""
        data = loads(markdown, mode="json")
        assert data["id"] == 123
        assert data["name"] == "test"

    def test_extract_yaml_from_markdown_codeblock(self):
        """Test extracting YAML from markdown code block."""
        markdown = """# Configuration

```yaml
database:
  host: localhost
  port: 5432
```
"""
        data = loads(markdown, mode="yaml")
        assert data["database"]["host"] == "localhost"
        assert data["database"]["port"] == 5432

    def test_extract_from_yml_codeblock(self):
        """Test extracting from yml (alternative YAML extension) code block."""
        markdown = """```yml
server: production
region: us-west
```"""
        data = loads(markdown, mode="yaml")
        assert data["server"] == "production"
        assert data["region"] == "us-west"


class TestLoadsEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_loads_with_nested_json_objects(self):
        """Test loads with deeply nested JSON."""
        json_text = """{"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}"""
        data = loads(json_text)
        assert data["level1"]["level2"]["level3"]["level4"]["value"] == "deep"

    def test_loads_with_nested_yaml_objects(self):
        """Test loads with deeply nested YAML."""
        yaml_text = """level1:
  level2:
    level3:
      level4:
        value: deep"""
        data = loads(yaml_text, mode="yaml")
        assert data["level1"]["level2"]["level3"]["level4"]["value"] == "deep"

    def test_loads_json_with_arrays_and_nested_objects(self):
        """Test loads with complex JSON structure."""
        json_text = """{
    "users": [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"}
    ],
    "metadata": {
        "total": 2,
        "page": 1
    }
}"""
        data = loads(json_text)
        assert len(data["users"]) == 2
        assert data["users"][0]["name"] == "Alice"
        assert data["metadata"]["total"] == 2

    def test_loads_yaml_with_arrays_and_nested_objects(self):
        """Test loads with complex YAML structure."""
        yaml_text = """users:
  - id: 1
    name: Alice
  - id: 2
    name: Bob
metadata:
  total: 2
  page: 1"""
        data = loads(yaml_text, mode="yaml")
        assert len(data["users"]) == 2
        assert data["users"][0]["name"] == "Alice"
        assert data["metadata"]["total"] == 2

        # Test nested JSON objects
        nested_text = 'Result: {"user": {"name": "David", "details": {"age": 40}}}'
        data = loads(nested_text, mode="json", extract_from_markdown=False)
        assert data == {"user": {"name": "David", "details": {"age": 40}}}
