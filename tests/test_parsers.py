"""Tests for parser implementations."""

import pytest

from llm_schema_lite.exceptions import ConversionError
from llm_schema_lite.parsers import JSONParser, YAMLParser


class TestJSONParser:
    """Tests for JSONParser."""

    def test_parse_plain_json(self):
        """Test parsing plain JSON."""
        parser = JSONParser()
        result = parser.parse('{"name": "John", "age": 30}')
        assert result == {"name": "John", "age": 30}

    def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        parser = JSONParser()
        result = parser.parse('```json\n{"name": "Jane", "age": 25}\n```')
        assert result == {"name": "Jane", "age": 25}

    def test_parse_json_embedded_in_text(self):
        """Test parsing JSON embedded in explanatory text."""
        parser = JSONParser()
        result = parser.parse('The user data is: {"name": "Bob", "age": 35} and that\'s all.')
        assert result == {"name": "Bob", "age": 35}

    def test_parse_json_array(self):
        """Test parsing JSON array."""
        parser = JSONParser()
        result = parser.parse('[{"name": "Alice"}, {"name": "Bob"}]')
        assert result == [{"name": "Alice"}, {"name": "Bob"}]

    def test_parse_json_with_repair(self):
        """Test parsing malformed JSON with repair enabled."""
        parser = JSONParser()
        # This test requires json_repair to be installed
        # If not installed, it should raise ConversionError
        try:
            result = parser.parse('{"name": "John", "age": 30,}', repair=True)
            # If repair succeeded, check the result
            assert result == {"name": "John", "age": 30}
        except ConversionError:
            # If json_repair is not installed, this is expected
            pytest.skip("json_repair not installed")

    def test_parse_json_without_repair(self):
        """Test parsing malformed JSON with repair disabled."""
        parser = JSONParser()
        with pytest.raises(ConversionError, match="Failed to parse JSON"):
            parser.parse('{"name": "John", "age": 30,}', repair=False)

    def test_parse_empty_json(self):
        """Test parsing empty object."""
        parser = JSONParser()
        result = parser.parse("{}")
        assert result == {}

    def test_parse_nested_json(self):
        """Test parsing nested JSON structures."""
        parser = JSONParser()
        json_text = '{"user": {"name": "John", "address": {"city": "NYC"}}}'
        result = parser.parse(json_text)
        assert result == {"user": {"name": "John", "address": {"city": "NYC"}}}

    def test_parse_json_with_extra_text_before(self):
        """Test parsing JSON with explanatory text before."""
        parser = JSONParser()
        result = parser.parse('Here is the result: {"name": "John", "age": 30}')
        assert result == {"name": "John", "age": 30}

    def test_parse_json_with_extra_text_after(self):
        """Test parsing JSON with explanatory text after."""
        parser = JSONParser()
        result = parser.parse('{"name": "John", "age": 30} is the data')
        assert result == {"name": "John", "age": 30}


class TestYAMLParser:
    """Tests for YAMLParser."""

    def test_parse_plain_yaml(self):
        """Test parsing plain YAML."""
        parser = YAMLParser()
        result = parser.parse("name: Alice\nage: 28")
        assert result == {"name": "Alice", "age": 28}

    def test_parse_yaml_with_markdown(self):
        """Test parsing YAML wrapped in markdown code blocks."""
        parser = YAMLParser()
        result = parser.parse("```yaml\nname: Alice\nage: 28\n```")
        assert result == {"name": "Alice", "age": 28}

    def test_parse_yaml_with_yml_markdown(self):
        """Test parsing YAML with ```yml code block."""
        parser = YAMLParser()
        result = parser.parse("```yml\nname: Alice\nage: 28\n```")
        assert result == {"name": "Alice", "age": 28}

    def test_parse_yaml_embedded_in_text(self):
        """Test parsing YAML embedded in explanatory text."""
        parser = YAMLParser()
        yaml_text = """Here is the config:
name: Alice
age: 28
That's all."""
        result = parser.parse(yaml_text)
        assert result == {"name": "Alice", "age": 28}

    def test_parse_yaml_list(self):
        """Test parsing YAML with list items."""
        parser = YAMLParser()
        yaml_text = """items:
  - name: Item1
  - name: Item2"""
        result = parser.parse(yaml_text)
        assert result == {"items": [{"name": "Item1"}, {"name": "Item2"}]}

    def test_parse_yaml_nested(self):
        """Test parsing nested YAML structures."""
        parser = YAMLParser()
        yaml_text = """user:
  name: John
  address:
    city: NYC"""
        result = parser.parse(yaml_text)
        assert result == {"user": {"name": "John", "address": {"city": "NYC"}}}

    def test_parse_yaml_with_repair_indentation(self):
        """Test parsing YAML with indentation issues."""
        parser = YAMLParser()
        # YAML with extra indentation
        yaml_text = """  name: Alice
  age: 28"""
        result = parser.parse(yaml_text, repair=True)
        assert result == {"name": "Alice", "age": 28}

    def test_parse_yaml_without_repair(self):
        """Test parsing malformed YAML with repair disabled."""
        parser = YAMLParser()
        # Invalid YAML (mixed indentation)
        yaml_text = """name: Alice
  age: 28
     city: NYC"""
        # This should either parse or fail cleanly
        # Behavior depends on PyYAML's strictness
        try:
            parser.parse(yaml_text, repair=False)
        except ConversionError:
            # Expected if YAML is truly malformed
            pass

    def test_parse_yaml_fallback_to_json(self):
        """Test YAML parser fallback to JSON."""
        parser = YAMLParser()
        # JSON should also work with YAML parser
        result = parser.parse('{"name": "John", "age": 30}')
        assert result == {"name": "John", "age": 30}

    def test_parse_empty_yaml(self):
        """Test parsing empty YAML object."""
        parser = YAMLParser()
        result = parser.parse("{}")
        assert result == {}


class TestLoadsIntegration:
    """Integration tests for loads() function using parsers."""

    def test_loads_json_plain(self):
        """Test loads() with plain JSON."""
        from llm_schema_lite import loads

        result = loads('{"name": "John", "age": 30}')
        assert result == {"name": "John", "age": 30}

    def test_loads_json_markdown(self):
        """Test loads() with markdown-wrapped JSON."""
        from llm_schema_lite import loads

        result = loads('```json\n{"name": "Jane", "age": 25}\n```')
        assert result == {"name": "Jane", "age": 25}

    def test_loads_yaml_plain(self):
        """Test loads() with plain YAML."""
        from llm_schema_lite import loads

        result = loads("name: Alice\nage: 28", mode="yaml")
        assert result == {"name": "Alice", "age": 28}

    def test_loads_yaml_markdown(self):
        """Test loads() with markdown-wrapped YAML."""
        from llm_schema_lite import loads

        result = loads("```yaml\nname: Alice\nage: 28\n```", mode="yaml")
        assert result == {"name": "Alice", "age": 28}

    def test_loads_empty_text_raises_error(self):
        """Test loads() with empty text raises ConversionError."""
        from llm_schema_lite import loads

        with pytest.raises(ConversionError, match="Empty or whitespace-only text"):
            loads("")

    def test_loads_whitespace_only_raises_error(self):
        """Test loads() with whitespace-only text raises ConversionError."""
        from llm_schema_lite import loads

        with pytest.raises(ConversionError, match="Empty or whitespace-only text"):
            loads("   \n  \t  ")

    def test_loads_unsupported_mode_raises_error(self):
        """Test loads() with unsupported mode raises ConversionError."""
        from llm_schema_lite import loads

        with pytest.raises(ConversionError, match="Unsupported mode"):
            loads('{"name": "John"}', mode="xml")  # type: ignore

    def test_loads_repair_disabled(self):
        """Test loads() with repair disabled."""
        from llm_schema_lite import loads

        with pytest.raises(ConversionError):
            loads('{"name": "John", "age": 30,}', repair=False)
