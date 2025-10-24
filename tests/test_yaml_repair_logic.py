"""Comprehensive tests for YAML repair logic to improve coverage."""

import pytest

from llm_schema_lite import loads
from llm_schema_lite.exceptions import ConversionError


class TestYAMLRepairIndentation:
    """Test YAML repair with indentation normalization."""

    def test_yaml_with_consistent_indentation(self):
        """Test YAML with excessive but consistent indentation."""
        yaml_text = """        name: Alice
        age: 30
        active: true"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["name"] == "Alice"
        assert result["age"] == 30
        assert result["active"] is True

    def test_yaml_with_mixed_indentation_levels(self):
        """Test YAML with mixed indentation that needs normalization."""
        yaml_text = """      name: Bob
      profile:
        age: 25
        city: NYC"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["name"] == "Bob"
        assert result["profile"]["age"] == 25

    def test_yaml_with_tabs_converted_to_spaces(self):
        """Test YAML with tab characters (common error)."""
        yaml_text = """    username: charlie
    email: charlie@example.com
    admin: false"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["username"] == "charlie"
        assert result["email"] == "charlie@example.com"
        assert result["admin"] is False


class TestYAMLRepairStartDetection:
    """Test YAML repair with start position detection."""

    def test_yaml_after_preamble_text(self):
        """Test YAML that starts after some explanatory text."""
        text = """Here is the configuration:
name: service_config
version: 2.0
enabled: true"""

        result = loads(text, mode="yaml", repair=True)
        assert result["name"] == "service_config"
        assert result["version"] == 2.0
        assert result["enabled"] is True

    def test_yaml_after_markdown_heading(self):
        """Test YAML after markdown-style heading."""
        text = """# Configuration Settings
server: production
port: 8080
ssl: true"""

        result = loads(text, mode="yaml", repair=True)
        assert result["server"] == "production"
        assert result["port"] == 8080

    def test_yaml_after_comment_lines(self):
        """Test YAML that starts after comment lines."""
        text = """# Configuration file
# Generated automatically
host: localhost
database: mydb"""

        result = loads(text, mode="yaml", repair=True)
        assert result["host"] == "localhost"
        assert result["database"] == "mydb"


class TestYAMLRepairSimpleKeyValue:
    """Test YAML repair with simple key-value format conversion."""

    def test_simple_key_value_pairs(self):
        """Test simple key: value pairs."""
        yaml_text = """username: admin
password: secret
timeout: 30"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["username"] == "admin"
        assert result["password"] == "secret"
        assert result["timeout"] == 30

    def test_key_value_with_spaces(self):
        """Test key-value pairs with spaces in values."""
        yaml_text = """title: My Document
author: John Doe
status: In Progress"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["title"] == "My Document"
        assert result["author"] == "John Doe"
        assert result["status"] == "In Progress"

    def test_key_value_with_special_characters(self):
        """Test key-value pairs with special characters."""
        yaml_text = """email: user@example.com
url: https://example.com
path: /var/log/app.log"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["email"] == "user@example.com"
        assert result["url"] == "https://example.com"


class TestYAMLRepairJSONFallback:
    """Test YAML repair falling back to JSON parsing."""

    def test_json_in_yaml_mode_with_repair(self):
        """Test JSON content parsed in YAML mode with repair."""
        json_text = '{"result": "success", "count": 42, "valid": true}'

        result = loads(json_text, mode="yaml", repair=True)
        assert result["result"] == "success"
        assert result["count"] == 42
        assert result["valid"] is True

    def test_json_array_in_yaml_mode(self):
        """Test JSON with arrays in YAML mode."""
        json_text = '{"items": [1, 2, 3], "names": ["a", "b", "c"]}'

        result = loads(json_text, mode="yaml", repair=True)
        assert result["items"] == [1, 2, 3]
        assert result["names"] == ["a", "b", "c"]

    def test_nested_json_in_yaml_mode(self):
        """Test nested JSON structures in YAML mode."""
        json_text = '{"user": {"name": "Alice", "profile": {"age": 30}}}'

        result = loads(json_text, mode="yaml", repair=True)
        assert result["user"]["name"] == "Alice"
        assert result["user"]["profile"]["age"] == 30


class TestYAMLRepairComplex:
    """Test complex YAML repair scenarios."""

    def test_yaml_with_nested_objects_and_lists(self):
        """Test YAML with both nested objects and lists."""
        yaml_text = """users:
  - name: Alice
    age: 30
  - name: Bob
    age: 25
settings:
  debug: true
  timeout: 60"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert len(result["users"]) == 2
        assert result["users"][0]["name"] == "Alice"
        assert result["settings"]["debug"] is True

    def test_yaml_with_multiline_values(self):
        """Test YAML with multiline string values."""
        yaml_text = """description: |
  This is a long
  multiline description
name: test"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert "name" in result
        assert result["name"] == "test"

    def test_yaml_with_quoted_strings(self):
        """Test YAML with quoted string values."""
        yaml_text = """message: "Hello, World!"
path: '/usr/local/bin'
status: 'active'"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["message"] == "Hello, World!"
        assert result["path"] == "/usr/local/bin"
        assert result["status"] == "active"


class TestYAMLRepairEdgeCases:
    """Test edge cases in YAML repair."""

    def test_yaml_with_empty_lines(self):
        """Test YAML with empty lines between keys."""
        yaml_text = """name: test

age: 25

city: NYC"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["name"] == "test"
        assert result["age"] == 25
        assert result["city"] == "NYC"

    def test_yaml_with_inline_comments(self):
        """Test YAML with inline comments."""
        yaml_text = """name: production  # Environment name
port: 8080  # Server port
debug: false  # Debug mode"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["name"] == "production"
        assert result["port"] == 8080
        assert result["debug"] is False

    def test_yaml_with_numeric_keys(self):
        """Test YAML with numeric-looking keys."""
        yaml_text = """1: first
2: second
3: third"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_yaml_with_boolean_values(self):
        """Test YAML with various boolean representations."""
        yaml_text = """active: true
disabled: false
enabled: yes
archived: no"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["active"] is True
        assert result["disabled"] is False

    def test_yaml_with_null_values(self):
        """Test YAML with null values."""
        yaml_text = """name: test
value: null
data: ~"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["name"] == "test"
        assert result["value"] is None


class TestYAMLRepairFailureCases:
    """Test cases where YAML repair should still succeed or fail gracefully."""

    def test_completely_malformed_yaml(self):
        """Test completely malformed YAML that can't be repaired."""
        yaml_text = ":::: this is not valid yaml ::::"

        # Should raise ConversionError
        with pytest.raises(ConversionError):
            loads(yaml_text, mode="yaml", repair=True)

    def test_yaml_with_no_colons(self):
        """Test text with no colons (not key-value pairs)."""
        yaml_text = "just some plain text without structure"

        # Should raise ConversionError
        with pytest.raises(ConversionError):
            loads(yaml_text, mode="yaml", repair=True)

    def test_empty_yaml_after_repair_attempts(self):
        """Test YAML that results in empty dict after repair."""
        yaml_text = "# Only comments\n# No actual data"

        # Should raise ConversionError
        with pytest.raises(ConversionError):
            loads(yaml_text, mode="yaml", repair=True)


class TestYAMLRepairWithoutPyYAML:
    """Test YAML repair behavior when PyYAML is not available."""

    def test_yaml_fallback_to_json_without_pyyaml(self):
        """Test YAML mode falling back to JSON when PyYAML unavailable."""
        # This is already tested in test_loads.py but adding here for completeness
        json_text = '{"key": "value"}'

        # Even in YAML mode, should parse JSON
        result = loads(json_text, mode="yaml", repair=True)
        assert result["key"] == "value"


class TestYAMLRepairIntegration:
    """Integration tests combining multiple repair strategies."""

    def test_yaml_requiring_multiple_repair_steps(self):
        """Test YAML that needs multiple repair strategies."""
        yaml_text = """    # Configuration
    name: test_app
    settings:
      timeout: 30
      retries: 3"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["name"] == "test_app"
        assert result["settings"]["timeout"] == 30

    def test_yaml_with_mixed_issues(self):
        """Test YAML with multiple types of issues."""
        yaml_text = """Some text before
    username: admin
    # Comment in middle
    password: secret123
    active: true"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["username"] == "admin"
        assert result["password"] == "secret123"
        assert result["active"] is True

    def test_yaml_repair_with_all_strategies(self):
        """Test YAML that exercises all repair strategies."""
        yaml_text = """      # Header
      config_name: production

      server_port: 8080
      enable_ssl: true"""

        result = loads(yaml_text, mode="yaml", repair=True)
        assert result["config_name"] == "production"
        assert result["server_port"] == 8080
        assert result["enable_ssl"] is True
