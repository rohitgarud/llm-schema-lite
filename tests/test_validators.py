"""Tests for JSON and YAML validators (Draft202012Validator, FormatChecker)."""

from pydantic import BaseModel

from llm_schema_lite import validate
from llm_schema_lite.validators.yaml_validators import YAMLValidator


class User(BaseModel):
    """Simple model for validator tests."""

    name: str
    age: int


class TestYAMLValidator:
    """Confirm YAMLValidator uses Draft202012Validator and FormatChecker with YAML."""

    def test_valid_yaml_string_passes(self):
        """Valid YAML string passes validation."""
        yaml_str = "name: Alice\nage: 30\n"
        ok, errs = validate(User, yaml_str, mode="yaml")
        assert ok is True
        assert errs is None

    def test_invalid_yaml_missing_required_fails(self):
        """Invalid YAML (missing required field) fails with expected errors."""
        yaml_str = "age: 25\n"
        ok, errs = validate(User, yaml_str, mode="yaml")
        assert ok is False
        assert errs is not None
        assert len(errs) >= 1
        assert any("name" in e.lower() or "required" in e.lower() for e in errs)

    def test_yaml_validator_parses_then_validates(self):
        """YAMLValidator.parse_data parses YAML; validate accepts parsed dict."""
        validator = YAMLValidator(User)
        parsed = validator.parse_data("name: Carol\nage: 28\n")
        assert parsed == {"name": "Carol", "age": 28}
        ok, errs = validator.validate(parsed)
        assert ok is True
        assert errs is None

    def test_schema_valid_for_draft202012(self):
        """Schema from Pydantic model is valid for Draft202012Validator."""
        from jsonschema import Draft202012Validator

        validator = YAMLValidator(User)
        Draft202012Validator.check_schema(validator._json_schema)

    def test_draft202012_and_format_checker_accept_parsed_yaml(self):
        """Draft202012Validator + FormatChecker accept data parsed from YAML."""
        from jsonschema import Draft202012Validator, FormatChecker

        validator = YAMLValidator(User)
        parsed = validator.parse_data("name: Bob\nage: 22\n")
        fc = FormatChecker()
        v = Draft202012Validator(validator._json_schema, format_checker=fc)
        errors = list(v.iter_errors(parsed))
        assert not errors
