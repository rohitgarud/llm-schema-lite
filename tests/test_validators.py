"""Tests for JSON and YAML validators (Draft202012Validator, FormatChecker)."""

from pydantic import BaseModel

from llm_schema_lite import validate
from llm_schema_lite.validators.json_validators import JSONValidator
from llm_schema_lite.validators.yaml_validators import YAMLValidator
from tests.conftest import ModelWithPriorityMetadata


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


# =============================================================================
# Enum alias validation
# =============================================================================


class TestEnumAliasValidation:
    """Validation with x-enum-aliases: aliases normalize to canonical and validate."""

    def test_json_validator_accepts_alias_for_enum(self):
        """Input 'urgent' (alias for 'critical') validates as critical."""
        validator = JSONValidator(ModelWithPriorityMetadata)
        ok, errs = validator.validate({"priority": "urgent"})
        assert ok is True
        assert errs is None

    def test_json_validator_accepts_second_alias(self):
        """Input 'blocker' (alias for 'critical') validates."""
        validator = JSONValidator(ModelWithPriorityMetadata)
        ok, errs = validator.validate({"priority": "blocker"})
        assert ok is True
        assert errs is None

    def test_json_validator_accepts_canonical_value(self):
        """Canonical value 'critical' still validates."""
        validator = JSONValidator(ModelWithPriorityMetadata)
        ok, errs = validator.validate({"priority": "critical"})
        assert ok is True
        assert errs is None

    def test_json_validator_rejects_invalid_alias(self):
        """Value that is not an enum value nor alias fails."""
        validator = JSONValidator(ModelWithPriorityMetadata)
        ok, errs = validator.validate({"priority": "unknown"})
        assert ok is False
        assert errs is not None
        assert len(errs) >= 1

    def test_yaml_validator_accepts_alias(self):
        """YAML validator normalizes alias to canonical."""
        validator = YAMLValidator(ModelWithPriorityMetadata)
        ok, errs = validator.validate("priority: urgent\n")
        assert ok is True
        assert errs is None

    def test_validate_function_accepts_alias_json(self):
        """validate(..., mode='json') accepts alias."""
        ok, errs = validate(ModelWithPriorityMetadata, '{"priority": "urgent"}', mode="json")
        assert ok is True
        assert errs is None

    def test_validate_function_accepts_alias_yaml(self):
        """validate(..., mode='yaml') accepts alias."""
        ok, errs = validate(ModelWithPriorityMetadata, "priority: blocker\n", mode="yaml")
        assert ok is True
        assert errs is None
