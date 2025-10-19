"""Tests for Enum and Dict field type handling."""

from enum import Enum

from pydantic import BaseModel, Field

from llm_schema_lite import simplify_schema


# Enum definitions
class Status(str, Enum):
    """Status enum."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class Priority(int, Enum):
    """Priority enum with int values."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Role(str, Enum):
    """Role enum."""

    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class TestEnumHandling:
    """Test enum field handling across all formatters."""

    def test_jsonish_enum_basic(self):
        """Test JSONish formatter with enum fields."""

        class Model(BaseModel):
            status: Status
            role: Role = Role.USER

        schema = simplify_schema(Model, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should show enum values
        assert "oneOf:" in output or "active" in output
        assert "inactive" in output
        assert "pending" in output
        assert "(defaults to user)" in output or "(defaults to Role.USER)" in output

    def test_typescript_enum_basic(self):
        """Test TypeScript formatter with enum fields."""

        class Model(BaseModel):
            status: Status
            priority: Priority

        schema = simplify_schema(Model, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        # Should show union of literals
        assert '"active"' in output
        assert '"inactive"' in output
        assert '"pending"' in output
        assert "|" in output

    def test_yaml_enum_basic(self):
        """Test YAML formatter with enum fields."""

        class Model(BaseModel):
            status: Status
            role: Role

        schema = simplify_schema(Model, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should show Literal type
        assert "Literal[" in output
        assert '"active"' in output or "active" in output

    def test_enum_with_default(self):
        """Test enum with default value."""

        class Model(BaseModel):
            role: Role = Role.USER
            status: Status = Status.ACTIVE

        # JSONish
        jsonish = simplify_schema(Model, format_type="jsonish", include_metadata=True)
        output_j = jsonish.to_string()
        assert "(defaults to" in output_j

        # TypeScript
        ts = simplify_schema(Model, format_type="typescript", include_metadata=True)
        output_ts = ts.to_string()
        assert "(defaults to" in output_ts

        # YAML
        yaml = simplify_schema(Model, format_type="yaml", include_metadata=True)
        output_y = yaml.to_string()
        assert "(defaults to" in output_y

    def test_optional_enum(self):
        """Test optional enum fields."""

        class Model(BaseModel):
            status: Status | None = None
            role: Role | None = None

        schema = simplify_schema(Model, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # Should handle optional enums
        assert "status:" in output

    def test_int_enum(self):
        """Test integer-based enums."""

        class Model(BaseModel):
            priority: Priority = Priority.MEDIUM

        # JSONish
        jsonish = simplify_schema(Model, format_type="jsonish", include_metadata=True)
        output_j = jsonish.to_string()
        # Should show enum values
        assert "1" in output_j or "2" in output_j or "3" in output_j

    def test_multiple_enums(self):
        """Test model with multiple enum fields."""

        class Model(BaseModel):
            status: Status
            role: Role
            priority: Priority

        # Test all formatters
        for format_type in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(Model, format_type=format_type, include_metadata=False)  # type: ignore
            output = schema.to_string()

            # Should have content for all enums
            assert len(output) > 50
            assert "status:" in output or "status;" in output
            assert "role:" in output or "role;" in output
            assert "priority:" in output or "priority;" in output


class TestDictHandling:
    """Test dict/Dict field type handling."""

    def test_dict_basic(self):
        """Test basic dict field."""

        class Model(BaseModel):
            metadata: dict
            settings: dict[str, str]

        # JSONish
        jsonish = simplify_schema(Model, format_type="jsonish", include_metadata=False)
        output_j = jsonish.to_string()
        assert "metadata: object" in output_j
        assert "settings: object" in output_j

        # TypeScript
        ts = simplify_schema(Model, format_type="typescript", include_metadata=False)
        output_ts = ts.to_string()
        assert "metadata: object" in output_ts
        assert "settings: object" in output_ts

        # YAML
        yaml = simplify_schema(Model, format_type="yaml", include_metadata=False)
        output_y = yaml.to_string()
        assert "metadata: object" in output_y
        assert "settings: object" in output_y

    def test_dict_with_types(self):
        """Test Dict with type parameters."""

        class Model(BaseModel):
            str_to_str: dict[str, str]
            str_to_int: dict[str, int]
            str_to_list: dict[str, list]

        schema = simplify_schema(Model, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # All dict types become object
        assert "str_to_str: object" in output
        assert "str_to_int: object" in output
        assert "str_to_list: object" in output

    def test_optional_dict(self):
        """Test optional dict fields."""

        class Model(BaseModel):
            metadata: dict[str, str] | None = None
            settings: dict | None = None

        schema = simplify_schema(Model, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # Should handle optional dicts
        assert "metadata:" in output
        assert "settings:" in output
        assert "null" in output or "None" in output or "or" in output

    def test_dict_with_default(self):
        """Test dict with default value."""

        class Model(BaseModel):
            metadata: dict = Field(default_factory=dict)
            settings: dict[str, str] = Field(default_factory=dict)

        schema = simplify_schema(Model, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should show defaults
        assert "metadata:" in output
        assert "settings:" in output


class TestEnumAndDictCombined:
    """Test models with both enums and dicts."""

    def test_complex_model(self):
        """Test model with both enum and dict fields."""

        class Config(BaseModel):
            name: str
            status: Status
            role: Role = Role.USER
            settings: dict[str, str]
            metadata: dict
            labels: dict[str, int] | None = None

        # Test all three formatters
        jsonish = simplify_schema(Config, format_type="jsonish", include_metadata=True)
        output_j = jsonish.to_string()

        # Enums should show values
        assert "oneOf:" in output_j or "active" in output_j

        # Dicts should show as object
        assert "object" in output_j

        # TypeScript
        ts = simplify_schema(Config, format_type="typescript", include_metadata=True)
        output_ts = ts.to_string()

        # Enums should be union literals
        assert '"active"' in output_ts
        assert "|" in output_ts

        # YAML
        yaml = simplify_schema(Config, format_type="yaml", include_metadata=True)
        output_y = yaml.to_string()

        # Enums should be Literal
        assert "Literal[" in output_y

        # Dicts should be object
        assert "object" in output_y

    def test_nested_with_enums(self):
        """Test nested models with enums."""

        class SubModel(BaseModel):
            status: Status
            priority: Priority

        class MainModel(BaseModel):
            name: str
            config: SubModel
            metadata: dict

        schema = simplify_schema(MainModel, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # Should handle nested model
        assert "config:" in output
        assert "metadata: object" in output
