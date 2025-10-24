"""Tests for formatter edge cases and missing coverage."""

from pydantic import BaseModel, Field

from llm_schema_lite import simplify_schema
from llm_schema_lite.formatters import JSONishFormatter, TypeScriptFormatter, YAMLFormatter


class TestFormatterEnumProcessing:
    """Test enum processing in formatters."""

    def test_enum_with_string_values(self):
        """Test enum with string values."""
        schema_dict = {
            "type": "object",
            "properties": {"status": {"enum": ["active", "inactive", "pending"]}},
            "$defs": {},
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            formatter_class = {
                "jsonish": JSONishFormatter,
                "typescript": TypeScriptFormatter,
                "yaml": YAMLFormatter,
            }[fmt]

            formatter = formatter_class(schema_dict, include_metadata=True)
            result = formatter.process_enum(schema_dict["properties"]["status"])
            assert isinstance(result, str)
            assert len(result) > 0

    def test_enum_with_integer_values(self):
        """Test enum with integer values."""
        schema_dict = {
            "type": "object",
            "properties": {"priority": {"enum": [1, 2, 3, 4, 5]}},
            "$defs": {},
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            formatter_class = {
                "jsonish": JSONishFormatter,
                "typescript": TypeScriptFormatter,
                "yaml": YAMLFormatter,
            }[fmt]

            formatter = formatter_class(schema_dict, include_metadata=True)
            result = formatter.process_enum(schema_dict["properties"]["priority"])
            assert isinstance(result, str)
            assert len(result) > 0

    def test_enum_with_mixed_values(self):
        """Test enum with mixed type values."""
        schema_dict = {
            "type": "object",
            "properties": {"value": {"enum": ["auto", 0, "manual", 1]}},
            "$defs": {},
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            formatter_class = {
                "jsonish": JSONishFormatter,
                "typescript": TypeScriptFormatter,
                "yaml": YAMLFormatter,
            }[fmt]

            formatter = formatter_class(schema_dict, include_metadata=True)
            result = formatter.process_enum(schema_dict["properties"]["value"])
            assert isinstance(result, str)


class TestFormatterReferenceHandling:
    """Test $ref handling in formatters."""

    def test_ref_with_enum_definition(self):
        """Test $ref pointing to enum definition."""
        schema_dict = {
            "type": "object",
            "properties": {"status": {"$ref": "#/$defs/Status"}},
            "$defs": {"Status": {"enum": ["active", "inactive"]}},
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert "status" in output

    def test_ref_with_properties_definition(self):
        """Test $ref pointing to object with properties."""
        schema_dict = {
            "type": "object",
            "properties": {"address": {"$ref": "#/$defs/Address"}},
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}, "city": {"type": "string"}},
                }
            },
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert "address" in output

    def test_ref_with_type_definition(self):
        """Test $ref pointing to simple type definition."""
        schema_dict = {
            "type": "object",
            "properties": {"id": {"$ref": "#/$defs/ID"}},
            "$defs": {"ID": {"type": "string"}},
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert "id" in output

    def test_missing_ref_definition(self):
        """Test handling of missing $ref definition."""
        schema_dict = {
            "type": "object",
            "properties": {"data": {"$ref": "#/$defs/MissingDefinition"}},
            "$defs": {},
        }

        # Should not crash, should fallback gracefully
        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert isinstance(output, str)


class TestFormatterAnyOfHandling:
    """Test anyOf handling in formatters."""

    def test_anyof_with_multiple_types(self):
        """Test anyOf with multiple type options."""
        schema_dict = {
            "type": "object",
            "properties": {
                "value": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}]}
            },
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert "value" in output

    def test_anyof_with_refs(self):
        """Test anyOf with $ref values."""
        schema_dict = {
            "type": "object",
            "properties": {
                "data": {"anyOf": [{"$ref": "#/$defs/StringType"}, {"$ref": "#/$defs/IntType"}]}
            },
            "$defs": {"StringType": {"type": "string"}, "IntType": {"type": "integer"}},
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert "data" in output

    def test_anyof_with_null(self):
        """Test anyOf including null type."""
        schema_dict = {
            "type": "object",
            "properties": {"optional_field": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert "optional_field" in output


class TestFormatterArrayHandling:
    """Test array handling in formatters."""

    def test_array_of_objects(self):
        """Test array with object items."""
        schema_dict = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
                    },
                }
            },
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert "items" in output

    def test_array_with_ref_items(self):
        """Test array with $ref items."""
        schema_dict = {
            "type": "object",
            "properties": {"users": {"type": "array", "items": {"$ref": "#/$defs/User"}}},
            "$defs": {"User": {"type": "object", "properties": {"username": {"type": "string"}}}},
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert "users" in output

    def test_nested_arrays(self):
        """Test nested array structures."""
        schema_dict = {
            "type": "object",
            "properties": {
                "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
            },
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert "matrix" in output


class TestFormatterMetadataHandling:
    """Test metadata inclusion and formatting."""

    def test_metadata_with_description(self):
        """Test metadata with field descriptions."""

        class ModelWithDescriptions(BaseModel):
            name: str = Field(..., description="User's full name")
            age: int = Field(..., description="User's age in years")

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema_with = simplify_schema(
                ModelWithDescriptions, format_type=fmt, include_metadata=True
            )
            schema_without = simplify_schema(
                ModelWithDescriptions, format_type=fmt, include_metadata=False
            )

            # With metadata should be longer
            assert len(schema_with.to_string()) > len(schema_without.to_string())

    def test_metadata_with_constraints(self):
        """Test metadata with validation constraints."""

        class ModelWithConstraints(BaseModel):
            email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
            age: int = Field(..., ge=0, le=150)
            score: float = Field(..., ge=0.0, le=100.0)

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(ModelWithConstraints, format_type=fmt, include_metadata=True)
            output = schema.to_string()
            assert "email" in output
            assert "age" in output
            assert "score" in output

    def test_metadata_with_defaults(self):
        """Test metadata with default values."""

        class ModelWithDefaults(BaseModel):
            active: bool = True
            count: int = 0
            name: str = "default"

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(ModelWithDefaults, format_type=fmt, include_metadata=True)
            output = schema.to_string()
            assert "active" in output
            assert "count" in output
            assert "name" in output


class TestFormatterEmptyAndEdgeCases:
    """Test empty schemas and edge cases."""

    def test_empty_properties(self):
        """Test schema with no properties."""
        schema_dict = {"type": "object", "properties": {}}

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert isinstance(output, str)

    def test_property_with_no_type(self):
        """Test property definition without explicit type."""
        schema_dict = {"type": "object", "properties": {"any_field": {}}}

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            output = schema.to_string()
            assert "any_field" in output

    def test_complex_nested_structure(self):
        """Test complex deeply nested structure."""

        class Address(BaseModel):
            street: str
            city: str
            country: str

        class Contact(BaseModel):
            email: str
            phone: str
            address: Address

        class Company(BaseModel):
            name: str
            contact: Contact

        class Person(BaseModel):
            name: str
            company: Company
            addresses: list[Address]

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(Person, format_type=fmt)
            output = schema.to_string()
            assert "name" in output
            assert "company" in output
            assert "addresses" in output
