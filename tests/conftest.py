"""Shared test fixtures and configuration for llm-schema-lite tests."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

import pytest
from pydantic import BaseModel, EmailStr, Field, HttpUrl

# ============================================================================
# Test Models and Schemas
# ============================================================================


# Simple models
class SimpleUser(BaseModel):
    """A simple user model."""

    name: str
    age: int
    email: str


# Enums
class Role(str, Enum):
    """User role enum."""

    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class OrderStatus(str, Enum):
    """Order status enum."""

    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


# Models with validation
class Address(BaseModel):
    """An address with validation."""

    street: str
    city: str
    state: str = Field(..., pattern=r"^[A-Z]{2}$", description="US state code")
    postal_code: str = Field(..., pattern=r"^\d{5}$", description="US zip code")
    country: str = Field(default="USA", description="Country name")


class ContactInfo(BaseModel):
    """Contact information."""

    email: EmailStr
    phone: str | None = Field(None, pattern=r"^\+1\d{10}$", description="US phone number")
    website: HttpUrl | None = None


# Complex nested models
class User(BaseModel):
    """A complex user model with nested structures."""

    id: int
    name: str = Field(
        ...,
        title="Full Name",
        max_length=100,
        description="The user's full name",
    )
    role: Role = Field(default=Role.USER, description="User role (admin, user, or guest)")
    signup_date: datetime = Field(default_factory=datetime.utcnow, description="Signup timestamp")
    is_active: bool = Field(..., description="Is the user currently active?")
    addresses: list[Address]
    contact_info: ContactInfo


class Product(BaseModel):
    """A product model."""

    product_id: str = Field(
        ...,
        alias="productId",
        pattern=r"^[A-Z]{3}-\d{4}$",
        description="Product code (e.g., ABC-1234)",
    )
    name: str = Field(..., description="Product name")
    price: float = Field(..., ge=0, description="Price must be non-negative")
    available: bool = Field(default=True, description="Is the product available?")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Order(BaseModel):
    """A complex order model."""

    order_id: int
    user: User
    products: list[Product]
    total_price: float = Field(..., ge=0, description="Total order price")
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")


# Models with optional fields and unions
class Location(BaseModel):
    """A location."""

    city: str
    state: str
    country: str


class Profile(BaseModel):
    """A user profile with optional fields."""

    name: str | None = None
    email: EmailStr | None = None
    profile_url: HttpUrl | None = None
    age: int = Field(..., ge=0, le=150, description="Age in years")
    profession: str = Field(..., description="User's profession")
    is_active: bool = True
    tags: list[str] | None = None
    ids: list[int] | None = None
    location: Location | None = None


# Model with various field types
class ComplexTypes(BaseModel):
    """Model with various complex field types."""

    # Basic types
    string_field: str = Field(..., description="A string field")
    int_field: int = Field(..., ge=0, description="An integer field")
    float_field: float = Field(..., ge=0.0, le=100.0, description="A float field")
    bool_field: bool = Field(default=False, description="A boolean field")

    # Arrays
    string_list: list[str] = Field(default_factory=list, description="List of strings")
    int_list: list[int] = Field(..., description="List of integers")

    # Optional fields
    optional_str: str | None = Field(None, description="Optional string")
    optional_int: int | None = None

    # Nested
    address: Address
    addresses: list[Address] = Field(default_factory=list)

    # Enums
    role: Role = Role.USER
    status: OrderStatus = OrderStatus.PENDING


# Model with default values
class ConfigModel(BaseModel):
    """Configuration model with various defaults."""

    name: str = Field(default="default_name", description="Configuration name")
    timeout: int = Field(default=30, ge=1, le=300, description="Timeout in seconds")
    enabled: bool = Field(default=True, description="Is configuration enabled")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    tags: list[str] = Field(default_factory=list, description="Configuration tags")


# Additional testing models for specific test cases
class ComplexOrder(BaseModel):
    """Complex order model with nested structures (for testing purposes)."""

    order_id: str = Field(..., description="Unique order identifier")
    customer: User = Field(..., description="Customer information")
    items: list[dict[str, Any]] = Field(..., description="Order items")
    total: float = Field(..., ge=0, description="Total order amount")
    status: str = Field("pending", enum=["pending", "confirmed", "shipped", "delivered"])


class NumericConstraints(BaseModel):
    """Model for testing numeric validation constraints."""

    int_field: int = Field(..., ge=0, le=100, multipleOf=5)
    float_field: float = Field(..., gt=0.0, lt=100.0)
    optional_int: int | None = Field(None, ge=0, le=50)


class StringConstraints(BaseModel):
    """Model for testing string validation constraints."""

    name: str = Field(..., minLength=1, maxLength=50)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    description: str | None = Field(None, maxLength=200)


class PatternConstraints(BaseModel):
    """Model for testing pattern validation."""

    phone: str = Field(..., pattern=r"^\+?1?\d{9,15}$")
    zip_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$")
    username: str = Field(..., pattern=r"^[a-zA-Z0-9_]{3,20}$")


class UnionTypes(BaseModel):
    """Model for testing union types."""

    id: int | str = Field(..., description="ID can be int or string")
    status: Literal["active"] | Literal["inactive"] | Literal["pending"] = Field(...)
    metadata: dict[str, Any] | list[str] | None = Field(None)


class OptionalWithDefault(BaseModel):
    """Model for testing optional fields with defaults."""

    name: str = Field(..., description="Required name")
    age: int | None = Field(None, description="Optional age")
    is_active: bool = Field(True, description="Default to active")
    tags: list[str] = Field(default_factory=list, description="Default empty list")


class NestedReferences(BaseModel):
    """Model for testing nested references."""

    user: User = Field(..., description="User reference")
    order: ComplexOrder = Field(..., description="Order reference")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Advanced JSON Schema Test Cases
# ============================================================================


class AdvancedFeatures(BaseModel):
    """Model for testing advanced JSON Schema features."""

    # anyOf example
    flexible_id: int | str = Field(..., description="ID can be int or string")

    # oneOf example
    status: Literal["active", "inactive", "pending"] = Field(..., description="Status enum")

    # allOf example (via Field constraints)
    constrained_string: str = Field(..., minLength=5, maxLength=20, pattern=r"^[A-Za-z]+$")

    # not example (via Field constraints - no numbers in name)
    name: str = Field(..., pattern=r"^[^0-9]*$", description="Name without numbers")


# ============================================================================
# Test Data and Schemas
# ============================================================================

# Empty schema for testing
EMPTY_SCHEMA = {"type": "object"}

# Complex schema with advanced features
COMPLEX_SCHEMA = {
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string", "minLength": 1},
                "email": {"type": "string", "format": "email"},
            },
            "required": ["id", "name", "email"],
        },
        "items": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}},
    },
    "required": ["user", "items"],
}

# Schema with dependencies
DEPENDENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "credit_card": {"type": "string"},
        "billing_address": {"type": "string"},
    },
    "dependencies": {"credit_card": ["billing_address"]},
}

# Schema with if/then/else
CONDITIONAL_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["user", "admin"]},
        "permissions": {"type": "array", "items": {"type": "string"}},
    },
    "if": {"properties": {"type": {"const": "admin"}}},
    "then": {"properties": {"permissions": {"minItems": 1}}},
    "else": {"properties": {"permissions": {"maxItems": 0}}},
}

# Schema with patternProperties
PATTERN_PROPERTIES_SCHEMA = {
    "type": "object",
    "patternProperties": {"^[a-z]+$": {"type": "string"}, "^[0-9]+$": {"type": "number"}},
}

# Schema with contains
CONTAINS_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
    "contains": {"type": "string", "pattern": "^admin_"},
}

# Schema with uniqueItems
UNIQUE_ITEMS_SCHEMA = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}

# Schema with propertyNames
PROPERTY_NAMES_SCHEMA = {"type": "object", "propertyNames": {"pattern": "^[a-zA-Z][a-zA-Z0-9_]*$"}}

# Schema with unevaluatedProperties
UNEVALUATED_PROPERTIES_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "unevaluatedProperties": False,
}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_user_model():
    """Fixture for SimpleUser model."""
    return SimpleUser


@pytest.fixture
def user_model():
    """Fixture for User model."""
    return User


@pytest.fixture
def complex_order_model():
    """Fixture for ComplexOrder model."""
    return ComplexOrder


@pytest.fixture
def numeric_constraints_model():
    """Fixture for NumericConstraints model."""
    return NumericConstraints


@pytest.fixture
def string_constraints_model():
    """Fixture for StringConstraints model."""
    return StringConstraints


@pytest.fixture
def pattern_constraints_model():
    """Fixture for PatternConstraints model."""
    return PatternConstraints


@pytest.fixture
def union_types_model():
    """Fixture for UnionTypes model."""
    return UnionTypes


@pytest.fixture
def optional_with_default_model():
    """Fixture for OptionalWithDefault model."""
    return OptionalWithDefault


@pytest.fixture
def nested_references_model():
    """Fixture for NestedReferences model."""
    return NestedReferences


@pytest.fixture
def advanced_features_model():
    """Fixture for AdvancedFeatures model."""
    return AdvancedFeatures


@pytest.fixture
def empty_schema():
    """Fixture for empty schema."""
    return EMPTY_SCHEMA


@pytest.fixture
def complex_schema():
    """Fixture for complex schema."""
    return COMPLEX_SCHEMA


@pytest.fixture
def dependency_schema():
    """Fixture for dependency schema."""
    return DEPENDENCY_SCHEMA


@pytest.fixture
def conditional_schema():
    """Fixture for conditional schema."""
    return CONDITIONAL_SCHEMA


@pytest.fixture
def pattern_properties_schema():
    """Fixture for pattern properties schema."""
    return PATTERN_PROPERTIES_SCHEMA


@pytest.fixture
def contains_schema():
    """Fixture for contains schema."""
    return CONTAINS_SCHEMA


@pytest.fixture
def unique_items_schema():
    """Fixture for unique items schema."""
    return UNIQUE_ITEMS_SCHEMA


@pytest.fixture
def property_names_schema():
    """Fixture for property names schema."""
    return PROPERTY_NAMES_SCHEMA


@pytest.fixture
def unevaluated_properties_schema():
    """Fixture for unevaluated properties schema."""
    return UNEVALUATED_PROPERTIES_SCHEMA


# ============================================================================
# Formatter Test Data
# ============================================================================


@pytest.fixture(params=["jsonish", "typescript", "yaml"])
def format_type(request):
    """Fixture providing all format types for parameterized tests."""
    return request.param


@pytest.fixture
def formatter_expected_patterns():
    """Expected patterns for each formatter type."""
    return {
        "jsonish": {
            "string": "string",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "array",
            "object": "object",
        },
        "typescript": {
            "string": "string",
            "integer": "number",
            "number": "number",
            "boolean": "boolean",
            "array": "Array",
            "object": "object",
        },
        "yaml": {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        },
    }


@pytest.fixture
def formatter_comment_symbols():
    """Comment symbols for each formatter type."""
    return {"jsonish": "//", "typescript": "//", "yaml": "#"}


# ============================================================================
# Test Data Factories
# ============================================================================


class TestDataFactory:
    """Factory for creating test data."""

    @staticmethod
    def create_simple_schema() -> dict[str, Any]:
        """Create a simple test schema."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["name", "age", "email"],
        }

    @staticmethod
    def create_enum_schema() -> dict[str, Any]:
        """Create a schema with enum values."""
        return {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["active", "inactive", "pending"]}},
        }

    @staticmethod
    def create_union_schema() -> dict[str, Any]:
        """Create a schema with union types."""
        return {
            "type": "object",
            "properties": {"id": {"anyOf": [{"type": "integer"}, {"type": "string"}]}},
        }

    @staticmethod
    def create_array_schema() -> dict[str, Any]:
        """Create a schema with array types."""
        return {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
        }

    @staticmethod
    def create_nested_schema() -> dict[str, Any]:
        """Create a schema with nested objects."""
        return {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                }
            },
        }


@pytest.fixture
def test_data_factory():
    """Fixture for TestDataFactory."""
    return TestDataFactory
