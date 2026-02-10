"""Shared test fixtures and configuration for llm-schema-lite tests."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

import pytest
from pydantic import BaseModel, EmailStr, Field, HttpUrl

# ============================================================================
# Test Models and Schemas
# ============================================================================


# ============================================================================
# Simple Models (one primary feature each)
# ============================================================================


# Primitives
class SimpleUser(BaseModel):
    """A simple user model."""

    name: str
    age: int
    email: str


class OnlyString(BaseModel):
    """Model with only a string field."""

    value: str


class OnlyInt(BaseModel):
    """Model with only an integer field."""

    value: int


class OnlyFloat(BaseModel):
    """Model with only a float field."""

    value: float


class OnlyBool(BaseModel):
    """Model with only a boolean field."""

    value: bool


# String constraints
class StringFormatEmail(BaseModel):
    """Model with email format."""

    email: EmailStr


class StringFormatUri(BaseModel):
    """Model with URI format."""

    website: HttpUrl


class StringPattern(BaseModel):
    """Model with pattern constraint."""

    code: str = Field(..., pattern=r"^[A-Z]{3}-\d{3}$")


class StringLength(BaseModel):
    """Model with min/max length constraints."""

    name: str = Field(..., min_length=3, max_length=50)


class StringConstraints(BaseModel):
    """Model for testing string validation constraints."""

    name: str = Field(..., min_length=1, max_length=50)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    description: str | None = Field(None, max_length=200)


class PatternConstraints(BaseModel):
    """Model for testing pattern validation."""

    phone: str = Field(..., pattern=r"^\+?1?\d{9,15}$")
    zip_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$")
    username: str = Field(..., pattern=r"^[a-zA-Z0-9_]{3,20}$")


# Numeric constraints
class NumericConstraints(BaseModel):
    """Model for testing numeric validation constraints."""

    int_field: int = Field(..., ge=0, le=100, multiple_of=5)
    float_field: float = Field(..., gt=0.0, lt=100.0)
    optional_int: int | None = Field(None, ge=0, le=50)


class ExclusiveMinMax(BaseModel):
    """Model with exclusive minimum/maximum (gt/lt)."""

    value: float = Field(..., gt=0.0, lt=100.0)
    count: int = Field(..., gt=0, lt=10)


# Array constraints
class ArrayOfStrings(BaseModel):
    """Model with array of strings."""

    items: list[str]


class ArrayMinMaxItems(BaseModel):
    """Model with min/max items constraints."""

    tags: list[str] = Field(..., min_length=1, max_length=5)


class ArrayLengthModel(BaseModel):
    """Model with array length constraints."""

    values: list[int] = Field(..., min_length=2, max_length=10)


class ArrayUniqueItems(BaseModel):
    """Model with unique items constraint."""

    unique_tags: set[str] = Field(..., description="Unique tags")


class UniqueListModel(BaseModel):
    """Model with unique list items."""

    items: set[int] = Field(..., description="Unique items")


# Object constraints
class ObjectRequiredOnly(BaseModel):
    """Model with all required fields."""

    name: str
    age: int


class ObjectWithDefaults(BaseModel):
    """Model with default values."""

    name: str = "default"
    count: int = 0


class ObjectAdditionalPropsFalse(BaseModel):
    """Model that forbids additional properties."""

    model_config = {"extra": "forbid"}
    name: str
    value: int


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


class Priority(int, Enum):
    """Integer enum for priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class IntEnumModel(BaseModel):
    """Model with integer enum."""

    priority: Priority


# Literal types
class LiteralSingle(BaseModel):
    """Model with single literal value."""

    api_version: Literal["v1"]


class LiteralUnion(BaseModel):
    """Model with union of literals."""

    status: Literal["draft", "published", "archived"]


class LiteralOnlyModel(BaseModel):
    """Model with multiple literal fields."""

    mode: Literal["read", "write", "execute"]
    level: Literal[1, 2, 3]


class ConstModel(BaseModel):
    """Model with const-like field (single literal)."""

    version: Literal["v1"] = "v1"


# Union types
class UnionTypes(BaseModel):
    """Model for testing union types."""

    id: int | str = Field(..., description="ID can be int or string")
    status: Literal["active"] | Literal["inactive"] | Literal["pending"] = Field(...)
    metadata: dict[str, Any] | list[str] | None = Field(None)


# Nullable/Optional types
class OptionalFields(BaseModel):
    """Model with optional fields."""

    name: str
    age: int | None = None
    email: str | None = None


class RequiredOptionalModel(BaseModel):
    """Model mixing required and optional fields."""

    required_field: str
    optional_field: str | None = None
    nullable_with_default: int | None = None


class OptionalWithDefault(BaseModel):
    """Model for testing optional fields with defaults."""

    name: str = Field(..., description="Required name")
    age: int | None = Field(None, description="Optional age")
    is_active: bool = Field(True, description="Default to active")
    tags: list[str] = Field(default_factory=list, description="Default empty list")


# Base models for nested references (defined early to avoid forward references)
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


# Nested models and references
class SingleNested(BaseModel):
    """Model with single nested object."""

    name: str
    address: Address


class PersonWithAddress(BaseModel):
    """Person model with nested address."""

    name: str
    address: Address


# Metadata models
class WithTitleDescription(BaseModel):
    """Model with title and description metadata."""

    model_config = {
        "title": "User Profile",
        "json_schema_extra": {"description": "A user profile model"},
    }

    name: str
    age: int


class WithFieldDescriptions(BaseModel):
    """Model with field-level descriptions."""

    name: str = Field(..., title="Full Name", description="The user's full name")
    email: EmailStr = Field(..., title="Email Address", description="Contact email")
    age: int = Field(..., ge=0, le=150, title="Age", description="Age in years")


# Alias models
class ModelWithAlias(BaseModel):
    """Model with field aliases."""

    user_id: str = Field(..., alias="userId")
    user_name: str = Field(..., alias="userName")


# Datetime models
class EventWithDate(BaseModel):
    """Model with datetime field."""

    name: str
    event_date: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Dict models
class DictOnlyModel(BaseModel):
    """Model with dict field."""

    metadata: dict[str, Any]
    config: dict[str, str] | None = None


# Models for formatter test equivalents
class SimpleFormatterModel(BaseModel):
    """Simple model for formatter tests."""

    name: str
    age: int = Field(..., ge=0, le=150)
    email: str | None = None


class OrderedFieldsModel(BaseModel):
    """Model with ordered fields for key-order tests."""

    first: str
    second: int
    third: bool


class ConstrainedFormatterModel(BaseModel):
    """Model with various constraints for formatter tests."""

    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    score: float = Field(..., ge=0.0, le=100.0)
    tags: list[str] = Field(default_factory=list)


# ============================================================================
# Complex Models (multiple features combined)
# ============================================================================


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
    locations: Location | None = None
    ids: list[int] | None = None


# Model with various field types (existing complex model)
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
    address: Address = Field(default_factory=Address, description="Address of the user")
    addresses: list[Address] | None = Field(
        default_factory=list,
        description="Addresses of the user",
    )

    # Enums
    role: Role = Role.USER
    status: OrderStatus = OrderStatus.PENDING


# Model with default values (existing complex model)
class ConfigModel(BaseModel):
    """Configuration model with various defaults."""

    name: str = Field(default="default_name", description="Configuration name")
    timeout: int = Field(default=30, ge=1, le=300, description="Timeout in seconds")
    enabled: bool = Field(default=True, description="Is configuration enabled")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    tags: list[str] = Field(default_factory=list, description="Configuration tags")


# Complex order (existing)
class ComplexOrder(BaseModel):
    """Complex order model with nested structures (for testing purposes)."""

    order_id: str = Field(..., description="Unique order identifier")
    customer: User = Field(..., description="Customer information")
    items: list[Product] = Field(..., description="Order items")
    total: float = Field(..., ge=0, description="Total order amount")
    status: Literal["pending", "confirmed", "shipped", "delivered"] = "pending"


class NestedReferences(BaseModel):
    """Model for testing nested references."""

    user: User = Field(..., description="User reference")
    order: ComplexOrder = Field(..., description="Order reference")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdvancedFeatures(BaseModel):
    """Model for testing advanced JSON Schema features."""

    # anyOf example
    flexible_id: int | str = Field(..., description="ID can be int or string")

    # oneOf example
    status: Literal["active", "inactive", "pending"] = Field(..., description="Status enum")

    # allOf example (via Field constraints)
    constrained_string: str = Field(..., min_length=5, max_length=20, pattern=r"^[A-Za-z]+$")

    # not example (via Field constraints - no numbers in name)
    name: str = Field(..., pattern=r"^[^0-9]*$", description="Name without numbers")


# New complex models combining multiple features
class DeepNested(BaseModel):
    """Model with deep nesting (A -> B -> C)."""

    class LevelC(BaseModel):
        """Innermost level."""

        value: str
        count: int

    class LevelB(BaseModel):
        """Middle level."""

        name: str
        level_c: "DeepNested.LevelC"

    id: int
    level_b: LevelB


class UnionHeavy(BaseModel):
    """Model with multiple union types."""

    id: int | str
    value: int | float | str
    status: Literal["active", "inactive"] | None
    data: dict[str, Any] | list[Any] | str | None


class BaseA(BaseModel):
    """Base model A for composition."""

    field_a: str
    count_a: int


class BaseB(BaseModel):
    """Base model B for composition."""

    field_b: str
    count_b: int


class AllOfLike(BaseA, BaseB):
    """Model that inherits from two bases (generates allOf in schema)."""

    own_field: str


class ListAndDict(BaseModel):
    """Model with list of objects and dict fields."""

    items: list[Address]
    metadata: dict[str, str]
    config: dict[str, int] | None = None


class FullFeaturedModel(BaseModel):
    """Kitchen sink model with many features."""

    # Primitives
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    score: float = Field(..., ge=0.0, le=100.0)
    is_active: bool = True

    # Enum and Literal
    role: Role = Role.USER
    status: Literal["draft", "published", "archived"] = "draft"

    # Union
    identifier: int | str

    # Nested
    address: Address | None = None

    # List of objects
    tags: list[str] = Field(default_factory=list)
    addresses: list[Address] = Field(default_factory=list)

    # Datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None

    # Optional with defaults
    description: str | None = Field(None, max_length=500)
    count: int = 0

    # Dict
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArrayOfRefsModel(BaseModel):
    """Model with arrays of referenced objects."""

    addresses: list[Address]
    products: list[Product]
    users: list[User] | None = None


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

# New schemas for composition and additional keywords

# Schema with allOf at top level
ALL_OF_SCHEMA = {
    "allOf": [
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        {"type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]},
    ]
}

# Schema with oneOf at top level
ONE_OF_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "type": {"const": "email"},
                "email": {"type": "string", "format": "email"},
            },
        },
        {
            "type": "object",
            "properties": {
                "type": {"const": "phone"},
                "phone": {"type": "string", "pattern": r"^\+?1?\d{9,15}$"},
            },
        },
    ]
}

# Schema with anyOf at top level
ANY_OF_SCHEMA = {
    "anyOf": [
        {"type": "object", "properties": {"id": {"type": "integer"}}},
        {"type": "object", "properties": {"id": {"type": "string"}}},
    ]
}

# Schema with const keyword
CONST_SCHEMA = {
    "type": "object",
    "properties": {"api_version": {"const": "v1.0"}, "name": {"type": "string"}},
    "required": ["api_version"],
}

# Schema with additionalItems (legacy)
ADDITIONAL_ITEMS_SCHEMA = {
    "type": "array",
    "items": [{"type": "string"}, {"type": "integer"}],
    "additionalItems": False,
}

# Schema with not keyword
NOT_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string", "not": {"pattern": "^test"}}},
}

# Schema with format keywords
FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "email": {"type": "string", "format": "email"},
        "uri": {"type": "string", "format": "uri"},
        "date_time": {"type": "string", "format": "date-time"},
        "uuid": {"type": "string", "format": "uuid"},
    },
}

# Schema with minItems/maxItems on array
MIN_MAX_ITEMS_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
        }
    },
}

# TDD schemas for not-yet-supported keywords (Draft 2020-12)

# Schema with prefixItems
PREFIX_ITEMS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "array",
    "prefixItems": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}],
    "items": {"type": "string"},
}

# Schema with dependentRequired
DEPENDENT_REQUIRED_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "credit_card": {"type": "string"},
        "billing_address": {"type": "string"},
        "cvv": {"type": "string"},
    },
    "dependentRequired": {"credit_card": ["billing_address", "cvv"]},
}

# Schema with dependentSchemas
DEPENDENT_SCHEMAS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "account_type": {"type": "string", "enum": ["personal", "business"]},
        "business_name": {"type": "string"},
    },
    "dependentSchemas": {
        "account_type": {
            "if": {"properties": {"account_type": {"const": "business"}}},
            "then": {"required": ["business_name"]},
        }
    },
}

# Schema with unevaluatedItems
UNEVALUATED_ITEMS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "array",
    "prefixItems": [{"type": "string"}, {"type": "integer"}],
    "unevaluatedItems": False,
}

# Schema with minContains/maxContains
MIN_MAX_CONTAINS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "array",
    "items": {"type": "string"},
    "contains": {"type": "string", "pattern": "^admin_"},
    "minContains": 1,
    "maxContains": 3,
}

# Schema with contentEncoding/contentMediaType
CONTENT_ENCODING_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "image": {"type": "string", "contentEncoding": "base64", "contentMediaType": "image/png"},
        "document": {"type": "string", "contentMediaType": "application/json"},
    },
}

# Schema with $anchor for TDD
ANCHOR_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$defs": {
        "user": {
            "$anchor": "userDef",
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
    },
    "type": "object",
    "properties": {"user": {"$ref": "#userDef"}},
}

# Schema with deprecated and examples
DEPRECATED_EXAMPLES_SCHEMA = {
    "type": "object",
    "properties": {
        "old_field": {
            "type": "string",
            "deprecated": True,
            "description": "This field is deprecated",
        },
        "new_field": {"type": "string", "examples": ["example1", "example2"]},
    },
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


# Fixtures for new simple models
@pytest.fixture
def only_string_model():
    """Fixture for OnlyString model."""
    return OnlyString


@pytest.fixture
def only_int_model():
    """Fixture for OnlyInt model."""
    return OnlyInt


@pytest.fixture
def only_float_model():
    """Fixture for OnlyFloat model."""
    return OnlyFloat


@pytest.fixture
def only_bool_model():
    """Fixture for OnlyBool model."""
    return OnlyBool


@pytest.fixture
def string_format_email_model():
    """Fixture for StringFormatEmail model."""
    return StringFormatEmail


@pytest.fixture
def string_format_uri_model():
    """Fixture for StringFormatUri model."""
    return StringFormatUri


@pytest.fixture
def string_pattern_model():
    """Fixture for StringPattern model."""
    return StringPattern


@pytest.fixture
def string_length_model():
    """Fixture for StringLength model."""
    return StringLength


@pytest.fixture
def exclusive_min_max_model():
    """Fixture for ExclusiveMinMax model."""
    return ExclusiveMinMax


@pytest.fixture
def array_of_strings_model():
    """Fixture for ArrayOfStrings model."""
    return ArrayOfStrings


@pytest.fixture
def array_min_max_items_model():
    """Fixture for ArrayMinMaxItems model."""
    return ArrayMinMaxItems


@pytest.fixture
def array_length_model():
    """Fixture for ArrayLengthModel model."""
    return ArrayLengthModel


@pytest.fixture
def array_unique_items_model():
    """Fixture for ArrayUniqueItems model."""
    return ArrayUniqueItems


@pytest.fixture
def unique_list_model():
    """Fixture for UniqueListModel model."""
    return UniqueListModel


@pytest.fixture
def object_required_only_model():
    """Fixture for ObjectRequiredOnly model."""
    return ObjectRequiredOnly


@pytest.fixture
def object_with_defaults_model():
    """Fixture for ObjectWithDefaults model."""
    return ObjectWithDefaults


@pytest.fixture
def object_additional_props_false_model():
    """Fixture for ObjectAdditionalPropsFalse model."""
    return ObjectAdditionalPropsFalse


@pytest.fixture
def int_enum_model():
    """Fixture for IntEnumModel model."""
    return IntEnumModel


@pytest.fixture
def literal_single_model():
    """Fixture for LiteralSingle model."""
    return LiteralSingle


@pytest.fixture
def literal_union_model():
    """Fixture for LiteralUnion model."""
    return LiteralUnion


@pytest.fixture
def literal_only_model():
    """Fixture for LiteralOnlyModel model."""
    return LiteralOnlyModel


@pytest.fixture
def const_model():
    """Fixture for ConstModel model."""
    return ConstModel


@pytest.fixture
def optional_fields_model():
    """Fixture for OptionalFields model."""
    return OptionalFields


@pytest.fixture
def required_optional_model():
    """Fixture for RequiredOptionalModel model."""
    return RequiredOptionalModel


@pytest.fixture
def single_nested_model():
    """Fixture for SingleNested model."""
    return SingleNested


@pytest.fixture
def person_with_address_model():
    """Fixture for PersonWithAddress model."""
    return PersonWithAddress


@pytest.fixture
def with_title_description_model():
    """Fixture for WithTitleDescription model."""
    return WithTitleDescription


@pytest.fixture
def with_field_descriptions_model():
    """Fixture for WithFieldDescriptions model."""
    return WithFieldDescriptions


@pytest.fixture
def model_with_alias():
    """Fixture for ModelWithAlias model."""
    return ModelWithAlias


@pytest.fixture
def event_with_date_model():
    """Fixture for EventWithDate model."""
    return EventWithDate


@pytest.fixture
def dict_only_model():
    """Fixture for DictOnlyModel model."""
    return DictOnlyModel


@pytest.fixture
def simple_formatter_model():
    """Fixture for SimpleFormatterModel model."""
    return SimpleFormatterModel


@pytest.fixture
def ordered_fields_model():
    """Fixture for OrderedFieldsModel model."""
    return OrderedFieldsModel


@pytest.fixture
def constrained_formatter_model():
    """Fixture for ConstrainedFormatterModel model."""
    return ConstrainedFormatterModel


# Fixtures for new complex models
@pytest.fixture
def deep_nested_model():
    """Fixture for DeepNested model."""
    return DeepNested


@pytest.fixture
def union_heavy_model():
    """Fixture for UnionHeavy model."""
    return UnionHeavy


@pytest.fixture
def all_of_like_model():
    """Fixture for AllOfLike model."""
    return AllOfLike


@pytest.fixture
def list_and_dict_model():
    """Fixture for ListAndDict model."""
    return ListAndDict


@pytest.fixture
def full_featured_model():
    """Fixture for FullFeaturedModel model."""
    return FullFeaturedModel


@pytest.fixture
def array_of_refs_model():
    """Fixture for ArrayOfRefsModel model."""
    return ArrayOfRefsModel


# Fixtures for new raw schemas
@pytest.fixture
def all_of_schema():
    """Fixture for allOf schema."""
    return ALL_OF_SCHEMA


@pytest.fixture
def one_of_schema():
    """Fixture for oneOf schema."""
    return ONE_OF_SCHEMA


@pytest.fixture
def any_of_schema():
    """Fixture for anyOf schema."""
    return ANY_OF_SCHEMA


@pytest.fixture
def const_schema():
    """Fixture for const schema."""
    return CONST_SCHEMA


@pytest.fixture
def additional_items_schema():
    """Fixture for additionalItems schema."""
    return ADDITIONAL_ITEMS_SCHEMA


@pytest.fixture
def not_schema():
    """Fixture for not schema."""
    return NOT_SCHEMA


@pytest.fixture
def format_schema():
    """Fixture for format schema."""
    return FORMAT_SCHEMA


@pytest.fixture
def min_max_items_schema():
    """Fixture for minItems/maxItems schema."""
    return MIN_MAX_ITEMS_SCHEMA


@pytest.fixture
def prefix_items_schema():
    """Fixture for prefixItems schema."""
    return PREFIX_ITEMS_SCHEMA


@pytest.fixture
def dependent_required_schema():
    """Fixture for dependentRequired schema."""
    return DEPENDENT_REQUIRED_SCHEMA


@pytest.fixture
def dependent_schemas_schema():
    """Fixture for dependentSchemas schema."""
    return DEPENDENT_SCHEMAS_SCHEMA


@pytest.fixture
def unevaluated_items_schema():
    """Fixture for unevaluatedItems schema."""
    return UNEVALUATED_ITEMS_SCHEMA


@pytest.fixture
def min_max_contains_schema():
    """Fixture for minContains/maxContains schema."""
    return MIN_MAX_CONTAINS_SCHEMA


@pytest.fixture
def content_encoding_schema():
    """Fixture for contentEncoding schema."""
    return CONTENT_ENCODING_SCHEMA


@pytest.fixture
def anchor_schema():
    """Fixture for $anchor schema."""
    return ANCHOR_SCHEMA


@pytest.fixture
def deprecated_examples_schema():
    """Fixture for deprecated/examples schema."""
    return DEPRECATED_EXAMPLES_SCHEMA


# Registry fixtures for all models and schemas
@pytest.fixture
def all_pydantic_models():
    """Registry of all Pydantic models (name, model_class)."""
    return [
        ("SimpleUser", SimpleUser),
        ("OnlyString", OnlyString),
        ("OnlyInt", OnlyInt),
        ("OnlyFloat", OnlyFloat),
        ("OnlyBool", OnlyBool),
        ("StringFormatEmail", StringFormatEmail),
        ("StringFormatUri", StringFormatUri),
        ("StringPattern", StringPattern),
        ("StringLength", StringLength),
        ("StringConstraints", StringConstraints),
        ("NumericConstraints", NumericConstraints),
        ("ExclusiveMinMax", ExclusiveMinMax),
        ("ArrayOfStrings", ArrayOfStrings),
        ("ArrayMinMaxItems", ArrayMinMaxItems),
        ("ArrayLengthModel", ArrayLengthModel),
        ("ArrayUniqueItems", ArrayUniqueItems),
        ("UniqueListModel", UniqueListModel),
        ("ObjectRequiredOnly", ObjectRequiredOnly),
        ("ObjectWithDefaults", ObjectWithDefaults),
        ("ObjectAdditionalPropsFalse", ObjectAdditionalPropsFalse),
        ("IntEnumModel", IntEnumModel),
        ("LiteralSingle", LiteralSingle),
        ("LiteralUnion", LiteralUnion),
        ("LiteralOnlyModel", LiteralOnlyModel),
        ("ConstModel", ConstModel),
        ("OptionalFields", OptionalFields),
        ("RequiredOptionalModel", RequiredOptionalModel),
        ("SingleNested", SingleNested),
        ("PersonWithAddress", PersonWithAddress),
        ("WithTitleDescription", WithTitleDescription),
        ("WithFieldDescriptions", WithFieldDescriptions),
        ("ModelWithAlias", ModelWithAlias),
        ("EventWithDate", EventWithDate),
        ("DictOnlyModel", DictOnlyModel),
        ("SimpleFormatterModel", SimpleFormatterModel),
        ("OrderedFieldsModel", OrderedFieldsModel),
        ("ConstrainedFormatterModel", ConstrainedFormatterModel),
        ("PatternConstraints", PatternConstraints),
        ("UnionTypes", UnionTypes),
        ("OptionalWithDefault", OptionalWithDefault),
        ("Address", Address),
        ("ContactInfo", ContactInfo),
        ("User", User),
        ("Product", Product),
        ("Order", Order),
        ("Location", Location),
        ("Profile", Profile),
        ("ComplexTypes", ComplexTypes),
        ("ConfigModel", ConfigModel),
        ("ComplexOrder", ComplexOrder),
        ("NestedReferences", NestedReferences),
        ("AdvancedFeatures", AdvancedFeatures),
        ("DeepNested", DeepNested),
        ("UnionHeavy", UnionHeavy),
        ("AllOfLike", AllOfLike),
        ("ListAndDict", ListAndDict),
        ("FullFeaturedModel", FullFeaturedModel),
        ("ArrayOfRefsModel", ArrayOfRefsModel),
    ]


@pytest.fixture
def all_simple_models():
    """Registry of simple Pydantic models (single features)."""
    return [
        ("SimpleUser", SimpleUser),
        ("OnlyString", OnlyString),
        ("OnlyInt", OnlyInt),
        ("OnlyFloat", OnlyFloat),
        ("OnlyBool", OnlyBool),
        ("StringFormatEmail", StringFormatEmail),
        ("StringFormatUri", StringFormatUri),
        ("StringPattern", StringPattern),
        ("StringLength", StringLength),
        ("StringConstraints", StringConstraints),
        ("NumericConstraints", NumericConstraints),
        ("ExclusiveMinMax", ExclusiveMinMax),
        ("ArrayOfStrings", ArrayOfStrings),
        ("ArrayMinMaxItems", ArrayMinMaxItems),
        ("ArrayLengthModel", ArrayLengthModel),
        ("ArrayUniqueItems", ArrayUniqueItems),
        ("UniqueListModel", UniqueListModel),
        ("ObjectRequiredOnly", ObjectRequiredOnly),
        ("ObjectWithDefaults", ObjectWithDefaults),
        ("ObjectAdditionalPropsFalse", ObjectAdditionalPropsFalse),
        ("IntEnumModel", IntEnumModel),
        ("LiteralSingle", LiteralSingle),
        ("LiteralUnion", LiteralUnion),
        ("LiteralOnlyModel", LiteralOnlyModel),
        ("ConstModel", ConstModel),
        ("OptionalFields", OptionalFields),
        ("RequiredOptionalModel", RequiredOptionalModel),
        ("SingleNested", SingleNested),
        ("PersonWithAddress", PersonWithAddress),
        ("WithTitleDescription", WithTitleDescription),
        ("WithFieldDescriptions", WithFieldDescriptions),
        ("ModelWithAlias", ModelWithAlias),
        ("EventWithDate", EventWithDate),
        ("DictOnlyModel", DictOnlyModel),
        ("SimpleFormatterModel", SimpleFormatterModel),
        ("OrderedFieldsModel", OrderedFieldsModel),
        ("ConstrainedFormatterModel", ConstrainedFormatterModel),
        ("PatternConstraints", PatternConstraints),
        ("UnionTypes", UnionTypes),
        ("OptionalWithDefault", OptionalWithDefault),
    ]


@pytest.fixture
def all_complex_models():
    """Registry of complex Pydantic models (multiple features)."""
    return [
        ("Address", Address),
        ("ContactInfo", ContactInfo),
        ("User", User),
        ("Product", Product),
        ("Order", Order),
        ("Location", Location),
        ("Profile", Profile),
        ("ComplexTypes", ComplexTypes),
        ("ConfigModel", ConfigModel),
        ("ComplexOrder", ComplexOrder),
        ("NestedReferences", NestedReferences),
        ("AdvancedFeatures", AdvancedFeatures),
        ("DeepNested", DeepNested),
        ("UnionHeavy", UnionHeavy),
        ("AllOfLike", AllOfLike),
        ("ListAndDict", ListAndDict),
        ("FullFeaturedModel", FullFeaturedModel),
        ("ArrayOfRefsModel", ArrayOfRefsModel),
    ]


@pytest.fixture
def all_pydantic_schemas(all_pydantic_models):
    """Registry of all Pydantic model schemas (name, schema_dict)."""
    return [(name, model.model_json_schema()) for name, model in all_pydantic_models]


@pytest.fixture
def all_simple_schemas(all_simple_models):
    """Registry of simple model schemas (name, schema_dict)."""
    return [(name, model.model_json_schema()) for name, model in all_simple_models]


@pytest.fixture
def all_raw_schemas():
    """Registry of all raw JSON schemas (name, schema_dict)."""
    return [
        ("EMPTY_SCHEMA", EMPTY_SCHEMA),
        ("COMPLEX_SCHEMA", COMPLEX_SCHEMA),
        ("DEPENDENCY_SCHEMA", DEPENDENCY_SCHEMA),
        ("CONDITIONAL_SCHEMA", CONDITIONAL_SCHEMA),
        ("PATTERN_PROPERTIES_SCHEMA", PATTERN_PROPERTIES_SCHEMA),
        ("CONTAINS_SCHEMA", CONTAINS_SCHEMA),
        ("UNIQUE_ITEMS_SCHEMA", UNIQUE_ITEMS_SCHEMA),
        ("PROPERTY_NAMES_SCHEMA", PROPERTY_NAMES_SCHEMA),
        ("UNEVALUATED_PROPERTIES_SCHEMA", UNEVALUATED_PROPERTIES_SCHEMA),
        ("ALL_OF_SCHEMA", ALL_OF_SCHEMA),
        ("ONE_OF_SCHEMA", ONE_OF_SCHEMA),
        ("ANY_OF_SCHEMA", ANY_OF_SCHEMA),
        ("CONST_SCHEMA", CONST_SCHEMA),
        ("ADDITIONAL_ITEMS_SCHEMA", ADDITIONAL_ITEMS_SCHEMA),
        ("NOT_SCHEMA", NOT_SCHEMA),
        ("FORMAT_SCHEMA", FORMAT_SCHEMA),
        ("MIN_MAX_ITEMS_SCHEMA", MIN_MAX_ITEMS_SCHEMA),
        ("PREFIX_ITEMS_SCHEMA", PREFIX_ITEMS_SCHEMA),
        ("DEPENDENT_REQUIRED_SCHEMA", DEPENDENT_REQUIRED_SCHEMA),
        ("DEPENDENT_SCHEMAS_SCHEMA", DEPENDENT_SCHEMAS_SCHEMA),
        ("UNEVALUATED_ITEMS_SCHEMA", UNEVALUATED_ITEMS_SCHEMA),
        ("MIN_MAX_CONTAINS_SCHEMA", MIN_MAX_CONTAINS_SCHEMA),
        ("CONTENT_ENCODING_SCHEMA", CONTENT_ENCODING_SCHEMA),
        ("ANCHOR_SCHEMA", ANCHOR_SCHEMA),
        ("DEPRECATED_EXAMPLES_SCHEMA", DEPRECATED_EXAMPLES_SCHEMA),
    ]


@pytest.fixture
def all_schemas(all_pydantic_schemas, all_raw_schemas):
    """Combined registry of all schemas (Pydantic + raw)."""
    return all_pydantic_schemas + all_raw_schemas


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
    """Factory for creating test data and valid instances."""

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

    # Schema + valid instance helpers
    @staticmethod
    def create_simple_schema_with_instance() -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a simple schema with a valid instance."""
        schema = TestDataFactory.create_simple_schema()
        instance = {"name": "John Doe", "age": 30, "email": "john@example.com"}
        return schema, instance

    @staticmethod
    def create_enum_schema_with_instance() -> tuple[dict[str, Any], dict[str, Any]]:
        """Create an enum schema with a valid instance."""
        schema = TestDataFactory.create_enum_schema()
        instance = {"status": "active"}
        return schema, instance

    @staticmethod
    def create_union_schema_with_instance() -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a union schema with a valid instance."""
        schema = TestDataFactory.create_union_schema()
        instance = {"id": 42}
        return schema, instance

    @staticmethod
    def create_array_schema_with_instance() -> tuple[dict[str, Any], dict[str, Any]]:
        """Create an array schema with a valid instance."""
        schema = TestDataFactory.create_array_schema()
        instance = {"items": ["item1", "item2", "item3"]}
        return schema, instance

    @staticmethod
    def create_nested_schema_with_instance() -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a nested schema with a valid instance."""
        schema = TestDataFactory.create_nested_schema()
        instance = {"user": {"name": "Jane Doe", "age": 25}}
        return schema, instance

    @staticmethod
    def create_string_format_schema(format_type: str = "email") -> dict[str, Any]:
        """Create a schema with string format constraint."""
        return {
            "type": "object",
            "properties": {"value": {"type": "string", "format": format_type}},
            "required": ["value"],
        }

    @staticmethod
    def create_array_items_schema(item_type: str = "string") -> dict[str, Any]:
        """Create a schema with array items constraint."""
        return {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": item_type}}},
            "required": ["items"],
        }

    @staticmethod
    def create_numeric_constraints_schema(minimum: int = 0, maximum: int = 100) -> dict[str, Any]:
        """Create a schema with numeric constraints."""
        return {
            "type": "object",
            "properties": {"value": {"type": "integer", "minimum": minimum, "maximum": maximum}},
            "required": ["value"],
        }

    @staticmethod
    def create_string_pattern_schema(pattern: str = r"^[A-Z]{3}-\d{3}$") -> dict[str, Any]:
        """Create a schema with string pattern constraint."""
        return {
            "type": "object",
            "properties": {"code": {"type": "string", "pattern": pattern}},
            "required": ["code"],
        }


@pytest.fixture
def test_data_factory():
    """Fixture for TestDataFactory."""
    return TestDataFactory
