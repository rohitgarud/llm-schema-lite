"""Shared test fixtures and configuration for llm-schema-lite tests."""

import contextlib
import importlib.util
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, EmailStr, Field, HttpUrl

# ============================================================================
# Offline tiktoken cache seeding (see thoughts/tasks/lsl-2026-09-05-008-*)
# ============================================================================

_ENCODING_PY = Path(__file__).resolve().parents[1] / "benchmarking/dspy_adapters/encoding.py"


def pytest_configure(config: pytest.Config) -> None:
    """Make tiktoken's `cl100k_base` table loadable offline, before anything is imported.

    Runs once per pytest process, before collection, so every test module, every doc
    block, and the `examples/` subprocess (which inherits `os.environ`) sees the same
    cache directory. Deliberately a hook and not a session fixture -- see the design
    note in the ticket folder.

    `encoding.py` is loaded by file path, never imported: `benchmarking` is not an
    installed package, and `import benchmarking.dspy_adapters.encoding` would execute the
    package `__init__`, pulling `dspy` (an optional extra) into every pytest session
    including the pre-commit `pytest -x` hook. The module is stdlib-only with no relative
    imports, so it loads standalone. Fails closed: any error leaves the environment alone.
    """
    with contextlib.suppress(Exception):
        spec = importlib.util.spec_from_file_location("_lsl_bench_encoding", _ENCODING_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.seed_tiktoken_cache()


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


# Enums with metadata (_descriptions, _aliases) for enhanced enum tests.
# Assign _descriptions/_aliases after class body so they are not str-Enum-coerced to string.
class PriorityWithMetadata(str, Enum):
    """Issue priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


PriorityWithMetadata._descriptions = {
    "LOW": "Non-urgent, can wait",
    "MEDIUM": "Normal priority",
    "HIGH": "Needs attention soon",
    "CRITICAL": "Urgent, blocking issue",
}
PriorityWithMetadata._aliases = {
    "CRITICAL": ["urgent", "blocker"],
}


class StatusWithDescriptionsOnly(str, Enum):
    """Status with only descriptions."""

    ACTIVE = "active"
    INACTIVE = "inactive"


StatusWithDescriptionsOnly._descriptions = {
    "ACTIVE": "Currently active",
    "INACTIVE": "Currently inactive",
}


class CategoryWithAliasesOnly(str, Enum):
    """Category with only aliases."""

    BUG = "bug"
    FEATURE = "feature"


CategoryWithAliasesOnly._aliases = {
    "BUG": ["issue", "error", "defect"],
}


class ModelWithPriorityMetadata(BaseModel):
    """Model with enum that has descriptions and aliases."""

    priority: PriorityWithMetadata


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


# Additional literal type models
class IntLiterals(BaseModel):
    """Model with integer literals."""

    priority: Literal[1, 2, 3, 4, 5]


class BoolLiterals(BaseModel):
    """Model with boolean literals."""

    flag: Literal[True, False]


class MixedTypeLiterals(BaseModel):
    """Model with mixed type literals (string, int, bool)."""

    status: Literal["active", "inactive"]
    level: Literal[1, 2, 3]
    enabled: Literal[True, False]


class SingleConstInt(BaseModel):
    """Model with single integer const."""

    version: Literal[1]


class IssueClassification(BaseModel):
    """Model for issue classification with multiple literal fields."""

    category: Literal["bug", "feature", "question"]
    priority: Literal[1, 2, 3, 4, 5]


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


# Recursive model fixtures (lsl-2026-09-04-014)
class TreeNode(BaseModel):
    """Directly self-referencing model (recursive list)."""

    name: str
    children: list["TreeNode"] = []


class MutualA(BaseModel):
    """Mutually recursive pair: A -> B -> A."""

    name: str
    b: "MutualB | None" = None


class MutualB(BaseModel):
    """Mutually recursive pair: B -> A -> B."""

    tag: str
    a: "MutualA | None" = None


class OptionalTree(BaseModel):
    """Self-referencing model through Optional."""

    value: str
    left: "OptionalTree | None" = None
    right: "OptionalTree | None" = None


TreeNode.model_rebuild()
MutualA.model_rebuild()
MutualB.model_rebuild()
OptionalTree.model_rebuild()


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
# Container Types (lsl-2026-09-04-015)
# ============================================================================


class Color(str, Enum):
    """Enum used as a dict-key type (propertyNames) in the container-types Root fixture."""

    RED = "red"
    GREEN = "green"


class SubModel(BaseModel):
    """Minimal nested model used as a dict-value type in the container-types Root fixture."""

    a: int
    b: str


class Strict(BaseModel):
    """Minimal `extra: forbid` model nested inside the container-types Root fixture."""

    model_config = {"extra": "forbid"}
    s: str


class Inner(BaseModel):
    """Nested model exercising dict/tuple fields one level below Root."""

    d: dict[str, int]
    t: tuple[int, str]


class Root(BaseModel):
    """Kitchen-sink fixture for lsl-2026-09-04-015 (dict/tuple/set/Any container rendering).

    Fields verbatim from the approved design (2026-09-04-design-discussion-v2.md 5.1).
    """

    extra: dict[str, int]
    dict_of_models: dict[str, SubModel]
    by_color: dict[Color, int]
    pair: tuple[int, str]
    var_tuple: tuple[int, ...]
    tags: set[str]
    anything: Any
    described: Any = Field(..., description="free form")
    opt_any: Any | None = None
    any_list: list[Any]
    opt_extra: dict[str, int] | None = None
    inner: Inner
    strict: Strict


# ============================================================================
# Closed-world markers on nested blocks (lsl-2026-09-04-006, R1)
# ============================================================================
# ``Root.strict`` only covers the required, non-nullable case. R1 also has to hold for a
# nullable nested block and for a list-wrapped one, and an ``extra="allow"`` sibling has to
# stay unmarked. These three fixtures give each of those a home.


class StrictSub(BaseModel):
    """``extra="forbid"`` nested model: its block must carry ``no additional properties``."""

    model_config = {"extra": "forbid"}
    s: str


class OpenSub(BaseModel):
    """``extra="allow"`` nested model: its block must carry NO closed-world note."""

    model_config = {"extra": "allow"}
    s: str


class R1Model(BaseModel):
    """R1 matrix: a closed-world nested block required, nullable, list-wrapped, and open."""

    strict: StrictSub
    opt: StrictSub | None = None
    many: list[StrictSub] = []
    open_one: OpenSub


# ============================================================================
# Test Data and Schemas
# ============================================================================

# Empty schema for testing
EMPTY_SCHEMA = {"type": "object"}

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

# Schema with prefixItems (Draft 2020-12)
PREFIX_ITEMS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "array",
    "prefixItems": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}],
    "items": {"type": "string"},
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
def deep_nested_model():
    """Fixture for DeepNested model."""
    return DeepNested


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
        ("TreeNode", TreeNode),
        ("MutualA", MutualA),
        ("OptionalTree", OptionalTree),
        ("UnionHeavy", UnionHeavy),
        ("AllOfLike", AllOfLike),
        ("ListAndDict", ListAndDict),
        ("FullFeaturedModel", FullFeaturedModel),
        ("ArrayOfRefsModel", ArrayOfRefsModel),
    ]


# ============================================================================
# Formatter Test Data
# ============================================================================


@pytest.fixture(params=["jsonish", "typescript", "yaml"])
def format_type(request):
    """Fixture providing all format types for parameterized tests."""
    return request.param


@pytest.fixture
def patient_model() -> type[BaseModel]:
    class Role(str, Enum):
        ADMIN = "admin"
        USER = "user"

    class Address(BaseModel):
        """A postal address."""

        street: str = Field(..., description="Street line", max_length=80)
        zip_code: str = Field(..., pattern=r"^\d{5}$", description="ZIP")

    class Patient(BaseModel):
        """A patient record."""

        name: str = Field(..., description="Full name", pattern=r"^[A-Za-z ]+$", max_length=50)
        age: int = Field(..., description="Age in years", ge=0, le=130)
        role: Role = Field(default=Role.USER, description="Role of the user")
        tags: list[str] = Field(default_factory=list, description="Tags", max_length=5)
        address: Address = Field(..., description="Home address")
        nickname: str | None = Field(default=None, description="Optional nickname")

    return Patient


class HashInPatternModel(BaseModel):
    """Regex patterns containing ``#`` (lsl-2026-09-05-006).

    Every field carries a SECOND metadata part on purpose: with pattern/format now owned
    by the YAML type token, a pattern-only field mints no deferred marker at all and would
    never enter ``_hoist_deferred_line``.
    """

    tag: str = Field(..., pattern=r"^#[0-9a-f]{6}$", description="Hex colour")
    shade: str = Field(default="#ffffff", pattern=r"^#[0-9a-f]{3,6}$")
    spaced: str = Field(..., pattern=r"^a #b$", description="Whitespace before the hash")


class MultiLineDescriptionInner(BaseModel):
    """Inner model for the nested-block and sequence-item multi-line cases."""

    step: str = Field(..., description="first step\nsecond step")


class MultiLineDescriptionModel(BaseModel):
    """Per-field descriptions containing real newlines (lsl-2026-09-05-006)."""

    summary: str = Field(..., description="line one\nline two")
    nested: MultiLineDescriptionInner
    items: list[MultiLineDescriptionInner]
