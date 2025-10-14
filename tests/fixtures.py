"""Test fixtures with various Pydantic models for testing formatters."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, EmailStr, Field, HttpUrl


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
    active: bool = Field(..., description="Is the user currently active?")
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
