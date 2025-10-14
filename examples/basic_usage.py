"""Basic usage examples for schema-lite."""

from pydantic import BaseModel, Field

from llm_schema_lite import simplify_schema


# Example 1: Simple model
class User(BaseModel):
    """A simple user model."""

    name: str
    age: int
    email: str


# Example 2: Model with metadata
class Product(BaseModel):
    """A product with validation constraints."""

    name: str = Field(..., description="Product name", min_length=1, max_length=100)
    price: float = Field(..., description="Product price", ge=0)
    quantity: int = Field(default=0, description="Available quantity", ge=0)
    tags: list[str] = Field(default_factory=list, description="Product tags")


# Example 3: Nested model
class Address(BaseModel):
    """An address."""

    street: str
    city: str
    zipcode: str


class Customer(BaseModel):
    """A customer with nested address."""

    name: str
    email: str
    address: Address


def main():
    """Run basic usage examples."""
    print("=" * 80)
    print("Schema-Lite Basic Usage Examples")
    print("=" * 80)

    # Example 1: Simple schema (JSONish format)
    print("\n1. Simple User Schema (JSONish Format):")
    print("-" * 80)
    user_schema = simplify_schema(User)
    print(user_schema.to_string())

    # Example 2: TypeScript format
    print("\n2. Simple User Schema (TypeScript Format):")
    print("-" * 80)
    user_ts = simplify_schema(User, format_type="typescript")
    print(user_ts.to_string())

    # Example 3: YAML format
    print("\n3. Simple User Schema (YAML Format):")
    print("-" * 80)
    user_yaml = simplify_schema(User, format_type="yaml")
    print(user_yaml.to_string())

    # Example 4: Schema with metadata (JSONish)
    print("\n4. Product Schema (JSONish with metadata):")
    print("-" * 80)
    product_schema = simplify_schema(Product, include_metadata=True)
    print(product_schema.to_string())

    # Example 5: Schema with metadata (TypeScript)
    print("\n5. Product Schema (TypeScript with metadata):")
    print("-" * 80)
    product_ts = simplify_schema(Product, include_metadata=True, format_type="typescript")
    print(product_ts.to_string())

    # Example 6: Schema with metadata (YAML)
    print("\n6. Product Schema (YAML with metadata):")
    print("-" * 80)
    product_yaml = simplify_schema(Product, include_metadata=True, format_type="yaml")
    print(product_yaml.to_string())

    # Example 7: Schema without metadata
    print("\n7. Product Schema (without metadata):")
    print("-" * 80)
    product_schema_no_meta = simplify_schema(Product, include_metadata=False)
    print(product_schema_no_meta.to_string())

    # Example 8: Nested schema (all formats)
    print("\n8. Nested Customer Schema (JSONish):")
    print("-" * 80)
    customer_schema = simplify_schema(Customer)
    print(customer_schema.to_string())

    print("\n9. Nested Customer Schema (TypeScript):")
    print("-" * 80)
    customer_ts = simplify_schema(Customer, format_type="typescript")
    print(customer_ts.to_string())

    print("\n10. Nested Customer Schema (YAML):")
    print("-" * 80)
    customer_yaml = simplify_schema(Customer, format_type="yaml")
    print(customer_yaml.to_string())

    # Example 11: JSON output
    print("\n11. JSON Output:")
    print("-" * 80)
    print(user_schema.to_json(indent=2))

    # Example 12: Dictionary output
    print("\n12. Dictionary Output:")
    print("-" * 80)
    print(user_schema.to_dict())

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
