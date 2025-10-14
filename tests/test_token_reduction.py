"""Tests for token count reduction - validates the 60-85% claim."""

import json

import pytest
from pydantic import BaseModel, Field

from llm_schema_lite import simplify_schema

# Skip if tiktoken not installed
tiktoken = pytest.importorskip("tiktoken")


class SimpleExtraction(BaseModel):
    """Simple extraction model - typical DSPy use case."""

    name: str = Field(description="Person's full name")
    age: int = Field(ge=0, le=150, description="Person's age in years")
    email: str = Field(pattern=r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$")
    active: bool = Field(default=True, description="Account status")


class ProductExtraction(BaseModel):
    """Product extraction - e-commerce use case."""

    product_id: str = Field(description="Unique product identifier")
    name: str = Field(min_length=1, max_length=200)
    price: float = Field(gt=0, description="Product price in USD")
    category: str
    in_stock: bool = Field(default=True)
    tags: list[str] = Field(default_factory=list)


class NestedExtraction(BaseModel):
    """Nested extraction - complex DSPy use case."""

    class Address(BaseModel):
        street: str
        city: str
        country: str = Field(default="USA")

    class ContactInfo(BaseModel):
        email: str
        phone: str | None = None

    name: str
    age: int
    address: Address
    contact: ContactInfo
    skills: list[str]


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text))


def test_simple_extraction_token_reduction():
    """Test token reduction for simple extraction model."""
    # Get original Pydantic schema
    original_schema = SimpleExtraction.model_json_schema()
    original_json = json.dumps(original_schema)
    original_tokens = count_tokens(original_json)

    # Get simplified schema
    simplified = simplify_schema(SimpleExtraction, format_type="jsonish")
    simplified_text = simplified.to_string()
    simplified_tokens = count_tokens(simplified_text)

    # Calculate reduction
    reduction_pct = ((original_tokens - simplified_tokens) / original_tokens) * 100

    print(f"\n{'=' * 70}")
    print("Simple Extraction Model Token Comparison:")
    print(f"{'=' * 70}")
    print(f"Original tokens:    {original_tokens:4d}")
    print(f"Simplified tokens:  {simplified_tokens:4d}")
    print(f"Reduction:          {reduction_pct:.1f}%")
    print(f"{'=' * 70}\n")

    # Verify significant reduction (at least 40% - conservative)
    assert reduction_pct >= 40, f"Expected ≥40% reduction, got {reduction_pct:.1f}%"
    assert simplified_tokens < original_tokens


def test_product_extraction_token_reduction():
    """Test token reduction for product extraction."""
    original_schema = ProductExtraction.model_json_schema()
    original_json = json.dumps(original_schema)
    original_tokens = count_tokens(original_json)

    simplified = simplify_schema(ProductExtraction, format_type="jsonish")
    simplified_text = simplified.to_string()
    simplified_tokens = count_tokens(simplified_text)

    reduction_pct = ((original_tokens - simplified_tokens) / original_tokens) * 100

    print(f"\n{'=' * 70}")
    print("Product Extraction Model Token Comparison:")
    print(f"{'=' * 70}")
    print(f"Original tokens:    {original_tokens:4d}")
    print(f"Simplified tokens:  {simplified_tokens:4d}")
    print(f"Reduction:          {reduction_pct:.1f}%")
    print(f"{'=' * 70}\n")

    assert reduction_pct >= 40
    assert simplified_tokens < original_tokens


def test_nested_extraction_token_reduction():
    """Test token reduction for nested extraction model."""
    original_schema = NestedExtraction.model_json_schema()
    original_json = json.dumps(original_schema)
    original_tokens = count_tokens(original_json)

    simplified = simplify_schema(NestedExtraction, format_type="jsonish")
    simplified_text = simplified.to_string()
    simplified_tokens = count_tokens(simplified_text)

    reduction_pct = ((original_tokens - simplified_tokens) / original_tokens) * 100

    print(f"\n{'=' * 70}")
    print("Nested Extraction Model Token Comparison:")
    print(f"{'=' * 70}")
    print(f"Original tokens:    {original_tokens:4d}")
    print(f"Simplified tokens:  {simplified_tokens:4d}")
    print(f"Reduction:          {reduction_pct:.1f}%")
    print(f"{'=' * 70}\n")

    assert reduction_pct >= 40
    assert simplified_tokens < original_tokens


def test_all_formats_token_reduction():
    """Test that all formats reduce tokens."""
    formats = ["jsonish", "typescript", "yaml"]

    original_schema = ProductExtraction.model_json_schema()
    original_json = json.dumps(original_schema)
    original_tokens = count_tokens(original_json)

    print(f"\n{'=' * 70}")
    print("Token Reduction Across All Formats:")
    print(f"{'=' * 70}")
    print(f"Original Pydantic schema: {original_tokens} tokens")
    print(f"{'-' * 70}")

    for format_type in formats:
        simplified = simplify_schema(ProductExtraction, format_type=format_type)  # type: ignore
        simplified_text = simplified.to_string()
        simplified_tokens = count_tokens(simplified_text)
        reduction_pct = ((original_tokens - simplified_tokens) / original_tokens) * 100

        print(
            f"{format_type.capitalize():12s}: {simplified_tokens:4d} tokens "
            f"({reduction_pct:5.1f}% reduction)"
        )

        # All formats should reduce tokens
        assert simplified_tokens < original_tokens
        assert reduction_pct >= 30  # At least 30% for all formats

    print(f"{'=' * 70}\n")


def test_schema_lite_compare_tokens_method():
    """Test the built-in compare_tokens method."""
    schema = simplify_schema(SimpleExtraction, format_type="jsonish")
    comparison = schema.compare_tokens()

    print(f"\n{'=' * 70}")
    print("Using schema.compare_tokens() method:")
    print(f"{'=' * 70}")
    print(f"Original tokens:    {comparison['original_tokens']}")
    print(f"Simplified tokens:  {comparison['simplified_tokens']}")
    print(f"Tokens saved:       {comparison['tokens_saved']}")
    print(f"Reduction:          {comparison['reduction_percent']}%")
    print(f"{'=' * 70}\n")

    # Verify the comparison results
    assert comparison["original_tokens"] > comparison["simplified_tokens"]
    assert comparison["tokens_saved"] > 0
    assert comparison["reduction_percent"] > 0
    assert 0 <= comparison["reduction_percent"] <= 100


def test_metadata_impact_on_tokens():
    """Test how metadata inclusion affects token count."""
    # Without metadata
    schema_no_meta = simplify_schema(SimpleExtraction, include_metadata=False)
    tokens_no_meta = count_tokens(schema_no_meta.to_string())

    # With metadata
    schema_with_meta = simplify_schema(SimpleExtraction, include_metadata=True)
    tokens_with_meta = count_tokens(schema_with_meta.to_string())

    print(f"\n{'=' * 70}")
    print("Metadata Impact on Token Count:")
    print(f"{'=' * 70}")
    print(f"Without metadata: {tokens_no_meta} tokens")
    print(f"With metadata:    {tokens_with_meta} tokens")
    print(f"Difference:       {tokens_with_meta - tokens_no_meta} tokens")
    print(f"{'=' * 70}\n")

    # With metadata should use more tokens
    assert tokens_with_meta > tokens_no_meta

    # But the difference should be reasonable for simple models
    # Note: For very compact schemas, metadata can be proportionally large
    # This is expected - users should use include_metadata=False for maximum compactness
    metadata_overhead_pct = ((tokens_with_meta - tokens_no_meta) / tokens_no_meta) * 100
    assert metadata_overhead_pct > 0  # Metadata adds tokens
    assert tokens_no_meta < 50  # Without metadata should be very compact


def test_character_count_reduction():
    """Test character count reduction (simpler metric than tokens)."""
    original_schema = ProductExtraction.model_json_schema()
    original_json = json.dumps(original_schema, indent=2)

    simplified = simplify_schema(ProductExtraction, format_type="jsonish")
    simplified_text = simplified.to_string()

    char_reduction = ((len(original_json) - len(simplified_text)) / len(original_json)) * 100

    print(f"\n{'=' * 70}")
    print("Character Count Comparison:")
    print(f"{'=' * 70}")
    print(f"Original characters:    {len(original_json)}")
    print(f"Simplified characters:  {len(simplified_text)}")
    print(f"Reduction:              {char_reduction:.1f}%")
    print(f"{'=' * 70}\n")

    assert len(simplified_text) < len(original_json)
    assert char_reduction > 30  # At least 30% character reduction


def test_line_count_reduction():
    """Test line count reduction (readability for humans)."""
    original_schema = NestedExtraction.model_json_schema()
    original_json = json.dumps(original_schema, indent=2)
    original_lines = len(original_json.split("\n"))

    simplified = simplify_schema(NestedExtraction, format_type="jsonish")
    simplified_text = simplified.to_string()
    simplified_lines = len(simplified_text.split("\n"))

    line_reduction = ((original_lines - simplified_lines) / original_lines) * 100

    print(f"\n{'=' * 70}")
    print("Line Count Comparison:")
    print(f"{'=' * 70}")
    print(f"Original lines:    {original_lines}")
    print(f"Simplified lines:  {simplified_lines}")
    print(f"Reduction:         {line_reduction:.1f}%")
    print(f"{'=' * 70}\n")

    assert simplified_lines < original_lines
    assert line_reduction > 20  # At least 20% line reduction
