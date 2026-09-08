#!/usr/bin/env python3
"""
Detailed analysis of schema simplification issues.
This script examines specific schemas to identify what's going wrong.
"""

import json

from llm_schema_lite import simplify_schema


def analyze_specific_schemas(
    schema_comparison_file: str = "schema_comparison.json", sample_size: int = 25
):
    """Analyze specific schemas to identify issues."""

    # Load the comparison data
    with open(schema_comparison_file, encoding="utf-8") as f:
        comparison_data = json.load(f)

    print(f"🔍 Analyzing {sample_size} schemas for detailed issues...")
    print("=" * 80)

    # Take first N schemas for detailed analysis
    sample_schemas = comparison_data[:sample_size]

    issues_found = {
        "over_simplified": [],
        "lost_structure": [],
        "incorrect_format": [],
        "missing_required": [],
        "type_issues": [],
    }

    for i, item in enumerate(sample_schemas):
        print(f"\n📋 Schema {i + 1}: {item['unique_id']} ({item['split']})")
        print(f"Token reduction: {item['token_reduction_percent']:.1f}%")
        print(f"Success: {item['success']}")

        if not item["success"]:
            print(f"❌ FAILED: {item.get('error', 'Unknown error')}")
            continue

        # Parse the original schema (it's stored as a double-encoded JSON string)
        try:
            # The original_schema is a JSON string containing a JSON string, so we need to
            # parse it twice
            schema_string = json.loads(item["original_schema"])  # First decode
            original_schema = json.loads(schema_string)  # Second decode
        except json.JSONDecodeError as e:
            print(f"❌ Could not parse original schema: {e}")
            continue

        simplified_schema = item["simplified_schema"]

        # Analyze the issues
        issues = analyze_schema_issues(original_schema, simplified_schema, item["unique_id"])

        for issue_type, issue_list in issues.items():
            if issue_list:
                issues_found[issue_type].extend(issue_list)

        # Show a sample of the comparison
        print("\n📄 Original Schema (first 200 chars):")
        print("-" * 50)
        print(item["original_schema"][:200] + "...")

        print("\n✨ Simplified Schema:")
        print("-" * 50)
        print(simplified_schema)

        # Check for specific problems
        if item["token_reduction_percent"] > 95:
            print("⚠️  OVER-SIMPLIFIED: Token reduction > 95% - might be losing important details")
            issues_found["over_simplified"].append(
                {
                    "id": item["unique_id"],
                    "reduction": item["token_reduction_percent"],
                    "original_tokens": item["original_tokens"],
                    "simplified_tokens": item["simplified_tokens"],
                }
            )

        if len(simplified_schema) < 20:
            print("⚠️  TOO SIMPLE: Simplified schema is very short - might be losing structure")
            issues_found["lost_structure"].append(
                {
                    "id": item["unique_id"],
                    "simplified_length": len(simplified_schema),
                    "simplified": simplified_schema,
                }
            )

        # Check if simplified schema looks like a proper schema
        if not looks_like_schema(simplified_schema):
            print("⚠️  INCORRECT FORMAT: Simplified schema doesn't look like a proper schema")
            issues_found["incorrect_format"].append(
                {"id": item["unique_id"], "simplified": simplified_schema}
            )

    # Generate summary report
    print("\n📊 ISSUE ANALYSIS SUMMARY")
    print("=" * 80)

    for issue_type, issues in issues_found.items():
        if issues:
            print(f"\n🔍 {issue_type.upper().replace('_', ' ')} ({len(issues)} cases):")
            for issue in issues[:3]:  # Show first 3 examples
                print(f"  - {issue['id']}: {issue}")

    return issues_found


def analyze_schema_issues(
    original_schema: dict, simplified_schema: str, schema_id: str
) -> dict[str, list]:
    """Analyze specific issues in a schema simplification."""
    issues = {
        "over_simplified": [],
        "lost_structure": [],
        "incorrect_format": [],
        "missing_required": [],
        "type_issues": [],
    }

    # Check if original has required fields
    if "required" in original_schema and original_schema["required"]:
        required_fields = original_schema["required"]
        if not any(field in simplified_schema for field in required_fields):
            issues["missing_required"].append(
                {
                    "id": schema_id,
                    "required_fields": required_fields,
                    "simplified": simplified_schema,
                }
            )

    # Check if original has properties
    if "properties" in original_schema and original_schema["properties"]:
        property_names = list(original_schema["properties"].keys())
        found_properties = sum(1 for prop in property_names if prop in simplified_schema)
        if found_properties < len(property_names) * 0.5:  # Less than 50% of properties found
            issues["lost_structure"].append(
                {
                    "id": schema_id,
                    "original_properties": property_names,
                    "found_properties": found_properties,
                    "total_properties": len(property_names),
                }
            )

    return issues


def looks_like_schema(simplified_schema: str) -> bool:
    """Check if the simplified schema looks like a proper schema."""
    # Basic checks for schema-like structure
    schema_indicators = [
        "{",
        "}",  # Object structure
        ":",  # Key-value pairs
        "string",
        "number",
        "boolean",
        "object",
        "array",  # Type indicators
        "*",  # Required field indicators
        "//",  # Comments
    ]

    return sum(1 for indicator in schema_indicators if indicator in simplified_schema) >= 3


def test_specific_problematic_cases():
    """Test specific cases that are likely to be problematic."""

    print("\n🧪 Testing specific problematic schema patterns...")
    print("=" * 60)

    # Test cases that are likely to cause issues
    test_cases = [
        {
            "name": "Complex nested object",
            "schema": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "address": {
                                "type": "object",
                                "properties": {
                                    "street": {"type": "string"},
                                    "city": {"type": "string"},
                                },
                                "required": ["street", "city"],
                            },
                        },
                        "required": ["name", "age", "address"],
                    }
                },
                "required": ["user"],
            },
        },
        {
            "name": "Array with complex items",
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "value": {"type": "number"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "value"],
                },
            },
        },
        {
            "name": "Schema with $ref",
            "schema": {
                "type": "object",
                "properties": {"data": {"$ref": "#/definitions/UserData"}},
                "definitions": {
                    "UserData": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string", "format": "email"},
                        },
                    }
                },
            },
        },
    ]

    for test_case in test_cases:
        print(f"\n🔬 Testing: {test_case['name']}")
        try:
            simplified = simplify_schema(test_case["schema"], format_type="jsonish")
            print(f"✅ Success: {simplified}")
        except Exception as e:
            print(f"❌ Failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting detailed schema analysis...")
    print("This will examine specific schemas to identify what's going wrong...")
    print()

    # Analyze the comparison data
    issues = analyze_specific_schemas(sample_size=25)

    # Test specific problematic cases
    test_specific_problematic_cases()

    print("\n🎯 ANALYSIS COMPLETE")
    print("=" * 80)
    print("Key findings:")
    print("1. Check if schemas are being over-simplified")
    print("2. Verify that important structure is preserved")
    print("3. Ensure the output format is correct")
    print("4. Look for patterns in failed cases")
