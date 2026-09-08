import json
import logging
from typing import Any

import tiktoken

from llm_schema_lite import simplify_schema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def analyze_dataset_coverage(dataset: list[dict[str, Any]]) -> None:
    """Analyze feature coverage of a given dataset."""

    # Handle invalid input
    if dataset is None:
        dataset = []
    elif not isinstance(dataset, list):
        dataset = []

    total_schemas = 0
    supported_schemas = 0
    token_reductions = []

    for schema in dataset:
        try:
            # Test if our formatter can handle this schema
            original_token_count = len(
                tiktoken.encoding_for_model("gpt-4o").encode(json.dumps(schema))
            )
            simplified_schema = simplify_schema(schema, format_type="jsonish")
            simplified_token_count = len(
                tiktoken.encoding_for_model("gpt-4o").encode(simplified_schema.to_string())
            )
            supported_schemas += 1
            token_reductions.append(
                (original_token_count - simplified_token_count) / original_token_count
            )

        except Exception as e:
            logger.error(f"Failed to process schema: {e}")
        total_schemas += 1

    coverage_percentage = supported_schemas / total_schemas * 100 if total_schemas > 0 else 0

    logger.info("\n📊 Dataset Coverage Analysis")
    logger.info("=" * 50)
    logger.info(
        f"✅ Supported schemas: {supported_schemas}/{total_schemas} ({coverage_percentage:.1f}%)"
    )
    logger.info(
        f"❌ Failed schemas: {total_schemas - supported_schemas}/{total_schemas} ({100 - coverage_percentage:.1f}%)"  # noqa: E501
    )

    logger.info(f"Dataset coverage: {coverage_percentage:.1f}%")

    if token_reductions:
        logger.info(
            f"Average token reduction: {(sum(token_reductions) / len(token_reductions)) * 100:.1f}%"
        )
        logger.info(f"Max token reduction: {max(token_reductions) * 100:.1f}%")
        logger.info(f"Min token reduction: {min(token_reductions) * 100:.1f}%")
    else:
        logger.info("Average token reduction: N/A (no successful schemas)")
        logger.info("Max token reduction: N/A (no successful schemas)")
        logger.info("Min token reduction: N/A (no successful schemas)")


def analyze_schema_features(schema: dict) -> list[str]:
    """Analyze which JSON Schema features are used in a schema (recursively)."""
    features = []

    def _check_schema_recursive(schema_obj: dict) -> None:
        """Recursively check schema for features."""
        if not isinstance(schema_obj, dict):
            return

        feature_checks = [
            # Core features
            ("type", lambda s: "type" in s),
            ("required", lambda s: "required" in s),
            ("properties", lambda s: "properties" in s),
            (
                "items",
                lambda s: "items" in s
                and (isinstance(s.get("items"), dict) or isinstance(s.get("items"), list)),
            ),
            ("additionalProperties", lambda s: "additionalProperties" in s),
            ("enum", lambda s: "enum" in s),
            ("$ref", lambda s: "$ref" in s),
            ("pattern", lambda s: "pattern" in s),
            ("format", lambda s: "format" in s),
            ("const", lambda s: "const" in s),
            # Union/Choice features
            ("oneOf", lambda s: "oneOf" in s),
            ("anyOf", lambda s: "anyOf" in s),
            ("allOf", lambda s: "allOf" in s),
            # Fixed: not should contain a valid schema
            ("not", lambda s: "not" in s and isinstance(s.get("not"), dict)),
            # Fixed: if/then/else should be checked as a group
            ("if", lambda s: "if" in s and "then" in s),  # if is only meaningful with then
            ("then", lambda s: "if" in s and "then" in s),  # then is only meaningful with if
            (
                "else",
                lambda s: "if" in s and "then" in s and "else" in s,
            ),  # else is optional but only meaningful with if/then
            # Array features
            ("contains", lambda s: "contains" in s),
            ("uniqueItems", lambda s: "uniqueItems" in s),
            ("additionalItems", lambda s: "additionalItems" in s),
            # Object features
            ("dependencies", lambda s: "dependencies" in s),
            ("patternProperties", lambda s: "patternProperties" in s),
            ("propertyNames", lambda s: "propertyNames" in s),
            ("unevaluatedProperties", lambda s: "unevaluatedProperties" in s),
            # Validation constraints
            ("multipleOf", lambda s: "multipleOf" in s),
            ("exclusiveMinimum", lambda s: "exclusiveMinimum" in s),
            ("exclusiveMaximum", lambda s: "exclusiveMaximum" in s),
            ("minLength", lambda s: "minLength" in s),
            ("maxLength", lambda s: "maxLength" in s),
            ("minimum", lambda s: "minimum" in s),
            ("maximum", lambda s: "maximum" in s),
            ("minItems", lambda s: "minItems" in s),
            ("maxItems", lambda s: "maxItems" in s),
            ("minProperties", lambda s: "minProperties" in s),
            ("maxProperties", lambda s: "maxProperties" in s),
            # Metadata features
            ("default", lambda s: "default" in s),
            ("description", lambda s: "description" in s),
            ("title", lambda s: "title" in s),
        ]

        for feature_name, check_func in feature_checks:
            if check_func(schema_obj) and feature_name not in features:
                features.append(feature_name)

        # Recursively check nested schemas
        if "properties" in schema_obj:
            for prop_schema in schema_obj["properties"].values():
                _check_schema_recursive(prop_schema)

        # Fixed: Handle both dict and array items
        if "items" in schema_obj:
            if isinstance(schema_obj["items"], dict):
                _check_schema_recursive(schema_obj["items"])
            elif isinstance(schema_obj["items"], list):
                for item_schema in schema_obj["items"]:
                    if isinstance(item_schema, dict):
                        _check_schema_recursive(item_schema)

        if "oneOf" in schema_obj:
            for oneof_schema in schema_obj["oneOf"]:
                _check_schema_recursive(oneof_schema)

        if "anyOf" in schema_obj:
            for anyof_schema in schema_obj["anyOf"]:
                _check_schema_recursive(anyof_schema)

        if "allOf" in schema_obj:
            for allof_schema in schema_obj["allOf"]:
                _check_schema_recursive(allof_schema)

        # Always recurse into if/then/else schemas to detect nested features
        if "if" in schema_obj and isinstance(schema_obj["if"], dict):
            _check_schema_recursive(schema_obj["if"])

        if "then" in schema_obj and isinstance(schema_obj["then"], dict):
            _check_schema_recursive(schema_obj["then"])

        if "else" in schema_obj and isinstance(schema_obj["else"], dict):
            _check_schema_recursive(schema_obj["else"])

        if "contains" in schema_obj and isinstance(schema_obj["contains"], dict):
            _check_schema_recursive(schema_obj["contains"])

        if "patternProperties" in schema_obj and isinstance(schema_obj["patternProperties"], dict):
            for pattern_schema in schema_obj["patternProperties"].values():
                if isinstance(pattern_schema, dict):
                    _check_schema_recursive(pattern_schema)

        # Added: Recursively check not schema
        if "not" in schema_obj and isinstance(schema_obj["not"], dict):
            _check_schema_recursive(schema_obj["not"])

        # Added: Recursively check dependencies
        if "dependencies" in schema_obj and isinstance(schema_obj["dependencies"], dict):
            for dep_value in schema_obj["dependencies"].values():
                if isinstance(dep_value, dict):
                    _check_schema_recursive(dep_value)

        # Added: Recursively check propertyNames
        if "propertyNames" in schema_obj and isinstance(schema_obj["propertyNames"], dict):
            _check_schema_recursive(schema_obj["propertyNames"])

        # Added: Recursively check unevaluatedProperties
        if "unevaluatedProperties" in schema_obj and isinstance(
            schema_obj["unevaluatedProperties"], dict
        ):
            _check_schema_recursive(schema_obj["unevaluatedProperties"])

    # Start recursive checking from root schema
    _check_schema_recursive(schema)
    return features


def get_feature_statistics(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """Get detailed feature statistics for a dataset.

    Returns:
        dict: Statistics including feature counts, percentages, and metadata
    """
    feature_counts = {}
    total_schemas = 0
    processed_schemas = 0
    schema_feature_map = {}  # Track which features each schema has

    for i, schema in enumerate(dataset):
        total_schemas += 1
        try:
            features = analyze_schema_features(schema)
            processed_schemas += 1
            schema_feature_map[i] = features

            # Count each feature
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        except Exception as e:
            logger.error(f"Failed to process schema {i}: {e}")
            schema_feature_map[i] = []

    # Calculate statistics
    stats = {
        "total_schemas": total_schemas,
        "processed_schemas": processed_schemas,
        "success_rate": (processed_schemas / total_schemas) * 100 if total_schemas > 0 else 0,
        "feature_counts": feature_counts,
        "feature_percentages": {},
        "schema_feature_map": schema_feature_map,
        "unique_features": set(feature_counts.keys()),
        "total_unique_features": len(feature_counts),
    }

    # Calculate percentages
    for feature, count in feature_counts.items():
        stats["feature_percentages"][feature] = (
            (count / processed_schemas) * 100 if processed_schemas > 0 else 0
        )

    # Add sorted lists for easy access
    stats["features_by_frequency"] = sorted(
        feature_counts.items(), key=lambda x: x[1], reverse=True
    )
    stats["features_by_name"] = sorted(feature_counts.items())

    return stats


def analyze_schema_feature_coverage(dataset: list[dict[str, Any]]) -> None:
    """Analyze feature coverage of a given dataset and display results."""
    # Get statistics using the reusable function
    stats = get_feature_statistics(dataset)

    # Display statistics
    logger.info("\n📊 Schema Feature Coverage Analysis")
    logger.info("=" * 60)
    logger.info(f"Total schemas processed: {stats['processed_schemas']}/{stats['total_schemas']}")
    logger.info(f"Success rate: {stats['success_rate']:.1f}%")

    if stats["feature_counts"]:
        logger.info("\n🔍 Feature Usage Statistics:")
        logger.info("-" * 40)

        # Sort features by usage count (descending)
        sorted_features = stats["features_by_frequency"]

        for feature, count in sorted_features:
            percentage = stats["feature_percentages"][feature]
            logger.info(f"{feature:20} | {count:4d} schemas ({percentage:5.1f}%)")

        logger.info("-" * 40)
        logger.info(f"{'Total features found':20} | {stats['total_unique_features']:4d}")
        logger.info(
            f"{'Most common feature':20} | {sorted_features[0][0]} ({sorted_features[0][1]} times)"
        )
        logger.info(
            f"{'Least common feature':20} | {sorted_features[-1][0]} "
            f"({sorted_features[-1][1]} times)"
        )
    else:
        logger.info("No features found in any schemas.")
