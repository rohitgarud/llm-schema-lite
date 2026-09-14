import json
import logging
import signal
import statistics
from types import FrameType
from typing import Any

import tiktoken

from llm_schema_lite import simplify_schema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class SchemaTimeout(Exception):
    """One schema exceeded the per-schema render budget."""


def _on_alarm(signum: int, frame: FrameType | None) -> None:
    """SIGALRM handler: turn the timer into an exception the render loop can catch."""
    raise SchemaTimeout


def analyze_dataset_coverage(
    dataset: list[dict[str, Any]],
    timeout_s: float | None = None,
    ids: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze feature coverage of a given dataset.

    Returns the measurements as well as logging them, so a per-config sweep can aggregate
    without re-deriving them -- the display below and any caller read the same numbers.

    ``timeout_s`` caps each schema individually. Ten schemas in this corpus render for
    20-25 minutes apiece (quadratic string post-processing under inline ``$ref``
    expansion, see the report's section 4), so an unbounded sweep does not finish in a
    sitting. A timeout is counted separately from an exception: the first is "too slow to
    use at prompt time", the second is "the library is broken on this input", and
    collapsing them would hide the distinction the report is built on.

    ponytail: SIGALRM, so this is Unix-only and main-thread-only. That is where the sweep
    runs; use a subprocess pool if it ever needs to be portable or threaded.
    """

    # Handle invalid input
    if dataset is None:
        dataset = []
    elif not isinstance(dataset, list):
        dataset = []

    total_schemas = 0
    supported_schemas = 0
    token_reductions = []
    failures: list[str] = []
    slow_ids: list[str] = []
    encoder = tiktoken.encoding_for_model("gpt-4o")
    if timeout_s:
        signal.signal(signal.SIGALRM, _on_alarm)

    for index, schema in enumerate(dataset):
        schema_id = ids[index] if ids is not None and index < len(ids) else str(index)
        if timeout_s:
            signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            # Test if our formatter can handle this schema
            original_token_count = len(encoder.encode(json.dumps(schema)))
            simplified_schema = simplify_schema(schema, format_type="jsonish")
            simplified_token_count = len(encoder.encode(simplified_schema.to_string()))
            supported_schemas += 1
            token_reductions.append(
                (original_token_count - simplified_token_count) / original_token_count
            )

        except SchemaTimeout:
            logger.error(f"Timed out after {timeout_s}s: {schema_id}")
            slow_ids.append(schema_id)
        except Exception as e:
            logger.error(f"Failed to process schema: {e}")
            failures.append(f"{schema_id}: {type(e).__name__}: {e}")
        finally:
            if timeout_s:
                signal.setitimer(signal.ITIMER_REAL, 0)
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

    return {
        "total_schemas": total_schemas,
        "supported_schemas": supported_schemas,
        "coverage_percentage": coverage_percentage,
        "mean_token_reduction": (
            sum(token_reductions) / len(token_reductions) * 100 if token_reductions else None
        ),
        "median_token_reduction": (
            statistics.median(token_reductions) * 100 if token_reductions else None
        ),
        "max_token_reduction": max(token_reductions) * 100 if token_reductions else None,
        "min_token_reduction": min(token_reductions) * 100 if token_reductions else None,
        "failures": failures,
        "timeouts": len(slow_ids),
        "slow_ids": slow_ids,
    }


# Every slot that holds a subschema directly, or a list of subschemas.
_SUBSCHEMA_SLOTS = (
    "items",
    "prefixItems",
    "additionalItems",
    "unevaluatedItems",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "propertyNames",
    "additionalProperties",
    "unevaluatedProperties",
    "oneOf",
    "anyOf",
    "allOf",
)

# Every slot that holds a ``{name: subschema}`` mapping.
_SUBSCHEMA_MAP_SLOTS = (
    "properties",
    "patternProperties",
    "definitions",
    "$defs",
    "dependencies",
    "dependentSchemas",
)


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

        # Recurse into every slot that can hold a subschema.
        #
        # This was a hand-written subset of those slots, and it omitted `definitions` and
        # `$defs` -- so any keyword living inside a definition was invisible to the count.
        # Measured on `patternProperties`: 416 schemas (4.4%) against the 687 (7.2%) the
        # fixed walk finds, with 934 of the missed occurrences under `definitions`, 19 under `$defs`
        # and 3 under `defs`. Every other keyword in the same table was a lower bound for
        # the same reason, so the fix is the whole slot list rather than the two keys that
        # happened to be noticed.
        for slot in _SUBSCHEMA_SLOTS:
            _recurse(schema_obj.get(slot))
        for slot in _SUBSCHEMA_MAP_SLOTS:
            mapping = schema_obj.get(slot)
            if isinstance(mapping, dict):
                for sub_schema in mapping.values():
                    _recurse(sub_schema)

    def _recurse(value: Any) -> None:
        """Descend one subschema slot, which may hold a schema or a list of schemas.

        A boolean schema (`additionalProperties: true`) and a `dependentRequired`-style
        list of property names are neither, and are skipped.
        """
        if isinstance(value, dict):
            _check_schema_recursive(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _check_schema_recursive(item)

    # Start recursive checking from root schema
    _check_schema_recursive(schema)
    return features


def get_feature_statistics(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """Get detailed feature statistics for a dataset.

    Returns:
        dict: Statistics including feature counts, percentages, and metadata
    """
    feature_counts: dict[str, int] = {}
    total_schemas = 0
    processed_schemas = 0
    schema_feature_map: dict[int, list[str]] = {}  # Track which features each schema has

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

    # Calculate statistics. Annotated because the heterogeneous literal below otherwise
    # infers a union that rejects the indexed assignments that follow.
    stats: dict[str, Any] = {
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
