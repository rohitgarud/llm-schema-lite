# benchmarking/jsonschemabench/schema_comparison.py

try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError("datasets not installed") from e

import json
import os
from typing import Any

import tiktoken

from llm_schema_lite import simplify_schema


def is_genuinely_minimal_schema(
    original_schema: dict[str, Any], original_tokens: int, simplified_tokens: int, reduction: float
) -> bool:
    """
    Check if a schema is genuinely minimal (has no meaningful structure to simplify).

    A schema is genuinely minimal if:
    - It has no properties, items, enum, or definitions
    - The original is small and simplified is very small

    This helps distinguish truly minimal schemas from problematic over-simplifications.
    """
    # Not minimal if it got longer (negative reduction)
    if reduction < 0:
        return False

    # Check schema structure
    has_properties = "properties" in original_schema and bool(original_schema.get("properties"))
    has_items = (
        "items" in original_schema
        and isinstance(original_schema.get("items"), dict)
        and bool(original_schema["items"])
    )
    has_enum = "enum" in original_schema and len(original_schema.get("enum", [])) > 0
    has_definitions = (
        "definitions" in original_schema and bool(original_schema.get("definitions"))
    ) or ("$defs" in original_schema and bool(original_schema.get("$defs")))
    has_pattern_properties = "patternProperties" in original_schema and bool(
        original_schema.get("patternProperties")
    )
    has_additional_properties_schema = (
        "additionalProperties" in original_schema
        and isinstance(original_schema.get("additionalProperties"), dict)
        and bool(original_schema.get("additionalProperties"))
    )

    # If schema has structure that should be preserved, it's not genuinely minimal
    if (
        has_properties
        or has_items
        or has_enum
        or has_definitions
        or has_pattern_properties
        or has_additional_properties_schema
    ):
        return False

    # If original is very small (<100 tokens) and simplified is 1-10 tokens,
    # likely genuinely minimal
    if original_tokens < 100 and simplified_tokens <= 10:
        return True

    # If original is small (<200 tokens) and simplified is 1-5 tokens,
    # likely genuinely minimal
    if original_tokens < 200 and simplified_tokens <= 5:
        return True

    return False


def create_schema_comparison_dataset(
    sample_size: int = 500, memory_optimized: bool = True, batch_size: int = 100
) -> list[dict[str, Any]]:
    """Create a dataset comparing original and simplified schemas."""

    # Load the dataset
    dataset = load_dataset("epfl-dlab/JSONSchemaBench")

    comparison_data = []
    token_encoding = tiktoken.encoding_for_model("gpt-4o")

    print(f"🔍 Analyzing {sample_size} schemas for comparison...")
    if memory_optimized:
        print("🧠 Memory optimization enabled - using aggressive filtering and cleanup")
        print(f"📦 Batch processing enabled - saving every {batch_size} schemas")

    processed_count = 0
    skipped_large = 0
    skipped_errors = 0
    batch_count = 0

    for split_name, split_data in dataset.items():
        if processed_count >= sample_size:
            break

        for item in split_data:
            if processed_count >= sample_size:
                break

            schema = item["json_schema"]
            unique_id = item["unique_id"]

            try:
                # Get original schema info
                original_json = json.dumps(schema, indent=2)
                original_tokens = len(token_encoding.encode(original_json))
                original_size = len(original_json)

                # Skip extremely large schemas that would cause unrealistic metrics
                max_tokens = 10000 if memory_optimized else 50000
                if original_tokens > max_tokens:
                    skipped_large += 1
                    if memory_optimized and skipped_large % 50 == 0:  # Only print every 50th skip
                        print(f"⏭️  Skipped {skipped_large} large schemas so far...")
                    elif not memory_optimized:
                        print(
                            f"⏭️  Skipping {split_name}/{unique_id}: "
                            f"Too large ({original_tokens} tokens)"
                        )
                    continue

                # Simplify the schema
                simplified_schema = simplify_schema(schema, format_type="jsonish")
                simplified_json = simplified_schema.to_string()
                simplified_tokens = len(token_encoding.encode(simplified_json))
                simplified_size = len(simplified_json)

                # Calculate metrics
                token_reduction = (
                    ((original_tokens - simplified_tokens) / original_tokens * 100)
                    if original_tokens > 0
                    else 0
                )
                size_reduction = (
                    ((original_size - simplified_size) / original_size * 100)
                    if original_size > 0
                    else 0
                )

                # Flag unrealistic reductions (over-simplified, under-simplified, or negative)
                # Updated threshold: 97% instead of 95% to reduce false positives
                is_unrealistic = (
                    token_reduction > 97  # >97% reduction is suspicious (over-simplified)
                    or token_reduction < 5  # <5% reduction is too low (under-simplified)
                    or token_reduction < 0  # Negative reduction (got longer)
                    or simplified_tokens < 10  # <10 tokens is too simple
                    or (
                        original_tokens > 1000 and simplified_tokens < 50
                    )  # Large schema -> very small result
                )

                # Check if schema is genuinely minimal (to exclude from unrealistic)
                schema_dict = json.loads(schema) if isinstance(schema, str) else schema
                is_genuinely_minimal = is_genuinely_minimal_schema(
                    schema_dict, original_tokens, simplified_tokens, token_reduction
                )

                # Memory-optimized entry (truncate very large schemas)
                comparison_entry = {
                    "unique_id": unique_id,
                    "split": split_name,
                    "original_schema": original_json[:5000] + "..."
                    if memory_optimized and len(original_json) > 5000
                    else original_json,
                    "simplified_schema": simplified_json,
                    "original_tokens": original_tokens,
                    "simplified_tokens": simplified_tokens,
                    "original_size_chars": original_size,
                    "simplified_size_chars": simplified_size,
                    "token_reduction_percent": round(token_reduction, 2),
                    "size_reduction_percent": round(size_reduction, 2),
                    "success": True,
                    "is_unrealistic": is_unrealistic,
                    "is_genuinely_minimal": is_genuinely_minimal,
                }

                comparison_data.append(comparison_entry)
                processed_count += 1

                # Memory cleanup
                if memory_optimized:
                    del original_json, simplified_json, simplified_schema

                # Batch processing and streaming save
                if memory_optimized and processed_count % batch_size == 0 and processed_count > 0:
                    batch_count += 1
                    filename = f"schema_comparison_batch_{batch_count}.json"
                    save_comparison_batch(comparison_data, filename, append=False)
                    print(f"💾 Saved batch {batch_count} with {len(comparison_data)} schemas")
                    comparison_data.clear()  # Clear memory
                    import gc

                    gc.collect()  # Force garbage collection

                # Progress tracking
                if processed_count % 50 == 0:  # Progress update every 50 schemas
                    print(f"📊 Processed {processed_count} schemas...")

                if is_unrealistic:
                    print(
                        f"⚠️  {split_name}/{unique_id}: "
                        f"{token_reduction:.1f}% reduction (UNREALISTIC)"
                    )
                else:
                    print(f"✅ {split_name}/{unique_id}: {token_reduction:.1f}% token reduction")

            except Exception as e:
                skipped_errors += 1
                error_schema = json.dumps(schema, indent=2)
                comparison_entry = {
                    "unique_id": unique_id,
                    "split": split_name,
                    "original_schema": error_schema[:5000] + "..."
                    if memory_optimized and len(error_schema) > 5000
                    else error_schema,
                    "simplified_schema": f"ERROR: {str(e)}",
                    "original_tokens": len(token_encoding.encode(error_schema)),
                    "simplified_tokens": 0,
                    "original_size_chars": len(error_schema),
                    "simplified_size_chars": 0,
                    "token_reduction_percent": 0,
                    "size_reduction_percent": 0,
                    "success": False,
                    "error": str(e),
                    "is_unrealistic": False,
                }
                comparison_data.append(comparison_entry)
                processed_count += 1
                print(f"❌ {split_name}/{unique_id}: {str(e)}")

    # Save final batch if there are remaining schemas
    if memory_optimized and comparison_data:
        batch_count += 1
        filename = f"schema_comparison_batch_{batch_count}.json"
        save_comparison_batch(comparison_data, filename, append=False)
        print(f"💾 Saved final batch {batch_count} with {len(comparison_data)} schemas")

    print(
        f"📈 Summary: Processed {processed_count}, "
        f"Skipped {skipped_large} large, {skipped_errors} errors"
    )
    return comparison_data


def save_comparison_dataset(
    comparison_data: list[dict[str, Any]], filename: str = "schema_comparison.json"
):
    """Save the comparison dataset to a file."""
    print(f"💾 Saving {len(comparison_data)} entries to {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved comparison dataset to {filename}")


def save_comparison_batch(
    batch_data: list[dict[str, Any]], filename: str = "schema_comparison.json", append: bool = True
):
    """Save a batch of comparison data, appending to existing file if needed."""
    mode = "a" if append else "w"
    with open(filename, mode, encoding="utf-8") as f:
        if append and os.path.exists(filename) and os.path.getsize(filename) > 0:
            f.write(",\n")
        json.dump(batch_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved batch of {len(batch_data)} entries to {filename}")


def merge_batch_files(output_filename: str = "schema_comparison_complete.json"):
    """Merge all batch files into a single complete dataset."""
    import glob

    batch_files = sorted(glob.glob("schema_comparison_batch_*.json"))
    if not batch_files:
        print("❌ No batch files found to merge")
        return

    print(f"🔗 Merging {len(batch_files)} batch files...")
    all_data = []

    for batch_file in batch_files:
        print(f"📖 Reading {batch_file}...")
        with open(batch_file, encoding="utf-8") as f:
            batch_data = json.load(f)
            all_data.extend(batch_data)

    # Save merged data
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged {len(all_data)} schemas into {output_filename}")

    # Clean up batch files
    for batch_file in batch_files:
        os.remove(batch_file)
    print(f"🧹 Cleaned up {len(batch_files)} batch files")


def analyze_comparison_results(comparison_data: list[dict[str, Any]]):
    """Analyze the comparison results."""
    successful = [item for item in comparison_data if item["success"]]
    failed = [item for item in comparison_data if not item["success"]]
    unrealistic = [item for item in successful if item.get("is_unrealistic", False)]

    print("\n📊 Comparison Analysis Results")
    print("=" * 50)
    print(f"Total schemas analyzed: {len(comparison_data)}")
    print(f"✅ Successful simplifications: {len(successful)}")
    print(f"❌ Failed simplifications: {len(failed)}")
    print(f"⚠️  Unrealistic reductions: {len(unrealistic)}")

    if successful:
        # Apply updated unrealistic detection logic dynamically
        # Updated threshold: 97% instead of 95% to reduce false positives
        def is_unrealistic_updated(item):
            token_reduction = item.get("token_reduction_percent", 0)
            simplified_tokens = item.get("simplified_tokens", 0)
            original_tokens = item.get("original_tokens", 0)

            return (
                token_reduction > 97  # >97% reduction is suspicious (over-simplified)
                or token_reduction < 5  # <5% reduction is too low (under-simplified)
                or token_reduction < 0  # Negative reduction (got longer)
                or simplified_tokens < 10  # <10 tokens is too simple
                or (
                    original_tokens > 1000 and simplified_tokens < 50
                )  # Large schema -> very small result
            )

        # Filter out genuinely minimal and unrealistic results
        # Genuinely minimal schemas are excluded from unrealistic category
        genuinely_minimal = [item for item in successful if item.get("is_genuinely_minimal", False)]
        unrealistic_updated = [
            item
            for item in successful
            if is_unrealistic_updated(item) and not item.get("is_genuinely_minimal", False)
        ]
        realistic = [
            item
            for item in successful
            if not is_unrealistic_updated(item) or item.get("is_genuinely_minimal", False)
        ]
        if realistic:
            token_reductions = [item["token_reduction_percent"] for item in realistic]
            size_reductions = [item["size_reduction_percent"] for item in realistic]

            print("\n📈 Token Reduction Statistics (Realistic Results Only):")
            print(f"  Average: {sum(token_reductions) / len(token_reductions):.1f}%")
            print(f"  Median: {sorted(token_reductions)[len(token_reductions) // 2]:.1f}%")
            print(f"  Max: {max(token_reductions):.1f}%")
            print(f"  Min: {min(token_reductions):.1f}%")

            print("\n📏 Size Reduction Statistics (Realistic Results Only):")
            print(f"  Average: {sum(size_reductions) / len(size_reductions):.1f}%")
            print(f"  Median: {sorted(size_reductions)[len(size_reductions) // 2]:.1f}%")
            print(f"  Max: {max(size_reductions):.1f}%")
            print(f"  Min: {min(size_reductions):.1f}%")

        # Show genuinely minimal schemas
        if genuinely_minimal:
            print(f"\n✨ Genuinely Minimal Schemas: {len(genuinely_minimal)} cases")
            print("  (Excluded from unrealistic category - these have no structure to simplify)")
            single_token_minimal = [
                item for item in genuinely_minimal if item.get("simplified_tokens", 0) == 1
            ]
            if single_token_minimal:
                print(f"    - Single token: {len(single_token_minimal)} cases")

        # Show unrealistic cases with updated logic
        if unrealistic_updated:
            print(f"\n⚠️  Unrealistic Reductions (Updated Logic): {len(unrealistic_updated)} cases")
            print("  Categories:")

            # Categorize unrealistic cases
            over_simplified = [
                item for item in unrealistic_updated if item.get("token_reduction_percent", 0) > 97
            ]
            under_simplified = [
                item
                for item in unrealistic_updated
                if 0 <= item.get("token_reduction_percent", 0) < 5
            ]
            negative_reduction = [
                item for item in unrealistic_updated if item.get("token_reduction_percent", 0) < 0
            ]
            too_simple = [
                item for item in unrealistic_updated if item.get("simplified_tokens", 0) < 10
            ]

            if over_simplified:
                print(f"    - Over-simplified (>97% reduction): {len(over_simplified)} cases")
            if under_simplified:
                print(f"    - Under-simplified (<5% reduction): {len(under_simplified)} cases")
            if negative_reduction:
                print(f"    - Negative reduction (got longer): {len(negative_reduction)} cases")
            if too_simple:
                print(f"    - Too simple (<10 tokens): {len(too_simple)} cases")

            # Show examples from each category
            print("\n  Examples:")
            for item in unrealistic_updated[:5]:  # Show first 5
                reduction = item.get("token_reduction_percent", 0)
                category = (
                    "over-simplified"
                    if reduction > 95
                    else "under-simplified"
                    if 0 <= reduction < 5
                    else "negative"
                    if reduction < 0
                    else "too-simple"
                )
                print(f"  - {item['unique_id']}: {reduction:.1f}% reduction ({category})")
                print(
                    f"    Original: {item['original_tokens']} tokens → "
                    f"Simplified: {item['simplified_tokens']} tokens"
                )

        # Find problematic cases (negative reductions)
        negative_reductions = [item for item in successful if item["token_reduction_percent"] < 0]
        if negative_reductions:
            print(f"\n⚠️  Schemas that got LONGER after simplification: {len(negative_reductions)}")
            for item in negative_reductions[:3]:  # Show first 3 examples
                print(
                    f"  - {item['unique_id']}: {item['token_reduction_percent']:.1f}% (got longer)"
                )

    if failed:
        print("\n❌ Failed simplifications:")
        error_counts = {}
        for item in failed:
            error = item.get("error", "Unknown error")
            error_counts[error] = error_counts.get(error, 0) + 1

        for error, count in error_counts.items():
            print(f"  - {error}: {count} cases")


def print_sample_comparisons(comparison_data: list[dict[str, Any]], num_samples: int = 3):
    """Print sample comparisons for manual inspection."""
    successful = [item for item in comparison_data if item["success"]]
    # Filter out unrealistic results for sample display
    realistic = [item for item in successful if not item.get("is_unrealistic", False)]

    print("\n🔍 Sample Comparisons (Realistic Results Only):")
    print("=" * 80)

    for i, item in enumerate(realistic[:num_samples]):
        print(f"\n📋 Sample {i + 1}: {item['unique_id']} ({item['split']})")
        print(f"Token reduction: {item['token_reduction_percent']:.1f}%")
        print(f"Size reduction: {item['size_reduction_percent']:.1f}%")
        print(
            f"Original tokens: {item['original_tokens']} → "
            f"Simplified tokens: {item['simplified_tokens']}"
        )

        print("\n📄 Original Schema:")
        print("-" * 40)
        print(
            item["original_schema"][:500] + "..."
            if len(item["original_schema"]) > 500
            else item["original_schema"]
        )

        print("\n✨ Simplified Schema:")
        print("-" * 40)
        print(
            item["simplified_schema"][:500] + "..."
            if len(item["simplified_schema"]) > 500
            else item["simplified_schema"]
        )


if __name__ == "__main__":
    print("🚀 Creating schema comparison dataset...")
    print("This will analyze original vs simplified schemas side by side...")
    print()

    # Create comparison dataset with memory optimization for entire dataset
    comparison_data = create_schema_comparison_dataset(
        sample_size=999999, memory_optimized=True, batch_size=50
    )

    # Merge all batch files if they exist
    merge_batch_files("schema_comparison_complete.json")

    # Load the complete dataset for analysis
    try:
        with open("schema_comparison_complete.json", encoding="utf-8") as f:
            complete_data = json.load(f)
        print(f"📊 Loaded complete dataset with {len(complete_data)} schemas")

        # Analyze results
        analyze_comparison_results(complete_data)

        # Show sample comparisons
        print_sample_comparisons(complete_data, num_samples=3)
    except FileNotFoundError:
        print("⚠️  No complete dataset found, analyzing current batch...")
        # Fallback to current data
        save_comparison_dataset(comparison_data)
        analyze_comparison_results(comparison_data)
        print_sample_comparisons(comparison_data, num_samples=3)

    print("\n🎯 Next steps:")
    print("1. Check the saved 'schema_comparison.json' file for detailed comparisons")
    print("2. Review unrealistic reductions to understand over-simplification cases")
    print("3. Look for patterns in schemas that got longer after simplification")
    print("4. Identify which schema features are causing issues")
    print("5. Consider adjusting the unrealistic detection thresholds if needed")
