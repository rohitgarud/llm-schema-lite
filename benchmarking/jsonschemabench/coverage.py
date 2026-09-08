# Analyze feature coverage of the JSONSchemaBench dataset

import json
import logging

try:
    from datasets import load_dataset

except ImportError as e:
    raise ImportError("datasets not installed") from e


from benchmarking.base import analyze_dataset_coverage, analyze_schema_feature_coverage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def analyze_coverage_jsonschemabench():
    """Analyze feature coverage of the JSONSchemaBench dataset."""
    # Load the dataset
    dataset = load_dataset("epfl-dlab/JSONSchemaBench")

    combined_dataset = []
    for _, split_data in dataset.items():
        for item in split_data:
            # Parse JSON string to Python dict
            try:
                schema_dict = json.loads(item["json_schema"])
                combined_dataset.append(schema_dict)
            except json.JSONDecodeError as e:
                logger.warning(f"Warning: Failed to parse schema: {e}")
                continue

    analyze_dataset_coverage(combined_dataset)
    analyze_schema_feature_coverage(combined_dataset)


if __name__ == "__main__":
    logger.info("🚀 Starting JSONSchemaBench feature coverage analysis...")
    logger.info("This may take a few minutes to download and process the dataset...")
    logger.info("")

    analyze_coverage_jsonschemabench()
