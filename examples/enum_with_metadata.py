"""Example: enums with _descriptions and _aliases for formatted output and validation."""

from enum import Enum

from pydantic import BaseModel

from llm_schema_lite import simplify_schema, validate


# Define an enum with per-value descriptions and aliases.
# Assign _descriptions and _aliases after the class body so they are not
# coerced to strings (str Enum stores all values as strings).
class Priority(str, Enum):
    """Issue priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


Priority._descriptions = {
    "LOW": "Non-urgent, can wait",
    "MEDIUM": "Normal priority",
    "HIGH": "Needs attention soon",
    "CRITICAL": "Urgent, blocking issue",
}
Priority._aliases = {
    "CRITICAL": ["urgent", "blocker"],
}


class Issue(BaseModel):
    """An issue with a priority that has descriptions and aliases."""

    title: str
    priority: Priority


def main():
    print("=" * 60)
    print("Enum with descriptions and aliases")
    print("=" * 60)

    # JSONish format shows OPTIONS with descriptions
    jsonish = simplify_schema(Issue, format_type="jsonish").to_string()
    print("\n--- JSONish ---\n")
    print(jsonish)

    # YAML format
    yaml_out = simplify_schema(Issue, format_type="yaml").to_string()
    print("\n--- YAML ---\n")
    print(yaml_out[:500] + "..." if len(yaml_out) > 500 else yaml_out)

    # Validation: aliases are accepted and normalized
    print("\n--- Validation with alias ---\n")
    ok, errs = validate(Issue, '{"title": "Bug", "priority": "urgent"}', mode="json")
    print(f"Input priority='urgent' (alias for 'critical'): valid={ok}, errors={errs}")

    ok2, errs2 = validate(Issue, '{"title": "Bug", "priority": "blocker"}', mode="json")
    print(f"Input priority='blocker' (alias for 'critical'): valid={ok2}, errors={errs2}")

    ok3, errs3 = validate(Issue, '{"title": "Bug", "priority": "unknown"}', mode="json")
    print(f"Input priority='unknown' (invalid): valid={ok3}, errors={errs3}")


if __name__ == "__main__":
    main()
