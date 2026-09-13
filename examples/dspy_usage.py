"""Use llm-schema-lite's compact schemas inside a DSPy program.

Runs offline: it configures the adapter, prints the system prompt the adapter
would send, and parses a hand-written reply. No LM is ever called, so this
script needs no API key and no local model server.

Requires the dspy extra:  pip install "llm-schema-lite[dspy]"
"""

import dspy
from pydantic import BaseModel, Field

from llm_schema_lite import ParseConfig
from llm_schema_lite.dspy_integration import OutputMode, StructuredOutputAdapter


class Contact(BaseModel):
    """A person pulled out of free text."""

    name: str = Field(description="Full name as written")
    email: str | None = None
    age: int | None = Field(default=None, ge=0, le=120)


class ExtractContact(dspy.Signature):
    """Extract the contact described in the text."""

    text: str = dspy.InputField()
    contact: Contact = dspy.OutputField()


def main() -> None:
    # JSONISH is the default: a JSON object described by the compact schema.
    # Passing a ParseConfig turns on the rescue tier; without it a reply that
    # misses the annotation raises instead, matching plain DSPy behaviour.
    adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH, parse_config=ParseConfig())
    dspy.configure(adapter=adapter)

    print("=== What the adapter puts in the system prompt ===")
    print(adapter.format_field_structure(ExtractContact))

    # A reply a small model might actually produce: the required marker leaks
    # through, and `age` arrives as a string. Both are handled by the rescue
    # tier above; drop the ParseConfig and this same reply raises.
    reply = '{"contact": {"name*": "Ada Lovelace", "email": "ada@example.com", "age": "36"}}'

    print("\n=== Parsed back into the Pydantic model ===")
    print(adapter.parse(ExtractContact, reply))


if __name__ == "__main__":
    main()
