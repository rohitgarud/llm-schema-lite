"""Tests for schema enrichment (enum metadata injection)."""

from enum import Enum

from pydantic import BaseModel

from llm_schema_lite.schema_enrichment import (
    enrich_schema_with_enum_metadata,
    extract_enum_metadata,
    inject_enum_metadata,
)
from tests.conftest import (
    CategoryWithAliasesOnly,
    ModelWithPriorityMetadata,
    PriorityWithMetadata,
    Role,
    StatusWithDescriptionsOnly,
)

# =============================================================================
# extract_enum_metadata
# =============================================================================


def test_extract_enum_metadata_with_descriptions_and_aliases():
    """Enum with both _descriptions and _aliases returns value-keyed dicts."""
    descs, aliases = extract_enum_metadata(PriorityWithMetadata)
    assert descs["low"] == "Non-urgent, can wait"
    assert descs["critical"] == "Urgent, blocking issue"
    assert aliases["critical"] == ["urgent", "blocker"]


def test_extract_enum_metadata_descriptions_only():
    """Enum with only _descriptions returns descriptions, empty aliases."""
    descs, aliases = extract_enum_metadata(StatusWithDescriptionsOnly)
    assert descs["active"] == "Currently active"
    assert descs["inactive"] == "Currently inactive"
    assert aliases == {}


def test_extract_enum_metadata_aliases_only():
    """Enum with only _aliases returns aliases, empty descriptions."""
    descs, aliases = extract_enum_metadata(CategoryWithAliasesOnly)
    assert descs == {}
    assert aliases["bug"] == ["issue", "error", "defect"]


def test_extract_enum_metadata_no_metadata():
    """Enum without _descriptions or _aliases returns empty dicts."""
    descs, aliases = extract_enum_metadata(Role)
    assert descs == {}
    assert aliases == {}


def test_extract_enum_metadata_invalid_descriptions_ignored():
    """Non-dict _descriptions is ignored gracefully."""
    from enum import Enum as StdEnum

    class BadDesc(str, StdEnum):
        A = "a"
        _descriptions = "not a dict"

    descs, aliases = extract_enum_metadata(BadDesc)
    assert descs == {}
    assert aliases == {}


def test_extract_enum_metadata_invalid_aliases_ignored():
    """Non-dict _aliases or non-list alias values ignored."""
    from enum import Enum as StdEnum

    class BadAlias(str, StdEnum):
        A = "a"
        _aliases = {"A": "not a list"}

    descs, aliases = extract_enum_metadata(BadAlias)
    assert aliases == {}


# =============================================================================
# inject_enum_metadata
# =============================================================================


def test_inject_enum_metadata_adds_extension_fields():
    """inject_enum_metadata mutates node with x-enum-descriptions and x-enum-aliases."""
    node = {"enum": ["low", "high"], "type": "string"}
    inject_enum_metadata(
        node,
        {"low": "Low priority", "high": "High priority"},
        {"high": ["urgent"]},
    )
    assert node["x-enum-descriptions"] == {"low": "Low priority", "high": "High priority"}
    assert node["x-enum-aliases"] == {"high": ["urgent"]}


def test_inject_enum_metadata_descriptions_only():
    """Only descriptions can be injected."""
    node = {"enum": ["a"]}
    inject_enum_metadata(node, {"a": "Desc"}, {})
    assert "x-enum-descriptions" in node
    assert "x-enum-aliases" not in node


def test_inject_enum_metadata_aliases_only():
    """Only aliases can be injected."""
    node = {"enum": ["a"]}
    inject_enum_metadata(node, {}, {"a": ["x"]})
    assert "x-enum-aliases" in node
    assert "x-enum-descriptions" not in node


# =============================================================================
# enrich_schema_with_enum_metadata
# =============================================================================


def test_enrich_schema_injects_into_defs():
    """Enrichment injects metadata into $defs enum entry."""
    schema = ModelWithPriorityMetadata.model_json_schema()
    assert "$defs" in schema
    assert "PriorityWithMetadata" in schema["$defs"]
    def_node = schema["$defs"]["PriorityWithMetadata"]
    assert "x-enum-descriptions" not in def_node
    assert "x-enum-aliases" not in def_node

    enrich_schema_with_enum_metadata(ModelWithPriorityMetadata, schema)
    def_node = schema["$defs"]["PriorityWithMetadata"]
    assert def_node["x-enum-descriptions"]["low"] == "Non-urgent, can wait"
    assert def_node["x-enum-aliases"]["critical"] == ["urgent", "blocker"]


def test_enrich_schema_backward_compatibility_no_metadata():
    """Enums without metadata are unchanged."""

    class ModelWithRole(BaseModel):
        role: Role

    schema = ModelWithRole.model_json_schema()
    enrich_schema_with_enum_metadata(ModelWithRole, schema)
    def_node = schema["$defs"]["Role"]
    assert "enum" in def_node
    assert "x-enum-descriptions" not in def_node
    assert "x-enum-aliases" not in def_node


def test_enrich_schema_dict_input_unchanged():
    """Passing dict as model does not mutate schema (no enrichment)."""
    schema = {"$defs": {"X": {"enum": ["a"], "type": "string"}}}
    enrich_schema_with_enum_metadata({"type": "object"}, schema)
    assert "x-enum-descriptions" not in schema["$defs"]["X"]


def test_enrich_schema_returns_same_schema():
    """enrich_schema_with_enum_metadata returns the schema dict."""
    schema = ModelWithPriorityMetadata.model_json_schema()
    out = enrich_schema_with_enum_metadata(ModelWithPriorityMetadata, schema)
    assert out is schema


# =============================================================================
# Recursive model termination (lsl-2026-09-04-014)
# =============================================================================


def test_enrich_terminates_on_self_referencing_model():
    """A directly self-referencing model enriches without raising RecursionError."""

    class SelfRefNode(BaseModel):
        name: str
        children: "list[SelfRefNode]" = []

    SelfRefNode.model_rebuild()
    schema = SelfRefNode.model_json_schema()
    result = enrich_schema_with_enum_metadata(SelfRefNode, schema)
    assert isinstance(result, dict)


def test_enrich_terminates_on_mutually_recursive_models():
    """A mutually recursive pair enriches without raising RecursionError."""

    class MutualRefA(BaseModel):
        name: str
        b: "MutualRefB | None" = None

    class MutualRefB(BaseModel):
        tag: str
        a: "MutualRefA | None" = None

    MutualRefA.model_rebuild()
    MutualRefB.model_rebuild()
    schema = MutualRefA.model_json_schema()
    result = enrich_schema_with_enum_metadata(MutualRefA, schema)
    assert isinstance(result, dict)


def test_enrich_collects_enum_nested_inside_recursive_model():
    """The visited set must not over-prune: a nested Enum is still collected."""

    class NestedStatus(Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    class RecursiveWithEnum(BaseModel):
        name: str
        status: NestedStatus
        children: "list[RecursiveWithEnum]" = []

    RecursiveWithEnum.model_rebuild()
    schema = RecursiveWithEnum.model_json_schema()
    enrich_schema_with_enum_metadata(RecursiveWithEnum, schema)
    assert "NestedStatus" in schema["$defs"]
    assert "enum" in schema["$defs"]["NestedStatus"]
