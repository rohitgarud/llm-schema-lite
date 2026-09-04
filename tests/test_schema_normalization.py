"""Tests for the pure, pydantic-free JSON-schema title normalization walker."""

import copy
from typing import Any

import pytest
from pydantic.json_schema import GenerateJsonSchema

from llm_schema_lite.schema_normalization import (
    auto_title_for_key,
    normalize_schema_titles,
)


class _TitleFromName(GenerateJsonSchema):
    """Minimal subclass to reach the bound instance method without generating a schema."""


_GENERATOR = _TitleFromName()


@pytest.mark.parametrize(
    "key",
    [
        "street",
        "postal_code",
        "productId",
        "product_id",
        "URL",
        "user_ID",
        "firstName",
    ],
)
def test_auto_title_for_key_table(key: str) -> None:
    assert auto_title_for_key(key) == _GENERATOR.get_title_from_name(key)


def test_normalize_schema_titles_does_not_mutate_input() -> None:
    original: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "title": "Name"},
        },
    }
    before = copy.deepcopy(original)

    result = normalize_schema_titles(original)

    assert original == before
    assert "title" in original["properties"]["name"]
    assert "title" not in result["properties"]["name"]


def test_normalize_schema_titles_raw_dict_input() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "street": {"type": "string", "title": "Street"},
            "zip": {"type": "string", "title": "Postal Code"},
        },
    }

    result = normalize_schema_titles(schema)

    assert "title" not in result["properties"]["street"]
    assert result["properties"]["zip"]["title"] == "Postal Code"


def test_normalize_schema_titles_root_r2() -> None:
    dropped = normalize_schema_titles(
        {
            "type": "object",
            "title": "SimpleFormatterModel",
            "description": "Simple model.",
            "properties": {},
        }
    )
    assert "title" not in dropped

    kept_non_identifier = normalize_schema_titles(
        {
            "type": "object",
            "title": "User Profile",
            "description": "A user profile model",
            "properties": {},
        }
    )
    assert kept_non_identifier["title"] == "User Profile"

    kept_no_description = normalize_schema_titles(
        {
            "type": "object",
            "title": "SimpleFormatterModel",
            "properties": {},
        }
    )
    assert kept_no_description["title"] == "SimpleFormatterModel"


def test_normalize_schema_titles_defs_class_title_untouched() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "$defs": {
            "Role": {
                "title": "Role",
                "enum": ["admin", "user"],
            }
        },
    }

    result = normalize_schema_titles(schema)

    assert result["$defs"]["Role"]["title"] == "Role"
