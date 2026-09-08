"""lsl-2026-09-05-013 -- rendering the same schema twice on ONE instance is idempotent.

Constructs the formatters DIRECTLY. ``simplify_schema(...).to_string()`` memoizes the
string (``core.py``: ``_string_representation``), so going through the public API can
never call ``transform_schema()`` twice and would make every assertion here vacuous.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_schema_lite import FormatterConfig
from llm_schema_lite.formatters.base import BaseFormatter
from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter
from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter
from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter
from tests.conftest import TreeNode

PREFIX = "# P\n"

FORMATTERS = (YAMLFormatter, TypeScriptFormatter, JSONishFormatter)
# ``TypeScriptFormatter`` never reads ``config.prefix`` on ANY path -- a pre-existing gap,
# orthogonal to this ticket, so it is excluded from the prefix assertion rather than fixed.
PREFIX_AWARE = (YAMLFormatter, JSONishFormatter)


class _Inner(BaseModel):
    a: int


class _Outer(BaseModel):
    """The measured drift shape: a model-valued mapping plus a scalar."""

    m: dict[str, _Inner]
    n: int


SCHEMAS = {
    "mapping_of_model": _Outer.model_json_schema(),
    "recursive": TreeNode.model_json_schema(),
}


@pytest.mark.parametrize("formatter_cls", FORMATTERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("schema_id", sorted(SCHEMAS))
@pytest.mark.parametrize("prefix", ["", PREFIX], ids=["no_prefix", "prefix"])
def test_second_render_equals_first(
    formatter_cls: type[BaseFormatter], schema_id: str, prefix: str
) -> None:
    """AC#2: two consecutive ``transform_schema()`` calls on one instance are equal."""
    formatter = formatter_cls(SCHEMAS[schema_id], config=FormatterConfig(prefix=prefix))

    first = formatter.transform_schema()
    second = formatter.transform_schema()

    assert second == first


@pytest.mark.parametrize("formatter_cls", PREFIX_AWARE, ids=lambda c: c.__name__)
@pytest.mark.parametrize("schema_id", sorted(SCHEMAS))
def test_prefix_survives_the_second_render(
    formatter_cls: type[BaseFormatter], schema_id: str
) -> None:
    """The bug was a *dropped prefix*, so equality alone is not enough.

    An equal-but-prefixless pair would satisfy ``test_second_render_equals_first``. This
    names the prefix, so a future change that drops it from both renders still fails.
    """
    formatter = formatter_cls(SCHEMAS[schema_id], config=FormatterConfig(prefix=PREFIX))

    formatter.transform_schema()
    second = formatter.transform_schema()

    assert second.startswith(PREFIX)
