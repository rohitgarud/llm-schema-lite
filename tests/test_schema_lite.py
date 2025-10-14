"""Basic tests for schema-lite package"""

import llm_schema_lite


def test_version():
    """Test that version is defined"""
    assert hasattr(llm_schema_lite, "__version__")
    assert isinstance(llm_schema_lite.__version__, str)


def test_import():
    """Test that package can be imported"""
    assert llm_schema_lite is not None
