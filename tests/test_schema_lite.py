"""Basic tests for schema-lite package"""

import schema_lite


def test_version():
    """Test that version is defined"""
    assert hasattr(schema_lite, "__version__")
    assert isinstance(schema_lite.__version__, str)


def test_import():
    """Test that package can be imported"""
    assert schema_lite is not None
