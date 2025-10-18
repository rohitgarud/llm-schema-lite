"""DSPy integration for llm-schema-lite.

This module provides adapters for integrating llm-schema-lite with DSPy,
enabling token-efficient schema representation in DSPy programs.
"""

from .adapters import OutputMode, StructuredOutputAdapter

__all__ = ["StructuredOutputAdapter", "OutputMode"]
