"""DSPy integration for llm-schema-lite.

This module provides adapters for integrating llm-schema-lite with DSPy,
enabling token-efficient schema representation in DSPy programs.
"""

from .adapters import OutputMode, PromptLayout, StructuredOutputAdapter
from .streaming import register_streaming_support

register_streaming_support()

__all__ = [
    "StructuredOutputAdapter",
    "OutputMode",
    "PromptLayout",
    "register_streaming_support",
]
