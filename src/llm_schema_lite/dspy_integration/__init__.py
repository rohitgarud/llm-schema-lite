"""DSPy integration for llm-schema-lite.

This module provides adapters for integrating llm-schema-lite with DSPy,
enabling token-efficient schema representation in DSPy programs.
"""

from .adapters import JevAdapter, JevLM, OutputMode, PromptLayout, SemIfLM, StructuredOutputAdapter
from .streaming import register_streaming_support

register_streaming_support()

__all__ = [
    "JevAdapter",
    "JevLM",
    "SemIfLM",
    "StructuredOutputAdapter",
    "OutputMode",
    "PromptLayout",
    "register_streaming_support",
]
