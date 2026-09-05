"""Register StructuredOutputAdapter (and its subclasses) with DSPy's StreamListener."""

import functools
import logging
from typing import Any

from .adapters import StructuredOutputAdapter

logger = logging.getLogger(__name__)

_REGISTERED_FLAG = "_llm_schema_lite_streaming_registered"
_JSON_ADAPTER_KEY = "JSONAdapter"


def _streamable_adapter_names() -> list[str]:
    """Return StructuredOutputAdapter's class name plus every subclass name, recursively.

    Evaluated at listener-construction time (inside the __init__ wrapper), not once at
    import, so subclasses defined after this module is imported are still picked up.

    Returns:
        Class names, deduplicated by name and order-preserving, starting with
        "StructuredOutputAdapter" followed by each subclass name in discovery order.
    """
    names: dict[str, None] = {StructuredOutputAdapter.__name__: None}
    stack: list[type[StructuredOutputAdapter]] = [StructuredOutputAdapter]
    while stack:
        for subclass in stack.pop().__subclasses__():
            if subclass.__name__ not in names:
                names[subclass.__name__] = None
                stack.append(subclass)
    return list(names)


def register_streaming_support() -> bool:
    """Teach dspy.streaming.StreamListener to recognise StructuredOutputAdapter by name.

    Wraps StreamListener.__init__ so that every listener constructed after this call gets
    additional adapter_identifiers entries - one per StructuredOutputAdapter subclass name,
    copied from the existing "JSONAdapter" entry on that same listener instance. Idempotent:
    a second call is a no-op. Never raises; degrades to a logged warning and returns False
    if dspy.streaming.streaming_listener.StreamListener cannot be imported or reshaped.

    Returns:
        True if this call performed the registration, False if it was already registered
        or if registration could not be performed (dspy absent, moved, or reshaped).
    """
    try:
        from dspy.streaming.streaming_listener import StreamListener

        original_init = StreamListener.__init__
        if getattr(original_init, _REGISTERED_FLAG, False):
            return False

        @functools.wraps(original_init)
        def __init__(self: StreamListener, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            table = getattr(self, "adapter_identifiers", None)
            if not isinstance(table, dict) or _JSON_ADAPTER_KEY not in table:
                return
            json_entry = table[_JSON_ADAPTER_KEY]
            for name in _streamable_adapter_names():
                table.setdefault(name, dict(json_entry))

        setattr(__init__, _REGISTERED_FLAG, True)
        StreamListener.__init__ = __init__
    except Exception:
        logger.warning(
            "Could not register llm-schema-lite adapters with dspy.streaming.StreamListener; "
            "per-field streaming will fall back to DSPy's unsupported-adapter error.",
            exc_info=True,
        )
        return False
    return True
