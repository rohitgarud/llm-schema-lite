"""Machine-independent provenance strings for the DSPy adapter benchmark.

Two concerns, both pure and both stdlib-only:

1. **Invocation.** :data:`PROG` is the one documented invocation form, and
   :func:`invocation_command` reconstructs a run's command line from ``sys.argv`` in that
   form -- never ``argv[0]``, which ``python -m`` sets to an absolute ``__main__.py`` path
   that leaks the generating machine's layout into a committed file.
2. **Redaction.** :func:`redact_lm_kwargs` and :func:`redact_url_userinfo` strip
   credential-shaped values out of a run's recorded configuration, so a results file is
   safe to commit by construction.

Stdlib-only and importing nothing from this package, exactly as ``encoding.py`` is: it can
therefore be imported by both ``cli.py`` and ``report.py`` with no cycle, and unit-tested by
``tests/test_bench_dspy_adapters_smoke.py`` -- which is forbidden by its own module contract
from importing ``cli`` -- without widening any import graph.

This module never *applies* either concern. ``cli.py`` calls
:func:`invocation_command`; ``report.RunMeta.__post_init__`` calls the two redactors. Keeping
application out of here is what keeps this module a set of pure functions.
"""

from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PROG: str = "python -m benchmarking.dspy_adapters"
"""The documented invocation form (README.md:69, Makefile's `bench-dspy` target).

Single source of truth: `cli.build_parser` passes this as argparse's `prog=`, so `--help`
output and the recorded provenance line cannot drift apart.
"""


REDACTED: str = "<redacted>"
"""Marker substituted for every credential-shaped value.

A module constant, not a literal, so tests assert against the constant and a reviewer can
positively grep a results file for it. No prior marker convention exists in this repo.
"""


SENSITIVE_KEY_SEGMENTS: frozenset[str] = frozenset(
    {
        "key",
        "keys",
        "apikey",
        "token",  # NOTE: "tokens" is DELIBERATELY absent -- see is_sensitive_key.
        "secret",
        "secrets",
        "auth",
        "authorization",
        "password",
        "passwd",
        "credential",
        "credentials",
        "bearer",
    }
)
"""Whole key segments that mark a value as a credential. Singular `token` only."""


_KEY_SEPARATOR_RE: re.Pattern[str] = re.compile(r"[^a-z0-9]+")
"""Splits a lowercased key into segments: `_`, `-`, `.`, spaces and `/` all separate."""


def invocation_command(argv: Sequence[str] | None = None) -> str:
    """Render the run's invocation as the documented, machine-independent command string.

    `argv[0]` is dropped -- `python -m` sets it to an absolute `__main__.py` path -- and
    replaced by `PROG`; the remaining arguments are appended `shlex.quote`d, so
    `--adapters "json, baml"` round-trips (legal input: `outcomes.resolve_ids` strips
    whitespace). `argv=None` reads `sys.argv`.

    `invocation_command(["/abs/.../__main__.py", "--offline"])`
        -> `"python -m benchmarking.dspy_adapters --offline"`
    """
    argv = list(sys.argv) if argv is None else list(argv)
    rest = argv[1:]
    if not rest:
        return PROG
    return PROG + " " + " ".join(shlex.quote(a) for a in rest)


def is_sensitive_key(key: str) -> bool:
    """Whether `key`, lowercased and split on non-alphanumerics, has a sensitive segment.

    Segment **equality**, never substring matching. `max_tokens` splits to
    `{"max", "tokens"}`, and `tokens` (plural) is not in `SENSITIVE_KEY_SEGMENTS`, so the
    measured `max_tokens=900` floor documented at `config.py:50-52` survives; a substring
    rule would destroy it. `token` singular *is* a credential, so `azure_ad_token` and
    `hf-token` match.

    Known gap, stated rather than closed: a separator-free compound such as `authtoken`
    yields the single segment `authtoken` and does **not** match. Adding a substring
    fallback would re-admit `max_tokens`, which is the case this rule exists to protect.

    The failure asymmetry is deliberate. A false positive costs one provenance value
    (`cache_key` would over-redact, harmlessly); a false negative commits a credential to
    git. The rule is tuned to over-redact at the margin.
    """
    segments = set(_KEY_SEPARATOR_RE.split(key.lower())) - {""}
    return bool(segments & SENSITIVE_KEY_SEGMENTS)


def _redact_value(value: Any) -> Any:
    """Recurse into a non-sensitive value: mappings and list/tuple elements only.

    `str` and `bytes` are `Sequence`s and MUST NOT be walked elementwise -- test
    `list | tuple` explicitly, never `isinstance(value, Sequence)`. A tuple is rebuilt as
    a tuple, a list as a list, so `repr()` in the provenance block is unchanged in shape.
    """
    if isinstance(value, Mapping):
        return redact_lm_kwargs(value)
    if isinstance(value, list | tuple):
        redacted = [_redact_value(item) for item in value]
        return tuple(redacted) if isinstance(value, tuple) else redacted
    return value


def redact_lm_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with every sensitive key's value replaced by `REDACTED`.

    Keys are preserved **verbatim** (original casing) -- the key name is provenance, only
    the value is secret, so `Authorization: <redacted>` still records which setting was in
    play. Matching is on the lowercased key.

    Recurses into nested mappings and into `list`/`tuple` elements, so
    `{"extra_headers": {"Authorization": "Bearer ..."}}` is closed; a sensitive key's whole
    subtree is replaced without descending into it. `json.loads` (config.py:106) bounds
    depth at parse time, so no extra depth guard is needed. Idempotent.
    """
    return {k: (REDACTED if is_sensitive_key(k) else _redact_value(v)) for k, v in kwargs.items()}


def redact_url_userinfo(url: str | None) -> str | None:
    """Replace a `user:pass@` userinfo component with `<redacted>@`.

    `None` and userinfo-free URLs pass through unchanged, so the committed
    `http://localhost:11434/v1` is byte-identical. Idempotent: re-splitting
    `https://<redacted>@host/v1` yields the same string (verified). A string with no
    scheme parses with an empty netloc and passes through untouched.
    """
    if url is None:
        return None
    parts = urlsplit(url)
    if not parts.netloc or "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=f"{REDACTED}@{host}"))
