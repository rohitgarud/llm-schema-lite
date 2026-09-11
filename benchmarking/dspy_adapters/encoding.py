"""Locate and seed an offline-usable ``cl100k_base`` tiktoken cache.

Stdlib-only, quarantined exactly as ``config.py`` is quarantined for
``LSL_BENCH_*``: no ``tiktoken``, no ``dspy``, no ``requests``, no relative
imports, and no ``os.environ`` access at module scope -- every read and
write happens inside ``seed_tiktoken_cache``. This is what keeps
``__init__.py``'s import-time "never touches ``os.environ`` or the
network" claim true: ``runner.py`` imports only the ``ENCODING_NAME``
constant, and ``cli.py`` calls ``seed_tiktoken_cache`` lazily, inside
``_run_offline``.

``tiktoken`` downloads the ``cl100k_base`` BPE table over HTTPS on a cold
cache (``tiktoken/load.py``). This module finds a copy that already ships
inside an installed dependency (today: litellm's bundled tokenizer
directory) and points ``TIKTOKEN_CACHE_DIR``/``CUSTOM_TIKTOKEN_CACHE_DIR``
at it, without ever importing ``tiktoken`` or the dependency itself, and
without overriding a cache directory that already works.

``tests/conftest.py`` loads this file by path (never as a package import, which would
run the ``benchmarking.dspy_adapters`` ``__init__`` and pull in ``dspy``), so it must
stay stdlib-only with no relative imports.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable, MutableMapping
from importlib.util import find_spec
from pathlib import Path

ENCODING_NAME = "cl100k_base"
CL100K_BLOBPATH = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
TIKTOKEN_CACHE_ENV = "TIKTOKEN_CACHE_DIR"
CUSTOM_TIKTOKEN_CACHE_ENV = "CUSTOM_TIKTOKEN_CACHE_DIR"
DATA_GYM_CACHE_ENV = "DATA_GYM_CACHE_DIR"


def cache_key(blobpath: str = CL100K_BLOBPATH) -> str:
    """Return tiktoken's cache filename for `blobpath`: `sha1(blobpath)` (load.py:52)."""
    return hashlib.sha1(blobpath.encode()).hexdigest()


def seed_tiktoken_cache(
    *,
    environ: MutableMapping[str, str] | None = None,
    candidates: Iterable[Path] | None = None,
) -> str | None:
    """Make the `cl100k_base` blob loadable offline without overriding a cache that works.

    Returns the cache directory tiktoken will use, or `None` when no cached blob can
    be found anywhere. `environ` defaults to `os.environ` and is the only thing this
    module ever mutates; `candidates` defaults to litellm's bundled tokenizer directory.
    Both are injectable so the contract can be tested against a plain `dict` and a
    `tmp_path`, with no real environment write and no dependency on litellm being
    installed. No network, no `tiktoken` import, no BPE parsing -- deliberately cheap
    enough for a `pytest_configure` hook.

    Order of operations:
      1. `TIKTOKEN_CACHE_DIR == ""` -> return `None`. The caller explicitly disabled
         tiktoken's cache, i.e. asked for a download every time; honour that.
      2. The blob is already where tiktoken would look -> return that directory
         unchanged. The lookup replicates `tiktoken/load.py:36-50`:
         `TIKTOKEN_CACHE_DIR` > `DATA_GYM_CACHE_DIR` > `<tempdir>/data-gym-cache`, and an
         empty directory means caching is disabled. A working cache -- the user's, or one
         a previous call seeded -- is never overridden, which is also what makes this
         function idempotent.
      3. Otherwise take the first candidate that actually holds the sha1-named blob --
         never trusting a directory blindly, which is what makes the undeclared litellm
         dependency safe to use -- and set **both** `TIKTOKEN_CACHE_DIR` and
         `CUSTOM_TIKTOKEN_CACHE_DIR` to it. The second is load-bearing: `import litellm`
         later does
         `os.environ["TIKTOKEN_CACHE_DIR"] = os.getenv("CUSTOM_TIKTOKEN_CACHE_DIR", <own dir>)`,
         so setting it *pins* our choice instead of racing litellm's rewrite.
      4. No candidate holds it -> return `None`, leaving the environment untouched.
    """
    env = os.environ if environ is None else environ
    if env.get(TIKTOKEN_CACHE_ENV) == "":
        return None

    key = cache_key()
    directory = env.get(
        TIKTOKEN_CACHE_ENV,
        env.get(DATA_GYM_CACHE_ENV, str(Path(tempfile.gettempdir()) / "data-gym-cache")),
    )
    if directory and os.path.isfile(os.path.join(directory, key)):
        return directory

    if candidates is None:
        # `find_spec` does **not** import litellm, so it does not trigger litellm's own
        # unconditional `TIKTOKEN_CACHE_DIR` rewrite (litellm_core_utils/default_encoding.py
        # :18-20). No candidate when litellm is absent or `find_spec` raises.
        try:
            spec = find_spec("litellm")
        except (ImportError, ValueError):
            spec = None
        candidates = (
            [Path(spec.origin).parent / "litellm_core_utils" / "tokenizers"]
            if spec is not None and spec.origin is not None
            else []
        )
    for candidate in candidates:
        if (Path(candidate) / key).is_file():
            # Safety: `import litellm` calls `tiktoken.get_encoding("cl100k_base")` at
            # *import* time (litellm_core_utils/default_encoding.py:23). A
            # CUSTOM_TIKTOKEN_CACHE_DIR pointing at a directory without the blob would make
            # `import litellm` itself fail. The `is_file()` check above has already
            # confirmed this directory holds the sha1-named blob -- that verification is
            # not optional.
            env[TIKTOKEN_CACHE_ENV] = env[CUSTOM_TIKTOKEN_CACHE_ENV] = str(candidate)
            return str(candidate)
    return None
