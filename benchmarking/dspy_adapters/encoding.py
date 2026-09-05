"""Locate and seed an offline-usable ``cl100k_base`` tiktoken cache.

Stdlib-only, quarantined exactly as ``config.py`` is quarantined for
``LSL_BENCH_*``: no ``tiktoken``, no ``dspy``, no ``requests``, no relative
imports, and no ``os.environ`` access at module scope -- every read and
write happens inside a function body. This is what keeps
``__init__.py``'s import-time "never touches ``os.environ`` or the
network" claim true: this module is imported by ``cli.py`` only, lazily,
inside ``_run_offline``.

``tiktoken`` downloads the ``cl100k_base`` BPE table over HTTPS on a cold
cache (``tiktoken/load.py``). This module finds a copy that already ships
inside an installed dependency (today: litellm's bundled tokenizer
directory) and points ``TIKTOKEN_CACHE_DIR``/``CUSTOM_TIKTOKEN_CACHE_DIR``
at it, without ever importing ``tiktoken`` or the dependency itself, and
without overriding a cache directory that already works.

``tests/conftest.py`` keeps a deliberate, self-contained stdlib copy of the
seeding logic in this module (see its docstring for why); keep the two in
sync -- this module is canonical.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable, Mapping, MutableMapping
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


def effective_cache_dir(environ: Mapping[str, str] | None = None) -> str:
    """Return the cache directory tiktoken would use, without importing tiktoken.

    Replicates the precedence at `tiktoken/load.py:36-46`:
    `TIKTOKEN_CACHE_DIR` > `DATA_GYM_CACHE_DIR` > `<tempdir>/data-gym-cache`.
    An empty string means the caller disabled caching entirely (`load.py:48-50`).
    `environ` defaults to `os.environ`.
    """
    env = os.environ if environ is None else environ
    if TIKTOKEN_CACHE_ENV in env:
        return env[TIKTOKEN_CACHE_ENV]
    if DATA_GYM_CACHE_ENV in env:
        return env[DATA_GYM_CACHE_ENV]
    return str(Path(tempfile.gettempdir()) / "data-gym-cache")


def blob_is_cached(
    environ: Mapping[str, str] | None = None, blobpath: str = CL100K_BLOBPATH
) -> bool:
    """Whether `blobpath`'s blob is already on disk where tiktoken would look for it.

    One `os.path.isfile` on `<effective_cache_dir>/<cache_key>`. No network, no
    `tiktoken` import, no BPE parsing -- deliberately cheap enough for a
    `pytest_configure` hook. Always `False` when caching is disabled.
    """
    directory = effective_cache_dir(environ)
    if not directory:
        return False
    return os.path.isfile(os.path.join(directory, cache_key(blobpath)))


def bundled_cache_dirs() -> tuple[Path, ...]:
    """Tiktoken cache directories that ship inside installed dependencies, best first.

    Today exactly one candidate: litellm's `litellm_core_utils/tokenizers`, located
    with `importlib.util.find_spec("litellm")`, which does **not** import litellm and
    therefore does not trigger litellm's own unconditional `TIKTOKEN_CACHE_DIR`
    rewrite (`litellm_core_utils/default_encoding.py:18-20`). Returns `()` when
    litellm is absent, when `find_spec` raises, or when the directory has moved.
    """
    try:
        spec = find_spec("litellm")
    except (ImportError, ValueError):
        return ()
    if spec is None or spec.origin is None:
        return ()
    candidate = Path(spec.origin).parent / "litellm_core_utils" / "tokenizers"
    return (candidate,) if candidate.is_dir() else ()


def find_cached_blob_dir(
    candidates: Iterable[Path], blobpath: str = CL100K_BLOBPATH
) -> Path | None:
    """First candidate directory that actually holds `blobpath`'s sha1-named blob.

    Pure: no environment access, no network. Verifying the specific filename -- never
    trusting a directory blindly -- is what makes the litellm dependency safe to use
    without declaring it.
    """
    key = cache_key(blobpath)
    for candidate in candidates:
        if (Path(candidate) / key).is_file():
            return candidate
    return None


def seed_tiktoken_cache(
    *,
    environ: MutableMapping[str, str] | None = None,
    candidates: Iterable[Path] | None = None,
    blobpath: str = CL100K_BLOBPATH,
) -> str | None:
    """Make `blobpath` loadable offline without overriding a cache that already works.

    Returns the cache directory tiktoken will use, or `None` when no cached blob can
    be found anywhere. `environ` defaults to `os.environ` and is the only thing this
    module ever mutates; `candidates` defaults to `bundled_cache_dirs()`. Both are
    injectable so the contract can be tested against a plain `dict` and a `tmp_path`,
    with no real environment write and no dependency on litellm being installed.

    Order of operations:
      1. `TIKTOKEN_CACHE_DIR == ""` -> return `None`. The caller explicitly disabled
         tiktoken's cache, i.e. asked for a download every time; honour that.
      2. `blob_is_cached(environ)` -> return the existing directory unchanged. A
         working cache -- the user's, or one a previous call seeded -- is never
         overridden, which is also what makes this function idempotent.
      3. Otherwise take the first candidate holding the blob and set **both**
         `TIKTOKEN_CACHE_DIR` and `CUSTOM_TIKTOKEN_CACHE_DIR` to it. The second is
         load-bearing: `import litellm` later does
         `os.environ["TIKTOKEN_CACHE_DIR"] = os.getenv("CUSTOM_TIKTOKEN_CACHE_DIR", <own dir>)`,
         so setting it *pins* our choice instead of racing litellm's rewrite.
      4. No candidate holds it -> return `None`, leaving the environment untouched.
    """
    env = os.environ if environ is None else environ
    if env.get(TIKTOKEN_CACHE_ENV) == "":
        return None

    if blob_is_cached(env, blobpath):
        return effective_cache_dir(env)

    pool = bundled_cache_dirs() if candidates is None else candidates
    found = find_cached_blob_dir(pool, blobpath)
    if found is None:
        return None
    # Safety: `import litellm` calls `tiktoken.get_encoding("cl100k_base")` at *import*
    # time (litellm_core_utils/default_encoding.py:23). A CUSTOM_TIKTOKEN_CACHE_DIR
    # pointing at a directory without the blob would make `import litellm` itself fail.
    # `find_cached_blob_dir` has already confirmed this directory holds the sha1-named
    # blob -- that verification is not optional.
    env[TIKTOKEN_CACHE_ENV] = env[CUSTOM_TIKTOKEN_CACHE_ENV] = str(found)
    return str(found)
