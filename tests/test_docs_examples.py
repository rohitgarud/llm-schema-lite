"""Execute every fenced ``python`` block in README.md and
src/llm_schema_lite/dspy_integration/README.md, plus every tracked examples/*.py
script, so the documentation cannot silently drift from the shipped API.

Deliberate deviation from the repo's module-level dspy-gating idiom
(``pytest.importorskip("dspy", minversion="3.3.1")`` at module scope, used by
``tests/dspy_helpers.py`` and the six ``tests/test_dspy_adapter_*.py`` modules):
this module imports no dspy at module scope, so no ``# noqa: E402`` is needed
anywhere. Gating happens per-block, at test-body time, keyed off whether the
literal substring "dspy" appears in the block's own source. A module-level
guard would also skip the top-level README Quick Start block (pure
simplify_schema/loads, no dspy) in any environment without the ``[dspy]``
extra -- see design v2 lsl-2026-09-04-018 section B.9.

No doc block is ever executed against a live LM. Blocks that would issue a
live LM request carry a mandatory, non-empty
``<!-- lsl-docs: skip: <reason> -->`` marker on the last non-blank line before
their opening fence and are skipped, never executed -- see section B.4/A.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# --------------------------------------------------------------------------
# Module-level constants
# --------------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
"""Repository root: this file lives at ``<root>/tests/test_docs_examples.py``."""

_SCANNED_DOCS: tuple[str, ...] = (
    "README.md",
    "src/llm_schema_lite/dspy_integration/README.md",
)
"""Doc files scanned for fenced python blocks, repo-root-relative. An explicit
tuple, never a ``**/*.md`` glob, so a parallel ticket's untracked markdown
(e.g. ``benchmarking/tests/README.md``) is never swept in."""

_EXAMPLE_SCRIPTS: tuple[str, ...] = ("examples/basic_usage.py",)
"""Tracked example scripts executed as subprocesses, repo-root-relative.
Explicit tuple so the untracked, out-of-scope
``examples/enum_with_metadata.py`` is never picked up."""

_OPEN_FENCE_RE: re.Pattern[str] = re.compile(r"^(?P<ticks>`{3,})(?P<info>[^`]*)$")
"""Matches a column-0 opening fence. ``info`` (stripped) is the language tag;
an empty string means untagged."""

_SKIP_MARKER_RE: re.Pattern[str] = re.compile(
    r"^<!--\s*lsl-docs:\s*skip:\s*(?P<reason>\S.*?)\s*-->$"
)
"""Matches a well-formed skip marker with a mandatory, non-empty reason."""

_DIRECTIVE_RE: re.Pattern[str] = re.compile(r"^<!--\s*lsl-docs:\s*(?P<directive>\S+)")
"""Matches any ``lsl-docs`` directive comment, well-formed or not, so a typo'd
directive (``sikp``, ``skip -``) is rejected loudly instead of silently
executing or silently skipping."""


# --------------------------------------------------------------------------
# Data shape
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DocCodeBlock:
    """One fenced code block discovered while scanning a markdown file.

    Attributes:
        rel_path: Repo-root-relative path of the file the block was found in.
        open_lineno: 1-based line number of the block's opening fence line.
        lang: The fence's language tag, exactly as written. Compared with
            ``==``, never as a family of aliases -- ``py``/``python3`` do not
            match ``"python"``.
        body: Every line strictly between the opening and closing fences.
        skip_reason: Reason captured from an immediately-preceding
            ``<!-- lsl-docs: skip: <reason> -->`` marker, else ``None``.
        requires_dspy: ``True`` iff ``"dspy"`` occurs anywhere in ``body``.
    """

    rel_path: str
    open_lineno: int
    lang: str
    body: str
    skip_reason: str | None
    requires_dspy: bool

    @property
    def test_id(self) -> str:
        """Return the ``"{rel_path}:L{open_lineno}"`` pytest parametrize id."""
        return f"{self.rel_path}:L{self.open_lineno}"

    @property
    def is_python(self) -> bool:
        """Return whether this block's language tag is exactly ``python``."""
        return self.lang == "python"


# --------------------------------------------------------------------------
# Scanning: a small line-based scanner, not a whole-file regex
# --------------------------------------------------------------------------


def _skip_reason_before(
    lines: list[str], fence_index: int, rel_path: str, open_lineno: int
) -> str | None:
    """Return the skip reason declared immediately before an opening fence.

    Walks backwards past blank lines only. Any other content between a marker
    and the fence means the marker does not apply.

    Raises:
        AssertionError: The last non-blank line is an ``lsl-docs`` directive
            that is not a well-formed, non-empty ``skip``.
    """
    index = fence_index - 1
    while index >= 0 and not lines[index].strip():
        index -= 1
    if index < 0:
        return None
    candidate = lines[index].strip()
    if _DIRECTIVE_RE.match(candidate) is None:
        return None
    marker = _SKIP_MARKER_RE.match(candidate)
    if marker is None:
        raise AssertionError(
            f"{rel_path}:{index + 1}: malformed lsl-docs directive {candidate!r} "
            f"before the fence opening at line {open_lineno}. The only valid form "
            f"is '<!-- lsl-docs: skip: <non-empty reason> -->'."
        )
    return marker.group("reason")


def _scan_fenced_blocks(text: str, rel_path: str) -> list[DocCodeBlock]:
    """Scan ``text`` line-by-line and return every column-0 fenced code block.

    Raises:
        AssertionError: An unterminated fence at EOF, or a malformed
            ``<!-- lsl-docs: ... -->`` directive.
    """
    lines = text.splitlines()
    blocks: list[DocCodeBlock] = []
    index = 0
    while index < len(lines):
        opening = _OPEN_FENCE_RE.match(lines[index])
        if opening is None:
            index += 1
            continue
        ticks = opening.group("ticks")
        lang = opening.group("info").strip()
        open_lineno = index + 1
        closing_re = re.compile(r"^`{" + str(len(ticks)) + r",}\s*$")
        cursor = index + 1
        while cursor < len(lines) and closing_re.match(lines[cursor]) is None:
            cursor += 1
        if cursor >= len(lines):
            raise AssertionError(
                f"{rel_path}:{open_lineno}: unterminated code fence "
                f"(no closing line of {len(ticks)} or more backticks)"
            )
        body = "\n".join(lines[index + 1 : cursor])
        blocks.append(
            DocCodeBlock(
                rel_path=rel_path,
                open_lineno=open_lineno,
                lang=lang,
                body=body,
                skip_reason=_skip_reason_before(lines, index, rel_path, open_lineno),
                requires_dspy="dspy" in body,
            )
        )
        index = cursor + 1
    return blocks


def _load_doc_blocks(rel_path: str) -> list[DocCodeBlock]:
    """Read ``_REPO_ROOT / rel_path`` and return its scanned fenced blocks."""
    path = _REPO_ROOT / rel_path
    return _scan_fenced_blocks(path.read_text(encoding="utf-8"), rel_path)


def _python_blocks(rel_path: str) -> list[DocCodeBlock]:
    """Return only the ``lang == "python"`` blocks of ``rel_path``."""
    return [block for block in _load_doc_blocks(rel_path) if block.is_python]


def _all_scanned_python_blocks() -> list[DocCodeBlock]:
    """Return every python block across ``_SCANNED_DOCS``, in file order."""
    blocks: list[DocCodeBlock] = []
    for rel_path in _SCANNED_DOCS:
        blocks.extend(_python_blocks(rel_path))
    return blocks


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def dspy_settings_guard() -> Iterator[None]:
    """Snapshot ``dspy.settings.config`` and restore it after each test.

    ``dspy.context()`` does *not* isolate ``dspy.configure``: calling
    ``dspy.configure(adapter=...)`` inside ``with dspy.context():`` leaves
    ``dspy.settings.adapter is None`` inside the block and then leaks the
    adapter globally after the context exits. Snapshot/restore is the only
    mechanism verified to round-trip cleanly.

    dspy is imported lazily here, never at module scope, so the module stays
    import-clean without the ``[dspy]`` extra installed.
    """
    try:
        import dspy
    except ImportError:
        yield
        return
    snapshot = dict(dspy.settings.config)
    try:
        yield
    finally:
        dspy.settings.configure(**snapshot)


@pytest.fixture(autouse=True)
def _chdir_tmp_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Chdir into ``tmp_path`` so a doc block that writes a file cannot
    collide with another xdist worker. A cheap forward guard -- nothing in
    either README or in ``examples/`` writes a file today."""
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    _all_scanned_python_blocks(),
    ids=lambda block: block.test_id,
)
def test_markdown_python_block_executes(block: DocCodeBlock) -> None:
    """Execute one fenced python block in a fresh, isolated namespace.

    The only assertion is "raises nothing". No doc block's printed output and
    no numeric value it computes may ever be asserted on -- token counts are
    call-order dependent (follow-up F5), so any numeric assertion would encode
    a bug.
    """
    if block.skip_reason is not None:
        pytest.skip(block.skip_reason)
    if block.requires_dspy:
        pytest.importorskip("dspy", minversion="3.3.1")
    # ``dont_inherit=True`` is load-bearing: this module has
    # ``from __future__ import annotations``, and ``compile()`` inherits the
    # caller's __future__ flags by default. Under PEP 563 a doc block's
    # ``address: Address`` is stored as the *string* "Address", which pydantic
    # then cannot resolve because the exec namespace's ``__name__``
    # ("__lsl_docs__") is not in ``sys.modules``. A doc block must compile with
    # the same semantics a user's own script would get, not this file's.
    code = compile(
        block.body,
        f"{block.rel_path}:{block.open_lineno}",
        "exec",
        dont_inherit=True,
    )
    namespace: dict[str, Any] = {"__name__": "__lsl_docs__"}
    exec(code, namespace)


@pytest.mark.parametrize("rel_path", _SCANNED_DOCS)
def test_scanned_file_has_executable_blocks(rel_path: str) -> None:
    """Fail if ``rel_path`` has zero non-skip-marked python blocks.

    This is the rail that stops the harness from passing vacuously: a
    fence-syntax change, an accidental indent, or mass skip-marking that
    leaves a file with nothing to execute must turn this red.
    """
    executable = [block for block in _python_blocks(rel_path) if block.skip_reason is None]
    assert executable, (
        f"{rel_path} contains no executable python blocks. The doc-example "
        f"harness would pass vacuously for this file."
    )


@pytest.mark.parametrize("rel_path", _EXAMPLE_SCRIPTS, ids=_EXAMPLE_SCRIPTS)
def test_example_script_runs(rel_path: str) -> None:
    """Run ``rel_path`` as a subprocess and assert it exits ``0``.

    A subprocess, never ``runpy``: an in-process runner would share this
    interpreter's global state, and importing
    ``llm_schema_lite.dspy_integration`` monkeypatches
    ``StreamListener.__init__`` process-wide. The subprocess is not measured
    by ``--cov=llm_schema_lite``; that is expected and harmless.
    """
    script = _REPO_ROOT / rel_path
    assert script.is_file(), f"{rel_path} does not exist"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"{rel_path} exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
