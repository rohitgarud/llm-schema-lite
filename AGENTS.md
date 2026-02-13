# AGENTS.md

## Quick Start
- **Package**: `llm-schema-lite` — LLM-ify JSON schemas (Pydantic, token-optimized formats)
- **Dependency manager**: `uv` with `pyproject.toml` (build: hatchling)
- **Common commands**: See `Makefile` for all available commands
- **Install**: `make sync` or `uv pip install -e ".[dev]"` for development
- **Run tests**: `make test`; **lint**: `make lint`

## Package Structure

- **Source**: `src/llm_schema_lite/` (installable package)
  - `core.py` — main schema conversion/validation entrypoints
  - `formatters/` — base formatter + JSON-ish, TypeScript, YAML formatters
  - `dspy_integration/` — optional DSPy adapters (extra: `[dspy]`)
  - `exceptions.py` — package-specific exceptions
- **Tests**: `tests/` (pytest); **Benchmarks**: `benchmarking/` (optional `[benchmark]`)
- **Examples**: `examples/` for basic usage

## Key Conventions

### Public API
- Export public functions/classes from `src/llm_schema_lite/__init__.py`
- Keep implementation details in modules; avoid leaking internals in the public API

### Formatters
- Formatters live in `formatters/` and extend the base formatter interface
- Add new formats by implementing the base formatter contract; keep formatting logic in formatters, not in `core`

### Optional Features
- **DSPy**: Install with `make install-dspy` or `uv pip install -e ".[dspy]"`; tests: `make test-dspy`
- **Benchmarking**: Optional `[benchmark]` deps; run with `make test-benchmarking`

### Testing & Code Quality
- **TDD**: Prefer writing failing tests first, then minimal code to pass (Red → Green → Refactor)
- Use **pytest** and fixtures in `tests/conftest.py`; keep tests isolated and independent
- **Type hints**: Use for function signatures, methods, and class attributes
- **Linting**: Run `make lint` before committing (ruff, mypy, bandit, pre-commit)
- Mark slow/integration tests with pytest markers (`@pytest.mark.slow`, `@pytest.mark.integration`)

## Git Rules
- Use **conventional commits** (type: description)
- Add detailed bulleted descriptions to the commits, highlighting the changes in the commit
- **Commit types**:
  - `feat`: Major user-facing features or substantial new capabilities
  - `chore`: Incremental improvements, internal changes, config, deps, tooling
  - `fix`: Bug fixes and corrections
  - `refactor`: Code restructuring without behavior changes
  - `docs`: Documentation updates
  - `test`: Adding or updating tests
  - `perf`: Performance improvements
- **Examples**:
  - ✅ `feat: add YAML formatter for Pydantic models`
  - ✅ `chore: add ruff rule for import sorting`
  - ✅ `fix: handle optional fields in jsonish formatter`
  - ❌ `feat: update pyproject` (use `chore`)

## Project-Specific Rules

### Do's
✅ Follow existing patterns for similar code (formatters, core, tests)
✅ Run `make lint` before committing
✅ Use type hints; keep the package typed (`py.typed` present)
✅ Prefer SOLID principles; keep formatters and core responsibilities clear
✅ Add tests for new behavior; mock external/optional deps (e.g. DSPy) when appropriate

### Don'ts
🚫 Change public API in `__init__.py` without considering backward compatibility
🚫 Put heavy or optional dependencies in core install (use extras: `dspy`, `benchmark`, `dev`)
🚫 Skip tests or lint for new code
🚫 Commit secrets or credentials

## Boundaries
✅ **Allowed**: Read files, run tests, format/lint, change `src/llm_schema_lite` and `tests`, update docs and examples

⚠️ **Ask first**: Add or bump dependencies, change build/publish config, alter supported Python versions

🚫 **Never**: Commit secrets, disable security or lint checks, break documented public API without a plan

### When in the Planning Mode
- Ask clarifying questions if not clear and then make the plan
- Read any directly mentioned files first
- Analyze and decompose the planning goal provided by user
- Identify any discrepancies or misunderstandings
- Note assumptions that need verification
- Identify specific components, patterns, or concepts and their connections to investigate, add to plan file
- Add exact paths in the plan to files the implementation of the plan will be touching, including line numbers/ranges
- If User provides some corrections, update the same plan accordingly
- Plan should be self-contained with all the research and necessary context, so that implementation agent should not have to open any files other than mentioned in the plan
- Explicitly list out-of-scope items to prevent scope creep
- Add verification step at the end which can include writing temporary or persistent tests, running relevant make commands

### When in the Agent Implementation Mode
- You should not to open any files other than mentioned in the plan as plan is comprehensive enough
- Follow existing patterns as per plan rather than creating new patterns wherever possible
