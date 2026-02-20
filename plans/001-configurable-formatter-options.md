# Plan: Configurable Formatter Options

## Overview
Add a `FormatterConfig` class to allow customization of formatter behavior including:
- Instruction prefix
- Enum/class hoisting behavior
- Union separators and markers
- Indentation
- Optional/required field markers
- Description/constraint inclusion toggles

## Scope
- **In Scope**: Formatter configuration, BaseFormatter updates, all three formatters (JSONish, TypeScript, YAML), core API integration, comprehensive tests
- **Out of Scope**: Changes to validators, parsers, or other non-formatter components

## Files to Modify

### 1. New File: `src/llm_schema_lite/formatters/config.py`
- Create `FormatterConfig` dataclass with the following options:
  - `prefix: str | None` - Instruction prefix (default: None)
  - `hoist_enums: bool | Literal["auto"]` - Enum hoisting (default: "auto")
  - `hoist_classes: bool | Literal["auto"] | list[str]` - Class/type hoisting (default: "auto")
  - `union_separator: str` - Union type separator (default: " | ")
  - `indent: int` - Indentation spaces (default: 2)
  - `optional_marker: str` - Optional field marker (default: "?")
  - `required_marker: str` - Required field marker (default: "*")
  - `include_descriptions: bool` - Include field descriptions (default: True)
  - `include_constraints: bool` - Include validation constraints (default: True)
  - `include_metadata: bool` - Include all metadata (default: True) - replaces current `include_metadata` param

### 2. Modify: `src/llm_schema_lite/formatters/base.py`
- **Lines ~55-75**: Update `__init__` signature:
  ```python
  def __init__(self, schema: dict[str, Any], config: FormatterConfig | None = None):
  ```
- Store config as `self.config` with defaults via `FormatterConfig()` if not provided
- **Lines ~300-310**: Update `format_field_name` to use `self.config.required_marker` and `self.config.optional_marker`
- **Lines ~360-375**: Update `get_required_fields_comment` to use config markers
- **Line ~68**: Replace `self.include_metadata` usage with `self.config.include_metadata`

### 3. Modify: `src/llm_schema_lite/formatters/jsonish_formatter.py`
- **Lines ~30-40**: Update `__init__` to accept config parameter and pass to super()
- Update union/anyOf processing to use `self.config.union_separator` (currently hardcoded as " OR ")
- Update enum formatting to respect `self.config.hoist_enums`
- Update prefix handling in `transform_schema()` to prepend `self.config.prefix` when set

### 4. Modify: `src/llm_schema_lite/formatters/typescript_formatter.py`
- **Lines ~30-40**: Update `__init__` to accept config parameter
- Update union type separator to use `self.config.union_separator` (currently hardcoded as " | ")
- Update `format_field_name` to use config markers
- Update hoisting logic to respect `self.config.hoist_classes`

### 5. Modify: `src/llm_schema_lite/formatters/yaml_formatter.py`
- **Lines ~60-75**: Update `__init__` to accept config parameter
- Update union separator to use `self.config.union_separator` (currently hardcoded as " OR ")
- Update prefix handling in `transform_schema()`

### 6. Modify: `src/llm_schema_lite/formatters/__init__.py`
- Add export for `FormatterConfig`
- Update `__all__` to include the new class

### 7. Modify: `src/llm_schema_lite/core.py`
- **Lines ~130-165**: Update `simplify_schema()` signature:
  ```python
  def simplify_schema(
      model: type["BaseModel"] | dict[str, Any] | str,
      config: FormatterConfig | None = None,  # NEW PARAMETER
      format_type: Literal["jsonish", "typescript", "yaml"] = "jsonish",
  ) -> SchemaLite:
  ```
- **Lines ~165-190**: Create formatter with config:
  ```python
  formatter: BaseFormatter
  formatter_kwargs = {"schema": original_schema, "config": config}
  if format_type == "jsonish":
      formatter = JSONishFormatter(**formatter_kwargs)
  # ... etc
  ```

### 8. Modify: `src/llm_schema_lite/__init__.py`
- Add export for `FormatterConfig`

## Implementation Details

### Hoisting Logic
- **"auto" mode**: Hoist enums/classes only when they are used multiple times in the schema
- **True**: Always hoist (move definitions to top-level with $ref)
- **False**: Never hoist (inline definitions)
- **list[str]**: Hoist only the specified class names

### Union Separator Mapping by Formatter
| Formatter | Default | Configurable |
|-----------|---------|--------------|
| JSONish   | " OR "  | union_separator |
| TypeScript| " \| "  | union_separator |
| YAML      | " OR "  | union_separator |

### Marker Configuration
| Marker | JSONish Default | TypeScript Default | YAML Default |
|--------|-----------------|-------------------|--------------|
| Required | "*" | "*" | "*" |
| Optional | "?" | "?" | "?" (not used) |

## Testing Strategy

### New Test File: `tests/test_formatter_config.py`
```python
import pytest
from pydantic import BaseModel
from llm_schema_lite import simplify_schema, FormatterConfig

class TestFormatterConfig:
    def test_prefix_option(self):
        config = FormatterConfig(prefix="Answer in this format:\n")
        result = simplify_schema(MyModel, config=config)
        assert result.to_string().startswith("Answer in this format:")

    def test_union_separator(self):
        config = FormatterConfig(union_separator=" or ")
        # Test with union type field

    def test_required_optional_markers(self):
        config = FormatterConfig(required_marker="!", optional_marker="~")
        # Verify markers appear in output

    def test_hoist_enums_true(self):
        config = FormatterConfig(hoist_enums=True)
        # Verify enums are hoisted

    def test_hoist_enums_false(self):
        config = FormatterConfig(hoist_enums=False)
        # Verify enums are inlined

    def test_include_descriptions_false(self):
        config = FormatterConfig(include_descriptions=False)
        # Verify no descriptions in output

    def test_include_constraints_false(self):
        config = FormatterConfig(include_constraints=False)
        # Verify no constraints (min, max, pattern, etc.) in output

    def test_indent_config(self):
        config = FormatterConfig(indent=4)
        # Verify 4-space indentation
```

### Update Existing Tests
- `tests/test_jsonish_formatter.py`: Add config parameter to formatter instantiations
- `tests/test_typescript_formatter.py`: Same
- `tests/test_yaml_formatter.py`: Same

## Backward Compatibility
- If `config=None` (default), use `FormatterConfig()` with all defaults
- If `include_metadata` is passed as separate param to `simplify_schema()`, map it to config
- Existing code continues to work without changes

## Usage Examples After Implementation
```python
from llm_schema_lite import simplify_schema, FormatterConfig
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    role: str  # will be enum

# Default behavior (unchanged)
schema = simplify_schema(User)

# Custom configuration
config = FormatterConfig(
    prefix="Answer in this format for a $400 tip:\n",
    hoist_enums=True,
    union_separator=" or ",
    indent=2,
    optional_marker="?",
    required_marker="*",
    include_descriptions=True,
    include_constraints=True,
)
schema = simplify_schema(User, config=config)

# From the spec example
schema = to_jsonish(
    MyModel,
    prefix="Answer in this format for a $400 tip:\n",
    hoist_enums=True,
    hoist_classes=True,
    union_separator=" or ",
    indent=2,
    optional_marker="?",
    required_marker="*",
    include_descriptions=True,
    include_constraints=True,
)
```

## Verification
1. Run existing tests: `make test` - all should pass
2. Run new tests: `pytest tests/test_formatter_config.py -v`
3. Check linting: `make lint`
4. Verify backward compatibility with existing usage patterns

## Implementation Order
1. Create `FormatterConfig` class
2. Update `BaseFormatter` to accept config
3. Update each formatter (JSONish, TypeScript, YAML) one by one
4. Update core API
5. Add exports
6. Write tests
7. Run full test suite
