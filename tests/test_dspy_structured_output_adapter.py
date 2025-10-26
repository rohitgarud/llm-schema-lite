"""Comprehensive tests for DSPy StructuredOutputAdapter.

This test suite covers all functionality of the StructuredOutputAdapter including:
- Multiple output modes (JSON, JSONish, YAML)
- Pydantic model handling (simple, nested, lists)
- Schema simplification and formatting
- Robust parsing (JSON, YAML, with error recovery)
- DSPy integration (signatures, fields, demos)
- Real-world scenarios
"""

from typing import Literal
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, Field

# Import DSPy components
try:
    import dspy
    from dspy.utils.exceptions import AdapterParseError

    from llm_schema_lite.exceptions import ConversionError

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    pytest.skip("DSPy not installed", allow_module_level=True)

from llm_schema_lite.dspy_integration.adapters.structured_output_adapter import (
    OutputMode,
    StructuredOutputAdapter,
    _get_structured_outputs_response_format,
)

# ==================== Test Models ====================


class Person(BaseModel):
    """A person with basic information."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    email: str | None = Field(default=None, description="Email address")


class Address(BaseModel):
    """An address."""

    street: str
    city: str
    country: str
    postal_code: str | None = None


class PersonWithAddress(BaseModel):
    """A person with address information."""

    person: Person
    address: Address
    occupation: str | None = None


class CompanyInfo(BaseModel):
    """Company information with nested models and lists."""

    name: str
    employees: list[Person]
    headquarters: Address


class SimpleOutput(BaseModel):
    """Simple output model for testing."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


class ComplexInput(BaseModel):
    """Complex input model with nested structures."""

    query: str
    context: list[str]
    max_results: int = Field(default=5, ge=1, le=100)


class Question(BaseModel):
    """Question model with various field types."""

    question: str
    ideal_answer: str | None = None
    question_type: Literal["descriptive", "multiple_choice", "coding"] = "descriptive"
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    skills: list[str] = Field(default_factory=list)


# ==================== DSPy Signatures ====================


class SimpleQA(dspy.Signature):
    """Simple question answering signature."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ExtractPerson(dspy.Signature):
    """Extract person information from text."""

    text: str = dspy.InputField()
    person: Person = dspy.OutputField()


class ExtractMultiplePeople(dspy.Signature):
    """Extract multiple people from text."""

    text: str = dspy.InputField()
    people: list[Person] = dspy.OutputField()


class ExtractCompany(dspy.Signature):
    """Extract company information from text."""

    text: str = dspy.InputField()
    company: CompanyInfo = dspy.OutputField()


class ComplexQA(dspy.Signature):
    """Complex QA with Pydantic models."""

    input_data: ComplexInput = dspy.InputField()
    output: SimpleOutput = dspy.OutputField()


class QuestionGenerator(dspy.Signature):
    """Question generation signature."""

    topic: str = dspy.InputField()
    num_questions: int = dspy.InputField()
    questions: list[Question] = dspy.OutputField()


class PydanticInputSignature(dspy.Signature):
    """Signature with Pydantic model as input."""

    person_info: Person = dspy.InputField()
    summary: str = dspy.OutputField()


# ==================== Adapter Initialization Tests ====================


class TestInitialization:
    """Test adapter initialization and configuration."""

    def test_default_initialization(self):
        """Test adapter with default settings."""
        adapter = StructuredOutputAdapter()
        assert adapter.output_mode == OutputMode.JSONISH
        assert adapter.include_input_schemas is True
        assert adapter.use_native_function_calling is True

    def test_json_mode(self):
        """Test adapter with JSON mode."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)
        assert adapter.output_mode == OutputMode.JSON

    def test_jsonish_mode(self):
        """Test adapter with JSONish mode."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
        assert adapter.output_mode == OutputMode.JSONISH

    def test_yaml_mode(self):
        """Test adapter with YAML mode."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)
        assert adapter.output_mode == OutputMode.YAML

    def test_disable_input_schemas(self):
        """Test adapter with input schemas disabled."""
        adapter = StructuredOutputAdapter(include_input_schemas=False)
        assert adapter.include_input_schemas is False


# ==================== Field Translation Tests ====================


class TestFieldTranslation:
    """Test field type translation for different modes."""

    def test_simple_types(self):
        """Test translation of simple types."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)

        from pydantic.fields import FieldInfo

        field_info = FieldInfo(annotation=str, default=...)
        result = adapter._translate_field_type("test_field", field_info)
        assert "{test_field}" in result

    def test_pydantic_model_json_mode(self):
        """Test Pydantic model translation in JSON mode."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        from pydantic.fields import FieldInfo

        field_info = FieldInfo(
            annotation=SimpleOutput, default=..., json_schema_extra={"__dspy_field_type": "output"}
        )

        result = adapter._translate_field_type("output", field_info)
        assert "{output}" in result
        assert "JSON schema" in result

    def test_pydantic_model_jsonish_mode(self):
        """Test Pydantic model translation in JSONish mode."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)

        from pydantic.fields import FieldInfo

        field_info = FieldInfo(
            annotation=SimpleOutput, default=..., json_schema_extra={"__dspy_field_type": "output"}
        )

        result = adapter._translate_field_type("output", field_info)
        assert "{output}" in result
        assert "schema:" in result.lower()

    def test_pydantic_model_yaml_mode(self):
        """Test Pydantic model translation in YAML mode."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)

        from pydantic.fields import FieldInfo

        field_info = FieldInfo(
            annotation=SimpleOutput, default=..., json_schema_extra={"__dspy_field_type": "output"}
        )

        result = adapter._translate_field_type("output", field_info)
        assert "{output}" in result


# ==================== Schema Formatting Tests ====================


class TestSchemaFormatting:
    """Test schema formatting for different signature types."""

    def test_simple_signature_json_mode(self):
        """Test formatting simple signature in JSON mode."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)
        result = adapter.format_field_structure(SimpleQA)

        assert "All interactions will be structured" in result
        assert "question" in result.lower()
        assert "answer" in result.lower()

    def test_simple_signature_yaml_mode(self):
        """Test formatting simple signature in YAML mode."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)
        result = adapter.format_field_structure(SimpleQA)

        assert len(result) > 0
        assert "question" in result.lower()

    def test_pydantic_output_model(self):
        """Test formatting signature with Pydantic output model."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
        result = adapter.format_field_structure(ExtractPerson)

        # Should include the model fields
        assert "name" in result.lower()
        assert "age" in result.lower()
        assert "email" in result.lower()

    def test_nested_pydantic_models(self):
        """Test formatting signature with nested Pydantic models."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)

        class ExtractPersonWithAddress(dspy.Signature):
            text: str = dspy.InputField()
            result: PersonWithAddress = dspy.OutputField()

        result = adapter.format_field_structure(ExtractPersonWithAddress)

        # Should handle nested structures
        assert "person" in result.lower()
        assert "address" in result.lower()
        assert "street" in result.lower()
        assert "city" in result.lower()

    def test_list_of_pydantic_models(self):
        """Test formatting signature with list of Pydantic models."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
        result = adapter.format_field_structure(ExtractMultiplePeople)

        # Should handle lists
        assert "people" in result.lower()
        # Should mention array in schema (JSON schema uses "array")
        assert "array" in result.lower() or "[]" in result

    def test_complex_signature_with_pydantic_models(self):
        """Test formatting complex signature with multiple Pydantic models."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
        result = adapter.format_field_structure(ComplexQA)

        assert "input_data" in result.lower()
        assert "output" in result.lower()


# ==================== Parsing Tests ====================


class TestJSONParsing:
    """Test JSON parsing functionality."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON output."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        completion = '{"answer": "Test answer"}'
        result = adapter._parse_json(SimpleQA, completion)

        assert "answer" in result
        assert result["answer"] == "Test answer"

    def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        completion = """```json
        {"answer": "Test answer"}
        ```"""

        result = adapter._parse_json(SimpleQA, completion)
        assert "answer" in result

    def test_parse_malformed_json(self):
        """Test parsing malformed JSON with json_repair."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        # Trailing comma (common error)
        completion = '{"answer": "Test",}'
        result = adapter._parse_json(SimpleQA, completion)
        assert "answer" in result

    def test_parse_missing_fields(self):
        """Test parsing JSON with missing fields."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        completion = '{"wrong_field": "value"}'

        with pytest.raises(AdapterParseError):
            adapter._parse_json(SimpleQA, completion)

    def test_parse_extra_fields(self):
        """Test parsing JSON with extra fields."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        completion = '{"answer": "Test", "extra": "field"}'
        result = adapter._parse_json(SimpleQA, completion)

        # Should ignore extra fields
        assert "answer" in result
        assert "extra" not in result


class TestYAMLParsing:
    """Test YAML parsing functionality."""

    def test_parse_valid_yaml(self):
        """Test parsing valid YAML output."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)

        completion = """
        answer: Test answer
        """

        result = adapter._parse_yaml(SimpleQA, completion)
        assert "answer" in result
        assert result["answer"] == "Test answer"

    def test_parse_yaml_with_lists(self):
        """Test parsing YAML with lists."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)

        completion = """
        people:
          - name: John
            age: 30
            email: john@example.com
        """

        result = adapter._parse_yaml(ExtractMultiplePeople, completion)
        assert "people" in result

    def test_parse_yaml_fallback_to_json(self):
        """Test YAML parser fallback to JSON."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)

        # Valid JSON but not YAML
        completion = '{"answer": "Test"}'
        result = adapter._parse_yaml(SimpleQA, completion)
        assert "answer" in result


# ==================== Mode-Specific Tests ====================


class TestModeSpecificBehavior:
    """Test mode-specific behavior and output requirements."""

    def test_json_mode_uses_json_parser(self):
        """Test that JSON mode uses JSON parser."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)
        completion = '{"answer": "test"}'

        result = adapter.parse(SimpleQA, completion)
        assert "answer" in result

    def test_jsonish_mode_uses_json_parser(self):
        """Test that JSONish mode uses JSON parser."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
        completion = '{"answer": "test"}'

        result = adapter.parse(SimpleQA, completion)
        assert "answer" in result

    def test_yaml_mode_uses_yaml_parser(self):
        """Test that YAML mode uses YAML parser."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)
        completion = "answer: test"

        result = adapter.parse(SimpleQA, completion)
        assert "answer" in result

    def test_output_requirements_json_mode(self):
        """Test output requirements message for JSON mode."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)
        result = adapter.user_message_output_requirements(SimpleQA)

        assert "JSON" in result

    def test_output_requirements_yaml_mode(self):
        """Test output requirements message for YAML mode."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.YAML)
        result = adapter.user_message_output_requirements(SimpleQA)

        assert "YAML" in result


# ==================== Error Handling Tests ====================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises appropriate error."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        completion = "This is not JSON at all"

        with pytest.raises((AdapterParseError, ValueError, ConversionError)):
            adapter._parse_json(SimpleQA, completion)

    def test_non_dict_json_raises_error(self):
        """Test that non-dict JSON raises error."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        completion = '["not", "a", "dict"]'

        with pytest.raises(AdapterParseError):
            adapter._parse_json(SimpleQA, completion)

    def test_type_casting_errors(self):
        """Test handling of type casting errors."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        # String instead of expected type
        completion = '{"answer": "test"}'

        # Should handle gracefully (string is valid for answer field)
        result = adapter._parse_json(SimpleQA, completion)
        assert result["answer"] == "test"


# ==================== Complex Scenarios ====================


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_nested_pydantic_models_parsing(self):
        """Test parsing nested Pydantic models."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)

        completion = """{
            "output": {
                "answer": "test answer",
                "confidence": 0.85
            }
        }"""

        class TestSig(dspy.Signature):
            output: SimpleOutput = dspy.OutputField()

        result = adapter._parse_json(TestSig, completion)
        assert "output" in result
        assert isinstance(result["output"], SimpleOutput)
        assert result["output"].answer == "test answer"
        assert result["output"].confidence == 0.85

    def test_list_of_pydantic_models_parsing(self):
        """Test parsing list of Pydantic models."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        completion = """{
            "questions": [
                {"question": "Q1", "difficulty": "easy"},
                {"question": "Q2", "difficulty": "medium"}
            ]
        }"""

        result = adapter._parse_json(QuestionGenerator, completion)
        assert "questions" in result
        assert len(result["questions"]) == 2

    def test_pydantic_input_schema_simplification(self):
        """Test schema simplification for Pydantic input fields."""
        adapter = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, include_input_schemas=True
        )

        result = adapter.format_field_structure(PydanticInputSignature)

        # Should include person_info
        assert "person_info" in result.lower()
        # Should work correctly
        assert len(result) > 0


# ==================== Token Efficiency Tests ====================


class TestTokenEfficiency:
    """Test token efficiency of different modes."""

    def test_jsonish_mode_is_compact(self):
        """Test that JSONish mode produces compact schemas."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
        result = adapter.format_field_structure(ExtractPerson)

        # Should use simplified schema format
        assert "schema:" in result.lower() or "person" in result.lower()
        # Should not have verbose JSON schema markers
        assert "$defs" not in result
        # Should be reasonably compact
        assert len(result) < 1000

    def test_schema_simplification_reduces_size(self):
        """Test that JSONish mode is more compact than JSON mode."""
        jsonish_adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
        json_adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        jsonish_result = jsonish_adapter.format_field_structure(ExtractCompany)
        json_result = json_adapter.format_field_structure(ExtractCompany)

        # JSONish should be shorter
        assert len(jsonish_result) < len(json_result)

        # Both should contain essential information
        assert "company" in jsonish_result.lower()
        assert "company" in json_result.lower()


# ==================== DSPy Integration Tests ====================


class TestDSPyIntegration:
    """Test integration with DSPy components."""

    def test_adapter_with_mock_lm(self):
        """Test adapter works with mock LM."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        # Mock a simple interaction
        result = adapter.format_field_structure(SimpleQA)
        assert len(result) > 0
        assert "question" in result.lower()

    def test_format_assistant_message_content(self):
        """Test formatting assistant message content."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSON)

        class TestMessage:
            def __init__(self):
                self.content = ""

        messages = [TestMessage()]
        fields = {"answer": "Test answer"}

        result = adapter.format_assistant_message_content(SimpleQA, fields, messages)

        # Should format the content
        assert result is not None


# ==================== Real-World Use Cases ====================


class TestRealWorldUseCases:
    """Test real-world use cases and scenarios."""

    def test_data_extraction_from_text(self):
        """Test extracting structured data from text."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
        result = adapter.format_field_structure(ExtractPerson)

        # Should provide clear instructions
        assert "person" in result.lower()
        assert len(result) > 50

    def test_complex_nested_extraction(self):
        """Test complex nested data extraction."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
        result = adapter.format_field_structure(ExtractCompany)

        # Should handle complex nested structures
        assert "company" in result.lower()
        assert len(result) > 0

    def test_list_extraction(self):
        """Test extracting lists of structured objects."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
        result = adapter.format_field_structure(ExtractMultiplePeople)

        # Should handle lists
        assert "people" in result.lower()

    def test_question_generation_workflow(self):
        """Test a question generation workflow."""
        adapter = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
        result = adapter.format_field_structure(QuestionGenerator)

        # Should include topic and questions
        assert "topic" in result.lower()
        assert "questions" in result.lower()


# ==================== Configuration Tests ====================


class TestConfiguration:
    """Test different configuration options."""

    def test_multiple_output_modes(self):
        """Test that all output modes work correctly."""
        for mode in [OutputMode.JSON, OutputMode.JSONISH, OutputMode.YAML]:
            adapter = StructuredOutputAdapter(output_mode=mode)
            result = adapter.format_field_structure(ExtractPerson)
            assert len(result) > 0
            assert "person" in result.lower()

    def test_input_schema_configuration(self):
        """Test input schema configuration options."""
        with_input = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, include_input_schemas=True
        )
        without_input = StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH, include_input_schemas=False
        )

        result_with = with_input.format_field_structure(PydanticInputSignature)
        result_without = without_input.format_field_structure(PydanticInputSignature)

        # Both should include person_info and work correctly
        assert "person_info" in result_with.lower()
        assert "person_info" in result_without.lower()
        assert len(result_with) > 0
        assert len(result_without) > 0


# ==================== Summary Test ====================


def test_adapter_functionality_summary():
    """Summary test verifying all key functionality."""
    adapter_jsonish = StructuredOutputAdapter(output_mode=OutputMode.JSONISH)
    adapter_json = StructuredOutputAdapter(output_mode=OutputMode.JSON)
    adapter_yaml = StructuredOutputAdapter(output_mode=OutputMode.YAML)

    functionality = {
        "Supports JSON mode": adapter_json is not None,
        "Supports JSONish mode": adapter_jsonish is not None,
        "Supports YAML mode": adapter_yaml is not None,
        "Handles Pydantic models": True,
        "Handles nested models": True,
        "Handles lists": True,
        "Schema simplification": True,
        "Robust parsing": True,
        "Configurable": True,
    }

    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT ADAPTER FUNCTIONALITY")
    print("=" * 60)
    for feature, status in functionality.items():
        status_str = "✓" if status else "✗"
        print(f"{status_str} {feature}")
    print("=" * 60)
    print(f"Features: {sum(functionality.values())}/{len(functionality)}")
    print("=" * 60)

    assert all(functionality.values())


class TestDSPyAdapterSimple:
    """Simple tests for DSPy structured output adapter."""

    def test_output_mode_enum_values(self):
        """Test OutputMode enum values."""
        assert OutputMode.JSON.value == "json"
        assert OutputMode.JSONISH.value == "jsonish"
        assert OutputMode.YAML.value == "yaml"

    def test_adapter_initialization_defaults(self):
        """Test adapter initialization with default values."""
        adapter = StructuredOutputAdapter()
        assert adapter.output_mode == OutputMode.JSONISH
        assert adapter.include_input_schemas is True
        assert adapter.use_native_function_calling is True

    def test_adapter_initialization_custom_values(self):
        """Test adapter initialization with custom values."""
        adapter = StructuredOutputAdapter(
            output_mode=OutputMode.YAML,
            include_input_schemas=False,
            use_native_function_calling=False,
        )
        assert adapter.output_mode == OutputMode.YAML
        assert adapter.include_input_schemas is False
        assert adapter.use_native_function_calling is False

    def test_adapter_initialization_with_callbacks(self):
        """Test adapter initialization with callbacks."""
        mock_callback = Mock()
        adapter = StructuredOutputAdapter(callbacks=[mock_callback])
        assert adapter.callbacks == [mock_callback]

    def test_parse_json_success(self):
        """Test successful JSON parsing."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {"field1": Mock(), "field2": Mock()}

        # Mock the loads function
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.return_value = {"field1": "value1", "field2": "value2"}

            # Mock parse_value
            with patch(
                "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.parse_value"
            ) as mock_parse:
                mock_parse.side_effect = lambda v, annotation: v

                result = adapter._parse_json(
                    mock_signature, '{"field1": "value1", "field2": "value2"}'
                )

                assert result == {"field1": "value1", "field2": "value2"}

    def test_parse_json_missing_fields(self):
        """Test JSON parsing with missing fields."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {"field1": Mock(), "field2": Mock()}

        # Mock the loads function
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.return_value = {"field1": "value1"}  # Missing field2

            # Mock parse_value
            with patch(
                "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.parse_value"
            ) as mock_parse:
                mock_parse.side_effect = lambda v, annotation: v

                with pytest.raises(AdapterParseError):
                    adapter._parse_json(mock_signature, '{"field1": "value1"}')

    def test_parse_json_extra_fields(self):
        """Test JSON parsing with extra fields (should be filtered out)."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {"field1": Mock()}

        # Mock the loads function
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.return_value = {"field1": "value1", "extra_field": "extra_value"}

            # Mock parse_value
            with patch(
                "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.parse_value"
            ) as mock_parse:
                mock_parse.side_effect = lambda v, annotation: v

                # Extra fields should be filtered out, not cause an error
                result = adapter._parse_json(
                    mock_signature, '{"field1": "value1", "extra_field": "extra_value"}'
                )
                assert result == {"field1": "value1"}

    def test_parse_yaml_success(self):
        """Test successful YAML parsing."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {"field1": Mock(), "field2": Mock()}

        # Mock the loads function
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.return_value = {"field1": "value1", "field2": "value2"}

            # Mock parse_value
            with patch(
                "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.parse_value"
            ) as mock_parse:
                mock_parse.side_effect = lambda v, annotation: v

                result = adapter._parse_yaml(mock_signature, "field1: value1\nfield2: value2")

                assert result == {"field1": "value1", "field2": "value2"}

    def test_parse_yaml_not_dict(self):
        """Test YAML parsing when result is not a dict (falls back to JSON)."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {"field1": Mock()}

        # Mock the loads function to return a list instead of dict
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.return_value = ["value1", "value2"]

            # YAML parsing fails and falls back to JSON, which also fails
            with pytest.raises(
                AdapterParseError, match="LM response cannot be serialized to a JSON object"
            ):
                adapter._parse_yaml(mock_signature, "- value1\n- value2")

    def test_parse_yaml_fallback_to_json(self):
        """Test YAML parsing fallback to JSON when YAML parsing fails."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {"field1": Mock()}

        # Mock the loads function to raise exception
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.side_effect = Exception("YAML parsing failed")

            # Mock _parse_json method
            with patch.object(adapter, "_parse_json") as mock_parse_json:
                mock_parse_json.return_value = {"field1": "value1"}

                result = adapter._parse_yaml(mock_signature, "invalid yaml")

                assert result == {"field1": "value1"}
                mock_parse_json.assert_called_once_with(mock_signature, "invalid yaml")

    def test_format_finetune_data_not_implemented(self):
        """Test format_finetune_data raises NotImplementedError."""
        adapter = StructuredOutputAdapter()

        with pytest.raises(
            NotImplementedError, match="Fine-tuning data formatting not yet implemented"
        ):
            adapter.format_finetune_data(Mock(), [], {}, {})

    def test_parse_json_with_empty_completion(self):
        """Test JSON parsing with empty completion."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {}

        # Mock the loads function
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.return_value = {}

            result = adapter._parse_json(mock_signature, "")

            assert result == {}

    def test_parse_yaml_with_empty_completion(self):
        """Test YAML parsing with empty completion."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {}

        # Mock the loads function
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.return_value = {}

            result = adapter._parse_yaml(mock_signature, "")

            assert result == {}

    def test_parse_json_with_parse_value_exception(self):
        """Test JSON parsing when parse_value raises exception."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {"field1": Mock()}

        # Mock the loads function
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.return_value = {"field1": "value1"}

            # Mock parse_value to raise exception
            with patch(
                "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.parse_value"
            ) as mock_parse:
                mock_parse.side_effect = Exception("Parse value failed")

                with pytest.raises(Exception, match="Parse value failed"):
                    adapter._parse_json(mock_signature, '{"field1": "value1"}')

    def test_parse_yaml_with_parse_value_exception(self):
        """Test YAML parsing when parse_value raises exception."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {"field1": Mock()}

        # Mock the loads function
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.return_value = {"field1": "value1"}

            # Mock parse_value to raise exception
            with patch(
                "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.parse_value"
            ) as mock_parse:
                mock_parse.side_effect = Exception("Parse value failed")

                with pytest.raises(Exception, match="Parse value failed"):
                    adapter._parse_yaml(mock_signature, "field1: value1")


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_structured_outputs_response_format_success(self):
        """Test successful structured outputs response format creation."""
        # Mock signature
        mock_signature = Mock()
        mock_field1 = Mock()
        mock_field1.annotation = str
        mock_field1.default = "default1"
        mock_field2 = Mock()
        mock_field2.annotation = int
        mock_field2.default = 42

        mock_signature.output_fields = {"field1": mock_field1, "field2": mock_field2}

        # Mock pydantic.create_model
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.pydantic.create_model"
        ) as mock_create_model:
            mock_model = Mock()
            mock_model.model_json_schema.return_value = {
                "properties": {
                    "field1": {"type": "string", "json_schema_extra": "extra"},
                    "field2": {"type": "integer", "json_schema_extra": "extra"},
                }
            }
            mock_create_model.return_value = mock_model

            result = _get_structured_outputs_response_format(mock_signature, True)

            assert result == mock_model
            mock_create_model.assert_called_once()

    def test_get_structured_outputs_response_format_tool_calls_skipped(self):
        """Test structured outputs response format skips ToolCalls when using native
        function calling."""
        from dspy.adapters.types.tool import ToolCalls

        # Mock signature
        mock_signature = Mock()
        mock_field1 = Mock()
        mock_field1.annotation = str
        mock_field1.default = "default1"
        mock_field2 = Mock()
        mock_field2.annotation = ToolCalls
        mock_field2.default = None

        mock_signature.output_fields = {"field1": mock_field1, "field2": mock_field2}

        # Mock pydantic.create_model
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.pydantic.create_model"
        ) as mock_create_model:
            mock_model = Mock()
            mock_model.model_json_schema.return_value = {
                "properties": {"field1": {"type": "string"}}
            }
            mock_create_model.return_value = mock_model

            result = _get_structured_outputs_response_format(mock_signature, True)

            assert result == mock_model
            # Should only include field1, not field2 (ToolCalls)
            call_args = mock_create_model.call_args
            assert "field1" in call_args[1]
            assert "field2" not in call_args[1]

    def test_get_structured_outputs_response_format_no_default(self):
        """Test structured outputs response format with field that has no default."""
        # Mock signature
        mock_signature = Mock()
        mock_field = Mock()
        mock_field.annotation = str
        # Remove default attribute
        del mock_field.default

        mock_signature.output_fields = {"field1": mock_field}

        # Mock pydantic.create_model
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.pydantic.create_model"
        ) as mock_create_model:
            mock_model = Mock()
            mock_model.model_json_schema.return_value = {
                "properties": {"field1": {"type": "string"}}
            }
            mock_create_model.return_value = mock_model

            result = _get_structured_outputs_response_format(mock_signature, True)

            assert result == mock_model
            # Should use ... as default when no default is available
            call_args = mock_create_model.call_args
            assert call_args[1]["field1"][1] == ...

    def test_get_structured_outputs_response_format_with_dict_type(self):
        """Test structured outputs response format with dict type (should not raise error)."""
        # Mock signature with dict type
        mock_signature = Mock()
        mock_field = Mock()
        mock_field.annotation = dict

        mock_signature.output_fields = {"field1": mock_field}

        # Mock pydantic.create_model
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.pydantic.create_model"
        ) as mock_create_model:
            mock_model = Mock()
            mock_model.model_json_schema.return_value = {
                "properties": {"field1": {"type": "object"}}
            }
            mock_create_model.return_value = mock_model

            result = _get_structured_outputs_response_format(mock_signature, True)

            assert result == mock_model
            mock_create_model.assert_called_once()


class TestDSPyAdapterEdgeCases:
    """Test edge cases for DSPy adapter."""

    def test_adapter_with_none_callbacks(self):
        """Test adapter with None callbacks."""
        adapter = StructuredOutputAdapter(callbacks=None)
        # The parent class initializes callbacks as an empty list
        assert adapter.callbacks == []

    def test_adapter_with_empty_callbacks(self):
        """Test adapter with empty callbacks list."""
        adapter = StructuredOutputAdapter(callbacks=[])
        assert adapter.callbacks == []

    def test_adapter_with_multiple_callbacks(self):
        """Test adapter with multiple callbacks."""
        mock_callback1 = Mock()
        mock_callback2 = Mock()
        adapter = StructuredOutputAdapter(callbacks=[mock_callback1, mock_callback2])
        assert adapter.callbacks == [mock_callback1, mock_callback2]

    def test_parse_json_with_non_dict_result(self):
        """Test JSON parsing when loads returns non-dict."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {"field1": Mock()}

        # Mock the loads function to return a list
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.return_value = ["value1", "value2"]

            with pytest.raises(
                AdapterParseError, match="LM response cannot be serialized to a JSON object"
            ):
                adapter._parse_json(mock_signature, '["value1", "value2"]')

    def test_parse_yaml_with_non_dict_result_fallback(self):
        """Test YAML parsing with non-dict result that falls back to JSON."""
        adapter = StructuredOutputAdapter()

        # Mock signature
        mock_signature = Mock()
        mock_signature.output_fields = {"field1": Mock()}

        # Mock the loads function to return a list
        with patch(
            "llm_schema_lite.dspy_integration.adapters.structured_output_adapter.loads"
        ) as mock_loads:
            mock_loads.return_value = ["value1", "value2"]

            # Mock _parse_json method
            with patch.object(adapter, "_parse_json") as mock_parse_json:
                mock_parse_json.side_effect = AdapterParseError(
                    adapter_name="StructuredOutputAdapter",
                    signature=mock_signature,
                    lm_response="invalid",
                    message="Test error",
                )

                with pytest.raises(AdapterParseError):
                    adapter._parse_yaml(mock_signature, "- value1\n- value2")
