"""
Tests for the FormatterAgent.

Tests format detection, markdown conversion, code formatting,
JSON/YAML formatting, and Mermaid diagram generation.
"""

import json
import pytest
from unittest.mock import patch, mock_open, MagicMock

from src.agents.formatter import (
    FormatterAgent,
    OutputFormat,
    create_formatter,
)


@pytest.fixture
def formatter(tmp_path):
    """Create a FormatterAgent with a temp output dir."""
    return FormatterAgent(
        system_prompt_path="nonexistent.md",
        output_dir=str(tmp_path),
    )


class TestFormatterInitialization:
    """Tests for FormatterAgent initialization."""

    def test_default_initialization(self, tmp_path):
        """Test default init parameters."""
        agent = FormatterAgent(
            system_prompt_path="nonexistent.md",
            output_dir=str(tmp_path),
        )
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.max_turns == 30

    def test_output_dir_created(self, tmp_path):
        """Test output directory is created."""
        output_dir = tmp_path / "new_output"
        agent = FormatterAgent(
            system_prompt_path="nonexistent.md",
            output_dir=str(output_dir),
        )
        assert output_dir.exists()

    def test_system_prompt_fallback(self, tmp_path):
        """Test fallback prompt."""
        agent = FormatterAgent(
            system_prompt_path="nonexistent.md",
            output_dir=str(tmp_path),
        )
        assert "Formatter" in agent.system_prompt

    def test_syntax_validation_configured(self, tmp_path):
        """Test syntax validation commands are configured."""
        agent = FormatterAgent(
            system_prompt_path="nonexistent.md",
            output_dir=str(tmp_path),
        )
        assert "python" in agent.syntax_validation

    def test_system_prompt_from_file(self, tmp_path):
        """Test loading from file."""
        with patch("builtins.open", mock_open(read_data="Formatter prompt")):
            agent = FormatterAgent(
                system_prompt_path="exists.md",
                output_dir=str(tmp_path),
            )
            assert agent.system_prompt == "Formatter prompt"


class TestFormat:
    """Tests for the format method."""

    def test_markdown_format(self, formatter):
        """Test markdown formatting."""
        result = formatter.format("# Hello World", target_format="markdown")
        assert result["format"] == "markdown"
        assert result["formatted_output"] is not None

    def test_json_format(self, formatter):
        """Test JSON formatting."""
        result = formatter.format({"key": "value"}, target_format="json")
        assert result["format"] == "json"
        parsed = json.loads(result["formatted_output"])
        assert parsed["key"] == "value"

    def test_text_format(self, formatter):
        """Test text formatting."""
        result = formatter.format("plain text", target_format="text")
        assert result["format"] == "text"
        assert result["formatted_output"] == "plain text"

    def test_code_format(self, formatter):
        """Test code formatting."""
        code = "def hello():\n    print('Hello')"
        result = formatter.format(code, target_format="code")
        assert result["format"] == "code"

    def test_mermaid_format(self, formatter):
        """Test Mermaid diagram formatting."""
        result = formatter.format("Process flow", target_format="mermaid")
        assert result["format"] == "mermaid"
        assert "mermaid" in result["formatted_output"]

    def test_unknown_format_inferred(self, formatter):
        """Test unknown format is inferred."""
        result = formatter.format("Hello", target_format="unknown_format")
        assert result["format"] is not None

    def test_metadata_included(self, formatter):
        """Test metadata is included in output."""
        result = formatter.format("Hello", target_format="text")
        assert "metadata" in result
        assert "timestamp" in result["metadata"]
        assert "size_bytes" in result["metadata"]


class TestMarkdownFormatting:
    """Tests for Markdown-specific formatting."""

    def test_dict_to_markdown(self, formatter):
        """Test dict-to-markdown conversion."""
        data = {"Title": "Hello", "Content": "World"}
        md = formatter._dict_to_markdown(data)
        assert "**Title:**" in md
        assert "Hello" in md

    def test_list_to_markdown(self, formatter):
        """Test list-to-markdown conversion."""
        data = ["Item 1", "Item 2", "Item 3"]
        md = formatter._list_to_markdown(data)
        assert "- Item 1" in md
        assert "- Item 2" in md

    def test_nested_dict_to_markdown(self, formatter):
        """Test nested dict conversion."""
        data = {"Section": {"Key": "Value"}}
        md = formatter._dict_to_markdown(data)
        assert "## Section" in md

    def test_ensure_markdown_passthrough(self, formatter):
        """Test already-formatted markdown passes through."""
        content = "# Title\n\n**Bold** text"
        result = formatter._ensure_markdown(content)
        assert "# Title" in result

    def test_plain_text_to_markdown(self, formatter):
        """Test plain text preservation."""
        content = "Just plain text\nWith lines"
        result = formatter._ensure_markdown(content)
        assert "Just plain text" in result


class TestCodeFormatting:
    """Tests for code-specific formatting."""

    def test_extract_code_from_blocks(self, formatter):
        """Test code extraction from markdown code blocks."""
        content = "```python\ndef hello():\n    pass\n```"
        code = formatter._extract_code(content)
        assert "def hello():" in code

    @pytest.mark.parametrize("file_path,expected_lang", [
        ("test.py", "python"),
        ("test.js", "javascript"),
        ("test.ts", "typescript"),
        ("test.java", "java"),
        ("test.go", "go"),
        ("test.rs", "rust"),
    ])
    def test_language_detection_from_extension(self, formatter, file_path, expected_lang):
        """Test language detection from file extension."""
        lang = formatter._detect_language("", file_path)
        assert lang == expected_lang

    def test_language_detection_from_content(self, formatter):
        """Test language detection from code content."""
        code = "def hello():\n    import os\n    pass"
        lang = formatter._detect_language(code, None)
        assert lang == "python"

    def test_python_syntax_validation_valid(self, formatter):
        """Test valid Python syntax passes."""
        result = formatter._validate_syntax("x = 1 + 2", "python")
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_python_syntax_validation_invalid(self, formatter):
        """Test invalid Python syntax fails."""
        result = formatter._validate_syntax("def x(:\n    pass", "python")
        assert result["valid"] is False
        assert len(result["errors"]) > 0


class TestJSONFormatting:
    """Tests for JSON formatting."""

    def test_dict_to_json(self, formatter):
        """Test dict to JSON formatting."""
        result = formatter._format_json({"key": "value"})
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_string_to_json(self, formatter):
        """Test string to JSON formatting."""
        result = formatter._format_json("hello world")
        parsed = json.loads(result)
        assert parsed["value"] == "hello world"

    def test_json_string_passthrough(self, formatter):
        """Test JSON string is parsed and re-formatted."""
        result = formatter._format_json('{"a": 1}')
        parsed = json.loads(result)
        assert parsed["a"] == 1


class TestFormatInference:
    """Tests for format inference."""

    def test_infer_json_for_dict(self, formatter):
        """Test dict infers JSON format."""
        fmt = formatter._infer_format({"key": "value"})
        assert fmt == OutputFormat.JSON_FMT

    def test_infer_code_for_code_content(self, formatter):
        """Test code content infers CODE format."""
        fmt = formatter._infer_format("def hello():\n    return 42")
        assert fmt == OutputFormat.CODE

    def test_infer_mermaid_for_diagram(self, formatter):
        """Test diagram keywords infer MERMAID format."""
        fmt = formatter._infer_format("Create a flowchart for the process")
        assert fmt == OutputFormat.MERMAID

    def test_infer_markdown_default(self, formatter):
        """Test default inference is MARKDOWN."""
        fmt = formatter._infer_format("Just some text")
        assert fmt == OutputFormat.MARKDOWN


class TestMermaidDiagrams:
    """Tests for Mermaid diagram generation."""

    def test_flowchart_generation(self, formatter):
        """Test flowchart generation."""
        result = formatter._generate_flowchart("Process flow", {})
        assert "flowchart" in result
        assert "mermaid" in result

    def test_sequence_diagram_generation(self, formatter):
        """Test sequence diagram generation."""
        result = formatter._generate_sequence_diagram("Interaction", {})
        assert "sequenceDiagram" in result

    def test_class_diagram_generation(self, formatter):
        """Test class diagram generation."""
        result = formatter._generate_class_diagram("Class structure", {})
        assert "classDiagram" in result

    def test_infer_mermaid_type(self, formatter):
        """Test Mermaid type inference."""
        assert formatter._infer_mermaid_type("process flow", None) == "flowchart"
        assert formatter._infer_mermaid_type("sequence interaction", None) == "sequence"
        assert formatter._infer_mermaid_type("class object", None) == "class"


class TestConvenienceFunction:
    """Tests for create_formatter convenience function."""

    def test_create_formatter(self):
        """Test convenience function creates a FormatterAgent."""
        agent = create_formatter(system_prompt_path="nonexistent.md")
        assert isinstance(agent, FormatterAgent)
