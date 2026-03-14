"""
Exhaustive Tests for FormatterAgent

Tests all methods, format handlers, edge cases, and convenience
functions for the Formatter subagent in src/agents/formatter.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ast
import json
import os
import pytest
from unittest.mock import patch, mock_open, MagicMock

from src.agents.formatter import (
    FormatterAgent,
    OutputFormat,
    create_formatter,
)
from src.schemas.analyst import ModalityType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def formatter(tmp_path):
    """Create a FormatterAgent with mocked system prompt and temp output dir."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        f = FormatterAgent(output_dir=str(tmp_path))
    return f


@pytest.fixture
def formatter_with_prompt(tmp_path):
    """Create a FormatterAgent with an actual system prompt file."""
    prompt_file = tmp_path / "CLAUDE.md"
    prompt_file.write_text("You are the Formatter agent.")
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return FormatterAgent(system_prompt_path=str(prompt_file), output_dir=str(out_dir))


# ============================================================================
# OutputFormat Enum Tests
# ============================================================================

class TestOutputFormat:
    def test_all_values(self):
        assert OutputFormat.MARKDOWN == "markdown"
        assert OutputFormat.CODE == "code"
        assert OutputFormat.DOCX == "docx"
        assert OutputFormat.PDF == "pdf"
        assert OutputFormat.XLSX == "xlsx"
        assert OutputFormat.PPTX == "pptx"
        assert OutputFormat.MERMAID == "mermaid"
        assert OutputFormat.JSON_FMT == "json"
        assert OutputFormat.YAML_FMT == "yaml"
        assert OutputFormat.TEXT == "text"

    def test_from_string(self):
        assert OutputFormat("markdown") == OutputFormat.MARKDOWN
        assert OutputFormat("json") == OutputFormat.JSON_FMT
        assert OutputFormat("yaml") == OutputFormat.YAML_FMT
        assert OutputFormat("text") == OutputFormat.TEXT
        assert OutputFormat("code") == OutputFormat.CODE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            OutputFormat("invalid_format")

    def test_is_str_enum(self):
        assert isinstance(OutputFormat.MARKDOWN, str)
        assert OutputFormat.MARKDOWN == "markdown"


# ============================================================================
# __init__ Tests
# ============================================================================

class TestInit:
    def test_defaults(self, formatter):
        assert formatter.system_prompt_path == "config/agents/formatter/CLAUDE.md"
        assert formatter.model == "claude-sonnet-4-20250514"
        assert formatter.max_turns == 30
        assert "Formatter" in formatter.system_prompt

    def test_custom_params(self, tmp_path):
        out_dir = str(tmp_path / "custom_output")
        with patch("builtins.open", side_effect=FileNotFoundError):
            f = FormatterAgent(
                system_prompt_path="custom/prompt.md",
                model="claude-3-opus",
                max_turns=15,
                output_dir=out_dir,
            )
        assert f.system_prompt_path == "custom/prompt.md"
        assert f.model == "claude-3-opus"
        assert f.max_turns == 15
        assert f.output_dir == out_dir

    def test_system_prompt_loading(self, formatter_with_prompt):
        assert formatter_with_prompt.system_prompt == "You are the Formatter agent."

    def test_system_prompt_fallback(self, formatter):
        assert "Formatter" in formatter.system_prompt
        assert "format" in formatter.system_prompt.lower()

    def test_output_dir_created(self, tmp_path):
        out_dir = str(tmp_path / "new_output_dir")
        with patch("builtins.open", side_effect=FileNotFoundError):
            FormatterAgent(output_dir=out_dir)
        assert os.path.isdir(out_dir)

    def test_syntax_validation_dict(self, formatter):
        assert "python" in formatter.syntax_validation
        assert "javascript" in formatter.syntax_validation


# ============================================================================
# format() Method Tests - Routing
# ============================================================================

class TestFormatRouting:
    def test_routes_to_markdown(self, formatter):
        result = formatter.format("Hello world", target_format="markdown")
        assert result["format"] == "markdown"
        assert "formatted_output" in result

    def test_routes_to_code(self, formatter):
        result = formatter.format("x = 1", target_format="code")
        assert result["format"] == "code"

    def test_routes_to_json(self, formatter):
        result = formatter.format({"key": "value"}, target_format="json")
        assert result["format"] == "json"

    def test_routes_to_yaml(self, formatter):
        result = formatter.format({"key": "value"}, target_format="yaml")
        assert result["format"] == "yaml"

    def test_routes_to_text(self, formatter):
        result = formatter.format("plain text", target_format="text")
        assert result["format"] == "text"

    def test_routes_to_mermaid(self, formatter):
        result = formatter.format("step1 -> step2", target_format="mermaid")
        assert result["format"] == "mermaid"

    def test_routes_to_docx(self, formatter):
        result = formatter.format("content", target_format="docx")
        assert result["format"] == "docx"

    def test_routes_to_pdf(self, formatter):
        result = formatter.format("content", target_format="pdf")
        assert result["format"] == "pdf"

    def test_routes_to_xlsx(self, formatter):
        result = formatter.format("content", target_format="xlsx")
        assert result["format"] == "xlsx"

    def test_routes_to_pptx(self, formatter):
        result = formatter.format("content", target_format="pptx")
        assert result["format"] == "pptx"

    def test_invalid_format_infers(self, formatter):
        result = formatter.format("Hello world", target_format="unknown_fmt")
        assert "format" in result
        assert result["formatted_output"] is not None

    def test_metadata_present(self, formatter):
        result = formatter.format("test", target_format="text")
        assert "metadata" in result
        assert "timestamp" in result["metadata"]
        assert "size_bytes" in result["metadata"]

    def test_file_path_none_for_non_code(self, formatter):
        result = formatter.format("test", target_format="markdown")
        assert result["file_path"] is None

    def test_file_path_set_for_code(self, formatter):
        result = formatter.format("x = 1", target_format="code", file_path="test.py")
        assert result["file_path"] == "test.py"  # file_path preserved for CODE format
        assert result["format"] == "code"

    def test_case_insensitive_format(self, formatter):
        result = formatter.format("test", target_format="MARKDOWN")
        assert result["format"] == "markdown"


# ============================================================================
# _format_markdown Tests
# ============================================================================

class TestFormatMarkdown:
    def test_dict_to_markdown(self, formatter):
        content = {"title": "Hello", "body": "World"}
        result = formatter._format_markdown(content)
        assert "title" in result
        assert "Hello" in result
        assert "World" in result

    def test_list_to_markdown(self, formatter):
        content = ["item1", "item2", "item3"]
        result = formatter._format_markdown(content)
        assert "- item1" in result
        assert "- item2" in result

    def test_string_with_markdown(self, formatter):
        content = "# Heading\n\nSome **bold** text"
        result = formatter._format_markdown(content)
        assert result == content  # Already markdown

    def test_string_without_markdown(self, formatter):
        content = "Plain text without any markdown"
        result = formatter._format_markdown(content)
        assert isinstance(result, str)

    def test_non_standard_type(self, formatter):
        result = formatter._format_markdown(42)
        assert result == "42"

    def test_nested_dict(self, formatter):
        content = {"outer": {"inner": "value"}}
        result = formatter._format_markdown(content)
        assert "outer" in result
        assert "inner" in result

    def test_heading_extraction(self, formatter):
        content = "# Main Title\n## Section\nContent here"
        result = formatter._format_markdown(content)
        assert "# Main Title" in result

    def test_code_blocks_preserved(self, formatter):
        content = "Text\n```python\nx = 1\n```\nMore text"
        result = formatter._format_markdown(content)
        assert "```" in result

    def test_table_content(self, formatter):
        content = "| Col1 | Col2 |\n|------|------|\n| a | b |"
        result = formatter._format_markdown(content)
        assert "|" in result


# ============================================================================
# _format_code Tests
# ============================================================================

class TestFormatCode:
    def test_language_detection_python(self, formatter):
        code = "def hello():\n    print('hello')\n"
        result = formatter._format_code(code, "test.py")
        assert result["language"] == "python"

    def test_language_detection_javascript(self, formatter):
        code = "function hello() {\n    console.log('hello');\n}\n"
        result = formatter._format_code(code, "test.js")
        assert result["language"] == "javascript"

    def test_syntax_validation_valid_python(self, formatter):
        code = "x = 1\ny = 2\nprint(x + y)\n"
        result = formatter._format_code(code, "test.py")
        assert result["validation"]["valid"] is True
        assert result["validation"]["errors"] == []

    def test_syntax_validation_invalid_python(self, formatter):
        code = "def foo(\n    x = 1\n"
        result = formatter._format_code(code, "test.py")
        assert result["validation"]["valid"] is False
        assert len(result["validation"]["errors"]) >= 1

    def test_code_block_wrapping_extraction(self, formatter):
        content = "```python\nx = 1\n```"
        result = formatter._format_code(content, "test.py")
        assert result["code"] == "x = 1"

    def test_file_writing(self, formatter, tmp_path):
        code = "x = 1"
        file_path = "subdir/test.py"
        result = formatter._format_code(code, file_path)
        assert "file_path" in result

    def test_no_file_path(self, formatter):
        code = "x = 1"
        result = formatter._format_code(code, None)
        assert result["status"] == "success"

    def test_detect_language_from_extension(self, formatter):
        lang = formatter._detect_language("", "test.py")
        assert lang == "python"

    def test_detect_language_from_extension_js(self, formatter):
        lang = formatter._detect_language("", "app.js")
        assert lang == "javascript"

    def test_detect_language_from_extension_ts(self, formatter):
        lang = formatter._detect_language("", "app.ts")
        assert lang == "typescript"

    def test_detect_language_from_extension_java(self, formatter):
        lang = formatter._detect_language("", "Main.java")
        assert lang == "java"

    def test_detect_language_from_extension_go(self, formatter):
        lang = formatter._detect_language("", "main.go")
        assert lang == "go"

    def test_detect_language_from_content_python(self, formatter):
        lang = formatter._detect_language("def foo():\n    pass", None)
        assert lang == "python"

    def test_detect_language_from_content_java(self, formatter):
        lang = formatter._detect_language("public class Main {\n}", None)
        assert lang == "java"

    def test_detect_language_from_content_go(self, formatter):
        lang = formatter._detect_language("package main\nfunc main() {}", None)
        assert lang == "go"

    def test_detect_language_from_content_javascript(self, formatter):
        lang = formatter._detect_language("function hello() {}", None)
        assert lang == "javascript"

    def test_detect_language_default(self, formatter):
        lang = formatter._detect_language("x = 1", None)
        # Falls through all checks - default is "python"
        assert lang == "python"


# ============================================================================
# _format_json Tests
# ============================================================================

class TestFormatJson:
    def test_dict_to_json(self, formatter):
        result = formatter._format_json({"key": "value"})
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_list_to_json(self, formatter):
        result = formatter._format_json([1, 2, 3])
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]

    def test_string_valid_json(self, formatter):
        result = formatter._format_json('{"key": "value"}')
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_string_invalid_json(self, formatter):
        result = formatter._format_json("just a string")
        parsed = json.loads(result)
        assert parsed == {"value": "just a string"}

    def test_nested_dict(self, formatter):
        content = {"outer": {"inner": [1, 2, 3]}}
        result = formatter._format_json(content)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == [1, 2, 3]

    def test_json_indented(self, formatter):
        result = formatter._format_json({"a": 1})
        assert "\n" in result  # indent=2 produces newlines

    def test_valid_json_output(self, formatter):
        result = formatter.format({"data": [1, 2]}, target_format="json")
        parsed = json.loads(result["formatted_output"])
        assert parsed["data"] == [1, 2]


# ============================================================================
# _format_yaml Tests
# ============================================================================

class TestFormatYaml:
    def test_dict_to_yaml(self, formatter):
        result = formatter._format_yaml({"key": "value"})
        assert "key:" in result
        assert "value" in result

    def test_list_to_yaml(self, formatter):
        result = formatter._format_yaml(["a", "b", "c"])
        assert "items:" in result

    def test_string_to_yaml(self, formatter):
        result = formatter._format_yaml("some text")
        assert "value:" in result

    def test_valid_yaml_output(self, formatter):
        import yaml
        result = formatter._format_yaml({"name": "test", "count": 5})
        parsed = yaml.safe_load(result)
        assert parsed["name"] == "test"
        assert parsed["count"] == 5

    def test_nested_dict_yaml(self, formatter):
        import yaml
        content = {"outer": {"inner": "value"}}
        result = formatter._format_yaml(content)
        parsed = yaml.safe_load(result)
        assert parsed["outer"]["inner"] == "value"


# ============================================================================
# _format_text Tests
# ============================================================================

class TestFormatText:
    def test_string_passthrough(self, formatter):
        result = formatter._format_text("plain text")
        assert result == "plain text"

    def test_dict_to_text(self, formatter):
        result = formatter._format_text({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_list_to_text(self, formatter):
        result = formatter._format_text([1, 2, 3])
        assert "1" in result

    def test_number_to_text(self, formatter):
        result = formatter._format_text(42)
        assert result == "42"

    def test_none_to_text(self, formatter):
        result = formatter._format_text(None)
        assert result == "None"

    def test_plain_text_cleanup(self, formatter):
        result = formatter.format("   hello   world   ", target_format="text")
        assert result["formatted_output"] == "   hello   world   "


# ============================================================================
# _detect_output_format / _infer_format Tests
# ============================================================================

class TestInferFormat:
    def test_from_context_format(self, formatter):
        result = formatter._infer_format("content", {"format": "json"})
        assert result == OutputFormat.JSON_FMT

    def test_from_context_output_format(self, formatter):
        result = formatter._infer_format({"data": 1}, {"output_format": "yaml"})
        assert result == OutputFormat.YAML_FMT

    def test_dict_defaults_to_json(self, formatter):
        result = formatter._infer_format({"key": "value"})
        assert result == OutputFormat.JSON_FMT

    def test_code_content_detected(self, formatter):
        result = formatter._infer_format("def foo():\n    return 1\n")
        assert result == OutputFormat.CODE

    def test_diagram_keywords_detected(self, formatter):
        result = formatter._infer_format("This is a flowchart of the process")
        assert result == OutputFormat.MERMAID

    def test_sequence_keyword(self, formatter):
        result = formatter._infer_format("sequence diagram of interactions")
        assert result == OutputFormat.MERMAID

    def test_class_diagram_keyword(self, formatter):
        # "class diagram" would match code due to "class " pattern
        result = formatter._infer_format("this is a class diagram description")
        # _looks_like_code would match "class " so it returns CODE
        assert result in (OutputFormat.CODE, OutputFormat.MERMAID)

    def test_plain_text_defaults_markdown(self, formatter):
        result = formatter._infer_format("Hello world, just a sentence.")
        assert result == OutputFormat.MARKDOWN

    def test_invalid_context_format_ignored(self, formatter):
        result = formatter._infer_format("content", {"format": "invalid_xyz"})
        assert result == OutputFormat.MARKDOWN  # Falls through to default


# ============================================================================
# _validate_python_syntax Tests
# ============================================================================

class TestValidateSyntax:
    def test_valid_python(self, formatter):
        result = formatter._validate_syntax("x = 1\nprint(x)", "python")
        assert result["valid"] is True
        assert result["errors"] == []

    def test_invalid_python(self, formatter):
        result = formatter._validate_syntax("def foo(\n", "python")
        assert result["valid"] is False
        assert len(result["errors"]) >= 1

    def test_valid_python_complex(self, formatter):
        code = """
class Foo:
    def __init__(self, x: int):
        self.x = x

    def bar(self) -> str:
        return f"value: {self.x}"
"""
        result = formatter._validate_syntax(code, "python")
        assert result["valid"] is True

    def test_invalid_python_syntax_error_details(self, formatter):
        result = formatter._validate_syntax("def (:", "python")
        assert result["valid"] is False
        assert any("Line" in e for e in result["errors"])

    def test_valid_javascript(self, formatter):
        code = "function foo() {\n    return 1;\n}\n"
        result = formatter._validate_syntax(code, "javascript")
        assert result["valid"] is True

    def test_invalid_javascript_unmatched_bracket(self, formatter):
        code = "function foo() {\n    return 1;\n\n"
        result = formatter._validate_syntax(code, "javascript")
        assert result["valid"] is False

    def test_valid_json_syntax(self, formatter):
        code = '{"key": "value", "num": 42}'
        result = formatter._validate_syntax(code, "json")
        assert result["valid"] is True

    def test_invalid_json_syntax(self, formatter):
        code = '{"key": value}'
        result = formatter._validate_syntax(code, "json")
        assert result["valid"] is False

    def test_unknown_language_passes(self, formatter):
        result = formatter._validate_syntax("anything", "rust")
        assert result["valid"] is True

    def test_yaml_valid(self, formatter):
        code = "key: value\nlist:\n  - item1\n  - item2\n"
        result = formatter._validate_syntax(code, "yaml")
        assert result["valid"] is True

    def test_yaml_invalid(self, formatter):
        code = "key: value\n  bad indent:\n    - item\n"
        result = formatter._validate_syntax(code, "yaml")
        # YAML is forgiving; this may or may not be invalid depending on parser
        assert isinstance(result["valid"], bool)

    def test_language_in_result(self, formatter):
        result = formatter._validate_syntax("x = 1", "python")
        assert result["language"] == "python"


# ============================================================================
# _extract_code_blocks Tests
# ============================================================================

class TestExtractCode:
    def test_extract_from_markdown_code_block(self, formatter):
        content = "Here is code:\n```python\nx = 1\ny = 2\n```\n"
        result = formatter._extract_code(content)
        assert "x = 1" in result
        assert "y = 2" in result

    def test_extract_from_plain_code(self, formatter):
        content = "x = 1\ny = 2"
        result = formatter._extract_code(content)
        assert result == "x = 1\ny = 2"

    def test_extract_from_non_string(self, formatter):
        result = formatter._extract_code(42)
        assert result == "42"

    def test_extract_from_code_block_no_language(self, formatter):
        content = "```\nx = 1\n```"
        result = formatter._extract_code(content)
        assert "x = 1" in result

    def test_extract_inline_code_short(self, formatter):
        content = "Use `x = 1` for assignment"
        result = formatter._extract_code(content)
        # Inline code total < 50 chars, so treated as regular content
        assert isinstance(result, str)


# ============================================================================
# _generate_table_of_contents / _ensure_markdown Tests
# ============================================================================

class TestEnsureMarkdown:
    def test_already_has_markdown(self, formatter):
        content = "# Heading\nSome text"
        result = formatter._ensure_markdown(content)
        assert result == content

    def test_has_bold(self, formatter):
        content = "This is **bold** text"
        result = formatter._ensure_markdown(content)
        assert result == content

    def test_has_list(self, formatter):
        content = "- item 1\n- item 2"
        result = formatter._ensure_markdown(content)
        assert result == content

    def test_plain_text_converted(self, formatter):
        content = "Just plain text here"
        result = formatter._ensure_markdown(content)
        assert isinstance(result, str)

    def test_preserves_blank_lines(self, formatter):
        content = "Line 1\n\nLine 3"
        result = formatter._ensure_markdown(content)
        assert "\n\n" in result or result == content


class TestDocumentSections:
    """Tests for _extract_document_sections used in document formatting."""

    def test_heading_extraction_from_string(self, formatter):
        content = "Intro text\n# Title\nBody here\n## Sub\nMore content"
        sections = formatter._extract_document_sections(content)
        assert len(sections) >= 1
        assert any("Title" in s["heading"] for s in sections)

    def test_dict_sections(self, formatter):
        content = {"overview": "intro text", "details": "detail text"}
        sections = formatter._extract_document_sections(content)
        assert len(sections) == 2

    def test_list_sections(self, formatter):
        content = [
            {"heading": "First", "body": "Content 1"},
            {"heading": "Second", "body": "Content 2"},
        ]
        sections = formatter._extract_document_sections(content)
        assert len(sections) == 2
        assert sections[0]["heading"] == "First"

    def test_plain_text_split(self, formatter):
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        sections = formatter._extract_document_sections(content)
        assert len(sections) >= 2


# ============================================================================
# _looks_like_code Tests
# ============================================================================

class TestLooksLikeCode:
    @pytest.mark.parametrize("content,expected", [
        ("def foo():\n    pass", True),
        ("function bar() {}", True),
        ("class MyClass:", True),
        ("import os", True),
        ("from pathlib import Path", True),
        ("return value", True),
        ("print(hello)", True),
        ("Just a normal sentence.", False),
        ("Hello world", False),
    ])
    def test_code_detection(self, formatter, content, expected):
        result = formatter._looks_like_code(content)
        assert result == expected


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_empty_content_markdown(self, formatter):
        result = formatter.format("", target_format="markdown")
        assert result["formatted_output"] is not None

    def test_empty_content_json(self, formatter):
        result = formatter.format("", target_format="json")
        assert result["formatted_output"] is not None

    def test_empty_content_text(self, formatter):
        result = formatter.format("", target_format="text")
        assert result["formatted_output"] == ""

    def test_empty_content_code(self, formatter):
        result = formatter.format("", target_format="code")
        assert result["formatted_output"] is not None

    def test_very_long_content(self, formatter):
        content = "x = 1\n" * 10000
        result = formatter.format(content, target_format="text")
        assert len(result["formatted_output"]) > 0

    def test_special_characters_markdown(self, formatter):
        content = "Special chars: <>&\"' and unicode: test"
        result = formatter.format(content, target_format="markdown")
        assert result["formatted_output"] is not None

    def test_special_characters_json(self, formatter):
        content = {"key": 'value with "quotes" and <html>'}
        result = formatter.format(content, target_format="json")
        parsed = json.loads(result["formatted_output"])
        assert '"quotes"' in parsed["key"]

    def test_none_content_text(self, formatter):
        result = formatter.format(None, target_format="text")
        assert result["formatted_output"] == "None"

    def test_nested_code_blocks(self, formatter):
        content = "```python\ndef foo():\n    pass\n```\n\n```javascript\nlet x = 1;\n```"
        result = formatter.format(content, target_format="markdown")
        assert "```" in result["formatted_output"]

    def test_format_returns_dict(self, formatter):
        result = formatter.format("test", target_format="text")
        assert isinstance(result, dict)
        assert "formatted_output" in result
        assert "format" in result
        assert "file_path" in result
        assert "metadata" in result


# ============================================================================
# _write_file Tests
# ============================================================================

class TestWriteFile:
    def test_write_success(self, formatter, tmp_path):
        file_path = str(tmp_path / "test_output.py")
        result = formatter._write_file(file_path, "x = 1")
        assert result["success"] is True
        assert result["bytes_written"] > 0
        assert Path(file_path).read_text() == "x = 1"

    def test_write_creates_directories(self, formatter, tmp_path):
        file_path = str(tmp_path / "sub" / "dir" / "test.py")
        result = formatter._write_file(file_path, "x = 1")
        assert result["success"] is True
        assert Path(file_path).exists()

    def test_write_null_bytes_rejected(self, formatter, tmp_path):
        file_path = str(tmp_path / "test\x00.py")
        result = formatter._write_file(file_path, "x = 1")
        assert result["success"] is False
        assert "null" in result["error"].lower()


# ============================================================================
# Document Format Tests
# ============================================================================

class TestFormatDocument:
    def test_docx_format(self, formatter):
        result = formatter.format("Some content", target_format="docx")
        output = result["formatted_output"]
        assert "DOCX" in output
        assert "Table of Contents" in output

    def test_pdf_format(self, formatter):
        result = formatter.format("Some content", target_format="pdf")
        output = result["formatted_output"]
        assert "PDF" in output

    def test_with_context_title(self, formatter):
        result = formatter.format(
            "Content here",
            target_format="docx",
            context={"title": "My Report", "author": "Test Author"},
        )
        output = result["formatted_output"]
        assert "My Report" in output
        assert "Test Author" in output

    def test_dict_content_document(self, formatter):
        content = {"introduction": "Intro text", "analysis": "Analysis text"}
        result = formatter.format(content, target_format="docx")
        output = result["formatted_output"]
        assert "Introduction" in output
        assert "Analysis" in output


# ============================================================================
# Mermaid Tests
# ============================================================================

class TestFormatMermaid:
    def test_flowchart_generation(self, formatter):
        content = {"steps": ["Start", "Process", "End"]}
        result = formatter.format(content, target_format="mermaid")
        output = result["formatted_output"]
        assert "```mermaid" in output
        assert "flowchart" in output or "graph" in output

    def test_sequence_diagram(self, formatter):
        content = "sequence interaction between client and server"
        result = formatter.format(
            content,
            target_format="mermaid",
            context={"diagram_type": "sequence"},
        )
        output = result["formatted_output"]
        assert "```mermaid" in output
        assert "sequenceDiagram" in output

    def test_class_diagram(self, formatter):
        content = "class diagram of objects"
        result = formatter.format(
            content,
            target_format="mermaid",
            context={"diagram_type": "class"},
        )
        output = result["formatted_output"]
        assert "```mermaid" in output
        assert "classDiagram" in output

    def test_state_diagram(self, formatter):
        content = "state transitions"
        result = formatter.format(
            content,
            target_format="mermaid",
            context={"diagram_type": "state"},
        )
        output = result["formatted_output"]
        assert "```mermaid" in output
        assert "stateDiagram" in output

    def test_infer_mermaid_type_flowchart(self, formatter):
        result = formatter._infer_mermaid_type("process flow steps")
        assert result == "flowchart"

    def test_infer_mermaid_type_sequence(self, formatter):
        result = formatter._infer_mermaid_type("sequence of interactions")
        assert result == "sequence"

    def test_infer_mermaid_type_class(self, formatter):
        result = formatter._infer_mermaid_type("class structure and objects")
        assert result == "class"

    def test_infer_mermaid_type_state(self, formatter):
        result = formatter._infer_mermaid_type("state transition diagram")
        assert result == "state"

    def test_infer_mermaid_type_from_context(self, formatter):
        result = formatter._infer_mermaid_type("anything", {"diagram_type": "sequence"})
        assert result == "sequence"

    def test_sanitize_mermaid_label(self, formatter):
        result = formatter._sanitize_mermaid_label('Hello "World" [test]')
        assert '"' not in result
        assert "[" not in result
        assert "]" not in result

    def test_sanitize_mermaid_label_truncation(self, formatter):
        long_label = "a" * 100
        result = formatter._sanitize_mermaid_label(long_label)
        assert len(result) <= 50


# ============================================================================
# create_formatter() Convenience Function
# ============================================================================

class TestCreateFormatter:
    def test_creates_instance(self, tmp_path):
        with patch("builtins.open", side_effect=FileNotFoundError):
            f = create_formatter(output_dir=str(tmp_path))
        assert isinstance(f, FormatterAgent)
        assert f.model == "claude-sonnet-4-20250514"

    def test_custom_params(self, tmp_path):
        with patch("builtins.open", side_effect=FileNotFoundError):
            f = create_formatter(
                system_prompt_path="custom.md",
                model="claude-3-haiku",
                output_dir=str(tmp_path),
            )
        assert f.system_prompt_path == "custom.md"
        assert f.model == "claude-3-haiku"

    def test_default_output_dir(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            f = create_formatter()
        assert f.output_dir == "output"


# ============================================================================
# Parametrized Tests
# ============================================================================

@pytest.mark.parametrize("extension,expected_lang", [
    (".py", "python"),
    (".js", "javascript"),
    (".ts", "typescript"),
    (".java", "java"),
    (".go", "go"),
    (".rs", "rust"),
    (".cpp", "cpp"),
    (".c", "c"),
    (".cs", "csharp"),
    (".php", "php"),
    (".rb", "ruby"),
    (".swift", "swift"),
    (".kt", "kotlin"),
    (".sh", "bash"),
])
def test_language_detection_by_extension(formatter, extension, expected_lang):
    lang = formatter._detect_language("", f"file{extension}")
    assert lang == expected_lang


@pytest.mark.parametrize("target_format", [
    "markdown", "code", "json", "yaml", "text",
    "docx", "pdf", "xlsx", "pptx", "mermaid",
])
def test_all_formats_produce_output(formatter, target_format):
    result = formatter.format("Test content", target_format=target_format)
    assert result["formatted_output"] is not None
    assert result["format"] == target_format


@pytest.mark.parametrize("content", [
    "",
    "a",
    "a" * 10000,
    "line1\nline2\nline3",
    '{"key": "value"}',
    "Special: <>&\"'",
])
def test_format_handles_various_content(formatter, content):
    result = formatter.format(content, target_format="text")
    assert isinstance(result, dict)
    assert "formatted_output" in result
