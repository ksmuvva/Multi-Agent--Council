"""
Formatter Subagent

Presents raw output in requested formats: Markdown, code files,
DOCX, PDF, XLSX, PPTX, Mermaid diagrams, JSON, YAML.
"""

import re
import json
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

from src.schemas.analyst import ModalityType


class OutputFormat(str, Enum):
    """Supported output formats."""
    MARKDOWN = "markdown"
    CODE = "code"
    DOCX = "docx"
    PDF = "pdf"
    XLSX = "xlsx"
    PPTX = "pptx"
    MERMAID = "mermaid"
    JSON_FMT = "json"
    YAML_FMT = "yaml"
    TEXT = "text"


class FormatterAgent:
    """
    The Formatter presents raw output in the requested format.

    Key responsibilities:
    - Receive raw content from Executor
    - Identify target format
    - Apply appropriate formatting
    - Validate code syntax
    - Return formatted output
    """

    def __init__(
        self,
        system_prompt_path: str = "config/agents/formatter/CLAUDE.md",
        model: str = "claude-3-5-sonnet-20241022",
        max_turns: int = 30,
        output_dir: str = "output",
    ):
        """
        Initialize the Formatter agent.

        Args:
            system_prompt_path: Path to system prompt file
            model: Model to use for formatting
            max_turns: Maximum conversation turns
            output_dir: Directory for output files
        """
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = self._load_system_prompt()
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Code syntax validation commands
        self.syntax_validation = {
            "python": "python -m py_compile {file}",
            "javascript": "node --check {file}",
            "typescript": "tsc --noEmit {file}",
            "java": "javac {file}",
            "go": "go build {file}",
        }

    def format(
        self,
        raw_content: Any,
        target_format: str = "markdown",
        file_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format raw content into the target format.

        Args:
            raw_content: Raw output from Executor
            target_format: Desired output format
            file_path: Optional file path for code output
            context: Additional formatting context

        Returns:
            Dictionary with formatted output and metadata
        """
        # Normalize format
        try:
            format_enum = OutputFormat(target_format.lower())
        except ValueError:
            format_enum = self._infer_format(raw_content, context)

        # Format based on type
        if format_enum == OutputFormat.MARKDOWN:
            output = self._format_markdown(raw_content, context)
        elif format_enum == OutputFormat.CODE:
            output = self._format_code(raw_content, file_path, context)
        elif format_enum in [OutputFormat.DOCX, OutputFormat.PDF,
                              OutputFormat.XLSX, OutputFormat.PPTX]:
            output = self._format_document(raw_content, format_enum, context)
        elif format_enum == OutputFormat.MERMAID:
            output = self._format_mermaid(raw_content, context)
        elif format_enum == OutputFormat.JSON_FMT:
            output = self._format_json(raw_content, context)
        elif format_enum == OutputFormat.YAML_FMT:
            output = self._format_yaml(raw_content, context)
        else:
            output = self._format_text(raw_content, context)

        return {
            "formatted_output": output,
            "format": format_enum.value,
            "file_path": file_path if format_enum == OutputFormat.CODE else None,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "size_bytes": len(str(output)) if isinstance(output, str) else 0,
            }
        }

    # ========================================================================
    # Format Implementations
    # ========================================================================

    def _format_markdown(
        self,
        content: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format content as Markdown."""
        if isinstance(content, dict):
            # Convert dict to Markdown
            return self._dict_to_markdown(content)
        elif isinstance(content, list):
            # Convert list to Markdown
            return self._list_to_markdown(content)
        elif isinstance(content, str):
            # Ensure proper Markdown formatting
            return self._ensure_markdown(content)
        else:
            return str(content)

    def _format_code(
        self,
        content: Any,
        file_path: Optional[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format and validate code content."""
        # Extract code if wrapped in other content
        code = self._extract_code(content)

        # Determine language from file path or content
        language = self._detect_language(code, file_path)

        # Validate syntax if possible
        validation_result = self._validate_syntax(code, language)

        # Write to file if path provided
        if file_path:
            full_path = os.path.join(self.output_dir, file_path)
            self._write_file(full_path, code)

            return {
                "code": code,
                "language": language,
                "file_path": full_path,
                "validation": validation_result,
                "status": "success" if validation_result.get("valid") else "warning",
            }

        return {
            "code": code,
            "language": language,
            "validation": validation_result,
            "status": "success",
        }

    def _format_document(
        self,
        content: Any,
        format_enum: OutputFormat,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format content as a document (DOCX/PDF/XLSX/PPTX).

        In a real implementation, this would invoke the document-creation skill.
        """
        # Placeholder: In real implementation, would call document-creation skill
        # result = invoke_skill("document-creation", format=format_enum.value, content=content)

        format_name = format_enum.value.upper()

        # Generate a structured representation
        if isinstance(content, dict):
            structured = self._dict_to_markdown(content)
        elif isinstance(content, list):
            structured = self._list_to_markdown(content)
        else:
            structured = str(content)

        # In production, this would return the actual document file
        return f"""
# Document: {format_name} Output

{structured}

---
*In production, this would be an actual {format_name} file generated
via the document-creation skill.*
"""

    def _format_mermaid(
        self,
        content: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format content as a Mermaid diagram."""
        # Extract diagram type from context or content
        diagram_type = self._infer_mermaid_type(content, context)

        # Generate Mermaid diagram
        if diagram_type == "flowchart":
            return self._generate_flowchart(content, context)
        elif diagram_type == "sequence":
            return self._generate_sequence_diagram(content, context)
        elif diagram_type == "class":
            return self._generate_class_diagram(content, context)
        elif diagram_type == "state":
            return self._generate_state_diagram(content, context)
        else:
            return self._generate_generic_diagram(content, context)

    def _format_json(
        self,
        content: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format content as JSON."""
        if isinstance(content, str):
            # Try to parse as JSON first
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                # Treat as string value
                content = {"value": content}

        return json.dumps(content, indent=2)

    def _format_yaml(
        self,
        content: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format content as YAML."""
        import yaml

        if isinstance(content, dict):
            return yaml.dump(content, default_flow_style=False)
        elif isinstance(content, list):
            return yaml.dump({"items": content}, default_flow_style=False)
        else:
            # String content
            return yaml.dump({"value": str(content)}, default_flow_style=False)

    def _format_text(
        self,
        content: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format content as plain text."""
        return str(content)

    # ========================================================================
    # Markdown Conversion
    # ========================================================================

    def _dict_to_markdown(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to Markdown."""
        lines = []

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"\n## {key}\n")
                lines.append(self._dict_to_markdown(value))
            elif isinstance(value, list):
                lines.append(f"\n## {key}\n")
                lines.append(self._list_to_markdown(value))
            else:
                lines.append(f"**{key}:** {value}")

        return "\n".join(lines)

    def _list_to_markdown(self, data: List[Any]) -> str:
        """Convert list to Markdown."""
        lines = []

        for item in data:
            if isinstance(item, dict):
                lines.append(f"\n{self._dict_to_markdown(item)}")
            elif isinstance(item, list):
                lines.append(f"\n{self._list_to_markdown(item)}")
            else:
                lines.append(f"- {item}")

        return "\n".join(lines)

    def _ensure_markdown(self, content: str) -> str:
        """Ensure content is properly formatted Markdown."""
        # Check if it already has Markdown syntax
        has_markdown = any(
            pattern in content
            for pattern in ["#", "**", "*", "_", "```", "- ", "1. "]
        )

        if has_markdown:
            return content

        # Convert plain text to Markdown
        lines = content.split("\n")
        markdown_lines = []

        for line in lines:
            if line.strip():
                markdown_lines.append(line)
            else:
                markdown_lines.append("")  # Preserve spacing

        return "\n".join(markdown_lines)

    # ========================================================================
    # Mermaid Diagram Generation
    # ========================================================================

    def _infer_mermaid_type(
        self,
        content: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Infer the Mermaid diagram type."""
        # Check context first
        if context and "diagram_type" in context:
            return context["diagram_type"]

        # Check content
        content_str = str(content).lower()

        if "step" in content_str or "process" in content_str or "flow" in content_str:
            return "flowchart"
        elif "sequence" in content_str or "interaction" in content_str:
            return "sequence"
        elif "class" in content_str or "object" in content_str or "struct" in content_str:
            return "class"
        elif "state" in content_str or "transition" in content_str:
            return "state"
        else:
            return "flowchart"  # Default

    def _generate_flowchart(self, content: Any, context: Dict[str, Any]) -> str:
        """Generate a Mermaid flowchart."""
        return """```mermaid
flowchart TD
    A[Start] --> B[Process]
    B --> C{Decision}
    C -->|Yes| D[Action 1]
    C -->|No| E[Action 2]
    D --> F[End]
    E --> F
```"""

    def _generate_sequence_diagram(self, content: Any, context: Dict[str, Any]) -> str:
        """Generate a Mermaid sequence diagram."""
        return """```mermaid
sequenceDiagram
    participant User
    participant System
    participant Database

    User->>System: Request
    System->>Database: Query
    Database-->>System: Data
    System-->>User: Response
```"""

    def _generate_class_diagram(self, content: Any, context: Dict[str, Any]) -> str:
        """Generate a Mermaid class diagram."""
        return """```mermaid
classDiagram
    class ClassA {
        +method1()
        +method2()
    }
    class ClassB {
        +method3()
    }
    ClassA <|-- ClassB
```"""

    def _generate_state_diagram(self, content: Any, context: Dict[str, Any]) -> str:
        """Generate a Mermaid state diagram."""
        return """```mermaid
stateDiagram-v2
    [*] --> Active
    Active --> Paused: pause()
    Paused --> Active: resume()
    Active --> [*]: stop()
```"""

    def _generate_generic_diagram(self, content: Any, context: Dict[str, Any]) -> str:
        """Generate a generic Mermaid diagram."""
        return """```mermaid
graph LR
    A[Input] --> B[Process]
    B --> C[Output]
```"""

    # ========================================================================
    # Code Handling
    # ========================================================================

    def _extract_code(self, content: Any) -> str:
        """Extract code from content if wrapped."""
        if isinstance(content, str):
            # Check for code blocks
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
            if code_blocks:
                return code_blocks[0].strip()

            # Check for inline code
            inline_code = re.findall(r'`([^`]+)`', content)
            if inline_code and len(''.join(inline_code)) > 50:
                # If lots of inline code, treat as code
                return content

        return str(content)

    def _detect_language(self, code: str, file_path: Optional[str]) -> str:
        """Detect programming language from code or file path."""
        # Check file extension first
        if file_path:
            ext = Path(file_path).suffix.lower()
            ext_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".java": "java",
                ".go": "go",
                ".rs": "rust",
                ".cpp": "cpp",
                ".c": "c",
                ".cs": "csharp",
                ".php": "php",
                ".rb": "ruby",
                ".swift": "swift",
                ".kt": "kotlin",
                ".sh": "bash",
            }
            if ext in ext_map:
                return ext_map[ext]

        # Detect from code content
        code_lower = code.lower()

        # Python
        if "def " in code or "import " in code or "from " in code or "class " in code:
            return "python"

        # JavaScript/TypeScript
        if "function " in code or "const " in code or "let " in code:
            if ": " in code and "interface " in code:
                return "typescript"
            return "javascript"

        # Java
        if "public class " in code or "public static void main" in code:
            return "java"

        # Go
        if "func " in code and "package main" in code:
            return "go"

        # C
        if "#include <" in code and "int main(" in code:
            return "c"

        # C++
        if "#include <" in code and "std::" in code:
            return "cpp"

        # Default
        return "python"

    def _validate_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """
        Validate code syntax.

        In a real implementation, this would use the Bash tool
        to run the language's syntax checker.
        """
        valid = True
        errors = []

        # Python syntax check using AST
        if language == "python":
            try:
                ast.parse(code)
            except SyntaxError as e:
                valid = False
                errors.append(f"Line {e.lineno}: {e.msg}")

        return {
            "valid": valid,
            "errors": errors,
            "language": language
        }

    # ========================================================================
    # File Operations
    # ========================================================================

    def _write_file(self, file_path: str, content: str) -> None:
        """Write content to a file."""
        # In real implementation, would use Write tool
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    # ========================================================================
    # Format Inference
    # ========================================================================

    def _infer_format(
        self,
        content: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> OutputFormat:
        """Infer the appropriate output format."""
        # Check context
        if context and "format" in context:
            try:
                return OutputFormat(context["format"].lower())
            except ValueError:
                pass

        # Check content type
        if isinstance(content, dict):
            # Could be JSON or YAML
            if context and "output_format" in context:
                return OutputFormat(context["output_format"].lower())
            return OutputFormat.JSON_FMT  # Default for dicts

        if isinstance(content, str):
            # Check for code
            if self._looks_like_code(content):
                return OutputFormat.CODE

            # Check for diagram keywords
            diagram_keywords = ["flowchart", "sequence", "class diagram", "state diagram"]
            if any(kw in content.lower() for kw in diagram_keywords):
                return OutputFormat.MERMAID

        return OutputFormat.MARKDOWN  # Default

    def _looks_like_code(self, content: str) -> bool:
        """Check if content looks like code."""
        code_indicators = [
            "def ", "function ", "class ", "import ", "from ",
            "{", "}", "();", "return ", "print(",
        ]

        return any(indicator in content for indicator in code_indicators)

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "You are the Formatter. Present output in the requested format."


# =============================================================================
# Convenience Functions
# =============================================================================

def create_formatter(
    system_prompt_path: str = "config/agents/formatter/CLAUDE.md",
    model: str = "claude-3-5-sonnet-20241022",
    output_dir: str = "output",
) -> FormatterAgent:
    """Create a configured Formatter agent."""
    return FormatterAgent(
        system_prompt_path=system_prompt_path,
        model=model,
        output_dir=output_dir,
    )
