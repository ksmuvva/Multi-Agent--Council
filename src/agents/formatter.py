"""
Formatter Subagent

Presents raw output in requested formats: Markdown, code files,
DOCX, PDF, XLSX, PPTX, Mermaid diagrams, JSON, YAML.
"""

import ast
import re
import json
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

from src.schemas.analyst import ModalityType
from src.utils.logging import get_agent_logger, AgentLogContext
from src.utils.events import emit_agent_started, emit_agent_completed, emit_error
from src.core.react import ReactLoop


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

        # Structured logger
        self.logger = get_agent_logger("formatter")
        self.logger.info("FormatterAgent initialized", model=model, output_dir=output_dir)

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
        mode: str = "react",
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
        if mode == "react":
            return self._react_format_output(raw_content, target_format, context)

        self.logger.info(
            "Formatting started",
            target_format=target_format,
            content_type=type(raw_content).__name__,
            content_length=len(str(raw_content)),
            has_file_path=file_path is not None,
        )
        emit_agent_started("formatter", phase="formatting")

        # Normalize format
        try:
            format_enum = OutputFormat(target_format.lower())
        except ValueError:
            format_enum = self._infer_format(raw_content, context)
            self.logger.info("Format inferred from content", inferred_format=format_enum.value)

        self.logger.debug("Format type selected", format=format_enum.value)

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

        result = {
            "formatted_output": output,
            "format": format_enum.value,
            "file_path": file_path if format_enum == OutputFormat.CODE else None,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "size_bytes": len(str(output)) if isinstance(output, str) else 0,
            }
        }
        self.logger.info(
            "Formatting completed",
            format=format_enum.value,
            output_size=result["metadata"]["size_bytes"],
        )
        emit_agent_completed("formatter", output_summary=f"Formatted as {format_enum.value}")
        return result

    def _react_format_output(
        self,
        raw_content: Any,
        target_format: str = "markdown",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run formatting using ReAct loop."""
        react_instruction = (
            "You are the Formatter. Transform raw content into the requested output format. "
            "Supported formats: markdown, code, JSON, YAML, text, and file formats "
            "(DOCX, PDF, XLSX, PPTX via python libraries). Optimize structure, add syntax "
            "highlighting for code, generate tables/diagrams as needed. Use Write to create "
            "output files if needed. Return the formatted content."
        )
        system_prompt = f"{self.system_prompt}\n\n{react_instruction}"

        task_input = f"Format this content as {target_format}:\n\n{raw_content}"

        if context:
            task_input += "\n\nContext:\n" + json.dumps(context, default=str)

        loop = ReactLoop(
            agent_name="formatter",
            system_prompt=system_prompt,
            allowed_tools=["Read", "Write", "Bash", "Skill"],
            output_schema=None,
            model=self.model,
            max_turns=self.max_turns,
        )

        result = loop.run(task_input)

        if result and result.get("status") == "success" and "output" in result:
            return {
                "formatted_output": result["output"],
                "format": target_format,
                "file_path": None,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "size_bytes": len(str(result["output"])),
                },
            }

        # Fallback to procedural logic
        self.logger.warning(
            "react_fallback",
            reason="ReAct loop did not succeed, falling back to local mode",
        )
        return self.format(raw_content, target_format, context=context, mode="local")

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
        Format content as a well-structured document in Markdown representation.

        Generates a complete document with title page, table of contents,
        numbered sections, and proper formatting. The output is Markdown since
        binary formats (DOCX/PDF) require external libraries, but the structure
        mirrors what a real document would contain.
        """
        format_name = format_enum.value.upper()
        title = (context or {}).get("title", f"{format_name} Document")
        author = (context or {}).get("author", "Multi-Agent Reasoning System")
        date_str = datetime.now().strftime("%B %d, %Y")

        # Normalize content into sections
        sections = self._extract_document_sections(content)

        # Build table of contents
        toc_lines = ["## Table of Contents\n"]
        for i, section in enumerate(sections, 1):
            heading = section["heading"]
            anchor = heading.lower().replace(" ", "-").replace(".", "")
            toc_lines.append(f"{i}. [{heading}](#{anchor})")
        toc_block = "\n".join(toc_lines)

        # Build section bodies
        body_parts = []
        for i, section in enumerate(sections, 1):
            heading = section["heading"]
            body = section["body"].strip()
            body_parts.append(f"## {i}. {heading}\n\n{body}")
        body_block = "\n\n---\n\n".join(body_parts)

        return (
            f"# {title}\n\n"
            f"**Format:** {format_name}  \n"
            f"**Author:** {author}  \n"
            f"**Date:** {date_str}  \n\n"
            f"---\n\n"
            f"{toc_block}\n\n"
            f"---\n\n"
            f"{body_block}\n\n"
            f"---\n\n"
            f"*Generated by the Formatter agent on {date_str}.*\n"
        )

    def _extract_document_sections(self, content: Any) -> List[Dict[str, str]]:
        """Extract or synthesize document sections from arbitrary content."""
        if isinstance(content, dict):
            sections = []
            for key, value in content.items():
                heading = str(key).replace("_", " ").title()
                if isinstance(value, dict):
                    body = self._dict_to_markdown(value)
                elif isinstance(value, list):
                    body = self._list_to_markdown(value)
                else:
                    body = str(value)
                sections.append({"heading": heading, "body": body})
            return sections if sections else [{"heading": "Content", "body": str(content)}]

        if isinstance(content, list):
            sections = []
            for i, item in enumerate(content):
                if isinstance(item, dict) and "heading" in item:
                    sections.append({
                        "heading": str(item["heading"]),
                        "body": str(item.get("body", item.get("content", "")))
                    })
                elif isinstance(item, dict):
                    heading = str(item.get("title", item.get("name", f"Section {i + 1}")))
                    body = self._dict_to_markdown(
                        {k: v for k, v in item.items() if k not in ("title", "name")}
                    ) if len(item) > 1 else str(list(item.values())[0]) if item else ""
                    sections.append({"heading": heading, "body": body})
                else:
                    sections.append({"heading": f"Item {i + 1}", "body": str(item)})
            return sections if sections else [{"heading": "Content", "body": str(content)}]

        # String content: split on markdown headings or double-newlines
        text = str(content)
        heading_splits = re.split(r'\n(#{1,3})\s+(.+)', text)

        if len(heading_splits) > 1:
            sections = []
            # heading_splits: [preamble, level, title, body, level, title, body, ...]
            preamble = heading_splits[0].strip()
            if preamble:
                sections.append({"heading": "Introduction", "body": preamble})
            idx = 1
            while idx + 2 <= len(heading_splits):
                title = heading_splits[idx + 1].strip()
                body = heading_splits[idx + 2].strip() if idx + 2 < len(heading_splits) else ""
                sections.append({"heading": title, "body": body})
                idx += 3
            return sections if sections else [{"heading": "Content", "body": text}]

        # Plain text: split on double newlines into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
        if len(paragraphs) > 1:
            sections = []
            sections.append({"heading": "Overview", "body": paragraphs[0]})
            for i, para in enumerate(paragraphs[1:], 1):
                sections.append({"heading": f"Section {i}", "body": para})
            return sections

        return [{"heading": "Content", "body": text}]

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

    def _sanitize_mermaid_label(self, text: str) -> str:
        """Sanitize a string for use as a Mermaid node label."""
        # Remove characters that break Mermaid syntax
        sanitized = re.sub(r'["\[\]{}|<>()#;]', '', text)
        return sanitized.strip()[:50]  # Truncate long labels

    def _extract_entities_and_relationships(self, content: Any) -> Dict[str, Any]:
        """Extract entities and relationships from content for diagram generation."""
        text = str(content) if not isinstance(content, str) else content
        entities = []
        relationships = []

        # If content is a dict with explicit entities/relationships keys, use them
        if isinstance(content, dict):
            if "entities" in content:
                entities = [str(e) for e in content["entities"]]
            if "relationships" in content:
                relationships = content["relationships"]
            if "steps" in content:
                steps = [str(s) for s in content["steps"]]
                entities = steps
                relationships = [
                    {"from": steps[i], "to": steps[i + 1], "label": ""}
                    for i in range(len(steps) - 1)
                ]
            if "nodes" in content:
                entities = [str(n) for n in content["nodes"]]
            if "edges" in content:
                relationships = content["edges"]
            # If dict had explicit structures, return early
            if entities or relationships:
                return {"entities": entities, "relationships": relationships}

            # Otherwise treat keys as entities
            entities = [str(k) for k in content.keys()]

        elif isinstance(content, list):
            entities = [str(item) if not isinstance(item, dict) else str(item.get("name", item.get("title", str(item)))) for item in content]

        # Fallback: extract capitalized phrases or quoted terms from text
        if not entities:
            # Look for quoted terms
            quoted = re.findall(r'"([^"]+)"', text)
            if quoted:
                entities = quoted
            else:
                # Extract sentences/phrases split by arrows, commas, or "then"/"to"
                arrow_parts = re.split(r'\s*(?:->|-->|=>|→|to|then)\s*', text, flags=re.IGNORECASE)
                if len(arrow_parts) > 1:
                    entities = [p.strip() for p in arrow_parts if p.strip()]
                else:
                    # Split by commas or newlines and use as entities
                    parts = re.split(r'[,\n]+', text)
                    entities = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]

        # If we still have nothing, use the whole content as a single entity
        if not entities:
            entities = [text[:50].strip()] if text.strip() else ["Start"]

        # Build sequential relationships if none were explicitly provided
        if not relationships and len(entities) > 1:
            relationships = [
                {"from": entities[i], "to": entities[i + 1], "label": ""}
                for i in range(len(entities) - 1)
            ]

        return {"entities": entities, "relationships": relationships}

    def _generate_flowchart(self, content: Any, context: Dict[str, Any]) -> str:
        """Generate a Mermaid flowchart from content."""
        extracted = self._extract_entities_and_relationships(content)
        entities = extracted["entities"]
        relationships = extracted["relationships"]

        if not entities:
            return "```mermaid\nflowchart TD\n    A[No content provided]\n```"

        # Build node ID mapping
        node_ids = {}
        for i, entity in enumerate(entities):
            node_ids[entity] = chr(65 + (i % 26)) + (str(i // 26) if i >= 26 else "")

        lines = ["```mermaid", "flowchart TD"]

        # Determine if any node should be a decision (contains '?' or 'decision'/'check')
        for entity in entities:
            nid = node_ids[entity]
            label = self._sanitize_mermaid_label(entity)
            lower = entity.lower()
            if "?" in entity or "decision" in lower or "check" in lower or "if " in lower:
                lines.append(f"    {nid}{{{{{label}}}}}")
            else:
                lines.append(f"    {nid}[{label}]")

        for rel in relationships:
            from_id = node_ids.get(rel["from"])
            to_id = node_ids.get(rel["to"])
            if from_id and to_id:
                label = rel.get("label", "")
                if label:
                    lines.append(f"    {from_id} -->|{self._sanitize_mermaid_label(label)}| {to_id}")
                else:
                    lines.append(f"    {from_id} --> {to_id}")

        lines.append("```")
        return "\n".join(lines)

    def _generate_sequence_diagram(self, content: Any, context: Dict[str, Any]) -> str:
        """Generate a Mermaid sequence diagram from content."""
        participants = []
        interactions = []

        if isinstance(content, dict):
            participants = [str(p) for p in content.get("participants", content.get("actors", []))]
            interactions = content.get("interactions", content.get("messages", []))
            if not participants and not interactions:
                # Treat keys as participants
                participants = [str(k) for k in content.keys()]

        elif isinstance(content, list):
            # Each item could be a message dict or a string describing an interaction
            for item in content:
                if isinstance(item, dict):
                    fr = str(item.get("from", item.get("sender", "")))
                    to = str(item.get("to", item.get("receiver", "")))
                    msg = str(item.get("message", item.get("action", item.get("label", ""))))
                    if fr and to:
                        interactions.append({"from": fr, "to": to, "message": msg})
                        if fr not in participants:
                            participants.append(fr)
                        if to not in participants:
                            participants.append(to)

        # Fallback: parse text for "A sends/calls/requests B" patterns
        if not interactions:
            text = str(content)
            # Look for "X -> Y: message" patterns
            arrow_matches = re.findall(r'(\w[\w\s]*\w)\s*(?:->|-->|sends?|calls?|requests?)\s*(\w[\w\s]*\w)(?:\s*:\s*(.+))?', text, re.IGNORECASE)
            for fr, to, msg in arrow_matches:
                fr, to = fr.strip(), to.strip()
                interactions.append({"from": fr, "to": to, "message": msg.strip() if msg else "request"})
                if fr not in participants:
                    participants.append(fr)
                if to not in participants:
                    participants.append(to)

        # If we still have nothing, create a minimal diagram
        if not participants:
            participants = ["Actor", "System"]
        if not interactions:
            interactions = [{"from": participants[0], "to": participants[-1] if len(participants) > 1 else participants[0], "message": "interaction"}]

        lines = ["```mermaid", "sequenceDiagram"]
        for p in participants:
            safe_name = re.sub(r'\s+', '_', p.strip())
            lines.append(f"    participant {safe_name}")

        for interaction in interactions:
            fr = re.sub(r'\s+', '_', str(interaction.get("from", participants[0])).strip())
            to = re.sub(r'\s+', '_', str(interaction.get("to", participants[-1])).strip())
            msg = self._sanitize_mermaid_label(str(interaction.get("message", "request")))
            resp = interaction.get("response")
            lines.append(f"    {fr}->>{to}: {msg}")
            if resp:
                lines.append(f"    {to}-->>{fr}: {self._sanitize_mermaid_label(str(resp))}")

        lines.append("```")
        return "\n".join(lines)

    def _generate_class_diagram(self, content: Any, context: Dict[str, Any]) -> str:
        """Generate a Mermaid class diagram from content."""
        classes = []  # list of {"name": str, "methods": list, "attributes": list}
        relationships = []

        if isinstance(content, dict):
            # Check for explicit classes key
            if "classes" in content:
                for cls in content["classes"]:
                    if isinstance(cls, dict):
                        classes.append({
                            "name": str(cls.get("name", "Unknown")),
                            "methods": [str(m) for m in cls.get("methods", [])],
                            "attributes": [str(a) for a in cls.get("attributes", [])],
                        })
                    else:
                        classes.append({"name": str(cls), "methods": [], "attributes": []})
            if "relationships" in content:
                relationships = content["relationships"]

            # Fallback: treat each key as a class
            if not classes:
                for key, value in content.items():
                    methods = []
                    attributes = []
                    if isinstance(value, dict):
                        methods = [str(m) for m in value.get("methods", [])]
                        attributes = [str(a) for a in value.get("attributes", value.get("fields", []))]
                        if not methods and not attributes:
                            # Treat sub-keys as attributes
                            attributes = [f"{k}: {type(v).__name__}" for k, v in value.items()]
                    elif isinstance(value, list):
                        methods = [str(m) for m in value]
                    classes.append({"name": str(key), "methods": methods, "attributes": attributes})

        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    name = str(item.get("name", item.get("class", f"Class{len(classes) + 1}")))
                    classes.append({
                        "name": name,
                        "methods": [str(m) for m in item.get("methods", [])],
                        "attributes": [str(a) for a in item.get("attributes", item.get("fields", []))],
                    })
                else:
                    classes.append({"name": str(item), "methods": [], "attributes": []})

        # Fallback: parse text for class-like identifiers
        if not classes:
            text = str(content)
            class_names = re.findall(r'\b([A-Z][a-zA-Z0-9]+)\b', text)
            seen = set()
            for name in class_names:
                if name not in seen and name not in ("True", "False", "None", "Any", "Dict", "List", "Optional"):
                    seen.add(name)
                    classes.append({"name": name, "methods": [], "attributes": []})

        if not classes:
            classes = [{"name": "Entity", "methods": [], "attributes": []}]

        lines = ["```mermaid", "classDiagram"]
        for cls in classes:
            safe_name = re.sub(r'\s+', '', cls["name"])
            lines.append(f"    class {safe_name} {{")
            for attr in cls.get("attributes", []):
                lines.append(f"        +{self._sanitize_mermaid_label(attr)}")
            for method in cls.get("methods", []):
                method_str = self._sanitize_mermaid_label(method)
                if not method_str.endswith(")"):
                    method_str += "()"
                lines.append(f"        +{method_str}")
            lines.append("    }")

        for rel in relationships:
            fr = re.sub(r'\s+', '', str(rel.get("from", "")))
            to = re.sub(r'\s+', '', str(rel.get("to", "")))
            rel_type = str(rel.get("type", "association")).lower()
            if fr and to:
                symbol_map = {
                    "inheritance": "<|--",
                    "extends": "<|--",
                    "composition": "*--",
                    "aggregation": "o--",
                    "dependency": "..>",
                    "implements": "<|..",
                    "association": "-->",
                }
                symbol = symbol_map.get(rel_type, "-->")
                label = rel.get("label", "")
                if label:
                    lines.append(f"    {fr} {symbol} {to} : {self._sanitize_mermaid_label(label)}")
                else:
                    lines.append(f"    {fr} {symbol} {to}")

        lines.append("```")
        return "\n".join(lines)

    def _generate_state_diagram(self, content: Any, context: Dict[str, Any]) -> str:
        """Generate a Mermaid state diagram from content."""
        states = []
        transitions = []

        if isinstance(content, dict):
            states = [str(s) for s in content.get("states", [])]
            transitions = content.get("transitions", [])
            if not states and not transitions:
                # Treat keys as states
                states = [str(k) for k in content.keys()]

        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    fr = str(item.get("from", ""))
                    to = str(item.get("to", ""))
                    label = str(item.get("label", item.get("trigger", item.get("event", ""))))
                    if fr and to:
                        transitions.append({"from": fr, "to": to, "label": label})
                        if fr not in states:
                            states.append(fr)
                        if to not in states:
                            states.append(to)
                else:
                    states.append(str(item))

        # Fallback: parse text
        if not states:
            text = str(content)
            # Look for state-like words
            state_matches = re.findall(r'\b([A-Z][a-zA-Z]+(?:ed|ing|ive|le)?)\b', text)
            seen = set()
            for s in state_matches:
                if s not in seen and s not in ("True", "False", "None"):
                    seen.add(s)
                    states.append(s)

        if not states:
            states = ["Initial", "Active", "Complete"]

        # Build sequential transitions if none provided
        if not transitions and len(states) > 1:
            transitions = [
                {"from": states[i], "to": states[i + 1], "label": ""}
                for i in range(len(states) - 1)
            ]

        lines = ["```mermaid", "stateDiagram-v2"]

        # Start transition
        if states:
            lines.append(f"    [*] --> {re.sub(r'[^a-zA-Z0-9_]', '_', states[0])}")

        for t in transitions:
            fr = re.sub(r'[^a-zA-Z0-9_]', '_', str(t["from"]))
            to = re.sub(r'[^a-zA-Z0-9_]', '_', str(t["to"]))
            label = t.get("label", "")
            if label:
                lines.append(f"    {fr} --> {to}: {self._sanitize_mermaid_label(label)}")
            else:
                lines.append(f"    {fr} --> {to}")

        # End transition
        if states:
            lines.append(f"    {re.sub(r'[^a-zA-Z0-9_]', '_', states[-1])} --> [*]")

        lines.append("```")
        return "\n".join(lines)

    def _generate_generic_diagram(self, content: Any, context: Dict[str, Any]) -> str:
        """Generate a generic Mermaid graph from content."""
        extracted = self._extract_entities_and_relationships(content)
        entities = extracted["entities"]
        relationships = extracted["relationships"]

        if not entities:
            return "```mermaid\ngraph LR\n    A[No content provided]\n```"

        node_ids = {}
        for i, entity in enumerate(entities):
            node_ids[entity] = chr(65 + (i % 26)) + (str(i // 26) if i >= 26 else "")

        lines = ["```mermaid", "graph LR"]

        for entity in entities:
            nid = node_ids[entity]
            label = self._sanitize_mermaid_label(entity)
            lines.append(f"    {nid}[{label}]")

        for rel in relationships:
            from_id = node_ids.get(rel["from"])
            to_id = node_ids.get(rel["to"])
            if from_id and to_id:
                label = rel.get("label", "")
                if label:
                    lines.append(f"    {from_id} -->|{self._sanitize_mermaid_label(label)}| {to_id}")
                else:
                    lines.append(f"    {from_id} --> {to_id}")

        lines.append("```")
        return "\n".join(lines)

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
        # Order matters: check more specific patterns before generic ones

        # Java (check before Python since "class " also matches Java)
        if "public class " in code or "public static void main" in code:
            return "java"

        # Go
        if "func " in code and "package main" in code:
            return "go"

        # C++
        if "#include <" in code and "std::" in code:
            return "cpp"

        # C
        if "#include <" in code and "int main(" in code:
            return "c"

        # JavaScript/TypeScript
        if "function " in code or "const " in code or "let " in code:
            if ": " in code and "interface " in code:
                return "typescript"
            return "javascript"

        # Python
        if "def " in code or "import " in code or "from " in code or "class " in code:
            return "python"

        # Default
        return "python"

    def _validate_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """
        Validate code syntax for supported languages.

        Performs in-process validation for Python, JavaScript, JSON, and YAML.
        Returns validation result with any errors found.
        """
        valid = True
        errors: List[str] = []

        if language == "python":
            try:
                ast.parse(code)
            except SyntaxError as e:
                valid = False
                errors.append(f"Line {e.lineno}: {e.msg}")

        elif language == "javascript":
            # Heuristic JS syntax checks (no full parser available in-process)
            bracket_stack: List[str] = []
            matching = {")": "(", "]": "[", "}": "{"}
            in_string: Optional[str] = None
            in_single_comment = False
            in_multi_comment = False
            prev_char = ""

            for line_num, line in enumerate(code.split("\n"), 1):
                in_single_comment = False
                for i, ch in enumerate(line):
                    next_char = line[i + 1] if i + 1 < len(line) else ""

                    if in_multi_comment:
                        if ch == "*" and next_char == "/":
                            in_multi_comment = False
                        continue
                    if in_single_comment:
                        continue
                    if in_string:
                        if ch == in_string and prev_char != "\\":
                            in_string = None
                        prev_char = ch
                        continue

                    if ch == "/" and next_char == "/":
                        in_single_comment = True
                        continue
                    if ch == "/" and next_char == "*":
                        in_multi_comment = True
                        continue
                    if ch in ('"', "'", "`"):
                        in_string = ch
                        prev_char = ch
                        continue

                    if ch in ("(", "[", "{"):
                        bracket_stack.append(ch)
                    elif ch in (")", "]", "}"):
                        if not bracket_stack:
                            valid = False
                            errors.append(f"Line {line_num}: Unexpected closing '{ch}'")
                        elif bracket_stack[-1] != matching[ch]:
                            valid = False
                            errors.append(f"Line {line_num}: Mismatched '{ch}', expected closing for '{bracket_stack[-1]}'")
                        else:
                            bracket_stack.pop()
                    prev_char = ch

            if bracket_stack:
                valid = False
                errors.append(f"Unclosed brackets: {''.join(bracket_stack)}")
            if in_string:
                valid = False
                errors.append("Unterminated string literal")

        elif language == "json":
            try:
                json.loads(code)
            except json.JSONDecodeError as e:
                valid = False
                errors.append(f"Line {e.lineno}, Col {e.colno}: {e.msg}")

        elif language == "yaml":
            try:
                import yaml
                yaml.safe_load(code)
            except ImportError:
                # yaml not available; log and skip validation
                self.logger.warning("PyYAML not installed; skipping YAML validation")
            except yaml.YAMLError as e:
                valid = False
                if hasattr(e, "problem_mark") and e.problem_mark is not None:
                    mark = e.problem_mark
                    errors.append(f"Line {mark.line + 1}, Col {mark.column + 1}: {getattr(e, 'problem', str(e))}")
                else:
                    errors.append(str(e))

        return {
            "valid": valid,
            "errors": errors,
            "language": language,
        }

    # ========================================================================
    # File Operations
    # ========================================================================

    def _write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a file with proper directory creation and error handling.

        Args:
            file_path: Absolute or relative path to write to.
            content: String content to write.

        Returns:
            Dictionary with success/failure status and metadata.
        """
        result: Dict[str, Any] = {
            "success": False,
            "file_path": file_path,
            "error": None,
            "bytes_written": 0,
        }

        # Validate the file path
        try:
            resolved = Path(file_path).resolve()
        except (OSError, ValueError) as e:
            result["error"] = f"Invalid file path: {e}"
            return result

        # Prevent writing outside the output directory when path is relative
        # (absolute paths are allowed for flexibility)
        if not Path(file_path).is_absolute():
            resolved = Path(self.output_dir).resolve() / file_path
        else:
            resolved = Path(file_path).resolve()

        # Security: reject paths with null bytes or overly long names
        if "\x00" in str(resolved):
            result["error"] = "File path contains null bytes"
            return result

        # Ensure the parent directory exists
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            result["error"] = f"Failed to create directory {resolved.parent}: {e}"
            return result

        # Write the file
        try:
            resolved.write_text(content, encoding="utf-8")
            result["success"] = True
            result["bytes_written"] = len(content.encode("utf-8"))
            result["file_path"] = str(resolved)
        except PermissionError:
            result["error"] = f"Permission denied writing to {resolved}"
        except OSError as e:
            result["error"] = f"OS error writing file: {e}"
        except Exception as e:
            result["error"] = f"Unexpected error writing file: {e}"

        return result

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
                self.logger.warning(
                    "Unknown output format, falling back to auto-detection",
                    requested_format=context["format"],
                )

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
