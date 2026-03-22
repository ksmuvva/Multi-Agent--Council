"""
ReAct Loop Engine - Reasoning + Acting for Multi-Agent System

Provides a reusable ReAct (Reason-Act-Observe) loop that all agents
use to perform LLM-driven reasoning with tool access. Uses the Claude
Agent SDK for LLM calls and tool execution.

Architecture:
    1. Agent system prompt + task → LLM
    2. LLM reasons (Thought) and optionally calls tools (Action)
    3. Tool results fed back as Observation
    4. Repeat until LLM produces structured Final Answer
    5. Final Answer validated against Pydantic schema
"""

import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from src.config import get_settings, get_model_for_agent
from src.utils.logging import get_logger
from src.utils.events import (
    get_event_emitter,
    EventType,
)

logger = get_logger("react")


# ============================================================================
# ReAct Event Types (emitted during loop execution)
# ============================================================================

REACT_THOUGHT = "react_thought"
REACT_ACTION = "react_action"
REACT_OBSERVATION = "react_observation"
REACT_FINAL_ANSWER = "react_final_answer"
REACT_ERROR = "react_error"


def _emit_react_event(
    event_subtype: str,
    agent_name: str,
    iteration: int,
    data: Dict[str, Any],
    session_id: Optional[str] = None,
) -> None:
    """Emit a ReAct loop event for logging and UI display."""
    try:
        emitter = get_event_emitter()
        emitter.emit(
            event_type=EventType.AGENT_PROGRESS,
            source=agent_name,
            data={
                "react_event": event_subtype,
                "iteration": iteration,
                **data,
            },
            session_id=session_id,
        )
    except Exception:
        pass  # Non-critical - don't break the loop for logging failures


# ============================================================================
# ReAct Loop using Claude Agent SDK
# ============================================================================

class ReactLoop:
    """
    Generic ReAct (Reasoning + Acting) execution loop using Claude Agent SDK.

    Each agent provides:
    - system_prompt: Role and instructions
    - allowed_tools: List of SDK tool names (Read, Write, Bash, etc.)
    - output_schema: Pydantic model for structured final answer

    The loop delegates all reasoning and tool use to the Claude Agent SDK,
    which handles the Thought → Action → Observation cycle natively.
    """

    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        allowed_tools: List[str],
        output_schema: Optional[Type[BaseModel]] = None,
        model: Optional[str] = None,
        max_turns: int = 30,
        max_budget_usd: Optional[float] = None,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools
        self.output_schema = output_schema
        self.model = model
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd

    def run(
        self,
        task_input: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the ReAct loop synchronously.

        For GLM provider, uses direct API calls (SDK not compatible).
        For other providers, tries Claude Agent SDK first with fallback to direct API.

        Args:
            task_input: The task/prompt for the agent
            context: Optional additional context dict
            session_id: Optional session ID for event correlation

        Returns:
            Dict with keys: status, output, raw_output, tokens_used, cost_usd, duration_ms
        """
        start_time = time.time()

        # Build the full prompt with context
        full_prompt = self._build_prompt(task_input, context)

        _emit_react_event(
            REACT_THOUGHT, self.agent_name, 0,
            {"message": f"Starting ReAct loop for {self.agent_name}"},
            session_id,
        )

        # Check provider - GLM uses direct API calls
        settings = get_settings()
        if settings.llm_provider.value == "glm":
            logger.info("react.using_direct_api_for_glm", agent=self.agent_name)
            result = self._run_with_direct_api(full_prompt, session_id)
        else:
            # Try SDK for other providers
            try:
                result = self._run_with_agent_sdk(full_prompt, session_id)
            except ImportError:
                logger.info("react.sdk_unavailable", agent=self.agent_name,
                            fallback="direct_api")
                result = self._run_with_direct_api(full_prompt, session_id)
            except Exception as e:
                # Catch SDK errors (like ProcessError when running in nested Claude Code session)
                logger.info("react.sdk_failed", agent=self.agent_name, error=str(e),
                            fallback="direct_api")
                result = self._run_with_direct_api(full_prompt, session_id)

        duration_ms = int((time.time() - start_time) * 1000)
        result["duration_ms"] = duration_ms

        # Try to parse output into schema
        if self.output_schema and result.get("status") == "success":
            parsed = self._parse_output(result.get("raw_output", ""))
            if parsed is not None:
                result["output"] = parsed
                result["schema_valid"] = True
            else:
                result["schema_valid"] = False

        _emit_react_event(
            REACT_FINAL_ANSWER, self.agent_name, 0,
            {
                "status": result.get("status"),
                "schema_valid": result.get("schema_valid", False),
                "duration_ms": duration_ms,
            },
            session_id,
        )

        return result

    async def arun(
        self,
        task_input: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the ReAct loop asynchronously using Claude Agent SDK.

        Args:
            task_input: The task/prompt for the agent
            context: Optional additional context dict
            session_id: Optional session ID for event correlation

        Returns:
            Dict with keys: status, output, raw_output, tokens_used, cost_usd, duration_ms
        """
        start_time = time.time()
        full_prompt = self._build_prompt(task_input, context)

        _emit_react_event(
            REACT_THOUGHT, self.agent_name, 0,
            {"message": f"Starting async ReAct loop for {self.agent_name}"},
            session_id,
        )

        try:
            result = await self._arun_with_agent_sdk(full_prompt, session_id)
        except ImportError:
            logger.info("react.sdk_unavailable_async", agent=self.agent_name,
                        fallback="anthropic_api")
            try:
                result = await self._arun_with_anthropic_api(full_prompt, session_id)
            except ImportError:
                result = self._run_simulated(full_prompt, session_id)

        duration_ms = int((time.time() - start_time) * 1000)
        result["duration_ms"] = duration_ms

        if self.output_schema and result.get("status") == "success":
            parsed = self._parse_output(result.get("raw_output", ""))
            if parsed is not None:
                result["output"] = parsed
                result["schema_valid"] = True
            else:
                result["schema_valid"] = False

        _emit_react_event(
            REACT_FINAL_ANSWER, self.agent_name, 0,
            {
                "status": result.get("status"),
                "schema_valid": result.get("schema_valid", False),
                "duration_ms": duration_ms,
            },
            session_id,
        )

        return result

    # ========================================================================
    # Claude Agent SDK execution
    # ========================================================================

    def _run_with_agent_sdk(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute via Claude Agent SDK (synchronous wrapper)."""
        from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

        options = self._build_sdk_options()

        raw_output = ""
        tokens_used = 0
        cost_usd = 0.0

        # Run the async SDK in a sync context
        async def _run():
            nonlocal raw_output, tokens_used, cost_usd
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    raw_output = message.result or ""

        # Use existing event loop or create new one
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - use nest_asyncio or run in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run())
                future.result(timeout=300)
        except RuntimeError:
            # No event loop running - create one
            asyncio.run(_run())

        return {
            "status": "success",
            "output": raw_output,
            "raw_output": raw_output,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
        }

    async def _arun_with_agent_sdk(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute via Claude Agent SDK (native async)."""
        from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

        options = self._build_sdk_options()

        raw_output = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                raw_output = message.result or ""

        return {
            "status": "success",
            "output": raw_output,
            "raw_output": raw_output,
            "tokens_used": 0,
            "cost_usd": 0.0,
        }

    def _build_sdk_options(self):
        """Build ClaudeAgentOptions for the Agent SDK."""
        from claude_agent_sdk import ClaudeAgentOptions

        opts = ClaudeAgentOptions(
            allowed_tools=self.allowed_tools,
            system_prompt=self.system_prompt,
            max_turns=self.max_turns,
        )

        if self.model:
            opts.model = self.model
        if self.max_budget_usd:
            opts.max_budget_usd = self.max_budget_usd
        if self.output_schema:
            opts.output_format = self.output_schema.model_json_schema()

        return opts

    # ========================================================================
    # Direct API execution (GLM or Anthropic based on provider)
    # ========================================================================

    def _run_with_direct_api(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute using direct API calls based on configured provider.

        Routes to GLM API or Anthropic API based on settings.
        """
        settings = get_settings()

        if settings.llm_provider.value == "glm":
            logger.info("react.using_glm_api", agent=self.agent_name)
            return self._run_with_glm_api(prompt, session_id)
        else:
            logger.info("react.using_anthropic_api", agent=self.agent_name)
            return self._run_with_anthropic_api(prompt, session_id)

    def _run_with_glm_api(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute via ZhipuAI GLM API (OpenAI-compatible endpoint).

        GLM doesn't support tool use in the same way as Claude, so we use
        a simplified approach: single completion call without tools.

        Includes retry logic for rate limiting (429 errors).
        """
        import httpx
        import time

        settings = get_settings()
        api_key = settings.get_api_key()
        base_url = settings.get_base_url() or "https://open.bigmodel.cn/api/paas/v4"
        model = self.model or get_model_for_agent(self.agent_name)

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Retry logic for rate limiting
        max_retries = 3
        base_delay = 2.0  # seconds

        for attempt in range(max_retries):
            try:
                # Add delay between retries (or before first request to avoid burst)
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.info(
                        "react.glm_retry",
                        agent=self.agent_name,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=delay,
                    )
                    time.sleep(delay)
                else:
                    time.sleep(0.3)  # Small initial delay to avoid rate limiting

                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        f"{base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )

                # Success - return the response
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    usage = data.get("usage", {})

                    total_tokens = usage.get("total_tokens", 0)
                    # GLM pricing estimate (approximate)
                    cost = total_tokens / 1_000_000 * 2.0

                    logger.info(
                        "react.glm_success",
                        agent=self.agent_name,
                        tokens=total_tokens,
                        cost=cost,
                    )

                    return {
                        "status": "success",
                        "output": content,
                        "raw_output": content,
                        "tokens_used": total_tokens,
                        "cost_usd": cost,
                    }

                # Rate limiting (429) - retry with backoff
                if resp.status_code == 429:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "react.glm_rate_limited",
                            agent=self.agent_name,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                        )
                        continue  # Retry after delay
                    else:
                        logger.error(
                            "react.glm_rate_limit_exceeded",
                            agent=self.agent_name,
                            max_retries=max_retries,
                        )
                        return {
                            "status": "error",
                            "output": None,
                            "raw_output": "",
                            "error": "GLM API rate limit exceeded (429). Please try again later.",
                            "tokens_used": 0,
                            "cost_usd": 0.0,
                        }

                # Other HTTP errors - don't retry
                logger.error(
                    "react.glm_api_error",
                    agent=self.agent_name,
                    status=resp.status_code,
                    body=resp.text[:200] if resp.text else "No response body",
                )
                return {
                    "status": "error",
                    "output": None,
                    "raw_output": "",
                    "error": f"GLM API error: {resp.status_code}",
                    "tokens_used": 0,
                    "cost_usd": 0.0,
                }

            except httpx.TimeoutError as e:
                logger.warning(
                    "react.glm_timeout",
                    agent=self.agent_name,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                if attempt == max_retries - 1:
                    return {
                        "status": "error",
                        "output": None,
                        "raw_output": "",
                        "error": "GLM API request timed out",
                        "tokens_used": 0,
                        "cost_usd": 0.0,
                    }

            except Exception as e:
                # Safely handle error messages that may contain non-ASCII characters
                # (e.g., Chinese characters in GLM error messages)
                error_msg = str(e)
                try:
                    # Try to encode as ASCII for safe logging
                    safe_error = error_msg.encode('ascii', errors='replace').decode('ascii')
                except (UnicodeDecodeError, UnicodeEncodeError):
                    safe_error = "[Error message contains unsupported characters]"

                logger.error(
                    "react.glm_exception",
                    agent=self.agent_name,
                    error=safe_error,
                    error_type=type(e).__name__,
                    attempt=attempt + 1,
                )
                # Don't retry on most exceptions, just return error
                return {
                    "status": "error",
                    "output": None,
                    "raw_output": "",
                    "error": error_msg,
                    "tokens_used": 0,
                    "cost_usd": 0.0,
                }

        # Should not reach here, but just in case
        return {
            "status": "error",
            "output": None,
            "raw_output": "",
            "error": "Max retries exceeded",
            "tokens_used": 0,
            "cost_usd": 0.0,
        }

    # ========================================================================
    # Anthropic API fallback (tool use agentic loop)
    # ========================================================================

    def _run_with_anthropic_api(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fallback: Use Anthropic API directly with manual tool use loop.

        Implements the agentic loop pattern from the Claude API docs:
        send message → if tool_use → execute tools → send results → repeat.
        """
        from anthropic import Anthropic

        settings = get_settings()
        client = Anthropic(api_key=settings.get_api_key())

        model = self.model or get_model_for_agent(self.agent_name)
        tools = self._build_anthropic_tools()
        messages = [{"role": "user", "content": prompt}]

        total_input_tokens = 0
        total_output_tokens = 0
        iteration = 0

        while iteration < self.max_turns:
            iteration += 1

            _emit_react_event(
                REACT_THOUGHT, self.agent_name, iteration,
                {"message": f"API call iteration {iteration}"},
                session_id,
            )

            create_kwargs = {
                "model": model,
                "max_tokens": 16384,
                "system": self.system_prompt,
                "messages": messages,
            }
            if tools:
                create_kwargs["tools"] = tools
            # Request structured JSON output if we have a schema
            if self.output_schema and not tools:
                create_kwargs["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": self._get_strict_schema(),
                    }
                }

            response = client.messages.create(**create_kwargs)

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # If Claude is done (no more tool calls)
            if response.stop_reason == "end_turn":
                raw_output = self._extract_text(response)
                cost = self._estimate_cost(total_input_tokens, total_output_tokens, model)
                return {
                    "status": "success",
                    "output": raw_output,
                    "raw_output": raw_output,
                    "tokens_used": total_input_tokens + total_output_tokens,
                    "cost_usd": cost,
                }

            # Server-side tool hit iteration limit
            if response.stop_reason == "pause_turn":
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.content},
                ]
                continue

            # Extract tool use blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                # No tools requested, treat as final answer
                raw_output = self._extract_text(response)
                cost = self._estimate_cost(total_input_tokens, total_output_tokens, model)
                return {
                    "status": "success",
                    "output": raw_output,
                    "raw_output": raw_output,
                    "tokens_used": total_input_tokens + total_output_tokens,
                    "cost_usd": cost,
                }

            # Append assistant response (including tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool and collect results
            tool_results = []
            for tool_block in tool_use_blocks:
                _emit_react_event(
                    REACT_ACTION, self.agent_name, iteration,
                    {"tool": tool_block.name, "input": str(tool_block.input)[:200]},
                    session_id,
                )

                result_text = self._execute_tool(tool_block.name, tool_block.input)

                _emit_react_event(
                    REACT_OBSERVATION, self.agent_name, iteration,
                    {"tool": tool_block.name, "result_length": len(result_text)},
                    session_id,
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        # Max turns reached
        raw_output = self._extract_text(response) if response else ""
        cost = self._estimate_cost(total_input_tokens, total_output_tokens, model)
        return {
            "status": "max_turns_reached",
            "output": raw_output,
            "raw_output": raw_output,
            "tokens_used": total_input_tokens + total_output_tokens,
            "cost_usd": cost,
        }

    async def _arun_with_anthropic_api(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async Anthropic API fallback."""
        from anthropic import AsyncAnthropic

        settings = get_settings()
        client = AsyncAnthropic(api_key=settings.get_api_key())

        model = self.model or get_model_for_agent(self.agent_name)
        tools = self._build_anthropic_tools()
        messages = [{"role": "user", "content": prompt}]

        total_input_tokens = 0
        total_output_tokens = 0
        iteration = 0

        while iteration < self.max_turns:
            iteration += 1

            create_kwargs = {
                "model": model,
                "max_tokens": 16384,
                "system": self.system_prompt,
                "messages": messages,
            }
            if tools:
                create_kwargs["tools"] = tools
            if self.output_schema and not tools:
                create_kwargs["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": self._get_strict_schema(),
                    }
                }

            response = await client.messages.create(**create_kwargs)

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            if response.stop_reason == "end_turn":
                raw_output = self._extract_text(response)
                cost = self._estimate_cost(total_input_tokens, total_output_tokens, model)
                return {
                    "status": "success",
                    "output": raw_output,
                    "raw_output": raw_output,
                    "tokens_used": total_input_tokens + total_output_tokens,
                    "cost_usd": cost,
                }

            if response.stop_reason == "pause_turn":
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.content},
                ]
                continue

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                raw_output = self._extract_text(response)
                cost = self._estimate_cost(total_input_tokens, total_output_tokens, model)
                return {
                    "status": "success",
                    "output": raw_output,
                    "raw_output": raw_output,
                    "tokens_used": total_input_tokens + total_output_tokens,
                    "cost_usd": cost,
                }

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tool_block in tool_use_blocks:
                result_text = self._execute_tool(tool_block.name, tool_block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result_text,
                })
            messages.append({"role": "user", "content": tool_results})

        raw_output = self._extract_text(response) if response else ""
        cost = self._estimate_cost(total_input_tokens, total_output_tokens, model)
        return {
            "status": "max_turns_reached",
            "output": raw_output,
            "raw_output": raw_output,
            "tokens_used": total_input_tokens + total_output_tokens,
            "cost_usd": cost,
        }

    # ========================================================================
    # Tool definitions and execution for Anthropic API
    # ========================================================================

    def _build_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Build Anthropic API tool definitions from allowed_tools list."""
        tool_defs = {
            "Read": {
                "name": "Read",
                "description": "Read a file from the filesystem. Returns file contents.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Absolute path to file"},
                        "offset": {"type": "integer", "description": "Line number to start reading from"},
                        "limit": {"type": "integer", "description": "Number of lines to read"},
                    },
                    "required": ["file_path"],
                },
            },
            "Write": {
                "name": "Write",
                "description": "Write content to a file. Creates or overwrites the file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Absolute path to file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["file_path", "content"],
                },
            },
            "Edit": {
                "name": "Edit",
                "description": "Edit a file by replacing old_string with new_string.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Absolute path to file"},
                        "old_string": {"type": "string", "description": "Text to replace"},
                        "new_string": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
            "Bash": {
                "name": "Bash",
                "description": "Execute a shell command and return output.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in milliseconds"},
                    },
                    "required": ["command"],
                },
            },
            "Glob": {
                "name": "Glob",
                "description": "Find files matching a glob pattern.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
                        "path": {"type": "string", "description": "Directory to search in"},
                    },
                    "required": ["pattern"],
                },
            },
            "Grep": {
                "name": "Grep",
                "description": "Search file contents using regex pattern.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "path": {"type": "string", "description": "File or directory to search"},
                        "glob": {"type": "string", "description": "Glob filter for files"},
                    },
                    "required": ["pattern"],
                },
            },
            "WebSearch": {
                "name": "WebSearch",
                "description": "Search the web for information.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
            "WebFetch": {
                "name": "WebFetch",
                "description": "Fetch and extract content from a URL.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "prompt": {"type": "string", "description": "What to extract from the page"},
                    },
                    "required": ["url", "prompt"],
                },
            },
            "Skill": {
                "name": "Skill",
                "description": "Load and read a skill file from .claude/skills/{name}/SKILL.md",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string", "description": "Skill name (e.g., 'code-generation')"},
                    },
                    "required": ["skill"],
                },
            },
        }

        return [tool_defs[t] for t in self.allowed_tools if t in tool_defs]

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool call and return the result as a string.

        This provides local execution of tools when using the Anthropic API
        directly (without the Agent SDK).
        """
        try:
            if tool_name == "Read":
                return self._tool_read(tool_input)
            elif tool_name == "Write":
                return self._tool_write(tool_input)
            elif tool_name == "Edit":
                return self._tool_edit(tool_input)
            elif tool_name == "Bash":
                return self._tool_bash(tool_input)
            elif tool_name == "Glob":
                return self._tool_glob(tool_input)
            elif tool_name == "Grep":
                return self._tool_grep(tool_input)
            elif tool_name == "WebSearch":
                return self._tool_websearch(tool_input)
            elif tool_name == "WebFetch":
                return self._tool_webfetch(tool_input)
            elif tool_name == "Skill":
                return self._tool_skill(tool_input)
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    # -- Tool implementations for Anthropic API fallback --

    @staticmethod
    def _tool_read(input: Dict[str, Any]) -> str:
        file_path = input.get("file_path", "")
        offset = input.get("offset")
        limit = input.get("limit")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if offset:
                lines = lines[max(0, offset - 1):]
            if limit:
                lines = lines[:limit]
            return "".join(lines)
        except Exception as e:
            return f"Error reading {file_path}: {e}"

    @staticmethod
    def _tool_write(input: Dict[str, Any]) -> str:
        file_path = input.get("file_path", "")
        content = input.get("content", "")
        try:
            import os
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} bytes to {file_path}"
        except Exception as e:
            return f"Error writing {file_path}: {e}"

    @staticmethod
    def _tool_edit(input: Dict[str, Any]) -> str:
        file_path = input.get("file_path", "")
        old_string = input.get("old_string", "")
        new_string = input.get("new_string", "")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if old_string not in content:
                return f"Error: old_string not found in {file_path}"
            content = content.replace(old_string, new_string, 1)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully edited {file_path}"
        except Exception as e:
            return f"Error editing {file_path}: {e}"

    @staticmethod
    def _tool_bash(input: Dict[str, Any]) -> str:
        import subprocess
        command = input.get("command", "")
        timeout_ms = input.get("timeout", 120000)
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout_ms / 1000,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout_ms}ms"
        except Exception as e:
            return f"Error running command: {e}"

    @staticmethod
    def _tool_glob(input: Dict[str, Any]) -> str:
        import glob as glob_module
        pattern = input.get("pattern", "")
        path = input.get("path", ".")
        try:
            import os
            full_pattern = os.path.join(path, pattern) if path != "." else pattern
            matches = sorted(glob_module.glob(full_pattern, recursive=True))
            if not matches:
                return "No files found matching pattern"
            return "\n".join(matches[:200])
        except Exception as e:
            return f"Error in glob: {e}"

    @staticmethod
    def _tool_grep(input: Dict[str, Any]) -> str:
        import subprocess
        pattern = input.get("pattern", "")
        path = input.get("path", ".")
        glob_filter = input.get("glob", "")
        try:
            cmd = ["grep", "-rn", pattern, path]
            if glob_filter:
                cmd = ["grep", "-rn", f"--include={glob_filter}", pattern, path]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            output = result.stdout
            if not output:
                return "No matches found"
            # Limit output size
            lines = output.split("\n")
            if len(lines) > 100:
                return "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"
            return output
        except Exception as e:
            return f"Error in grep: {e}"

    def _tool_websearch(self, input: Dict[str, Any]) -> str:
        """
        Execute WebSearch using available web search capabilities.

        This provides real web search functionality for the Researcher agent.
        Falls back to simulated results if no web search is available.
        """
        query_text = input.get("query", "")
        max_results = input.get("max_results", 5)

        # Try using the web_tools module
        try:
            from src.tools.web_tools import web_search_tool

            results = web_search_tool(query=query_text, max_results=max_results)

            if results and isinstance(results, list):
                formatted_results = []
                for i, result in enumerate(results[:5], 1):
                    url = result.get("url", "")
                    title = result.get("title", "")
                    snippet = result.get("snippet", result.get("body", ""))[:200]
                    formatted_results.append(
                        f"{i}. {title}\n"
                        f"   URL: {url}\n"
                        f"   {snippet}..."
                    )
                return "\n\n".join(formatted_results) if formatted_results else "No results found."
        except ImportError as e:
            logger.warning("react.websearch_import_failed", error=str(e))
        except Exception as e:
            logger.warning("react.websearch_failed", error=str(e))

        # Try using the WebSearch tool from the environment if available
        try:
            # Import and use the WebSearch tool from the MCP server
            from src.tools.web_tools import web_search_tool

            results = web_search_tool(query=query_text, max_results=5)

            if results and isinstance(results, list):
                formatted_results = []
                for i, result in enumerate(results[:5], 1):
                    url = result.get("url", "")
                    title = result.get("title", "")
                    snippet = result.get("snippet", result.get("body", ""))[:200]
                    formatted_results.append(
                        f"{i}. {title}\n"
                        f"   URL: {url}\n"
                        f"   {snippet}..."
                    )
                return "\n\n".join(formatted_results) if formatted_results else "No results found."
        except ImportError:
            pass  # Fall through to next attempt
        except Exception as e:
            logger.warning("react.websearch_failed", error=str(e))

        # Try using the environment's WebSearch capability
        try:
            # The environment may have WebSearch available
            import subprocess
            import json

            # Try to use a web search API or service
            # This is a placeholder - in production, this would call a real web search API
            result = subprocess.run(
                ["echo", f"WebSearch for: {query_text}"],
                capture_output=True,
                text=True,
                timeout=5
            )

            # For now, provide informative output about search being simulated
            # In production with SDK, this would return real results
            return (
                f"[WebSearch results for: {query_text}]\n"
                f"Note: When using Claude Agent SDK, WebSearch provides real results. "
                f"Current fallback mode: search would be performed via SDK.\n"
                f"To enable real web search: Ensure API key is configured and use SDK mode."
            )
        except Exception as e:
            logger.warning("react.websearch_fallback_failed", error=str(e))

        # Final fallback
        return (
            f"[WebSearch results for: {query_text}]\n"
            "Note: Real web search requires the Claude Agent SDK with proper API key. "
            "Ensure MAS_LLM_PROVIDER is set correctly and API keys are configured."
        )

    def _tool_webfetch(self, input: Dict[str, Any]) -> str:
        """
        Execute WebFetch to retrieve content from a URL.

        This provides real web content fetching for the Researcher agent.
        Falls back to simulated content if fetching fails.
        """
        url = input.get("url", "")
        prompt = input.get("prompt", "Extract the main content from this page")

        if not url:
            return "Error: No URL provided for WebFetch"

        # Try using the web_tools module
        try:
            from src.tools.web_tools import web_fetch_tool

            content = web_fetch_tool(url=url, prompt=prompt)
            return content
        except ImportError as e:
            logger.warning("react.webfetch_import_failed", error=str(e))
        except Exception as e:
            logger.warning("react.webfetch_failed", url=url, error=str(e))

        if not url:
            return "Error: No URL provided for WebFetch"

        # Try using real web fetching
        try:
            import httpx
            from html.parser import HTMLParser
            from io import StringIO

            # Fetch the page
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; Multi-Agent-Researcher/1.0)"
            }

            with httpx.Client(timeout=30) as client:
                response = client.get(url, headers=headers, follow_redirects=True)

                if response.status_code == 200:
                    # Extract text content from HTML
                    html_content = response.text

                    # Simple HTML tag removal for content extraction
                    import re
                    # Remove script and style tags
                    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                    # Remove HTML tags
                    text_content = re.sub(r'<[^>]+>', '\n', html_content)
                    # Clean up whitespace
                    text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
                    text_content = text_content.strip()

                    # Return first 2000 characters
                    if len(text_content) > 2000:
                        text_content = text_content[:2000] + "...\n\n[Content truncated due to length]"

                    return f"Content from {url}:\n\n{text_content}"
                else:
                    return f"Error: Failed to fetch {url} (HTTP {response.status_code})"

        except httpx.TimeoutError:
            return f"Error: Timeout while fetching {url}"
        except Exception as e:
            logger.warning("react.webfetch_failed", url=url, error=str(e))

        # Fallback message
        return (
            f"[WebFetch for: {url}]\n"
            f"Prompt: {prompt}\n"
            "Note: Real web fetch requires network access. When using Claude Agent SDK, "
            "WebFetch provides real content. Ensure network access is available."
        )

    @staticmethod
    def _tool_skill(input: Dict[str, Any]) -> str:
        """
        Execute Skill to load and read a skill file.

        Skills are stored in .claude/skills/{name}/SKILL.md and provide
        specialized instructions and capabilities to agents.
        """
        skill_name = input.get("skill", "")

        if not skill_name:
            return "Error: No skill name provided for Skill tool"

        # Try multiple possible skill file locations
        import os
        possible_paths = [
            # Project-local skills
            f".claude/skills/{skill_name}/SKILL.md",
            f"skills/{skill_name}/SKILL.md",
            # User-level skills
            os.path.expanduser(f"~/.claude/skills/{skill_name}/SKILL.md"),
            os.path.expanduser(f"~/.claude/project/{skill_name}/SKILL.md"),
            # Module-relative skills
            os.path.join(os.path.dirname(__file__), "..", "skills", skill_name, "SKILL.md"),
        ]

        for skill_path in possible_paths:
            try:
                # Resolve relative paths against current working directory
                if not os.path.isabs(skill_path):
                    skill_path = os.path.abspath(skill_path)

                if os.path.exists(skill_path):
                    with open(skill_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Truncate very long skill files
                    if len(content) > 5000:
                        content = content[:5000] + "\n\n... [Skill content truncated due to length]"

                    return f"Skill: {skill_name}\nPath: {skill_path}\n\n{content}"
            except Exception as e:
                continue  # Try next path

        # Skill not found
        return (
            f"Error: Skill '{skill_name}' not found.\n"
            f"Searched paths:\n" + "\n".join(f"  - {p}" for p in possible_paths) + "\n"
            f"Hint: Skills should be in .claude/skills/{{name}}/SKILL.md"
        )

    # ========================================================================
    # Simulation fallback (no API available)
    # ========================================================================

    def _run_simulated(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Simulate a response when no API is available (dev/testing)."""
        logger.warning("react.simulated", agent=self.agent_name)

        if self.output_schema:
            # Generate a minimal valid instance
            simulated = self._generate_minimal_schema()
            return {
                "status": "simulated",
                "output": simulated,
                "raw_output": json.dumps(simulated) if isinstance(simulated, dict) else str(simulated),
                "tokens_used": 0,
                "cost_usd": 0.0,
            }

        return {
            "status": "simulated",
            "output": f"[Simulated ReAct output for {self.agent_name}]",
            "raw_output": f"[Simulated ReAct output for {self.agent_name}]",
            "tokens_used": 0,
            "cost_usd": 0.0,
        }

    # ========================================================================
    # Helper methods
    # ========================================================================

    def _build_prompt(self, task_input: str, context: Optional[Dict[str, Any]]) -> str:
        """Build the full prompt with context."""
        parts = [task_input]
        if context:
            ctx_str = json.dumps(context, indent=2, default=str)
            parts.append(f"\n\nAdditional context:\n```json\n{ctx_str}\n```")
        if self.output_schema:
            schema_json = json.dumps(self.output_schema.model_json_schema(), indent=2)
            parts.append(
                f"\n\nYou MUST return your final answer as valid JSON matching this schema:\n"
                f"```json\n{schema_json}\n```\n"
                "Return ONLY the JSON object, no surrounding text or markdown."
            )
        return "\n".join(parts)

    def _parse_output(self, raw_output: str) -> Any:
        """Try to parse raw output into the output schema."""
        if not self.output_schema or not raw_output:
            return None

        # Try direct JSON parse
        try:
            # Strip markdown code blocks if present
            text = raw_output.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first and last ``` lines
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)

            data = json.loads(text)
            return self.output_schema.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

        # Try to find JSON within the output
        try:
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw_output[start:end])
                return self.output_schema.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

        return None

    def _get_strict_schema(self) -> Dict[str, Any]:
        """Get a strict JSON schema for structured output."""
        if not self.output_schema:
            return {}
        schema = self.output_schema.model_json_schema()
        # Ensure additionalProperties: false for structured output
        schema["additionalProperties"] = False
        return schema

    @staticmethod
    def _extract_text(response) -> str:
        """Extract text from an Anthropic API response."""
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost based on token usage and model."""
        # Pricing per million tokens
        pricing = {
            "claude-opus-4-6": (5.0, 25.0),
            "claude-sonnet-4-6": (3.0, 15.0),
            "claude-haiku-4-5": (1.0, 5.0),
        }
        # Default to Sonnet pricing
        input_rate, output_rate = pricing.get(model, (3.0, 15.0))
        return (input_tokens / 1_000_000 * input_rate) + (output_tokens / 1_000_000 * output_rate)

    def _generate_minimal_schema(self) -> Dict[str, Any]:
        """Generate a minimal valid instance of the output schema for simulation."""
        if not self.output_schema:
            return {}
        try:
            schema = self.output_schema.model_json_schema()
            return self._fill_schema_defaults(schema)
        except Exception:
            return {}

    def _fill_schema_defaults(self, schema: Dict[str, Any]) -> Any:
        """Recursively fill a JSON schema with minimal default values."""
        schema_type = schema.get("type", "object")
        if schema_type == "object":
            result = {}
            props = schema.get("properties", {})
            for key, prop in props.items():
                if "default" in prop:
                    result[key] = prop["default"]
                elif prop.get("type") == "string":
                    result[key] = f"[simulated_{key}]"
                elif prop.get("type") == "integer":
                    result[key] = 0
                elif prop.get("type") == "number":
                    result[key] = 0.0
                elif prop.get("type") == "boolean":
                    result[key] = False
                elif prop.get("type") == "array":
                    result[key] = []
                elif prop.get("type") == "object":
                    result[key] = {}
                elif "enum" in prop:
                    result[key] = prop["enum"][0] if prop["enum"] else ""
                elif "anyOf" in prop:
                    result[key] = None
                else:
                    result[key] = None
            return result
        elif schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type in ("integer", "number"):
            return 0
        elif schema_type == "boolean":
            return False
        return None
