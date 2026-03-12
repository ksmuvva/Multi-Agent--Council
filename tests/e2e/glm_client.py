"""
GLM-4 API Client Wrapper

Uses the ZhipuAI GLM-4 model via OpenAI-compatible API for E2E testing.
GLM-4 is accessed through https://open.bigmodel.cn/api/paas/v4 which is
OpenAI-compatible, so we use the openai SDK with a custom base_url.
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)

# GLM-4 Configuration
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
GLM_MODEL = "glm-4-plus"
GLM_FLASH_MODEL = "glm-4-flash"


@dataclass
class GLMResponse:
    """Structured response from GLM-4 API."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    finish_reason: str
    latency_ms: float
    raw_response: Optional[Any] = None
    error: Optional[str] = None
    success: bool = True


@dataclass
class GLMDefect:
    """Captured defect from E2E testing."""
    defect_id: str
    scenario: str
    test_name: str
    severity: str  # critical, high, medium, low
    category: str  # api_error, logic_error, schema_error, timeout, etc.
    description: str
    expected: str
    actual: str
    prompt: str = ""
    response: str = ""
    traceback: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defect_id": self.defect_id,
            "scenario": self.scenario,
            "test_name": self.test_name,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "expected": self.expected,
            "actual": self.actual,
            "prompt": self.prompt[:500],
            "response": self.response[:500],
            "traceback": self.traceback[:1000],
            "timestamp": self.timestamp,
        }


class DefectCollector:
    """Collects and reports defects found during E2E testing."""

    def __init__(self):
        self.defects: List[GLMDefect] = []
        self._counter = 0

    def add_defect(
        self,
        scenario: str,
        test_name: str,
        severity: str,
        category: str,
        description: str,
        expected: str,
        actual: str,
        prompt: str = "",
        response: str = "",
        traceback: str = "",
    ) -> GLMDefect:
        self._counter += 1
        defect = GLMDefect(
            defect_id=f"DEF-{self._counter:04d}",
            scenario=scenario,
            test_name=test_name,
            severity=severity,
            category=category,
            description=description,
            expected=expected,
            actual=actual,
            prompt=prompt,
            response=response,
            traceback=traceback,
        )
        self.defects.append(defect)
        logger.warning(
            f"DEFECT {defect.defect_id} [{severity.upper()}] "
            f"{scenario}/{test_name}: {description}"
        )
        return defect

    def get_summary(self) -> Dict[str, Any]:
        by_severity = {}
        by_category = {}
        for d in self.defects:
            by_severity[d.severity] = by_severity.get(d.severity, 0) + 1
            by_category[d.category] = by_category.get(d.category, 0) + 1

        return {
            "total_defects": len(self.defects),
            "by_severity": by_severity,
            "by_category": by_category,
            "defects": [d.to_dict() for d in self.defects],
        }

    def has_critical(self) -> bool:
        return any(d.severity == "critical" for d in self.defects)


class GLMClient:
    """
    Client for interacting with the GLM-4 API via OpenAI-compatible interface.

    Wraps the openai SDK and provides methods that simulate agent behaviors.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("GLM_API_KEY", "")
        self.base_url = base_url or GLM_BASE_URL
        self.model = model or GLM_MODEL
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "GLM API key is required. Set GLM_API_KEY env var or pass api_key."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        self.defect_collector = DefectCollector()
        self._call_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0

    def chat(
        self,
        prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> GLMResponse:
        """
        Send a chat completion request to GLM-4.

        Args:
            prompt: User message
            system_prompt: System message
            model: Model override
            temperature: Sampling temperature
            max_tokens: Max output tokens
            response_format: JSON response format spec

        Returns:
            GLMResponse with content and metadata
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(**kwargs)
            latency_ms = (time.time() - start_time) * 1000

            choice = response.choices[0]
            usage = response.usage

            self._call_count += 1
            self._total_tokens += usage.total_tokens
            self._total_latency_ms += latency_ms

            return GLMResponse(
                content=choice.message.content or "",
                model=response.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                finish_reason=choice.finish_reason or "stop",
                latency_ms=latency_ms,
                raw_response=response,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return GLMResponse(
                content="",
                model=model or self.model,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                finish_reason="error",
                latency_ms=latency_ms,
                error=str(e),
                success=False,
            )

    def chat_json(
        self,
        prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> GLMResponse:
        """Send a chat request expecting JSON output."""
        json_system = (
            system_prompt
            + "\n\nIMPORTANT: You MUST respond with valid JSON only. "
            "No markdown, no code fences, no explanations outside JSON."
        )
        response = self.chat(
            prompt=prompt,
            system_prompt=json_system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Attempt to clean JSON from markdown fences
        if response.success and response.content:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            response.content = content.strip()

        return response

    def simulate_agent(
        self,
        agent_name: str,
        agent_system_prompt: str,
        task_prompt: str,
        output_schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> GLMResponse:
        """
        Simulate an agent call using GLM-4.

        Args:
            agent_name: Name of the agent being simulated
            agent_system_prompt: The agent's system prompt
            task_prompt: The task for the agent
            output_schema: Expected output JSON schema
            model: Model override

        Returns:
            GLMResponse from the simulated agent
        """
        system = f"You are the {agent_name} agent.\n\n{agent_system_prompt}"

        if output_schema:
            schema_str = json.dumps(output_schema, indent=2)
            system += (
                f"\n\nYour output MUST conform to this JSON schema:\n{schema_str}\n"
                "Respond with valid JSON only."
            )
            return self.chat_json(
                prompt=task_prompt,
                system_prompt=system,
                model=model,
            )
        else:
            return self.chat(
                prompt=task_prompt,
                system_prompt=system,
                model=model,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get client usage statistics."""
        return {
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": (
                self._total_latency_ms / self._call_count
                if self._call_count > 0
                else 0
            ),
            "defects_found": len(self.defect_collector.defects),
        }
