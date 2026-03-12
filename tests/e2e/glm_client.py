"""
GLM-4 Client Wrapper for E2E Testing

Provides a lightweight client for ZhipuAI's GLM-4 model via their
OpenAI-compatible API endpoint. Used as the LLM backbone for all
E2E test scenarios.

API docs: https://open.bigmodel.cn/dev/api
"""

import json
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
GLM_API_KEY = "85cf3935c0b843738d461fec7cb2b515.dFTF3tjsPnXLaglE"
GLM_MODEL = "glm-4-plus"
GLM_FLASH_MODEL = "glm-4-flash"

REQUEST_TIMEOUT = 120  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds


@dataclass
class GLMResponse:
    """Parsed response from GLM-4 API."""
    content: str
    model: str
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.finish_reason in ("stop", "eos")

    def as_json(self) -> Optional[Dict[str, Any]]:
        """Try to parse content as JSON."""
        try:
            # Handle markdown code blocks
            text = self.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            return None


@dataclass
class GLMDefect:
    """A defect captured during E2E testing."""
    test_name: str
    category: str  # api_error, schema_violation, logic_error, timeout, etc.
    severity: str  # critical, high, medium, low
    description: str
    prompt: str
    response: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    error_trace: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")


class DefectTracker:
    """Collects and reports defects found during E2E testing."""

    def __init__(self):
        self.defects: List[GLMDefect] = []

    def add(self, defect: GLMDefect):
        self.defects.append(defect)
        logger.warning(
            f"DEFECT [{defect.severity.upper()}] {defect.category}: "
            f"{defect.description} (test={defect.test_name})"
        )

    def get_by_severity(self, severity: str) -> List[GLMDefect]:
        return [d for d in self.defects if d.severity == severity]

    def get_by_category(self, category: str) -> List[GLMDefect]:
        return [d for d in self.defects if d.category == category]

    @property
    def critical_count(self) -> int:
        return len(self.get_by_severity("critical"))

    @property
    def total_count(self) -> int:
        return len(self.defects)

    def summary(self) -> Dict[str, Any]:
        by_severity = {}
        by_category = {}
        for d in self.defects:
            by_severity[d.severity] = by_severity.get(d.severity, 0) + 1
            by_category[d.category] = by_category.get(d.category, 0) + 1
        return {
            "total": self.total_count,
            "by_severity": by_severity,
            "by_category": by_category,
            "defects": [
                {
                    "test": d.test_name,
                    "category": d.category,
                    "severity": d.severity,
                    "description": d.description,
                }
                for d in self.defects
            ],
        }


# Global defect tracker shared across all E2E tests
defect_tracker = DefectTracker()


class GLMClient:
    """
    Client for GLM-4 API calls.

    Supports chat completions with system prompts, JSON mode,
    temperature control, and automatic retry with backoff.
    """

    def __init__(
        self,
        api_key: str = GLM_API_KEY,
        base_url: str = GLM_BASE_URL,
        model: str = GLM_MODEL,
        timeout: int = REQUEST_TIMEOUT,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self.call_count = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0

    def chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, str]] = None,
        stop: Optional[List[str]] = None,
    ) -> GLMResponse:
        """
        Send a chat completion request to GLM-4.

        Args:
            user_prompt: The user message
            system_prompt: Optional system message
            model: Model override (default: glm-4-plus)
            temperature: Sampling temperature
            max_tokens: Max output tokens
            response_format: e.g. {"type": "json_object"} for JSON mode
            stop: Stop sequences

        Returns:
            GLMResponse with parsed results
        """
        model = model or self.default_model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(MAX_RETRIES):
            start = time.time()
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                latency_ms = (time.time() - start) * 1000

                if resp.status_code != 200:
                    error_body = resp.text
                    logger.error(
                        f"GLM API error {resp.status_code}: {error_body} "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    last_error = f"HTTP {resp.status_code}: {error_body}"
                    if resp.status_code in (429, 500, 502, 503):
                        time.sleep(RETRY_BACKOFF * (attempt + 1))
                        continue
                    break

                data = resp.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                usage = data.get("usage", {})

                self.call_count += 1
                tokens = usage.get("total_tokens", 0)
                self.total_tokens += tokens
                self.total_latency_ms += latency_ms

                return GLMResponse(
                    content=message.get("content", ""),
                    model=data.get("model", model),
                    finish_reason=choice.get("finish_reason", "unknown"),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=tokens,
                    latency_ms=latency_ms,
                    raw=data,
                )

            except httpx.TimeoutException as e:
                latency_ms = (time.time() - start) * 1000
                last_error = f"Timeout after {latency_ms:.0f}ms: {e}"
                logger.warning(f"GLM timeout (attempt {attempt + 1}): {e}")
                time.sleep(RETRY_BACKOFF * (attempt + 1))
            except httpx.HTTPError as e:
                latency_ms = (time.time() - start) * 1000
                last_error = f"HTTP error: {e}"
                logger.warning(f"GLM HTTP error (attempt {attempt + 1}): {e}")
                time.sleep(RETRY_BACKOFF * (attempt + 1))

        # All retries exhausted
        return GLMResponse(
            content=f"[ERROR] {last_error}",
            model=model,
            finish_reason="error",
            latency_ms=0,
        )

    def chat_json(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> GLMResponse:
        """Chat with JSON response format enforced."""
        # Append JSON instruction to system prompt
        json_system = system_prompt
        if json_system:
            json_system += "\n\nYou MUST respond with valid JSON only. No markdown, no extra text."
        else:
            json_system = "You MUST respond with valid JSON only. No markdown, no extra text."

        return self.chat(
            user_prompt=user_prompt,
            system_prompt=json_system,
            model=model,
            temperature=temperature,
        )

    def stats(self) -> Dict[str, Any]:
        """Return usage stats."""
        return {
            "calls": self.call_count,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": (
                self.total_latency_ms / self.call_count
                if self.call_count > 0
                else 0
            ),
        }

    def close(self):
        self._client.close()
