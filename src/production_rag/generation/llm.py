"""Provider-agnostic LLM client.

Both providers use their SDK's built-in retries with exponential backoff and a
hard timeout, and every call reports token usage so cost can be tracked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from anthropic import Anthropic
from openai import OpenAI


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLM(Protocol):
    model: str

    def complete(
        self, system: str, prompt: str, *, max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse: ...


class OpenAILLM:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def complete(
        self, system: str, prompt: str, *, max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"} if json_mode else {"type": "text"},
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
        )


class AnthropicLLM:
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self._client = Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def complete(
        self, system: str, prompt: str, *, max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        if json_mode:
            system = f"{system}\n\nRespond with a single valid JSON object and nothing else."
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
        )
