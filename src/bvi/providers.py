"""Optional SDK transports; explicit model IDs and credentials are caller-owned.

Wire formats checked against official docs on 2026-09-14:
https://developers.openai.com/api/docs/guides/images-vision
https://developers.openai.com/api/docs/guides/structured-outputs
https://platform.claude.com/docs/en/api/messages/create

SDKs are imported only on generate(). Retries are disabled so every attempt is
accounted for by VLMCoordinator. These adapters are offline-contract-tested;
live account access and model support must be validated in an authorized run.
"""
from __future__ import annotations

import base64
import json
from typing import Any

from .coordinator import VLMRequest, VLMResponse


class OpenAITransport:
    provider = "openai"

    def __init__(self, model: str, timeout_seconds: float = 60.0, client: Any = None,
                 image_detail: str = "low", reasoning_effort: str | None = None):
        if not model:
            raise ValueError("An explicit model ID is required")
        self.model, self.timeout_seconds, self._client = model, timeout_seconds, client
        if image_detail not in ("low", "high", "auto", "original"):
            raise ValueError("Unsupported image detail")
        self.image_detail, self.reasoning_effort = image_detail, reasoning_effort
        self.request_options = {"image_detail": image_detail, "reasoning_effort": reasoning_effort}

    def generate(self, request: VLMRequest) -> VLMResponse:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(max_retries=0, timeout=self.timeout_seconds)
        content = [{"type": "input_text", "text": request.prompt}]
        for frame in request.images:
            encoded = base64.b64encode(frame.data).decode("ascii")
            content.append({"type": "input_image", "detail": self.image_detail,
                            "image_url": f"data:{frame.media_type};base64,{encoded}"})
        options = {} if self.reasoning_effort is None else {"reasoning": {"effort": self.reasoning_effort}}
        response = self._client.responses.create(
            model=self.model, instructions=request.system,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "skill_request",
                              "strict": True, "schema": dict(request.schema)}},
            max_output_tokens=request.max_output_tokens, store=False, **options,
        )
        usage = response.usage.model_dump() if response.usage is not None else None
        return VLMResponse(response.output_text, response.id, usage,
                           getattr(response, "status", None))


class AnthropicTransport:
    provider = "anthropic"

    def __init__(self, model: str, timeout_seconds: float = 60.0, client: Any = None):
        if not model:
            raise ValueError("An explicit model ID is required")
        self.model, self.timeout_seconds, self._client = model, timeout_seconds, client

    def generate(self, request: VLMRequest) -> VLMResponse:
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(max_retries=0, timeout=self.timeout_seconds)
        content = [{"type": "text", "text": request.prompt}]
        for frame in request.images:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": frame.media_type,
                "data": base64.b64encode(frame.data).decode("ascii")}})
        response = self._client.messages.create(
            model=self.model, system=request.system,
            messages=[{"role": "user", "content": content}],
            tools=[{"name": "request_skill", "description": "Request one admissible skill",
                    "input_schema": dict(request.schema)}],
            tool_choice={"type": "tool", "name": "request_skill"},
            thinking={"type": "disabled"},
            max_tokens=request.max_output_tokens,
        )
        calls = [block for block in response.content
                 if block.type == "tool_use" and block.name == "request_skill"]
        # Zero/multiple tool calls yield an invalid output, preserving usage in
        # the coordinator's log before the parser rejects the response.
        text = json.dumps(calls[0].input) if len(calls) == 1 else ""
        usage = response.usage.model_dump() if response.usage is not None else None
        return VLMResponse(text, response.id, usage, response.stop_reason)
