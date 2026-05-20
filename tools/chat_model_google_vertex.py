"""LangChain chat model adapter backed by Google Gen AI SDK on Vertex AI."""

from __future__ import annotations

import base64
import asyncio
import mimetypes
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from google.genai import types
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from utils.google_vertex import create_genai_client
from utils.rate_limiter import RateLimiter


class ChatGoogleVertexAI(BaseChatModel):
    """Minimal LangChain chat adapter for Gemini models served by Vertex AI."""

    model: str = "gemini-3.1-pro-preview"
    project: Optional[str] = None
    location: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    candidate_count: Optional[int] = None
    api_version: str = "v1"
    max_requests_per_minute: Optional[int] = None
    max_requests_per_day: Optional[int] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    _client: Any = PrivateAttr()
    _rate_limiter: Optional[RateLimiter] = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._client = create_genai_client(
            project=self.project,
            location=self.location,
            use_vertex_ai=True,
            api_version=self.api_version,
        )
        if self.max_requests_per_minute or self.max_requests_per_day:
            self._rate_limiter = RateLimiter(
                max_requests_per_minute=self.max_requests_per_minute,
                max_requests_per_day=self.max_requests_per_day,
            )

    @property
    def _llm_type(self) -> str:
        return "google_vertex_genai"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "project": self.project,
            "location": self.location,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._acquire_rate_limit_sync()
        contents, system_instruction = self._messages_to_contents(messages)
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._build_config(stop, system_instruction, kwargs),
        )
        return self._to_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        contents, system_instruction = self._messages_to_contents(messages)
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._build_config(stop, system_instruction, kwargs),
        )
        return self._to_chat_result(response)

    def _build_config(
        self,
        stop: Optional[List[str]],
        system_instruction: Optional[str],
        runtime_kwargs: Dict[str, Any],
    ) -> Optional[types.GenerateContentConfig]:
        config_kwargs = dict(self.model_kwargs)
        config_kwargs.update(runtime_kwargs)

        for key in ("temperature", "top_p", "top_k", "candidate_count"):
            value = getattr(self, key)
            if value is not None:
                config_kwargs[key] = value

        max_output_tokens = self.max_output_tokens or self.max_tokens
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens

        if stop:
            config_kwargs["stop_sequences"] = stop

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        return types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    def _acquire_rate_limit_sync(self):
        if not self._rate_limiter:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._rate_limiter.acquire())

    def _messages_to_contents(
        self,
        messages: Sequence[BaseMessage],
    ) -> Tuple[List[types.Content], Optional[str]]:
        system_chunks: List[str] = []
        contents: List[types.Content] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                system_chunks.append(_content_to_text(message.content))
                continue

            role = "user" if isinstance(message, HumanMessage) else "model"
            parts = _content_to_parts(message.content)
            if parts:
                contents.append(types.Content(role=role, parts=parts))

        return contents, "\n\n".join(chunk for chunk in system_chunks if chunk) or None

    def _to_chat_result(self, response: Any) -> ChatResult:
        message = AIMessage(
            content=_response_text(response),
            response_metadata=_response_metadata(response),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                chunks.append(item.get("text", ""))
        return "\n".join(chunk for chunk in chunks if chunk)

    return str(content)


def _content_to_parts(content: Any) -> List[types.Part]:
    if isinstance(content, str):
        return [types.Part.from_text(text=content)]

    if not isinstance(content, list):
        return [types.Part.from_text(text=str(content))]

    parts: List[types.Part] = []
    for item in content:
        if isinstance(item, str):
            parts.append(types.Part.from_text(text=item))
            continue

        if not isinstance(item, dict):
            parts.append(types.Part.from_text(text=str(item)))
            continue

        item_type = item.get("type")
        if item_type in {"text", "input_text"}:
            parts.append(types.Part.from_text(text=item.get("text", "")))
        elif item_type == "image_url":
            url_value = item.get("image_url", "")
            url = url_value.get("url", "") if isinstance(url_value, dict) else url_value
            parts.append(_image_url_to_part(url))

    return parts


def _image_url_to_part(url: str) -> types.Part:
    if url.startswith("data:"):
        header, payload = url.split(",", 1)
        mime_type = header.split(";", 1)[0].replace("data:", "") or "application/octet-stream"
        return types.Part.from_bytes(data=base64.b64decode(payload), mime_type=mime_type)

    if url.startswith("gs://") or url.startswith("http://") or url.startswith("https://"):
        mime_type, _ = mimetypes.guess_type(url)
        return types.Part.from_uri(file_uri=url, mime_type=mime_type or "image/png")

    if os.path.exists(url):
        mime_type, _ = mimetypes.guess_type(url)
        with open(url, "rb") as image_file:
            return types.Part.from_bytes(
                data=image_file.read(),
                mime_type=mime_type or "application/octet-stream",
            )

    raise ValueError(f"Unsupported image URL for Vertex AI content: {url}")


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text

    chunks: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
    return "".join(chunks)


def _response_metadata(response: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        metadata["usage_metadata"] = usage

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason

    return metadata
