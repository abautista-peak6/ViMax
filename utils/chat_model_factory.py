"""Chat model factory for ViMax inference providers."""

from typing import Any, Dict

from langchain.chat_models import init_chat_model

from tools.chat_model_google_vertex import ChatGoogleVertexAI
from utils.provider_presets import resolve_chat_model_config


GOOGLE_VERTEX_PROVIDERS = {"google_vertex", "google_vertexai", "vertex", "vertexai"}
_VERTEX_DIRECT_KWARGS = {
    "model",
    "project",
    "location",
    "temperature",
    "max_tokens",
    "max_output_tokens",
    "top_p",
    "top_k",
    "candidate_count",
    "api_version",
}


def create_chat_model(init_args: Dict[str, Any]):
    """Create the configured chat model, routing Google Vertex locally."""
    args = resolve_chat_model_config(init_args)
    provider = args.get("model_provider", "google_vertex")

    if provider in GOOGLE_VERTEX_PROVIDERS:
        vertex_args = {
            key: value
            for key, value in args.items()
            if key in _VERTEX_DIRECT_KWARGS and value is not None
        }
        vertex_args.setdefault("model", "gemini-2.5-flash")
        model_kwargs = {
            key: value
            for key, value in args.items()
            if key not in _VERTEX_DIRECT_KWARGS
            and key not in {"model_provider", "api_key", "base_url"}
            and value is not None
        }
        if model_kwargs:
            vertex_args["model_kwargs"] = model_kwargs
        return ChatGoogleVertexAI(**vertex_args)

    return init_chat_model(**args)
