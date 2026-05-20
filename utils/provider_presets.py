"""
Provider preset system for ViMax chat model configuration.

Supports auto-detection and resolution of LLM provider settings,
allowing users to specify a provider name (e.g., ``minimax``) instead
of manually configuring base_url and model details.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "google_vertex": {
        "env_project": "GOOGLE_CLOUD_PROJECT",
        "env_location": "GOOGLE_CLOUD_LOCATION",
        "default_model": "gemini-3.1-pro-preview",
        "default_location": "global",
        "models": [
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
        ],
        "temperature_range": (0.0, 2.0),
        "rewrite_model_provider": "google_vertex",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "env_key": "MINIMAX_API_KEY",
        "default_model": "MiniMax-M2.7",
        "models": [
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
        ],
        "temperature_range": (0.0, 1.0),
        "rewrite_model_provider": "openai",
    },
}


def resolve_chat_model_config(init_args: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve provider presets and return final chat model factory kwargs.

    If ``model_provider`` matches a known preset (e.g. ``google_vertex`` or
    ``minimax``), the returned dict will have provider defaults filled in.
    OpenAI-compatible providers such as ``minimax`` are rewritten to
    ``"openai"`` for LangChain compatibility.

    * ``model`` defaulted to the preset's default model when not already set
    * ``temperature`` clamped to the provider's supported range

    For unknown providers the dict is returned unchanged.
    """
    args = dict(init_args)  # shallow copy
    provider = args.get("model_provider") or "google_vertex"
    args["model_provider"] = provider

    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        return args

    # base_url
    if preset.get("base_url") and not args.get("base_url"):
        args["base_url"] = preset["base_url"]

    # api_key – fall back to env var
    if not args.get("api_key"):
        env_key = preset.get("env_key", "")
        env_val = os.environ.get(env_key, "")
        if env_val:
            args["api_key"] = env_val
            logger.info("Using %s API key from environment variable %s", provider, env_key)

    # Vertex project/location
    if not args.get("project"):
        env_project = preset.get("env_project", "")
        project = os.environ.get(env_project, "")
        if project:
            args["project"] = project
            logger.info("Using Google Vertex project from environment variable %s", env_project)

    if preset.get("default_location") and not args.get("location"):
        env_location = preset.get("env_location", "")
        args["location"] = os.environ.get(env_location, "") or preset["default_location"]

    # default model
    if not args.get("model"):
        args["model"] = preset["default_model"]
        logger.info("Defaulting to model %s for provider %s", args["model"], provider)

    # temperature clamping
    temp_range = preset.get("temperature_range")
    if temp_range and "temperature" in args and args["temperature"] is not None:
        lo, hi = temp_range
        original = args["temperature"]
        args["temperature"] = max(lo, min(hi, original))
        if args["temperature"] != original:
            logger.warning(
                "Clamped temperature %.2f -> %.2f for provider %s",
                original, args["temperature"], provider,
            )

    rewrite_provider = preset.get("rewrite_model_provider")
    if rewrite_provider:
        args["model_provider"] = rewrite_provider

    return args


def detect_provider_from_env() -> Optional[str]:
    """Return the name of a provider whose API key is found in the environment.

    Checks ``PROVIDER_PRESETS`` in definition order and returns the first
    match, or ``None`` if no key is set.
    """
    for name, preset in PROVIDER_PRESETS.items():
        env_key = preset.get("env_key", "")
        if env_key and os.environ.get(env_key):
            return name
        env_project = preset.get("env_project", "")
        if env_project and os.environ.get(env_project):
            return name
    return None
