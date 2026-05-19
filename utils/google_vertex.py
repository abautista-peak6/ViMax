"""Helpers for Google Vertex AI inference clients."""

import os
from typing import Optional

from google import genai
from google.genai import types


DEFAULT_VERTEX_LOCATION = "global"


def resolve_vertex_project(project: Optional[str] = None) -> str:
    """Resolve the Google Cloud project used for Vertex AI calls."""
    resolved = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if not resolved:
        raise ValueError(
            "Google Vertex AI requires a project. Set init_args.project or "
            "GOOGLE_CLOUD_PROJECT."
        )
    return resolved


def resolve_vertex_location(location: Optional[str] = None) -> str:
    """Resolve the Vertex AI region/location for inference calls."""
    return (
        location
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or os.environ.get("GOOGLE_CLOUD_REGION")
        or DEFAULT_VERTEX_LOCATION
    )


def create_genai_client(
    *,
    project: Optional[str] = None,
    location: Optional[str] = None,
    api_key: Optional[str] = None,
    use_vertex_ai: bool = True,
    api_version: str = "v1",
):
    """Create a Google Gen AI SDK client for Vertex AI by default."""
    http_options = types.HttpOptions(api_version=api_version) if api_version else None

    if use_vertex_ai:
        return genai.Client(
            vertexai=True,
            project=resolve_vertex_project(project),
            location=resolve_vertex_location(location),
            http_options=http_options,
        )

    if not api_key:
        raise ValueError("Gemini Developer API mode requires api_key.")

    return genai.Client(
        api_key=api_key,
        http_options=http_options,
    )
