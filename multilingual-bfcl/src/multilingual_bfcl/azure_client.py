"""
Azure AI Foundry client + request helpers — the single place the project talks to Azure.

The project targets **Azure AI Foundry via its OpenAI-compatible v1 endpoint** as the
sole LLM provider. This is the route the Foundry portal's sample code uses: the plain
`openai` client pointed at `https://<resource>.services.ai.azure.com/openai/v1`, with
NO `api-version` query parameter (unlike the older `AzureOpenAI` client). A "model" here
is an Azure **deployment name**, passed as the `model` argument.

Environment (put in multilingual-bfcl/.env):
    AZURE_OPENAI_ENDPOINT   the resource endpoint. Either the v1 base URL
                            (https://<resource>.services.ai.azure.com/openai/v1) or the
                            resource root (https://<resource>.services.ai.azure.com) —
                            "/openai/v1" is appended automatically if missing.
    AZURE_OPENAI_API_KEY    the resource key (sent as a bearer token by the openai SDK).

Two kinds of model are distinguished on the command line via --model-type:
  - standard  : ordinary chat models (gpt-4o, gpt-4.1, ...). Accept `temperature`
                and `max_tokens`.
  - reasoning : reasoning models (o1/o3/o4/gpt-5, ...). Reject `temperature` and use
                `max_completion_tokens` instead of `max_tokens`.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any
from urllib.parse import urlparse

# Foundry route suffixes (all under the same resource host).
_V1_SUFFIX = "/openai/v1"        # OpenAI-compatible chat/completions + tools
_ANTHROPIC_SUFFIX = "/anthropic"  # native Anthropic Messages API (Claude models)


class ModelType(str, Enum):
    """How to route/shape a request for the selected deployment.

    STANDARD/REASONING go through the OpenAI-compatible v1 route (shaping only
    differs by whether `temperature`/`max_completion_tokens` are sent). CLAUDE routes
    through the Foundry `/anthropic` route (native Anthropic message format) and is
    handled by AnthropicFoundryFCHandler — used only by the robustness eval.
    """

    STANDARD = "standard"
    REASONING = "reasoning"
    CLAUDE = "claude"


# Param-shaping model types valid on the OpenAI v1 route (excludes CLAUDE).
OPENAI_MODEL_TYPES = (ModelType.STANDARD, ModelType.REASONING)


def default_deployment() -> str | None:
    """Deployment name used when --model is not given (AZURE_OPENAI_DEPLOYMENT)."""
    return os.getenv("AZURE_OPENAI_DEPLOYMENT")


def _resource_host() -> str:
    """The Foundry resource root (scheme://host) from AZURE_OPENAI_ENDPOINT.

    Accepts whatever the portal shows — a route URL (…/openai/v1, …/anthropic), the
    resource root, or the project endpoint (…/api/projects/<project>) — and reduces it
    to scheme://host, off which the individual route suffixes are appended.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is not set. Add it to multilingual-bfcl/.env, e.g. "
            "https://<resource>.services.ai.azure.com/openai/v1"
        )
    endpoint = endpoint.rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return endpoint


def _base_url() -> str:
    """The Foundry OpenAI-compatible v1 base URL (https://<host>/openai/v1)."""
    return _resource_host() + _V1_SUFFIX


def anthropic_base_url() -> str:
    """The Foundry native-Anthropic base URL (https://<host>/anthropic)."""
    return _resource_host() + _ANTHROPIC_SUFFIX


def _api_key() -> str:
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY is not set. Add it to multilingual-bfcl/.env."
        )
    return key


def make_sync_client(**kwargs: Any):
    """A synchronous OpenAI client pointed at the Foundry v1 endpoint."""
    from openai import OpenAI

    return OpenAI(base_url=_base_url(), api_key=_api_key(), **kwargs)


def make_async_client(**kwargs: Any):
    """An asynchronous OpenAI client pointed at the Foundry v1 endpoint."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=_base_url(), api_key=_api_key(), **kwargs)


def make_anthropic_client(**kwargs: Any):
    """A synchronous AnthropicFoundry client for the Foundry `/anthropic` route.

    Uses the resource key (AZURE_OPENAI_API_KEY) for auth, so no azure-identity /
    Entra ID token provider is required. The Foundry `/anthropic` gateway accepts the
    resource key as a **bearer token** (`Authorization: Bearer <key>`) — NOT as the
    `api-key` header the SDK would otherwise use — so we set that header explicitly.
    """
    from anthropic import AnthropicFoundry

    key = _api_key()
    headers = {"Authorization": f"Bearer {key}", **kwargs.pop("default_headers", {})}
    return AnthropicFoundry(
        base_url=anthropic_base_url(), api_key=key, default_headers=headers, **kwargs
    )


def build_chat_params(
    deployment: str,
    messages: list[dict[str, Any]],
    *,
    model_type: ModelType = ModelType.STANDARD,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble kwargs for a chat.completions request (direct call or a batch body).

    Handles the standard/reasoning parameter differences in one place so the direct
    SDK path, the Batch API path, and the eval handler stay consistent.
    """
    params: dict[str, Any] = {"model": deployment, "messages": messages}
    if model_type == ModelType.REASONING:
        if max_tokens is not None:
            params["max_completion_tokens"] = max_tokens
    else:
        params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
    if tools:
        params["tools"] = tools
    return params
