"""
Azure OpenAI client + request helpers — the single place the project talks to Azure.

The project now targets **Azure AI Foundry via its OpenAI-compatible endpoint** as the
sole LLM provider. A "model" everywhere in this codebase is an Azure **deployment name**
(what you called the deployment in the Foundry/Azure OpenAI portal), passed as the
`model` argument to the OpenAI-compatible API.

Environment (put in multilingual-bfcl/.env):
    AZURE_OPENAI_ENDPOINT      e.g. https://<resource>.openai.azure.com/
    AZURE_OPENAI_API_KEY       the resource key
    AZURE_OPENAI_API_VERSION   optional; defaults to DEFAULT_API_VERSION below
    AZURE_OPENAI_DEPLOYMENT    optional; default deployment when --model is omitted

Two kinds of model are distinguished on the command line via --model-type:
  - standard  : ordinary chat models (gpt-4o, gpt-4.1, ...). Accept `temperature`
                and `max_tokens`.
  - reasoning : reasoning models (o1/o3/o4/gpt-5 reasoning, ...). Reject `temperature`
                and use `max_completion_tokens` instead of `max_tokens`.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

# A recent GA api-version that supports tools + the Batch API. Override with
# AZURE_OPENAI_API_VERSION (e.g. a preview version for the newest models).
DEFAULT_API_VERSION = "2024-10-21"


class ModelType(str, Enum):
    """How to shape request parameters for the selected deployment."""

    STANDARD = "standard"
    REASONING = "reasoning"


def default_deployment() -> str | None:
    """Deployment name used when --model is not given (AZURE_OPENAI_DEPLOYMENT)."""
    return os.getenv("AZURE_OPENAI_DEPLOYMENT")


def _endpoint() -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is not set. Add it to multilingual-bfcl/.env "
            "(e.g. https://<resource>.openai.azure.com/)."
        )
    return endpoint


def _api_version() -> str:
    return os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)


def _api_key() -> str:
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY is not set. Add it to multilingual-bfcl/.env."
        )
    return key


def make_sync_client(**kwargs: Any):
    """A synchronous AzureOpenAI client configured from the environment."""
    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=_endpoint(),
        api_key=_api_key(),
        api_version=_api_version(),
        **kwargs,
    )


def make_async_client(**kwargs: Any):
    """An asynchronous AsyncAzureOpenAI client configured from the environment."""
    from openai import AsyncAzureOpenAI

    return AsyncAzureOpenAI(
        azure_endpoint=_endpoint(),
        api_key=_api_key(),
        api_version=_api_version(),
        **kwargs,
    )


def make_langchain_chat(
    deployment: str,
    *,
    model_type: ModelType = ModelType.STANDARD,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
):
    """A LangChain AzureChatOpenAI bound to `deployment`.

    Reasoning deployments omit `temperature` (only the default is accepted) and pass
    the token cap as `max_completion_tokens` via `model_kwargs`.
    """
    from langchain_openai import AzureChatOpenAI

    params: dict[str, Any] = {
        "azure_deployment": deployment,
        "azure_endpoint": _endpoint(),
        "api_key": _api_key(),
        "api_version": _api_version(),
    }
    if model_type == ModelType.REASONING:
        if max_tokens is not None:
            params.setdefault("model_kwargs", {})["max_completion_tokens"] = max_tokens
    else:
        params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
    params.update(kwargs)
    return AzureChatOpenAI(**params)


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
