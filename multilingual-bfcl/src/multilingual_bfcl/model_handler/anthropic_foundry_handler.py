"""
Anthropic (Claude) function-calling handler for the robustness harness, via the
Azure AI Foundry `/anthropic` route.

Subclasses BFCL's ClaudeHandler (native Anthropic Messages format: separate `system`
field, `content` blocks, `tool_result` user messages) but points the client at
Foundry's `AnthropicFoundry` client instead of the public Anthropic API:

  - The client is `anthropic.AnthropicFoundry(base_url=…/anthropic, api_key=<resource key>)`.
  - `model` is the Azure **deployment name** (the harness registers the handler under
    "<deployment>-FC"; we strip that suffix for the API call).
  - `temperature` is not sent — Foundry Claude reasoning models (e.g. claude-opus-5)
    reject sampling params, and translation/eval want deterministic output anyway.

Selected by `--model-type claude` in run_robustness_eval. The other scripts stay on
the OpenAI-compatible v1 route (AzureOpenAIFCHandler).
"""

from __future__ import annotations

from anthropic.types import TextBlock, ToolUseBlock
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.api_inference.claude import ClaudeHandler
from bfcl_eval.model_handler.base_handler import BaseHandler

from multilingual_bfcl.azure_client import make_anthropic_client


class AnthropicFoundryFCHandler(ClaudeHandler):
    # max_tokens is required by the Anthropic Messages API.
    DEFAULT_MAX_TOKENS = 8192
    REQUEST_TIMEOUT = 1200

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        *,
        max_tokens: int | None = None,
        **kwargs,
    ) -> None:
        # Skip ClaudeHandler.__init__: it eagerly builds `Anthropic(api_key=ANTHROPIC_API_KEY)`.
        # Initialise the grandparent directly, then attach the Foundry Anthropic client.
        BaseHandler.__init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.ANTHROPIC
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.client = make_anthropic_client()

    def _api_model_name(self) -> str:
        """The Azure deployment name (registry name minus the trailing '-FC')."""
        name = self.model_name
        return name[:-3] if name.endswith("-FC") else name

    def _get_max_tokens(self) -> int:
        # ClaudeHandler._get_max_tokens raises for models it doesn't know; use our cap.
        return self.max_tokens

    def _query_FC(self, inference_data: dict):
        messages = inference_data["message"]
        inference_data["inference_input_log"] = {
            "message": repr(messages),
            "tools": inference_data["tools"],
            "system_prompt": inference_data.get("system_prompt", []),
        }
        kwargs = {
            "model": self._api_model_name(),
            "max_tokens": self._get_max_tokens(),
            "tools": inference_data["tools"],
            "messages": messages,
            "timeout": self.REQUEST_TIMEOUT,
        }
        if "system_prompt" in inference_data:
            kwargs["system"] = inference_data["system_prompt"]
        return self.generate_with_backoff(**kwargs)

    # --- Format adapters used by the robustness harness (see run_robustness_eval) ---

    def extract_turn(self, api_response) -> tuple[list[str], list[dict]]:
        """Pull assistant text + tool calls out of an Anthropic Messages response."""
        texts = [b.text for b in api_response.content if isinstance(b, TextBlock)]
        tool_calls = [
            {"name": b.name, "arguments": b.input, "id": b.id}
            for b in api_response.content
            if isinstance(b, ToolUseBlock)
        ]
        return texts, tool_calls

    def apply_language_prefix(self, inference_data: dict, text: str) -> None:
        """Claude uses a separate `system` field — prepend the prefix there."""
        prefix = {"type": "text", "text": text}
        existing = inference_data.get("system_prompt", [])
        inference_data["system_prompt"] = [prefix] + existing
