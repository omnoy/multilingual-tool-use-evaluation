"""
Azure OpenAI function-calling handler for the robustness harness.

Subclasses BFCL's OpenAICompletionsHandler (chat.completions + OpenAI-style
`tools`/`tool_calls`) but points the client at Azure AI Foundry and adapts request
parameters:

  - The client is a plain OpenAI client aimed at the Foundry v1 endpoint (no
    api-version); endpoint/key come from the environment via make_sync_client().
  - `model` is the Azure **deployment name** (the harness registers the handler under
    "<deployment>-FC"; we strip that suffix for the API call).
  - Reasoning deployments omit `temperature` and send `max_completion_tokens`; standard
    deployments send `temperature` + `max_tokens`.
  - `store` (an OpenAI-only convenience) is not sent — Azure does not accept it.

Everything else (tool compilation, response parsing, chat-history bookkeeping,
decode_ast/decode_execute) is inherited unchanged, so the harness drives it exactly
as it drove ClaudeHandler.
"""

from __future__ import annotations

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from bfcl_eval.model_handler.base_handler import BaseHandler

from multilingual_bfcl.azure_client import ModelType, make_sync_client


class AzureOpenAIFCHandler(OpenAICompletionsHandler):
    # Default cap on generated tokens per turn (function-calling replies are short).
    DEFAULT_MAX_TOKENS = 8192
    # Long timeout so large tool schemas / contexts do not auto-error.
    REQUEST_TIMEOUT = 1200

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        *,
        model_type: ModelType = ModelType.STANDARD,
        max_tokens: int | None = None,
        **kwargs,
    ) -> None:
        # Skip OpenAICompletionsHandler.__init__: it eagerly builds an OpenAI() client
        # (which needs OPENAI_API_KEY). Initialise the grandparent directly, then attach
        # an Azure client and the OpenAI completions model style the parent's methods use.
        BaseHandler.__init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.model_type = model_type
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.client = make_sync_client()

    def _api_model_name(self) -> str:
        """The Azure deployment name (registry name minus the trailing '-FC')."""
        name = self.model_name
        return name[:-3] if name.endswith("-FC") else name

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "messages": message,
            "model": self._api_model_name(),
            "timeout": self.REQUEST_TIMEOUT,
        }
        if self.model_type == ModelType.REASONING:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["temperature"] = self.temperature
            kwargs["max_tokens"] = self.max_tokens

        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    # --- Format adapters used by the robustness harness (see run_robustness_eval) ---

    def extract_turn(self, api_response) -> tuple[list[str], list[dict]]:
        """Pull assistant text + tool calls out of an OpenAI chat.completions response."""
        import json

        message = api_response.choices[0].message
        texts = [message.content] if message.content else []
        tool_calls = []
        for tc in (message.tool_calls or []):
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = tc.function.arguments  # keep raw if unparseable
            tool_calls.append({"name": tc.function.name, "arguments": arguments, "id": tc.id})
        return texts, tool_calls

    def apply_language_prefix(self, inference_data: dict, text: str) -> None:
        """OpenAI FC has no separate system field — prepend a system-role message."""
        inference_data["message"].insert(0, {"role": "system", "content": text})
