"""
Azure OpenAI Responses-API function-calling handler for the robustness harness.

Some reasoning deployments (e.g. gpt-6-astra) refuse function tools combined with
`reasoning_effort` on /v1/chat/completions and require /v1/responses instead. This
handler subclasses BFCL's OpenAIResponsesHandler (`responses.create`, `function_call` /
`function_call_output` items) and points it at Azure AI Foundry:

  - The client is a plain OpenAI client aimed at the Foundry v1 endpoint, so requests
    go to .../openai/v1/responses; endpoint/key come from make_sync_client().
  - `model` is the Azure **deployment name** ("<deployment>-FC" minus the suffix).
  - `temperature` is never sent (reasoning models reject it); `reasoning.effort` is
    sent only when --reasoning-effort is given, else the deployment default applies.
  - `store=False` + `include=["reasoning.encrypted_content"]` so the reasoning items
    can be replayed statelessly on each retry turn.

Tool compilation, response parsing, chat-history bookkeeping and decode_ast are
inherited from BFCL unchanged.
"""

from __future__ import annotations

import json

from bfcl_eval.model_handler.api_inference.openai_response import (
    OpenAIResponsesHandler,
)
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.constants.enums import ModelStyle

from multilingual_bfcl.azure_client import make_sync_client


class AzureOpenAIResponsesFCHandler(OpenAIResponsesHandler):
    # Default cap on generated tokens per turn (includes reasoning tokens).
    DEFAULT_MAX_TOKENS = 8192
    # Long timeout so large tool schemas / long reasoning do not auto-error.
    REQUEST_TIMEOUT = 1200

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        *,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> None:
        # Skip OpenAIResponsesHandler.__init__: it builds an OpenAI() client from
        # OPENAI_* env vars. Initialise the grandparent directly and attach an Azure client.
        BaseHandler.__init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_RESPONSES
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.reasoning_effort = reasoning_effort
        self.client = make_sync_client()

    def _api_model_name(self) -> str:
        """The Azure deployment name (registry name minus the trailing '-FC')."""
        name = self.model_name
        return name[:-3] if name.endswith("-FC") else name

    def _query_FC(self, inference_data: dict):
        message: list = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        reasoning: dict = {"summary": "auto"}
        if self.reasoning_effort:
            reasoning["effort"] = self.reasoning_effort

        kwargs = {
            "input": message,
            "model": self._api_model_name(),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": reasoning,
            "max_output_tokens": self.max_tokens,
            "timeout": self.REQUEST_TIMEOUT,
        }
        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    # --- Format adapters used by the robustness harness (see run_robustness_eval) ---

    def extract_turn(self, api_response) -> tuple[list[str], list[dict]]:
        """Pull assistant text + function calls out of a Responses API response."""
        texts = [api_response.output_text] if api_response.output_text else []
        tool_calls = []
        for item in api_response.output:
            if item.type != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = item.arguments  # keep raw if unparseable
            tool_calls.append({"name": item.name, "arguments": arguments, "id": item.call_id})
        return texts, tool_calls

    def apply_language_prefix(self, inference_data: dict, text: str) -> None:
        """Responses input has no separate system field — prepend a developer message."""
        inference_data["message"].insert(0, {"role": "developer", "content": text})
