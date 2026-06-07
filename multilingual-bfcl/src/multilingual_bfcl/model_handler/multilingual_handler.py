"""
MultilingualHandler wraps any BFCL BaseHandler and injects a language-awareness
prefix into the system prompt before inference.

The wrapper is intentionally thin: it does not change how the model is called
or how the response is parsed. The only mutation is adding a one-sentence
language instruction so the model knows to expect a non-English query.
"""

from __future__ import annotations

from typing import Any

from bfcl_eval.model_handler.base_handler import BaseHandler

from multilingual_bfcl.localization.locale_config import Locale


# Template injected at the START of the system prompt.
# Kept short and factual — models handle language switching well when told explicitly.
_LANG_PREFIX_TEMPLATE = (
    "The user will write in {language}. "
    "Understand the request in {language} and respond with the correct function call "
    "exactly as specified in the tool definitions (which remain in English)."
)


class MultilingualHandler:
    """
    Wraps a BFCL handler to inject locale context into every inference call.

    Example:
        base = ClaudeHandler("claude-opus-4-8", temperature=0.0, ...)
        handler = MultilingualHandler(base, locale=get_locale("he"))
        result = handler.inference_single_turn_FC(test_entry, include_input_log=True)
    """

    def __init__(self, base_handler: BaseHandler, locale: Locale) -> None:
        self._base = base_handler
        self.locale = locale
        self._lang_prefix = _LANG_PREFIX_TEMPLATE.format(language=locale.name)

    # ------------------------------------------------------------------
    # Delegate attribute access to the wrapped handler so the rest of the
    # pipeline (eval_runner, result writers, etc.) can treat this object
    # as a BaseHandler without modification.
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    # ------------------------------------------------------------------
    # Inference entry points — patch system prompt then delegate
    # ------------------------------------------------------------------

    def inference_single_turn_FC(self, test_entry: dict, **kwargs) -> tuple:
        test_entry = self._patch_system_prompt(test_entry)
        return self._base.inference_single_turn_FC(test_entry, **kwargs)

    def inference_single_turn_prompting(self, test_entry: dict, **kwargs) -> tuple:
        test_entry = self._patch_system_prompt(test_entry)
        return self._base.inference_single_turn_prompting(test_entry, **kwargs)

    def inference_multi_turn_FC(self, test_entry: dict, **kwargs) -> tuple:
        test_entry = self._patch_system_prompt(test_entry)
        return self._base.inference_multi_turn_FC(test_entry, **kwargs)

    def inference_multi_turn_prompting(self, test_entry: dict, **kwargs) -> tuple:
        test_entry = self._patch_system_prompt(test_entry)
        return self._base.inference_multi_turn_prompting(test_entry, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _patch_system_prompt(self, test_entry: dict) -> dict:
        """
        Return a shallow copy of test_entry with the language prefix prepended
        to the initial_config system prompt (if present), or stored in a new key.
        """
        import copy
        entry = copy.deepcopy(test_entry)

        if "initial_config" in entry and "system_prompt" in entry["initial_config"]:
            original = entry["initial_config"]["system_prompt"]
            entry["initial_config"]["system_prompt"] = (
                self._lang_prefix + "\n\n" + original
            )
        else:
            # Single-turn entries don't always have initial_config;
            # store under a key the handler will prepend if it reads it.
            entry.setdefault("initial_config", {})
            entry["initial_config"]["language_hint"] = self._lang_prefix

        return entry
