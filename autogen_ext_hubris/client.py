"""Hubris chat completion client for AutoGen."""

from __future__ import annotations

from typing import Any

from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

BASE_URL = "https://api.hubris.pw/v1"

# A representative subset of the Hubris catalog (https://hubris.pw/models).
# `family` maps to the closest known AutoGen ModelFamily bucket (used for
# prompt-formatting hints, not an exact version match) — pass `model_info`
# explicitly for any model not listed here.
_MODEL_INFO: dict[str, ModelInfo] = {
    "claude-sonnet-5": ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        structured_output=True,
        family=ModelFamily.CLAUDE_4_SONNET,
    ),
    "claude-opus-5": ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        structured_output=True,
        family=ModelFamily.CLAUDE_4_OPUS,
    ),
    "gpt-5.6": ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        structured_output=True,
        family=ModelFamily.GPT_5,
    ),
    "gemini-3.5-pro": ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        structured_output=True,
        family=ModelFamily.GEMINI_2_5_PRO,
    ),
}


class HubrisChatCompletionClient(OpenAIChatCompletionClient):
    """An :class:`OpenAIChatCompletionClient` preconfigured for `Hubris <https://hubris.pw>`__.

    Hubris is a ruble-billed, OpenAI-compatible LLM gateway giving access to
    400+ models (OpenAI, Anthropic, Google, and more) behind a single API key.

    Example:
        .. code-block:: python

            from autogen_ext_hubris import HubrisChatCompletionClient

            client = HubrisChatCompletionClient(
                model="claude-sonnet-5",
                api_key="sk-gw-...",
            )

    For a model not in the built-in registry, pass ``model_info`` explicitly —
    see the full catalog at https://hubris.pw/models::

        from autogen_core.models import ModelFamily, ModelInfo

        client = HubrisChatCompletionClient(
            model="some-new-model",
            api_key="sk-gw-...",
            model_info=ModelInfo(
                vision=False,
                function_calling=True,
                json_output=True,
                structured_output=True,
                family=ModelFamily.UNKNOWN,
            ),
        )
    """

    def __init__(self, **kwargs: Any) -> None:
        if "model" not in kwargs:
            msg = "model is required"
            raise ValueError(msg)
        model = kwargs["model"]

        if "model_info" not in kwargs:
            info = _MODEL_INFO.get(model)
            if info is None:
                msg = (
                    f"No built-in model_info for {model!r}. Pass model_info "
                    "explicitly — see the HubrisChatCompletionClient docstring "
                    "or the catalog at https://hubris.pw/models."
                )
                raise ValueError(msg)
            kwargs["model_info"] = info

        kwargs.setdefault("base_url", BASE_URL)
        super().__init__(**kwargs)
