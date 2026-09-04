"""Unit tests for HubrisChatCompletionClient."""

import pytest
from autogen_core.models import ModelFamily, ModelInfo

from autogen_ext_hubris import HubrisChatCompletionClient
from autogen_ext_hubris.client import BASE_URL


def test_known_model_resolves_info() -> None:
    client = HubrisChatCompletionClient(model="claude-sonnet-5", api_key="test-key")
    assert client.model_info["family"] == ModelFamily.CLAUDE_4_SONNET
    assert client.model_info["function_calling"] is True


def test_defaults_to_hubris_base_url() -> None:
    client = HubrisChatCompletionClient(model="claude-sonnet-5", api_key="test-key")
    assert str(client._client.base_url).rstrip("/") == BASE_URL  # noqa: SLF001


def test_explicit_base_url_is_respected() -> None:
    client = HubrisChatCompletionClient(
        model="claude-sonnet-5",
        api_key="test-key",
        base_url="https://example.com/v1",
    )
    assert str(client._client.base_url).rstrip("/") == "https://example.com/v1"  # noqa: SLF001


def test_unknown_model_without_model_info_raises() -> None:
    with pytest.raises(ValueError, match="No built-in model_info"):
        HubrisChatCompletionClient(model="not-in-registry", api_key="test-key")


def test_unknown_model_with_explicit_model_info_works() -> None:
    client = HubrisChatCompletionClient(
        model="not-in-registry",
        api_key="test-key",
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=True,
            family=ModelFamily.UNKNOWN,
        ),
    )
    assert client.model_info["family"] == ModelFamily.UNKNOWN


def test_missing_model_raises() -> None:
    with pytest.raises(ValueError, match="model is required"):
        HubrisChatCompletionClient(api_key="test-key")
