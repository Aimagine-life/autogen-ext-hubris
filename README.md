# autogen-ext-hubris

An [AutoGen](https://microsoft.github.io/autogen/) extension client preconfigured for [Hubris](https://hubris.pw) — a ruble-billed, OpenAI-compatible LLM gateway giving access to 400+ models (OpenAI, Anthropic, Google, and more) behind a single API key.

## Installation

```bash
pip install autogen-ext-hubris
```

## Usage

```python
from autogen_ext_hubris import HubrisChatCompletionClient

client = HubrisChatCompletionClient(
    model="claude-sonnet-5",
    api_key="sk-gw-...",  # create one at https://hubris.pw/keys
)
```

`HubrisChatCompletionClient` is a thin subclass of `OpenAIChatCompletionClient` — it presets `base_url` to Hubris's endpoint and resolves `model_info` for known catalog models. For a model not in the built-in registry, pass `model_info` explicitly:

```python
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext_hubris import HubrisChatCompletionClient

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
```

See the full model catalog at [hubris.pw/models](https://hubris.pw/models) and docs at [hubris.pw/docs/integrations/autogen](https://hubris.pw/docs/integrations/autogen).
