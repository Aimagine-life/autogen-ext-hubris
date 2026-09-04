from importlib import metadata

from autogen_ext_hubris.client import HubrisChatCompletionClient

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Package is not installed
    __version__ = ""
del metadata

__all__ = [
    "HubrisChatCompletionClient",
    "__version__",
]
