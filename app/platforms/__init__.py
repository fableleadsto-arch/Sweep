"""Platform adapters — Reddit, GitHub, YouTube, X, LinkedIn, Instagram."""

from .registry import get_adapter_for_url, get_adapter_by_platform, list_adapters

__all__ = [
    "get_adapter_for_url",
    "get_adapter_by_platform",
    "list_adapters",
]
