"""Platform registry — auto-detects URL platform and routes to the correct adapter."""

from __future__ import annotations

from typing import Optional

from .base import PlatformAdapter
from .reddit import RedditAdapter
from .github import GitHubAdapter
from .youtube import YouTubeAdapter
from .x import XAdapter
from .linkedin import LinkedInAdapter
from .instagram import InstagramAdapter

# All adapters in priority order (more specific first)
_ADAPTERS: list[PlatformAdapter] = [
    YouTubeAdapter(),
    XAdapter(),
    InstagramAdapter(),
    LinkedInAdapter(),
    RedditAdapter(),
    GitHubAdapter(),
]

# Build hostname → adapter map
def _build_host_map() -> dict[str, PlatformAdapter]:
    mapping: dict[str, PlatformAdapter] = {}
    for adapter in _ADAPTERS:
        if isinstance(adapter, YouTubeAdapter):
            for h in ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"):
                mapping[h] = adapter
        elif isinstance(adapter, XAdapter):
            for h in ("x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"):
                mapping[h] = adapter
        elif isinstance(adapter, InstagramAdapter):
            for h in ("instagram.com", "www.instagram.com", "m.instagram.com"):
                mapping[h] = adapter
        elif isinstance(adapter, LinkedInAdapter):
            for h in ("linkedin.com", "www.linkedin.com"):
                mapping[h] = adapter
        elif isinstance(adapter, RedditAdapter):
            for h in ("reddit.com", "www.reddit.com", "old.reddit.com", "redd.it"):
                mapping[h] = adapter
        elif isinstance(adapter, GitHubAdapter):
            for h in ("github.com", "www.github.com"):
                mapping[h] = adapter
    return mapping


_HOST_MAP = _build_host_map()


def get_adapter_for_url(url: str) -> Optional[PlatformAdapter]:
    """Return the platform adapter that can handle this URL, or None."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    # Strip www. prefix
    if hostname.startswith("www."):
        hostname = hostname[4:]
    adapter = _HOST_MAP.get(hostname)
    if adapter and adapter.can_handle(url):
        return adapter
    return None


def get_adapter_by_platform(platform_name: str) -> Optional[PlatformAdapter]:
    """Return the adapter for a given platform name (e.g. 'youtube', 'x')."""
    for adapter in _ADAPTERS:
        if adapter.platform.value == platform_name:
            return adapter
    return None


def list_adapters() -> list[dict]:
    """Return info about all registered adapters."""
    return [
        {
            "platform": a.platform.value,
            "class": type(a).__name__,
        }
        for a in _ADAPTERS
    ]
