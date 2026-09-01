"""Base platform adapter protocol — defines the interface all adapters implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..core.types import PageData, SearchResult, SearchAccessMode, SurfPlatform


class PlatformAdapter(ABC):
    """Abstract base for platform-specific search and extraction."""

    @property
    @abstractmethod
    def platform(self) -> SurfPlatform:
        ...

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Whether this adapter can handle a URL."""
        ...

    @abstractmethod
    async def search(
        self, query: str, *, limit: int = 5, subreddit: Optional[str] = None,
    ) -> tuple[list[SearchResult], Optional[str], SearchAccessMode]:
        """Native/platform search. Returns (results, note, access_mode)."""
        ...

    @abstractmethod
    async def extract_page(self, url: str) -> Optional[PageData]:
        """Extract a platform page (post, repo, video, etc.)."""
        ...
