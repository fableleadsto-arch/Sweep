"""GitHub connector — repo metadata, README, releases and CHANGELOG.

Uses the public REST API (rate-limited) with an optional ``GITHUB_TOKEN``
rate-limit upgrade. Everything is read-only. RawItem URLs point at the real
GitHub pages so every ingested fact stays traceable.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..models import IngestSource, RawItem, SourceKind
from .base import IngestConnector, http_json, http_text, parse_dt

_REPO_RE = re.compile(r"(?:github\.com/|^)([^/\s]+)/([^/\s#?]+)", re.IGNORECASE)


def _parse_repo(source: IngestSource) -> Optional[tuple[str, str]]:
    match = _REPO_RE.search(source.url)
    if not match:
        return None
    owner, repo = match.group(1), match.group(2)
    repo = re.sub(r"\.git$", "", repo)
    return owner, repo


class GitHubConnector(IngestConnector):
    kind = SourceKind.GITHUB

    def can_handle(self, source: IngestSource) -> bool:
        return _parse_repo(source) is not None

    @property
    def _api_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        return headers

    async def fetch(
        self, source: IngestSource, client: Any
    ) -> list[RawItem]:
        repo = _parse_repo(source)
        if repo is None:
            return []
        owner, name = repo
        items: list[RawItem] = []
        items.extend(await self._repo_card(owner, name, source, client))
        items.extend(await self._readme(owner, name, source, client))
        items.extend(await self._releases(owner, name, source, client))
        items.extend(await self._changelog(owner, name, source, client))
        return items

    async def _repo_card(
        self, owner: str, name: str, source: IngestSource, client: Any
    ) -> list[RawItem]:
        data = await http_json(
            client, f"https://api.github.com/repos/{owner}/{name}", headers=self._api_headers
        )
        if not isinstance(data, dict) or data.get("id") is None:
            return []
        description = str(data.get("description") or "")
        topics = " ".join(data.get("topics") or [])
        content = (
            f"{data.get('full_name', '')} — {description}\n"
            f"Topics: {topics}\n"
            f"Language: {data.get('language', '')}\n"
            f"Stars: {data.get('stargazers_count', 0)} · Forks: {data.get('forks_count', 0)}\n"
            f"Default branch: {data.get('default_branch', '')}"
        )
        published = parse_dt(data.get("pushed_at"))
        updated = parse_dt(data.get("updated_at"))
        return [
            RawItem(
                title=data.get("full_name", source.name),
                url=data.get("html_url") or source.url,
                content=content,
                summary=description,
                published_at=updated or published,
                external_id=f"{owner}/{name}",
                last_modified=data.get("updated_at"),
            )
        ]

    async def _readme(
        self, owner: str, name: str, source: IngestSource, client: Any
    ) -> list[RawItem]:
        for branch in ("HEAD",):
            url = f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/README.md"
            text = await http_text(client, url, headers={"User-Agent": self.user_agent})
            if text:
                return [
                    RawItem(
                        title=f"{owner}/{name} — README",
                        url=f"https://github.com/{owner}/{name}#readme",
                        content=text,
                        summary=f"README of {owner}/{name}",
                        published_at=parse_dt(source.last_modified),
                        external_id=f"{owner}/{name}:readme",
                    )
                ]
        return []

    async def _releases(
        self, owner: str, name: str, source: IngestSource, client: Any
    ) -> list[RawItem]:
        releases = await http_json(
            client,
            f"https://api.github.com/repos/{owner}/{name}/releases",
            params={"per_page": "10"},
            headers=self._api_headers,
        )
        if not isinstance(releases, list):
            return []
        items: list[RawItem] = []
        for release in releases:
            if not isinstance(release, dict):
                continue
            body = str(release.get("body") or "").strip()
            if not body:
                body = f"Release {release.get('tag_name', '')}"
            published = parse_dt(release.get("published_at"))
            items.append(
                RawItem(
                    title=f"{owner}/{name} — {release.get('name') or release.get('tag_name')}",
                    url=release.get("html_url") or source.url,
                    content=body,
                    summary=release.get("tag_name", ""),
                    published_at=published,
                    external_id=f"{owner}/{name}:release:{release.get('tag_name')}",
                )
            )
        return items

    async def _changelog(
        self, owner: str, name: str, source: IngestSource, client: Any
    ) -> list[RawItem]:
        url = f"https://raw.githubusercontent.com/{owner}/{name}/HEAD/CHANGELOG.md"
        text = await http_text(client, url, headers={"User-Agent": self.user_agent})
        if not text:
            return []
        return [
            RawItem(
                title=f"{owner}/{name} — CHANGELOG",
                url=f"https://github.com/{owner}/{name}/blob/HEAD/CHANGELOG.md",
                content=text,
                summary="Changelog of " + f"{owner}/{name}",
                external_id=f"{owner}/{name}:changelog",
            )
        ]
