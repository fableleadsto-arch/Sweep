"""Hugging Face connector — model + dataset cards from the public API."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

from ..models import IngestSource, RawItem, SourceKind
from .base import IngestConnector, http_json, parse_dt

_API_MODELS = "https://huggingface.co/api/models/{id}"
_API_DATASETS = "https://huggingface.co/api/datasets/{id}"


def _hf_id(source: IngestSource) -> Optional[str]:
    parsed = urlparse(source.url)
    if "huggingface.co" not in (parsed.netloc or ""):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in ("models", "datasets"):
        return f"{parts[-2]}/{parts[-1]}"
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return None


class HuggingFaceConnector(IngestConnector):
    kind = SourceKind.HUGGINGFACE

    async def fetch(
        self, source: IngestSource, client: Any
    ) -> list[RawItem]:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        model_id = _hf_id(source)
        if model_id is None:
            return await self._search(source, client, headers)
        items: list[RawItem] = []
        model = await http_json(client, _API_MODELS.format(id=model_id), headers=headers)
        if isinstance(model, dict) and model.get("id"):
            items.append(self._card(model, source, kind="model"))
        dataset = await http_json(client, _API_DATASETS.format(id=model_id), headers=headers)
        if isinstance(dataset, dict) and dataset.get("id") and dataset.get("id") != (model or {}).get("id"):
            items.append(self._card(dataset, source, kind="dataset"))
        return items

    async def _search(
        self, source: IngestSource, client: Any, headers: dict[str, str]
    ) -> list[RawItem]:
        query = source.config.get("query") or source.name or source.url
        items: list[RawItem] = []
        models = await http_json(client, "https://huggingface.co/api/models", params={"search": query, "limit": "10"}, headers=headers)
        if isinstance(models, list):
            for model in models[:10]:
                if isinstance(model, dict) and model.get("id"):
                    items.append(self._card(model, source, kind="model"))
        datasets = await http_json(client, "https://huggingface.co/api/datasets", params={"search": query, "limit": "5"}, headers=headers)
        if isinstance(datasets, list):
            for dataset in datasets[:5]:
                if isinstance(dataset, dict) and dataset.get("id"):
                    items.append(self._card(dataset, source, kind="dataset"))
        return items

    def _card(self, data: dict, source: IngestSource, *, kind: str) -> RawItem:
        hf_id = str(data.get("id") or source.name)
        tags = " ".join(data.get("tags") or [])
        card_text = str(data.get("cardData") or {}).get("text", "") if isinstance(data.get("cardData"), dict) else ""
        downloads = data.get("downloads") or 0
        likes = data.get("likes") or 0
        last_modified = data.get("lastModified")
        content = (
            f"{kind.title()} {hf_id}\n"
            f"Tags: {tags}\n"
            f"Downloads: {downloads} · Likes: {likes}\n\n{card_text}"
        ).strip()
        return RawItem(
            title=f"{kind.title()}: {hf_id}",
            url=f"https://huggingface.co/{hf_id}",
            content=content,
            summary=card_text[:400],
            published_at=parse_dt(last_modified),
            external_id=hf_id,
        )
