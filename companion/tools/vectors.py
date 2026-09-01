"""Vector-database capabilities — Qdrant and Milvus.

Both connectors use the official client libraries against the configured
server. Qdrant settings (URL + key) already exist in Relay's config; Milvus
reads MILVUS_URI / MILVUS_TOKEN. Capabilities list collections, and search
when the caller supplies a vector — the honest, real-client path.
"""

from __future__ import annotations

from typing import Any

from .common import CapabilityUnavailable, load


# ── Qdrant ───────────────────────────────────────────────────────────────


def run_qdrant(payload: dict[str, Any]) -> dict[str, Any]:
    """Qdrant — collections, info and vector search via qdrant-client."""
    params = payload.get("params") or {}
    settings = payload.get("_settings")
    mode = str(params.get("mode") or "collections").lower()

    qdrant_client = load("qdrant_client")
    url = str(params.get("url") or (settings and getattr(settings, "qdrant_api_url", "")) or "http://localhost:6333")
    api_key = str(params.get("api_key") or (settings and getattr(settings, "qdrant_api_key", "")) or "")
    try:
        if api_key:
            client = qdrant_client.QdrantClient(url=url, api_key=api_key)
        else:
            client = qdrant_client.QdrantClient(url=url)
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"Qdrant client failed to connect: {exc}") from exc

    try:
        if mode == "collections":
            collections = client.get_collections().collections
            return {
                "result": {
                    "engine": "qdrant",
                    "url": url,
                    "collections": [{"name": c.name, "points_count": _points_count(client, c.name)} for c in collections[:25]],
                    "count": len(collections),
                },
                "summary": f"Qdrant at {url} has {len(collections)} collection(s).",
                "libraries_used": ["qdrant_client"],
            }

        if mode == "search":
            collection = str(params.get("collection") or settings and getattr(settings, "qdrant_collection", "") or "")
            vector = params.get("vector")
            if not collection:
                raise ValueError("Qdrant search needs `params.collection`.")
            if not isinstance(vector, list) or not vector:
                raise ValueError("Qdrant search needs `params.vector` (a list of floats).")
            limit = int(params.get("limit") or 5)
            hits = client.search(collection_name=collection, query_vector=[float(v) for v in vector[:2048]], limit=limit)
            return {
                "result": {
                    "engine": "qdrant",
                    "collection": collection,
                    "hits": [
                        {"id": str(h.id), "score": round(float(h.score), 4), "payload": h.payload or {}}
                        for h in hits
                    ],
                    "count": len(hits),
                },
                "summary": f"Qdrant searched {collection} and returned {len(hits)} hit(s).",
                "libraries_used": ["qdrant_client"],
            }

        if mode == "info":
            info = client.get_collection(str(params.get("collection") or settings and getattr(settings, "qdrant_collection", "") or ""))
            return {
                "result": {
                    "engine": "qdrant",
                    "collection": info.name,
                    "points_count": getattr(info, "points_count", None),
                    "vectors_config": str(getattr(info, "config", {}).get("params", {}).get("vectors", {})) if getattr(info, "config", None) else None,
                },
                "summary": f"Qdrant collection '{info.name}' inspected.",
                "libraries_used": ["qdrant_client"],
            }
    except CapabilityUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"Qdrant operation failed: {exc}") from exc

    raise ValueError("qdrant mode must be 'collections', 'search' or 'info'.")


def _points_count(client: Any, collection: str) -> int:
    try:
        info = client.get_collection(collection)
        return int(getattr(info, "points_count", 0) or 0)
    except Exception:  # noqa: BLE001 - best-effort count
        return 0


# ── Milvus ───────────────────────────────────────────────────────────────


def run_milvus(payload: dict[str, Any]) -> dict[str, Any]:
    """Milvus — collections and search via pymilvus."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "collections").lower()

    pymilvus = load("pymilvus")
    uri = str(params.get("uri") or "http://localhost:19530")
    token = str(params.get("token") or "")

    try:
        from pymilvus import MilvusClient

        client_kwargs: dict[str, Any] = {"uri": uri}
        if token:
            client_kwargs["token"] = token
        client = MilvusClient(**client_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"pymilvus failed to connect to {uri}: {exc}") from exc

    try:
        if mode == "collections":
            names = client.list_collections()
            return {
                "result": {"engine": "milvus", "uri": uri, "collections": names[:25], "count": len(names)},
                "summary": f"Milvus at {uri} has {len(names)} collection(s).",
                "libraries_used": ["pymilvus"],
            }

        if mode == "search":
            collection = str(params.get("collection") or "")
            vector = params.get("vector")
            if not collection:
                raise ValueError("Milvus search needs `params.collection`.")
            if not isinstance(vector, list) or not vector:
                raise ValueError("Milvus search needs `params.vector` (a list of floats).")
            limit = int(params.get("limit") or 5)
            hits = client.search(
                collection_name=collection,
                data=[[float(v) for v in vector[:2048]]],
                limit=limit,
                output_fields=["*"],
            )
            rows = hits[0] if hits else []
            return {
                "result": {
                    "engine": "milvus",
                    "collection": collection,
                    "hits": [
                        {"id": str(h.get("id")), "distance": round(float(h.get("distance", 0)), 4), "entity": h.get("entity") or {}}
                        for h in rows
                    ],
                    "count": len(rows),
                },
                "summary": f"Milvus searched {collection} and returned {len(rows)} hit(s).",
                "libraries_used": ["pymilvus"],
            }
    except CapabilityUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"Milvus operation failed: {exc}") from exc

    raise ValueError("milvus mode must be 'collections' or 'search'.")
