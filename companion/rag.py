"""RAG knowledge retrieval — mirrors `src/RelAI/knowledge/rag.server.ts`.

Flow (identical to the TypeScript stack):
  1. Embed the query (Gemini → OpenAI fallback)
  2. Vector search via Supabase RPC `search_knowledge_chunks` (pgvector)
  3. Fall back to keyword scoring over recent chunks when vector search fails
  4. Format as a prompt-ready context block with source citations
"""

from __future__ import annotations

import re
from typing import Any, Optional

import httpx

from .config import BrainSettings
from .embeddings import embed_one
from .schemas import RagResult, SourceRef

RAG_TABLE_CHUNKS = "knowledge_chunks"
RAG_TABLE_DOCUMENTS = "knowledge_documents"
RAG_SEARCH_RPC = "search_knowledge_chunks"
CORPUS_SEARCH_RPC = "search_brain_corpus"
MAX_CHUNK_CHARS = 2000
TERM_MIN_LENGTH = 2


class RagService:
    """Retrieve and format knowledge-base context for AI prompts."""

    def __init__(self, settings: BrainSettings) -> None:
        self.settings = settings

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self.settings.supabase_key,
            "Authorization": f"Bearer {self.settings.supabase_key}",
            "Content-Type": "application/json",
        }

    async def retrieve(
        self,
        workspace_id: str,
        query: str,
        *,
        max_chunks: int = 6,
        min_score: float = 0.45,
        client: Optional[httpx.AsyncClient] = None,
    ) -> RagResult:
        if not self.settings.supabase_url or not self.settings.supabase_key:
            return RagResult(context="", found=False)

        embedding = await embed_one(query, self.settings)
        if embedding is not None:
            try:
                rows = await self._vector_search(
                    workspace_id, query, embedding, max_chunks, min_score, client
                )
                if rows:
                    return await self._format_vector(rows, client)
            except httpx.HTTPError:
                pass

        # Knowledge base unavailable (auth, network, rate limit, bad body) —
        # degrade to an empty context instead of failing the whole turn.
        return await self._text_search(workspace_id, query, max_chunks, client)

    # ── vector path ────────────────────────────────────────────────────

    async def _vector_search(
        self,
        workspace_id: str,
        query: str,
        embedding: list[float],
        max_chunks: int,
        min_score: float,
        client: Optional[httpx.AsyncClient],
    ) -> list[dict[str, Any]]:
        payload = {
            "p_workspace_id": workspace_id,
            "p_embedding": embedding,
            "p_match_threshold": min_score,
            "p_match_count": max_chunks,
        }
        url = f"{self.settings.supabase_url}/rest/v1/rpc/{RAG_SEARCH_RPC}"
        own_client = client is None
        client = client or httpx.AsyncClient(timeout=60.0)
        try:
            resp = await client.post(url, headers=self._headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        finally:
            if own_client:
                await client.aclose()

    async def _format_vector(
        self, rows: list[dict[str, Any]], client: Optional[httpx.AsyncClient]
    ) -> RagResult:
        doc_ids = sorted({str(r.get("document_id")) for r in rows if r.get("document_id")})
        doc_names: dict[str, tuple[str, str]] = {}
        if doc_ids:
            url = f"{self.settings.supabase_url}/rest/v1/{RAG_TABLE_DOCUMENTS}"
            own_client = client is None
            client = client or httpx.AsyncClient(timeout=60.0)
            try:
                resp = await client.get(
                    url,
                    headers=self._headers,
                    params={
                        "select": "id,name,file_type",
                        "id": f"in.({','.join(doc_ids)})",
                    },
                )
                if resp.is_success:
                    for doc in resp.json() or []:
                        doc_names[str(doc.get("id"))] = (
                            str(doc.get("name", "Unknown")),
                            str(doc.get("file_type", "text")),
                        )
            except httpx.HTTPError:
                pass
            finally:
                if own_client:
                    await client.aclose()

        sources: list[SourceRef] = []
        lines = ["Relevant knowledge base context:"]
        for row in rows:
            name, doc_type = doc_names.get(
                str(row.get("document_id")), ("Unknown", "text")
            )
            relevance = round(float(row.get("similarity") or 0.0) * 100)
            lines.append(f'--- From "{name}" ({doc_type}) [relevance: {relevance}%] ---')
            lines.append(str(row.get("content", ""))[:MAX_CHUNK_CHARS])
            sources.append(
                SourceRef(name=name, type=doc_type, chunk_index=int(row.get("chunk_index") or 0))
            )

        return RagResult(
            context="\n\n".join(lines),
            found=True,
            source_count=len(sources),
            sources=sources,
        )

    # ── keyword fallback path ─────────────────────────────────────────

    async def _text_search(
        self,
        workspace_id: str,
        query: str,
        limit: int,
        client: Optional[httpx.AsyncClient],
    ) -> RagResult:
        terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > TERM_MIN_LENGTH]
        if not terms:
            return RagResult(context="", found=False)

        url = f"{self.settings.supabase_url}/rest/v1/{RAG_TABLE_CHUNKS}"
        own_client = client is None
        client = client or httpx.AsyncClient(timeout=60.0)
        try:
            resp = await client.get(
                url,
                headers=self._headers,
                params={
                    "select": "*,documents:document_id(name,file_type)",
                    "workspace_id": f"eq.{workspace_id}",
                    "limit": "300",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            chunks = data if isinstance(data, list) else []
        except (httpx.HTTPError, ValueError):
            return RagResult(context="", found=False)
        finally:
            if own_client:
                await client.aclose()

        scored = []
        for chunk in chunks:
            content = str(chunk.get("content", "")).lower()
            score = sum(content.count(t) * 2 for t in terms)
            heading = str(chunk.get("heading", "")).lower()
            if heading:
                score += sum(5 for t in terms if t in heading)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        return self._format_keyword(scored[:limit])

    def _format_keyword(self, scored: list[tuple[dict[str, Any], int]]) -> RagResult:
        if not scored:
            return RagResult(context="", found=False)

        sources: list[SourceRef] = []
        lines = ["Relevant knowledge base context:"]
        for chunk, score in scored:
            docs = chunk.get("documents") or {}
            name = docs.get("name") if isinstance(docs, dict) else None
            doc_type = docs.get("file_type") if isinstance(docs, dict) else None
            name = str(name or "Unknown")
            doc_type = str(doc_type or "text")
            relevance = round(score * 5)
            lines.append(f'--- From "{name}" ({doc_type}) [relevance: {relevance}%] ---')
            lines.append(str(chunk.get("content", ""))[:MAX_CHUNK_CHARS])
            sources.append(
                SourceRef(name=name, type=doc_type, chunk_index=int(chunk.get("chunk_index") or 0))
            )

        return RagResult(
            context="\n\n".join(lines),
            found=True,
            source_count=len(sources),
            sources=sources,
        )

    # ── shared corpus path (brain_corpus RPC) ───────────────────────────

    async def corpus(
        self,
        query: str,
        *,
        max_chunks: int = 3,
        min_score: float = 0.45,
        client: Optional[httpx.AsyncClient] = None,
    ) -> RagResult:
        """Search the shared reference corpus (`search_brain_corpus` RPC).

        Unlike the per-workspace knowledge base, the corpus is global curated
        knowledge (framework docs, Wikipedia, …) that every workspace's brain
        can query. Degrades to an empty result, never raises.
        """
        if not self.settings.enable_corpus_search:
            return RagResult(context="", found=False)
        if not self.settings.supabase_url or not self.settings.supabase_key:
            return RagResult(context="", found=False)

        embedding = await embed_one(query, self.settings)
        if embedding is None:
            return RagResult(context="", found=False)

        url = f"{self.settings.supabase_url}/rest/v1/rpc/{CORPUS_SEARCH_RPC}"
        payload = {
            "p_embedding": embedding,
            "p_match_threshold": min_score,
            "p_match_count": max_chunks,
        }
        own_client = client is None
        client = client or httpx.AsyncClient(timeout=60.0)
        try:
            resp = await client.post(url, headers=self._headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            rows = data if isinstance(data, list) else []
        except httpx.HTTPError:
            return RagResult(context="", found=False)
        finally:
            if own_client:
                await client.aclose()

        if not rows:
            return RagResult(context="", found=False)

        sources: list[SourceRef] = []
        lines = ["Relevant shared corpus context:"]
        for row in rows:
            name = str(row.get("name") or "Unknown")
            source = str(row.get("source") or "corpus")
            source_url = str(row.get("source_url") or "")
            relevance = round(float(row.get("similarity") or 0.0) * 100)
            label = f'--- From "{name}" ({source}) [relevance: {relevance}%] ---'
            if source_url:
                label += f"\nSource: {source_url}"
            lines.append(label)
            lines.append(str(row.get("content", ""))[:MAX_CHUNK_CHARS])
            sources.append(
                SourceRef(name=name, type=source, chunk_index=int(row.get("chunk_index") or 0))
            )

        return RagResult(
            context="\n\n".join(lines),
            found=True,
            source_count=len(sources),
            sources=sources,
        )

    # ── continuous-knowledge ingestion path ─────────────────────────────
    # Mirrors `corpus` but reads the global ingestion tables (ingest_chunks +
    # search_ingest_chunks RPC). Pure retrieval — the write side lives in
    # companion/ingest/.

    INGEST_SEARCH_RPC = "search_ingest_chunks"
    INGEST_TABLE_CHUNKS = "ingest_chunks"
    INGEST_TABLE_DOCUMENTS = "ingest_documents"

    async def ingest_retrieve(
        self,
        query: str,
        *,
        max_chunks: int = 4,
        min_score: float = 0.45,
        client: Optional[httpx.AsyncClient] = None,
    ) -> RagResult:
        """Search continuously ingested global knowledge (best-effort)."""
        if not self.settings.supabase_url or not self.settings.supabase_key:
            return RagResult(context="", found=False)

        embedding = await embed_one(query, self.settings)
        if embedding is None:
            return RagResult(context="", found=False)

        url = f"{self.settings.supabase_url}/rest/v1/rpc/{self.INGEST_SEARCH_RPC}"
        payload = {
            "p_embedding": embedding,
            "p_match_threshold": min_score,
            "p_match_count": max_chunks,
        }
        own_client = client is None
        client = client or httpx.AsyncClient(timeout=60.0)
        try:
            resp = await client.post(url, headers=self._headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            rows = data if isinstance(data, list) else []
        except httpx.HTTPError:
            return RagResult(context="", found=False)
        finally:
            if own_client:
                await client.aclose()

        if not rows:
            return RagResult(context="", found=False)

        doc_ids = sorted({str(r.get("document_id")) for r in rows if r.get("document_id")})
        doc_names: dict[str, tuple[str, str]] = {}
        if doc_ids:
            url = f"{self.settings.supabase_url}/rest/v1/{self.INGEST_TABLE_DOCUMENTS}"
            own_client = client is None
            client = client or httpx.AsyncClient(timeout=60.0)
            try:
                resp = await client.get(
                    url,
                    headers=self._headers,
                    params={"select": "id,name,source_kind", "id": f"in.({','.join(doc_ids)})"},
                )
                if resp.is_success:
                    for doc in resp.json() or []:
                        doc_names[str(doc.get("id"))] = (
                            str(doc.get("name", "Unknown")),
                            str(doc.get("source_kind", "ingest")),
                        )
            except httpx.HTTPError:
                pass
            finally:
                if own_client:
                    await client.aclose()

        sources: list[SourceRef] = []
        lines = ["Relevant continuous-knowledge context:"]
        for row in rows:
            name, kind = doc_names.get(str(row.get("document_id")), ("Unknown", "ingest"))
            relevance = round(float(row.get("similarity") or 0.0) * 100)
            lines.append(f'--- From "{name}" ({kind}) [relevance: {relevance}%] ---')
            lines.append(str(row.get("content", ""))[:MAX_CHUNK_CHARS])
            sources.append(
                SourceRef(name=name, type=kind, chunk_index=int(row.get("chunk_index") or 0))
            )

        return RagResult(
            context="\n\n".join(lines),
            found=True,
            source_count=len(sources),
            sources=sources,
        )
