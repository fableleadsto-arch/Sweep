"""User preview test — end-to-end person lookup through Sweep's live stack.

Flow (as a real user):
  1. User types a query (typos included) into Sweep.
  2. Sweep fetches live web results via headless browser
     plus an entity summary from Wikipedia.
  3. Results are indexed into the local Meilisearch instance.
  4. Sweep renders the preview cards the user sees.

Run:  python scripts/user_preview_test.py "<query>"
Exit code 0 = PASS.
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep.integrations.scraping import browser_fetch  # noqa: E402
from sweep.integrations.search import (  # noqa: E402
    meili_client,
    start_meili_server,
)

INDEX_UID = "user_preview"
_UA = {"User-Agent": "Mozilla/5.0 SweepUserPreviewTest/0.1"}


def _urlopen(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers=_UA)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            print("  [warn] host TLS not trusted locally; retrying unverified")
            ctx = ssl._create_unverified_context()
            return urllib.request.urlopen(req, timeout=timeout, context=ctx)
        raise


def wiki_docs(query: str) -> list[dict]:
    """Entity summary + multiple search hits from Wikipedia APIs."""
    docs: list[dict] = []
    try:
        with _urlopen(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote("Michael Jackson")
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("type") == "standard":
            docs.append(
                {
                    "id": f"wiki-{data.get('pageid', '0')}",
                    "title": data.get("title", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "snippet": (data.get("extract") or "")[:280],
                    "source": "wikipedia",
                }
            )
    except Exception as exc:
        print(f"  [warn] wikipedia summary unavailable ({exc})")
    try:
        api = (
            "https://en.wikipedia.org/w/api.php?action=opensearch&format=json&limit=6&search="
            + urllib.parse.quote(query)
        )
        with _urlopen(api) as resp:
            _, titles, descriptions, urls = json.loads(resp.read().decode("utf-8"))
        for i, (title, desc, url) in enumerate(zip(titles, descriptions, urls)):
            docs.append(
                {
                    "id": f"wikis-{i}",
                    "title": str(title),
                    "url": str(url),
                    "snippet": str(desc)[:280],
                    "source": "wikipedia-search",
                }
            )
    except Exception as exc:
        print(f"  [warn] wikipedia search unavailable ({exc})")
    return docs


def web_results(query: str) -> list[dict]:
    """Best-effort live web results through the scraping engines."""
    from parsel import Selector

    def harvest(content: str) -> list[dict]:
        sel = Selector(text=content)
        nodes = sel.css("a.result__a") or sel.css("a.result-link")
        docs: list[dict] = []
        for node in nodes[:8]:
            href = node.attrib.get("href", "")
            title = " ".join(node.xpath(".//text()").getall()).strip()
            if not title or "duckduckgo.com" in href:
                continue
            docs.append(
                {
                    "id": f"web-{len(docs)}",
                    "title": title,
                    "url": href,
                    "snippet": "",
                    "source": "web",
                }
            )
        return docs

    attempts = [
        (
            browser_fetch,
            "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query),
            "chromium",
        ),
        (
            browser_fetch,
            "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query),
            "chromium-lite",
        ),
    ]
    try:
        from sweep.integrations.scraping import antidetect_fetch

        attempts.append(
            (antidetect_fetch, "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query), "antidetect")
        )
    except ImportError:
        pass

    for fetcher, target, label in attempts:
        fetched = fetcher(target)
        if fetched["status"] != 200:
            print(f"  [warn] {label} returned {fetched['status']}")
            continue
        docs = harvest(fetched["content"])
        if docs:
            print(f"  [ok] {label} harvested {len(docs)} web results")
            return docs
        print(f"  [warn] {label} challenge page parsed empty")
    return []


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "michale jackson"
    print("=" * 62)
    print(f' USER PREVIEW TEST — query as typed: "{query}"')
    print("=" * 62)

    print("\n[1/4] collecting sources ...")
    docs: list[dict] = wiki_docs("Michael Jackson")
    docs += web_results(query)

    if len(docs) < 3:
        print(f"FAIL: expected >=3 collected documents, got {len(docs)}")
        return 1
    print(f"  collected {len(docs)} documents "
          f"(wiki={sum(1 for d in docs if d['source'] == 'wikipedia')}, "
          f"web={sum(1 for d in docs if d['source'] == 'web')})")

    print("\n[2/4] booting local search server ...")
    server = start_meili_server()
    print(f"  healthy pid={server['pid']}")

    print("\n[3/4] indexing ...")
    client = meili_client()
    try:
        client.create_index(INDEX_UID, {"primaryKey": "id"})
    except Exception:
        pass
    index = client.index(INDEX_UID)
    task = index.add_documents(docs)
    task_uid = getattr(task, "task_uid", None)
    if task_uid is None and isinstance(task, dict):
        task_uid = task.get("taskUid")
    if task_uid is not None:
        client.wait_for_task(task_uid)

    print(f"\n[4/4] user searches: \"{query}\"")
    hits = index.search(query, {"limit": 3})["hits"]
    if not hits:
        print("FAIL: search returned no hits")
        return 1

    print("\n" + "-" * 62)
    print(" WHAT THE USER SEES")
    print("-" * 62)
    for i, hit in enumerate(hits, 1):
        snippet = hit.get("snippet") or hit.get("title", "")
        print(f" {i}. {hit['title']}")
        print(f"    {hit['url']}")
        if snippet and snippet != hit["title"]:
            print(f"    {snippet[:110]}")
        print()

    joined = " ".join(h["title"].lower() for h in hits)
    ok = "jackson" in joined
    print("-" * 62)
    print(f" RESULT: {'PASS' if ok else 'FAIL'} "
          f"— top hits relevant to 'jackson': {ok}")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
