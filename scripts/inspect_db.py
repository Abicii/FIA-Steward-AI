"""
ChromaDB Inspector Script.

Inspects collections, total chunk counts, citations, and stored records.
Works both:
  1. Outside Docker (from your host terminal via port 8001 or 8000)
  2. Inside Docker (via `docker compose exec api python scripts/inspect_db.py`)

Usage:
    python scripts/inspect_db.py [--collection fia_regulations] [--limit 5] [--article B1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_COLLECTION = "fia_regulations"


def inspect_via_chromadb_client(
    host: str,
    port: int,
    collection_name: str,
    limit: int = 5,
    filter_article: str | None = None,
) -> bool:
    """Attempts to inspect using the Python chromadb library if installed."""
    try:
        import chromadb
    except ImportError:
        return False

    try:
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()
    except Exception:
        return False

    try:
        col = client.get_collection(collection_name)
    except Exception as exc:
        print(f"Collection '{collection_name}' not found: {exc}")
        return True

    count = col.count()
    print(f"\n=======================================================")
    print(f" ChromaDB Inspection (Connected to {host}:{port})")
    print(f" Collection Name : {collection_name}")
    print(f" Total Chunks    : {count}")
    print(f"=======================================================\n")

    if count == 0:
        print("Collection is currently empty.")
        return True

    where_filter = {"article_number": filter_article.upper()} if filter_article else None
    peek_data = col.get(limit=limit, where=where_filter, include=["documents", "metadatas"])

    print(f"Showing {len(peek_data.get('ids', []))} sample chunks (limit={limit}):\n")
    for idx, (cid, doc, meta) in enumerate(
        zip(peek_data.get("ids", []), peek_data.get("documents", []), peek_data.get("metadatas", [])),
        start=1,
    ):
        citation = meta.get("citation", "No Citation")
        art_num = meta.get("article_number", "N/A")
        sub_sec = meta.get("sub_section", "N/A")
        snippet = doc.replace("\n", " ").strip()[:200]

        print(f"[{idx}] ID: {cid}")
        print(f"    Citation    : {citation}")
        print(f"    Article     : {art_num} | Sub-section: {sub_sec}")
        print(f"    Preview     : {snippet}...")
        print("-" * 55)

    return True


def inspect_via_rest_api(
    host: str,
    port: int,
    collection_name: str,
    limit: int = 5,
) -> bool:
    """Fallback: Inspects via Chroma's native REST API without needing Python chromadb."""
    base_url = f"http://{host}:{port}/api/v1"
    try:
        # 1. Get collection list
        req = urllib.request.Request(f"{base_url}/collections")
        with urllib.request.urlopen(req, timeout=5) as resp:
            collections = json.loads(resp.read().decode("utf-8"))

        target = next((c for c in collections if c.get("name") == collection_name), None)
        if not target:
            names = [c.get("name") for c in collections]
            print(f"Collection '{collection_name}' not found. Available collections: {names}")
            return True

        col_id = target["id"]

        # 2. Get count
        count_req = urllib.request.Request(f"{base_url}/collections/{col_id}/count")
        with urllib.request.urlopen(count_req, timeout=5) as resp:
            count = json.loads(resp.read().decode("utf-8"))

        print(f"\n=======================================================")
        print(f" ChromaDB Inspection via REST API (Connected to {host}:{port})")
        print(f" Collection Name : {collection_name}")
        print(f" Collection UUID : {col_id}")
        print(f" Total Chunks    : {count}")
        print(f"=======================================================\n")

        if count == 0:
            print("Collection is empty.")
            return True

        # 3. Get peek items
        get_req = urllib.request.Request(
            f"{base_url}/collections/{col_id}/get",
            data=json.dumps({"limit": limit, "include": ["documents", "metadatas"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(get_req, timeout=5) as resp:
            peek_data = json.loads(resp.read().decode("utf-8"))

        for idx, (cid, doc, meta) in enumerate(
            zip(peek_data.get("ids", []), peek_data.get("documents", []), peek_data.get("metadatas", [])),
            start=1,
        ):
            citation = meta.get("citation", "No Citation")
            art_num = meta.get("article_number", "N/A")
            sub_sec = meta.get("sub_section", "N/A")
            snippet = (doc or "").replace("\n", " ").strip()[:200]

            print(f"[{idx}] ID: {cid}")
            print(f"    Citation    : {citation}")
            print(f"    Article     : {art_num} | Sub-section: {sub_sec}")
            print(f"    Preview     : {snippet}...")
            print("-" * 55)

        return True

    except urllib.error.URLError as exc:
        return False
    except Exception as exc:
        print(f"REST API error: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Inspect ChromaDB contents and citations")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection name")
    parser.add_argument("--limit", type=int, default=5, help="Number of sample chunks to display")
    parser.add_argument("--article", default=None, help="Optional filter by article number (e.g. 'B1', 'B5')")
    parser.add_argument("--port", type=int, default=None, help="Chroma port (default: 8001 on host, 8000 in docker)")
    parser.add_argument("--host", default=None, help="Chroma host (default: localhost or chroma)")
    args = parser.parse_args()

    # Determine connection targets:
    # 1. First try host='localhost', port=8001 (standard host-to-docker mapping)
    # 2. Then try host='chroma', port=8000 (inside docker network)
    # 3. Then try host='localhost', port=8000 (local or mapped 8000)
    candidates = []
    if args.host and args.port:
        candidates.append((args.host, args.port))
    else:
        env_host = os.getenv("CHROMA_HOST")
        if env_host:
            candidates.append((env_host, int(os.getenv("CHROMA_PORT", 8000))))
        candidates.extend([("localhost", 8001), ("chroma", 8000), ("localhost", 8000)])

    inspected = False
    for h, p in candidates:
        # Try native chromadb client first
        if inspect_via_chromadb_client(h, p, args.collection, limit=args.limit, filter_article=args.article):
            inspected = True
            break
        # Try REST API fallback
        if inspect_via_rest_api(h, p, args.collection, limit=args.limit):
            inspected = True
            break

    if not inspected:
        print("\n[ERROR] Could not connect to ChromaDB on any of the attempted endpoints:")
        for h, p in candidates:
            print(f"  * http://{h}:{p}")
        print("\nMake sure Docker is running with: docker compose up -d\n")


if __name__ == "__main__":
    main()
