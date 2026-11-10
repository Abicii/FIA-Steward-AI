"""
Script to clear/reset the ChromaDB collection.

Can be run:
  1. Outside Docker via HTTP: python scripts/clear_db.py
  2. Inside Docker: docker compose exec api python scripts/clear_db.py
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

DEFAULT_COLLECTION = "fia_regulations"


def clear_via_http(api_url: str = "http://localhost:8000", collection_name: str = DEFAULT_COLLECTION) -> bool:
    """Calls DELETE /clear on the FastAPI server."""
    url = f"{api_url.rstrip('/')}/clear?collection_name={collection_name}"
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[OK] {data.get('message', 'Cleared successfully.')}")
            return True
    except urllib.error.URLError as exc:
        print(f"[ERROR] Could not connect to API at {api_url}: {exc}")
        return False
    except Exception as exc:
        print(f"[ERROR] Failed to clear collection: {exc}")
        return False


def clear_via_chromadb_client(host: str = "localhost", port: int = 8001, collection_name: str = DEFAULT_COLLECTION) -> bool:
    """Clears directly using chromadb Python library if available."""
    try:
        import chromadb

        client = chromadb.HttpClient(host=host, port=port)
        client.delete_collection(collection_name)
        print(f"[OK] Collection '{collection_name}' deleted directly via Chroma client ({host}:{port}).")
        return True
    except Exception as exc:
        return False


def main():
    parser = argparse.ArgumentParser(description="Clear ChromaDB collection")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection name to clear")
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPI base URL")
    args = parser.parse_args()

    print(f"Clearing collection '{args.collection}'...")
    # Try HTTP first
    if clear_via_http(api_url=args.api_url, collection_name=args.collection):
        return

    # Fallback to direct Chroma ports
    for h, p in [("localhost", 8001), ("chroma", 8000)]:
        if clear_via_chromadb_client(host=h, port=p, collection_name=args.collection):
            return

    print("Could not clear database. Make sure your Docker container is running (docker compose up).")


if __name__ == "__main__":
    main()
