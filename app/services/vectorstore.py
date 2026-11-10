"""
Vector store service managing FastEmbed embeddings and ChromaDB storage.

Supports both remote Chroma server (Docker Compose) and local persistent client fallback,
with batched ingestion and similarity search returning rich citation metadata.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError:
    chromadb = None

from app.services.chunking import DocumentChunk

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_COLLECTION = "fia_regulations"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedWrapper:
    """Lightweight wrapper for FastEmbed embedding generation."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding

                logger.info(f"Loading FastEmbed model: {self.model_name}")
                self._embedder = TextEmbedding(model_name=self.model_name)
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed package is not installed. Please install fastembed or run via Docker."
                ) from exc
        return self._embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embedder = self._get_embedder()
        embeddings = list(embedder.embed(texts))
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


_embedding_instance: FastEmbedWrapper | None = None


def get_embedding_function() -> FastEmbedWrapper:
    """Singleton getter for the FastEmbed embedding instance."""
    global _embedding_instance
    if _embedding_instance is None:
        model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
        _embedding_instance = FastEmbedWrapper(model_name=model_name)
    return _embedding_instance


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Returns a ChromaDB client.
    Attempts connection to the configured HTTP host (Docker/Remote).
    Falls back gracefully to a persistent local client if the remote server is unavailable.
    """
    if chromadb is None:
        raise RuntimeError(
            "The 'chromadb' package is not installed in this Python environment.\n"
            "If using Docker, run this command inside Docker with:\n"
            "  docker compose exec api python scripts/evaluate_retrieval.py\n"
            "Or run the evaluation via the HTTP API using:\n"
            "  python scripts/evaluate_retrieval.py --api-url http://localhost:8000"
        )
    chroma_host = os.getenv("CHROMA_HOST")
    if chroma_host:
        port_default = 8000 if chroma_host != "localhost" else 8001
        chroma_port = int(os.getenv("CHROMA_PORT", port_default))
        try:
            client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            client.heartbeat()
            logger.info(f"Connected to ChromaDB HttpClient at {chroma_host}:{chroma_port}")
            return client
        except Exception as exc:
            logger.warning(
                f"Could not connect to ChromaDB at {chroma_host}:{chroma_port} ({exc}). "
                f"Falling back to local PersistentClient."
            )

    local_dir = PROJECT_ROOT / "data" / "chroma_db"
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using local ChromaDB PersistentClient at {local_dir}")
    return chromadb.PersistentClient(path=str(local_dir))


def get_or_create_collection(
    collection_name: str = DEFAULT_COLLECTION,
    client: chromadb.ClientAPI | None = None,
) -> chromadb.Collection:
    """Gets an existing collection or creates a new one with cosine distance."""
    cli = client or get_chroma_client()
    return cli.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_chunks(
    chunks: list[DocumentChunk],
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 64,
    purge_existing_source: bool = True,
) -> int:
    """
    Generates embeddings and upserts document chunks into ChromaDB in batches.
    Uses deterministic IDs and purges existing source chunks to eliminate duplicates.
    """
    if not chunks:
        return 0

    embedder = get_embedding_function()
    collection = get_or_create_collection(collection_name)

    # Purge existing chunks for this source file to prevent duplicate accumulation
    if purge_existing_source:
        sources = {c.metadata.get("source") for c in chunks if c.metadata.get("source")}
        for src in sources:
            try:
                collection.delete(where={"source": src})
                logger.info(f"Purged existing chunks for source '{src}' before re-ingestion.")
            except Exception:
                pass

    total_ingested = 0
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        metadatas = [c.metadata for c in batch]

        # Deterministic IDs: sha256 hash of (source + chunk_index + text preview)
        ids = []
        for idx, c in enumerate(batch, start=i):
            raw_key = f"{c.metadata.get('source', '')}_{c.metadata.get('chunk_index', idx)}_{c.page_content[:60]}"
            content_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
            doc_type = c.metadata.get("doc_type", "doc")
            sec = c.metadata.get("section", "sec")
            art = c.metadata.get("article_number", "art")
            ids.append(f"{doc_type}_{sec}_{art}_{content_hash}")

        embeddings = embedder.embed_documents(texts)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        total_ingested += len(batch)
        logger.info(f"Ingested {total_ingested}/{total_chunks} chunks into '{collection_name}'")

    return total_ingested


def search_similar(
    query: str,
    k: int = 4,
    collection_name: str = DEFAULT_COLLECTION,
    doc_type: str | None = None,
    section: str | None = None,
) -> list[dict[str, Any]]:
    """
    Queries ChromaDB and returns top-k matching chunks with similarity scores and citation metadata.
    Supports optional metadata filtering by doc_type ('sporting', 'technical') and section ('B', 'C').
    """
    embedder = get_embedding_function()
    collection = get_or_create_collection(collection_name)

    query_embedding = embedder.embed_query(query)

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }

    # Apply metadata filter if provided
    where_clauses: list[dict[str, Any]] = []
    if doc_type:
        where_clauses.append({"doc_type": doc_type.lower().strip()})
    if section:
        where_clauses.append({"section": section.upper().strip()})

    if len(where_clauses) == 1:
        query_kwargs["where"] = where_clauses[0]
    elif len(where_clauses) > 1:
        query_kwargs["where"] = {"$and": where_clauses}

    results = collection.query(**query_kwargs)

    formatted: list[dict[str, Any]] = []
    if not results or not results.get("documents"):
        return formatted

    documents = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        # Convert cosine distance to cosine similarity score (score = 1 - distance)
        similarity_score = round(1.0 - float(dist), 4) if dist is not None else None
        formatted.append(
            {
                "content": doc,
                "citation": meta.get("citation", ""),
                "doc_type": meta.get("doc_type", ""),
                "section": meta.get("section", ""),
                "article_number": meta.get("article_number", ""),
                "article_title": meta.get("article_title", ""),
                "sub_section": meta.get("sub_section", ""),
                "source": meta.get("source", ""),
                "similarity_score": similarity_score,
                "metadata": meta,
            }
        )

    return formatted
