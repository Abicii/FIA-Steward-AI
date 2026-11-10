"""
High-level indexing service coordinating markdown chunking and vector storage ingestion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.services.chunking import chunk_document
from app.services.vectorstore import DEFAULT_COLLECTION, ingest_chunks

logger = logging.getLogger(__name__)


def index_markdown_text(
    markdown_text: str,
    filename: str = "document.md",
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> dict[str, Any]:
    """
    Chunks and indexes markdown text directly from memory into ChromaDB.
    Returns indexing metrics including total chunks and unique articles identified.
    """
    chunks = chunk_document(
        markdown_text=markdown_text,
        filename=filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        return {
            "source": filename,
            "status": "empty",
            "chunks_indexed": 0,
            "articles": [],
        }

    total_ingested = ingest_chunks(chunks=chunks, collection_name=collection_name)

    # Collect unique citations/articles
    unique_articles = sorted(
        {
            f"{c.metadata.get('article_type', 'ARTICLE')} {c.metadata.get('article_number', '')}: {c.metadata.get('article_title', '')}"
            for c in chunks
            if c.metadata.get("article_number") and c.metadata.get("article_number") != "0"
        }
    )

    return {
        "source": filename,
        "status": "success",
        "chunks_indexed": total_ingested,
        "collection": collection_name,
        "total_articles": len(unique_articles),
        "articles": unique_articles,
    }


def index_markdown_file(
    file_path: Path,
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> dict[str, Any]:
    """Reads a .md file from disk and indexes its content into ChromaDB."""
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Markdown file not found at: {file_path}")

    logger.info(f"Indexing markdown file: {file_path}")
    content = file_path.read_text(encoding="utf-8")
    return index_markdown_text(
        markdown_text=content,
        filename=file_path.name,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def index_all_parsed_files(
    parsed_dir: Path,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """Scans the parsed data directory and indexes all markdown files."""
    if not parsed_dir.exists():
        return []

    results = []
    for md_file in parsed_dir.glob("*.md"):
        try:
            res = index_markdown_file(md_file, collection_name=collection_name)
            results.append(res)
        except Exception as exc:
            logger.error(f"Failed to index {md_file.name}: {exc}")
            results.append({"source": md_file.name, "status": "error", "detail": str(exc)})

    return results
