"""Tests for indexer and ingestion logic."""

from pathlib import Path
from app.services.chunking import chunk_document
from app.services.indexer import index_markdown_text

SAMPLE_TEXT = """# ARTICLE B1: ORGANISATION OF A COMPETITION

## B1.1 General Principles
The competition commences 4 hours before FP1.

## B1.2 FIA Delegates
FIA will nominate delegates.
"""

def test_index_markdown_text_empty():
    res = index_markdown_text("", filename="empty.md")
    assert res["status"] == "empty"
    assert res["chunks_indexed"] == 0

def test_chunking_citation_consistency():
    chunks = chunk_document(SAMPLE_TEXT, filename="sample.md", chunk_size=80, chunk_overlap=10)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata["article_number"] == "B1"
        assert "ORGANISATION OF A COMPETITION" in c.metadata["article_title"]
        assert c.metadata["citation"].startswith("ARTICLE B1")
