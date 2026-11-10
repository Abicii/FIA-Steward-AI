"""
Text chunking service tailored for regulatory documents (e.g., FIA Formula 1 Regulations).

Splits markdown text semantically by Article and Sub-Article, generating metadata-rich
chunks containing full citation paths (Article number, title, sub-section) for reliable RAG retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """Represents a text chunk with citation metadata."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
        }


# Regex to detect Article and Appendix headings in markdown
# Handles: '# ARTICLE B1: ...', '## ARTICLE B1 ...', '### ARTICLE ...', and 'APPENDIX B1 ...'
ARTICLE_HEADER_PATTERN = re.compile(
    r"(?m)^#{1,3}\s+(?:\*{0,2}|~~)?(ARTICLE|APPENDIX)\s+([A-Z0-9]+)[:\s]+([^~\*\n\r]+?)(?:\*{0,2}|~~)?\s*$"
)

# Regex to detect sub-sections within an article (e.g. 'B1.1 General Principles', 'B1.2 FIA Delegates')
SUB_SECTION_PATTERN = re.compile(
    r"(?:^|\n)(?:\|?\s*|#{1,3}\s+|\*{2})([A-Z0-9]+\.[0-9]+(?:\.[0-9]+)?)\s*[:\|*]?\s*([^\n\r\|]{3,80})"
)


def _split_text_recursively(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursively splits text using a list of separators, preserving markdown structure.
    Falls back gracefully if langchain is not present.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        pass

    # Built-in lightweight fallback if running outside Docker without langchain installed
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    seps = separators or ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    chosen_sep = ""
    for sep in seps:
        if sep in text:
            chosen_sep = sep
            break

    splits = text.split(chosen_sep) if chosen_sep else list(text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for part in splits:
        part_len = len(part) + (len(chosen_sep) if current_chunk else 0)
        if current_length + part_len > chunk_size and current_chunk:
            combined = chosen_sep.join(current_chunk).strip()
            if combined:
                chunks.append(combined)
            current_chunk = []
            current_length = 0

        current_chunk.append(part)
        current_length += part_len

    if current_chunk:
        combined = chosen_sep.join(current_chunk).strip()
        if combined:
            chunks.append(combined)

    return chunks


def extract_article_sections(markdown_text: str) -> list[dict[str, Any]]:
    """
    Segments the markdown document into distinct Article/Appendix blocks.
    Any text before the first article (like title/TOC) is captured as 'PREAMBLE'.
    """
    matches = list(ARTICLE_HEADER_PATTERN.finditer(markdown_text))
    if not matches:
        return [
            {
                "article_type": "DOCUMENT",
                "article_number": "ALL",
                "article_title": "Full Document",
                "text": markdown_text,
            }
        ]

    sections: list[dict[str, Any]] = []

    # Text before the first matched article (usually preamble or TOC)
    if matches[0].start() > 0:
        preamble_text = markdown_text[: matches[0].start()].strip()
        if preamble_text:
            sections.append(
                {
                    "article_type": "PREAMBLE",
                    "article_number": "0",
                    "article_title": "Overview & Table of Contents",
                    "text": preamble_text,
                }
            )

    for i, match in enumerate(matches):
        start_idx = match.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        section_text = markdown_text[start_idx:end_idx].strip()

        article_type = match.group(1).strip().upper()
        article_number = match.group(2).strip().upper()
        article_title = match.group(3).strip()

        sections.append(
            {
                "article_type": article_type,
                "article_number": article_number,
                "article_title": article_title,
                "text": section_text,
            }
        )

    return sections


def find_nearest_sub_section(chunk_text: str, current_sub_section: str = "") -> str:
    """Detects if a new sub-section header exists within the chunk."""
    match = SUB_SECTION_PATTERN.search(chunk_text)
    if match:
        code = match.group(1).strip()
        title = match.group(2).strip().strip("*").strip()
        return f"{code} {title}".strip()
    return current_sub_section


def chunk_document(
    markdown_text: str,
    filename: str = "document.md",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """
    Main chunking pipeline for a regulatory markdown file:
    1. Splits document by Article sections.
    2. Recursively splits each Article into manageable chunks (~1000 chars).
    3. Injects citation metadata (Article number, Article title, sub-section, source).
    """
    sections = extract_article_sections(markdown_text)
    chunks: list[DocumentChunk] = []
    chunk_counter = 0

    for section in sections:
        article_type = section["article_type"]
        article_num = section["article_number"]
        article_title = section["article_title"]
        section_text = section["text"]

        base_citation = (
            f"{article_type} {article_num}: {article_title}"
            if article_type != "PREAMBLE"
            else "Preamble & Table of Contents"
        )

        sub_chunks = _split_text_recursively(
            section_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        active_sub_section = ""
        for raw_chunk in sub_chunks:
            chunk_text = raw_chunk.strip()
            if not chunk_text:
                continue

            active_sub_section = find_nearest_sub_section(chunk_text, active_sub_section)

            # Build full citation for easy LLM attribution
            full_citation = (
                f"{base_citation} > {active_sub_section}"
                if active_sub_section
                else base_citation
            )

            metadata = {
                "source": filename,
                "article_type": article_type,
                "article_number": article_num,
                "article_title": article_title,
                "sub_section": active_sub_section,
                "citation": full_citation,
                "chunk_index": chunk_counter,
                "chunk_size": len(chunk_text),
            }

            chunks.append(DocumentChunk(page_content=chunk_text, metadata=metadata))
            chunk_counter += 1

    return chunks
