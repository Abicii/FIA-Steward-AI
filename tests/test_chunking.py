"""Unit tests for the regulatory chunking and citation extraction service."""

from app.services.chunking import (
    chunk_document,
    extract_article_sections,
    find_nearest_sub_section,
)

SAMPLE_REGULATION_MD = """# SECTION B: SPORTING REGULATIONS

## CONTENTS
### ARTICLE B1: ORGANISATION OF A COMPETITION 4

# ARTICLE B1: ORGANISATION OF A COMPETITION

## B1.1 General Principles & Provisions

### B1.1.1
A Competition commences four (4) hours before FP1 is scheduled to start and ends at the time for the lodging of a protest.

### B1.1.2 Instructions And Communications
The stewards or Race Director may give instructions to Competitors by means of special circulars.

## B1.2 FIA Delegates

### B1.2.1
For each Competition the FIA will nominate the following delegates:
Safety Delegate, Sporting Delegate, Medical Delegate.

# ARTICLE B6: TYRE LIMITATIONS

## B6.1 Supply Of Tyres
Pirelli will be the sole tyre supplier for all Events during the Championship.
"""


def test_extract_article_sections():
    sections = extract_article_sections(SAMPLE_REGULATION_MD)
    assert len(sections) >= 2

    # Verify Article B1 and Article B6 were extracted
    numbers = [s["article_number"] for s in sections]
    assert "B1" in numbers
    assert "B6" in numbers


def test_chunk_document_metadata_and_citations():
    chunks = chunk_document(
        SAMPLE_REGULATION_MD,
        filename="FIA_2027_Sporting.md",
        chunk_size=300,
        chunk_overlap=50,
    )

    assert len(chunks) > 0

    # Ensure all chunks carry essential citation metadata
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "citation" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["source"] == "FIA_2027_Sporting.md"

    # Verify that a chunk under Article B1 carries the correct citation
    b1_chunks = [c for c in chunks if c.metadata.get("article_number") == "B1"]
    assert len(b1_chunks) > 0
    assert "ARTICLE B1: ORGANISATION OF A COMPETITION" in b1_chunks[0].metadata["citation"]

    # Verify sub-section tracking
    sub_sections = [c.metadata.get("sub_section") for c in b1_chunks if c.metadata.get("sub_section")]
    assert any("B1.1" in s or "B1.2" in s for s in sub_sections)


def test_find_nearest_sub_section():
    text = "## B5.12 Virtual Safety Car (VSC)\nWhen the VSC is deployed..."
    sub_sec = find_nearest_sub_section(text)
    assert "B5.12" in sub_sec
    assert "Virtual Safety Car" in sub_sec
