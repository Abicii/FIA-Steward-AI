import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.services.indexer import index_all_parsed_files, index_markdown_file
from app.services.vectorstore import (
    DEFAULT_COLLECTION,
    get_chroma_client,
    search_similar,
)
from scripts.clean_markdown import clean_markdown
from scripts.llamaparse import LlamaParseConfigError, parse_pdf

app = FastAPI(title="FIA Steward AI")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_DATA_DIR = PROJECT_ROOT / "parsed data"


class IngestRequest(BaseModel):
    filename: str | None = None
    collection_name: str = DEFAULT_COLLECTION


def _safe_stem(filename: str | None) -> str:
    stem = Path(filename or "document").stem
    cleaned = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem).strip()
    return cleaned or "document"


def save_markdown(filename: str | None, markdown: str) -> Path:
    PARSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARSED_DATA_DIR / f"{_safe_stem(filename)}.md"
    if out_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = PARSED_DATA_DIR / f"{_safe_stem(filename)}_{stamp}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return out_path


@app.get("/health")
def health_check():
    try:
        client = get_chroma_client()
        heartbeat = client.heartbeat()
        return {"api": "ok", "chroma": heartbeat}
    except Exception as exc:
        return {"api": "ok", "chroma": "unreachable", "error": str(exc)}


def _is_pdf(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    return content_type == "application/pdf" or filename.endswith(".pdf")


@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    """
    Parses and cleans an uploaded PDF, writing the resulting .md file to 'parsed data/'.
    Does not perform any chunking or embedding.
    """
    if not _is_pdf(file):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)

        markdown = clean_markdown(parse_pdf(tmp_path))
        saved_path = save_markdown(file.filename, markdown)
    except LlamaParseConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="LlamaParse failed to parse the PDF") from exc
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    return {
        "filename": file.filename,
        "saved_path": str(saved_path),
        "markdown": markdown,
    }


@app.get("/parsed-files")
def list_parsed_files():
    """Lists all cleaned markdown files available in 'parsed data/'."""
    if not PARSED_DATA_DIR.exists():
        return {"files": []}
    files = [
        {
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "modified_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
        }
        for p in PARSED_DATA_DIR.glob("*.md")
    ]
    return {"count": len(files), "files": files}


@app.post("/ingest")
def ingest_cleaned_markdown(req: IngestRequest | None = None):
    """
    Independent ingestion stage:
    Reads cleaned .md file(s) from 'parsed data/', splits into chunks with Article citation
    metadata, generates FastEmbed embeddings, and stores them in ChromaDB.
    """
    collection = req.collection_name if req else DEFAULT_COLLECTION
    filename = req.filename if req else None

    if filename and filename.lower() != "all":
        target_path = PARSED_DATA_DIR / filename
        if not target_path.exists():
            target_path = PARSED_DATA_DIR / f"{filename}.md"
        if not target_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' not found in '{PARSED_DATA_DIR.name}' directory.",
            )
        try:
            return index_markdown_file(target_path, collection_name=collection)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    results = index_all_parsed_files(PARSED_DATA_DIR, collection_name=collection)
    return {
        "status": "success",
        "collection": collection,
        "files_indexed": results,
    }


# Backwards compatibility alias for /index
@app.post("/index", include_in_schema=False)
def index_alias(req: IngestRequest | None = None):
    return ingest_cleaned_markdown(req)


@app.get("/search")
def search_regulations(
    q: str = Query(..., description="Query to search regulations for."),
    k: int = Query(default=4, ge=1, le=20, description="Number of chunks to return."),
    collection_name: str = Query(
        default=DEFAULT_COLLECTION,
        description="Collection to query.",
    ),
):
    """
    Semantic similarity search across indexed regulations.
    Returns matching chunks along with complete citation metadata.
    """
    try:
        results = search_similar(query=q, k=k, collection_name=collection_name)
        return {
            "query": q,
            "count": len(results),
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
