import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile

from scripts.clean_markdown import clean_markdown
from scripts.llamaparse import LlamaParseConfigError, parse_pdf

app = FastAPI(title="FIA Steward AI")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_DATA_DIR = PROJECT_ROOT / "parsed data"


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
    client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST", "localhost"), port=8000)
    return {"api": "ok", "chroma": client.heartbeat()}


def _is_pdf(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    return content_type == "application/pdf" or filename.endswith(".pdf")


@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
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
