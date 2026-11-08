import os 
import chromadb
from fastapi import FastAPI

app = FastAPI(title="FIA Steward AI")

@app.get("/health")

def health_check():
    client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST", "localhost"), port=8000)
    return {"api": "ok", "chroma": client.heartbeat()}