from __future__ import annotations
from typing import List, Optional, Any, Dict
from functools import lru_cache
import os
import httpx

from fastapi import APIRouter, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL        = os.getenv("QDRANT_URL", "").rstrip("/")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "")
DENSE_VECTOR_NAME = os.getenv("DENSE_VECTOR_NAME", "text")

GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "")
DENSE_EMBEDDING_MODEL   = os.getenv("DENSE_EMBEDDING_MODEL", "models/gemini-embedding-001")
GEMINI_TASK_TYPE  = os.getenv("GEMINI_TASK_TYPE", "RETRIEVAL_QUERY")
GEMINI_OUTPUT_DIM = os.getenv("GEMINI_OUTPUT_DIM")

SPARSE_K = int(os.getenv("SPARSE_K", "20"))
DENSE_K  = int(os.getenv("DENSE_K", "20"))
LIMIT    = int(os.getenv("LIMIT", "10"))

BM25_MODEL     = os.getenv("BM25_MODEL", "Qdrant/bm25")
BM25_CACHE_DIR = os.getenv("BM25_CACHE_DIR", ".cache_fastembed")

RETRIEVAL_API_KEY = (os.getenv("RETRIEVAL_API_KEY") or "").strip()

router = APIRouter(tags=["retrieval"])

def _check_auth(x_api_key: Optional[str]):
    if RETRIEVAL_API_KEY and (x_api_key or "") != RETRIEVAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@lru_cache(maxsize=1)
def _sparse_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name=BM25_MODEL, cache_dir=BM25_CACHE_DIR)

def _embed_sparse_indices_values(query: str):
    emb = next(_sparse_model().embed([query]))
    indices = emb.indices.tolist() if hasattr(emb.indices, "tolist") else list(emb.indices)
    values  = emb.values.tolist()  if hasattr(emb.values,  "tolist")  else list(emb.values)
    return [int(i) for i in indices], [float(v) for v in values]

def _google_model_path() -> str:
    return DENSE_EMBEDDING_MODEL if DENSE_EMBEDDING_MODEL.startswith("models/") else f"models/{DENSE_EMBEDDING_MODEL}"

def _embed_dense_google(query: str) -> List[float]:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set")
    model_path = _google_model_path()
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:embedContent"
    body: Dict[str, Any] = {
        "model": model_path,
        "content": {"parts": [{"text": query}]},
        "taskType": GEMINI_TASK_TYPE,
    }
    if GEMINI_OUTPUT_DIM:
        try:
            body["outputDimensionality"] = int(GEMINI_OUTPUT_DIM)
        except ValueError:
            pass

    headers = {"x-goog-api-key": GOOGLE_API_KEY, "Content-Type": "application/json"}
    with httpx.Client(timeout=httpx.Timeout(20.0)) as client:
        r = client.post(url, headers=headers, json=body)
        r.raise_for_status()
        j = r.json()
        values = (j.get("embedding") or {}).get("values")
        if values is None and "embeddings" in j:
            values = j["embeddings"][0]["values"]
        if not isinstance(values, list):
            raise HTTPException(status_code=502, detail="Gemini embedding response missing 'values'")
        return [float(x) for x in values]

class HybridRequest(BaseModel):
    query: str = Field(..., description="User query")
    category: Optional[str] = Field(None, description="Exact match to metadata.category")
    limit: Optional[int] = None
    sparse_k: Optional[int] = None
    dense_k: Optional[int] = None

class Hit(BaseModel):
    id: Any
    score: float
    text: Optional[str] = None
    metadata: Dict[str, Any] = {}

class HybridResponse(BaseModel):
    points: List[Hit]

@router.get("/retrieve/health")
def retrieve_health():
    return {
        "status": "ok",
        "qdrant_url_set": bool(QDRANT_URL),
        "collection": QDRANT_COLLECTION or None,
        "dense_vector_name": DENSE_VECTOR_NAME,
        "bm25_model": BM25_MODEL,
        "gemini_model": _google_model_path(),
    }

@router.post("/retrieve/hybrid", response_model=HybridResponse)
def retrieve_hybrid(body: HybridRequest, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
    _check_auth(x_api_key)
    if not (QDRANT_URL and QDRANT_API_KEY and QDRANT_COLLECTION):
        raise HTTPException(status_code=500, detail="Qdrant env (URL/API_KEY/COLLECTION) not set")

    s_idx, s_val = _embed_sparse_indices_values(body.query)
    d_vec = _embed_dense_google(body.query)

    q_filter = None
    if body.category:
        q_filter = {"must": [{"key": "metadata.category", "match": {"value": body.category}}]}

    sparse_k = int(body.sparse_k or SPARSE_K)
    dense_k  = int(body.dense_k  or DENSE_K)
    limit    = int(body.limit    or LIMIT)

    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/query"
    payload = {
        "prefetch": [
            {"query": {"indices": s_idx, "values": s_val}, "using": "bm25", "limit": sparse_k, "filter": q_filter},
            {"query": d_vec, "limit": dense_k, "filter": q_filter},
        ],
        "query": {"fusion": "rrf"},
        "limit": limit,
        "with_payload": True
    }
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    pts = data.get("result", {}).get("points") or data.get("points") or []
    out: List[Hit] = []
    for p in pts:
        pid = p.get("id")
        score = float(p.get("score", 0.0))
        payload = p.get("payload") or {}
        text = payload.get("text") or payload.get("content")
        meta = payload.get("metadata") or payload
        out.append(Hit(id=pid, score=score, text=text, metadata=meta))
    return HybridResponse(points=out)

def attach_retrieval_endpoints(app: FastAPI) -> None:
    app.include_router(router)
