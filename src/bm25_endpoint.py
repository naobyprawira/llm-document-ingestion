"""
BM25 sparse vector endpoints to be attached to the existing FastAPI app.

Usage (edit src/api.py):
    from .bm25_endpoint import attach_bm25_endpoints
    attach_bm25_endpoints(app)

Routes added:
    GET  /bm25/health
    POST /sparse/bm25           body: {"text": "..."}          -> {"indices":[...], "values":[...]}
    POST /sparse/bm25/batch     body: {"texts": ["...","..."]} -> [{"indices":[...],"values":[...]}, ...]

Requires:
    pip install fastembed>=0.3.5

Env (optional):
    BM25_API_KEY   - if set, requests must include header: X-API-Key: <value>
    BM25_MODEL     - default "Qdrant/bm25"
    BM25_CACHE_DIR - default ".cache_fastembed"
"""
from typing import List, Optional
from fastapi import APIRouter, Header, HTTPException, FastAPI

from pydantic import BaseModel
from functools import lru_cache
import os
from dotenv import load_dotenv
load_dotenv()

BM25_MODEL = os.getenv("BM25_MODEL", "Qdrant/bm25")
BM25_CACHE_DIR = os.getenv("BM25_CACHE_DIR", ".cache_fastembed")
BM25_API_KEY = (os.getenv("BM25_API_KEY") or "your-secret-bm25-api-key").strip()

router = APIRouter(tags=["bm25"])

def _check_auth(x_api_key: Optional[str]):
    if BM25_API_KEY and (x_api_key or "") != BM25_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@lru_cache(maxsize=1)
def _sparse_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name=BM25_MODEL, cache_dir=BM25_CACHE_DIR)

# Pydantic response model (optional but nice: docs + validation)
class SparseVector(BaseModel):
    indices: List[int]
    values: List[float]

class SparseRequest(BaseModel):
    text: str

class SparseBatchRequest(BaseModel):
    texts: List[str]

@router.get("/bm25/health")
def bm25_health():
    return {"status": "ok", "model": BM25_MODEL}

@router.post("/sparse/bm25", response_model=SparseVector)
def sparse_single(req: SparseRequest, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
    _check_auth(x_api_key)
    emb = next(_sparse_model().embed([req.text]))
    # 👇 Convert NumPy dtypes to plain Python types
    indices = emb.indices.tolist() if hasattr(emb.indices, "tolist") else list(emb.indices)
    values  = emb.values.tolist()  if hasattr(emb.values,  "tolist") else list(emb.values)
    indices = [int(i) for i in indices]
    values  = [float(v) for v in values]
    return {"indices": indices, "values": values}

@router.post("/sparse/bm25/batch", response_model=List[SparseVector])
def sparse_batch(req: SparseBatchRequest, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
    _check_auth(x_api_key)
    out = []
    for emb in _sparse_model().embed(req.texts):
        idx = emb.indices.tolist() if hasattr(emb.indices, "tolist") else list(emb.indices)
        val = emb.values.tolist()  if hasattr(emb.values,  "tolist") else list(emb.values)
        out.append({"indices": [int(i) for i in idx], "values": [float(v) for v in val]})
    return out


def attach_bm25_endpoints(app: FastAPI) -> None:
    """Attach the router to your existing FastAPI app."""
    app.include_router(router)
