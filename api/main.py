"""
Run with:
    uvicorn api.main:app --reload

Then:
    curl "http://localhost:8000/search?q=training+a+model&alpha=0.5&top_k=5"
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from search.hybrid import HybridSearcher

app = FastAPI(title="Hybrid Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

searcher: HybridSearcher | None = None


@app.on_event("startup")
def load_index():
    global searcher
    searcher = HybridSearcher("data")


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="0 = keyword only, 1 = semantic only"),
    top_k: int = Query(10, ge=1, le=50),
):
    results = searcher.search(q, alpha=alpha, top_k=top_k)
    return {"query": q, "alpha": alpha, "results": results}


@app.get("/health")
def health():
    return {"status": "ok", "docs_indexed": len(searcher.docs) if searcher else 0}
