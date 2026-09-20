"""
Loads the artifacts built by indexer/build_index.py and answers queries by
fusing BM25 (keyword) scores with FAISS (semantic) scores.

Usage as a library:
    from search.hybrid import HybridSearcher
    hs = HybridSearcher("data")
    hs.search("training a recommendation model", alpha=0.5, top_k=5)

alpha=0   -> pure keyword search
alpha=1   -> pure semantic search
alpha=0.5 -> equal blend (a reasonable default)
"""

import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from indexer.build_index import tokenize


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridSearcher:
    def __init__(self, data_dir="data", model_name="all-MiniLM-L6-v2"):
        with open(f"{data_dir}/bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(f"{data_dir}/docs.pkl", "rb") as f:
            self.docs = pickle.load(f)
        self.faiss_index = faiss.read_index(f"{data_dir}/faiss.index")
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, alpha: float = 0.5, top_k: int = 10):
        # Keyword scores over the whole corpus.
        bm25_scores = np.array(self.bm25.get_scores(tokenize(query)))

        # Semantic scores: FAISS returns top-N by similarity; we ask for all
        # docs so we can fuse scores fairly rather than fusing two separate
        # top-k lists.
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        n = self.faiss_index.ntotal
        sem_scores_all, idx_all = self.faiss_index.search(q_emb, n)
        sem_scores = np.zeros(n, dtype="float32")
        sem_scores[idx_all[0]] = sem_scores_all[0]

        kw_norm = min_max_normalize(bm25_scores)
        sem_norm = min_max_normalize(sem_scores)
        hybrid = (1 - alpha) * kw_norm + alpha * sem_norm

        ranked = np.argsort(-hybrid)[:top_k]
        results = []
        for i in ranked:
            results.append(
                {
                    "title": self.docs[i]["title"],
                    "url": self.docs[i]["url"],
                    "snippet": self.docs[i]["text"][:200],
                    "score": float(hybrid[i]),
                    "keyword_score": float(kw_norm[i]),
                    "semantic_score": float(sem_norm[i]),
                }
            )
        return results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "how does a search engine rank pages"
    hs = HybridSearcher("data")
    for r in hs.search(query, alpha=0.5, top_k=5):
        print(f"{r['score']:.3f}  {r['title']}  ({r['url']})")
