"""
Reads data/pages.jsonl (one JSON object per line: url, title, text) and builds:
  - data/bm25.pkl        BM25Okapi index over tokenized documents
  - data/embeddings.npy  sentence embeddings for every document
  - data/faiss.index     FAISS index over those embeddings (cosine via inner product)
  - data/docs.pkl        the document metadata (url, title, text) in index order

Usage:
    python indexer/build_index.py --input data/pages.jsonl
"""

import argparse
import json
import pickle
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

STOPWORDS = set(
    "a an the of and to for in on is are with using how what was were be by "
    "this that it as at from or not can will".split()
)


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t and t not in STOPWORDS]


def load_docs(path):
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pages.jsonl")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    docs = load_docs(args.input)
    if not docs:
        raise SystemExit(f"No documents found in {args.input}")
    print(f"Loaded {len(docs)} documents")

    # --- Keyword index (BM25) ---
    tokenized = [tokenize(d["title"] + " " + d["text"]) for d in docs]
    bm25 = BM25Okapi(tokenized)
    with open(f"{args.out_dir}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("Built BM25 index")

    # --- Semantic index (sentence embeddings + FAISS) ---
    model = SentenceTransformer(args.model)
    texts = [d["title"] + ". " + d["text"][:2000] for d in docs]
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on normalized vecs = cosine
    index.add(embeddings)

    np.save(f"{args.out_dir}/embeddings.npy", embeddings)
    faiss.write_index(index, f"{args.out_dir}/faiss.index")
    print(f"Built FAISS index with {index.ntotal} vectors, dim {embeddings.shape[1]}")

    with open(f"{args.out_dir}/docs.pkl", "wb") as f:
        pickle.dump(docs, f)

    print("Done. Artifacts written to", args.out_dir)


if __name__ == "__main__":
    main()
