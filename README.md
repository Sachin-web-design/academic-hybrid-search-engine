# Hybrid academic search engine (keyword + semantic)

A small end-to-end search engine over research papers: fetch papers,
index them two ways (BM25 keyword search and sentence-embedding
semantic search), and serve fused results over an API.

```
crawler/    arxiv_fetcher.py (primary) + spider.py (generic web fallback)
indexer/    Builds the BM25 index and the FAISS semantic index
search/     Fuses keyword + semantic scores at query time
api/        FastAPI endpoint that serves search results
data/       pages.jsonl (sample data included) + generated index files
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Fetch papers (optional — sample data is already in data/pages.jsonl)

Data comes from the arXiv API — structured, public, and built for this
exact use case (better fit than scraping HTML for academic content).

```bash
# Information retrieval papers
python crawler/arxiv_fetcher.py --query "cat:cs.IR" --max-results 500

# Or free-text search
python crawler/arxiv_fetcher.py --query "large language models" --max-results 300
```

Useful category codes: `cs.IR` (information retrieval), `cs.CL` (NLP),
`cs.LG` (machine learning), `cs.AI` (AI general). Combine with `AND`/`OR`,
e.g. `--query "cat:cs.IR AND cat:cs.CL"`.

If you'd rather crawl a specific set of paper-listing or university
research pages directly, `crawler/spider.py` (Scrapy) still works the
same way as before — just point `seeds.txt` at those pages.

## 3. Build the index

```bash
python indexer/build_index.py --input data/pages.jsonl
```

This writes `data/bm25.pkl`, `data/embeddings.npy`, `data/faiss.index`,
and `data/docs.pkl`. The first run downloads the `all-MiniLM-L6-v2`
sentence-transformers model (~90MB), so it needs internet access once.

## 4. Search from the command line

```bash
python -m search.hybrid "training a recommendation model"
```

## 5. Run the API

```bash
uvicorn api.main:app --reload
curl "http://localhost:8000/search?q=training+a+model&alpha=0.5&top_k=5"
```

`alpha` controls the keyword/semantic blend: `0` = pure BM25,
`1` = pure semantic, `0.5` = even split.

## Notes for your project report

- BM25 handles exact term matches well but misses synonyms/paraphrases.
- The semantic layer (MiniLM embeddings + FAISS cosine similarity) catches
  meaning-based matches — e.g. a query about "vehicles" can surface a page
  about "self-driving cars" even without shared vocabulary.
- Fusing the two (rather than picking one) is what production search
  systems generally do, and it's a good talking point in interviews:
  explain the trade-off each method makes and why blending helps.
- Reasonable extensions: reciprocal rank fusion instead of min-max score
  blending, spelling correction, query autocomplete, PageRank-style link
  authority scoring from the crawl graph.
