"""
Fetches papers from the arXiv API (export.arxiv.org) and writes them in the
same {url, title, text} JSONL format the indexer expects, so the rest of the
pipeline (build_index.py, hybrid.py, api/main.py) needs no changes.

arXiv's API is the right tool here instead of scraping HTML: it's public,
well-documented, gives clean structured metadata, and has a clear rate-limit
policy (max ~1 request per 3 seconds) that's easy to respect.

Usage:
    python crawler/arxiv_fetcher.py --query "cat:cs.IR" --max-results 500
    python crawler/arxiv_fetcher.py --query "large language models" --max-results 300

Docs: https://info.arxiv.org/help/api/user-manual.html
"""

import argparse
import json
import time
import urllib.parse
import urllib.request

import feedparser

ARXIV_API = "http://export.arxiv.org/api/query"
PAGE_SIZE = 100          # arXiv recommends batching requests
DELAY_SECONDS = 3        # be a polite API citizen


def fetch_batch(query: str, start: int, max_results: int) -> feedparser.FeedParserDict:
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    return feedparser.parse(raw)


def paper_to_doc(entry) -> dict:
    authors = [a.name for a in getattr(entry, "authors", [])]
    categories = [t["term"] for t in getattr(entry, "tags", [])]
    published = getattr(entry, "published", "")[:10]  # just the date part
    return {
        "url": entry.link,
        "title": entry.title.strip().replace("\n", " "),
        "text": entry.summary.strip().replace("\n", " "),  # clean abstract, used for search/embeddings
        "authors": authors,
        "categories": categories,
        "published": published,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="cat:cs.IR",
        help='arXiv query, e.g. "cat:cs.IR" (Information Retrieval), '
        '"cat:cs.CL" (NLP), "cat:cs.LG" (ML), or free text like "search engines"',
    )
    parser.add_argument("--max-results", type=int, default=300)
    parser.add_argument("--out", default="data/pages.jsonl")
    args = parser.parse_args()

    fetched = 0
    with open(args.out, "w") as f:
        while fetched < args.max_results:
            batch_size = min(PAGE_SIZE, args.max_results - fetched)
            feed = fetch_batch(args.query, fetched, batch_size)
            if not feed.entries:
                break
            for entry in feed.entries:
                doc = paper_to_doc(entry)
                f.write(json.dumps(doc) + "\n")
                fetched += 1
            print(f"Fetched {fetched} papers so far...")
            time.sleep(DELAY_SECONDS)

    print(f"Done. Wrote {fetched} papers to {args.out}")


if __name__ == "__main__":
    main()
