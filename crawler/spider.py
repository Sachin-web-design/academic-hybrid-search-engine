"""
Crawls a list of seed URLs (and their same-domain links, up to max_pages)
and writes one JSON object per line to the output file:
    {"url": ..., "title": ..., "text": ...}

Usage:
    scrapy runspider crawler/spider.py -a seeds=seeds.txt -a max_pages=200 \
        -o data/pages.jsonl -t jsonlines

seeds.txt: one URL per line.
"""

import re
from urllib.parse import urlparse

import scrapy
from scrapy.linkextractors import LinkExtractor


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SiteSpider(scrapy.Spider):
    name = "site_spider"

    # Be a polite crawler.
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 1.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "USER_AGENT": "MyPlacementProjectBot/1.0 (+contact: you@example.com)",
        "DEPTH_LIMIT": 3,
    }

    def __init__(self, seeds=None, max_pages=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not seeds:
            raise ValueError("Pass -a seeds=seeds.txt (one URL per line)")
        with open(seeds) as f:
            self.start_urls = [line.strip() for line in f if line.strip()]
        self.allowed_domains = list(
            {urlparse(u).netloc for u in self.start_urls}
        )
        self.max_pages = int(max_pages)
        self.seen = 0
        self.link_extractor = LinkExtractor(allow_domains=self.allowed_domains)

    def parse(self, response):
        if self.seen >= self.max_pages:
            return
        if "text/html" not in response.headers.get("Content-Type", b"").decode(
            "latin-1", "ignore"
        ):
            return

        title = response.css("title::text").get(default="").strip()

        # Strip script/style/nav/footer before grabbing visible text.
        for bad in response.css("script, style, nav, footer, header"):
            bad.root.getparent().remove(bad.root) if bad.root.getparent() is not None else None

        paragraphs = response.css(
            "p::text, li::text, h1::text, h2::text, h3::text"
        ).getall()
        text = clean_text(" ".join(paragraphs))

        if len(text) > 200:  # skip near-empty pages
            self.seen += 1
            yield {
                "url": response.url,
                "title": title,
                "text": text[:20000],  # cap page size
            }

        if self.seen < self.max_pages:
            for link in self.link_extractor.extract_links(response):
                yield response.follow(link, callback=self.parse)
