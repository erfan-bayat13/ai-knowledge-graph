# kg/crawlers/arxiv.py
# arXiv RSS crawler — feeds configurable from the CLI
#
# Usage:
#   python -m kg.crawlers.arxiv                              # default feeds
#   python -m kg.crawlers.arxiv --feeds cs.CV,cs.LG         # custom feeds
#   python -m kg.crawlers.arxiv --list-feeds                 # show all available categories
#   python -m kg.crawlers.arxiv --max 200                    # papers per feed

import asyncio
import argparse
import httpx
import feedparser
import logging
from typing import List, Dict, Optional
from datetime import datetime

from kg.crawlers.base import BaseCrawler

logger = logging.getLogger(__name__)

# ── Default feeds (edit here OR override via --feeds at CLI) ──────────────────
ARXIV_FEEDS = [
    "https://export.arxiv.org/rss/cs.LG",    # Machine Learning
    "https://export.arxiv.org/rss/cs.AI",    # Artificial Intelligence
    "https://export.arxiv.org/rss/cs.CL",    # Computation and Language (NLP)
    "https://export.arxiv.org/rss/cs.NE",    # Neural and Evolutionary Computing
    "https://export.arxiv.org/rss/cs.CV",    # Computer Vision
    "https://export.arxiv.org/rss/cs.DC",    # Distributed, Parallel Computing
    "https://export.arxiv.org/rss/cs.SE",    # Software Engineering
    "https://export.arxiv.org/rss/cs.CR",    # Cryptography and Security
    "https://export.arxiv.org/rss/stat.ML",  # Statistical Machine Learning
]

# ── All available arXiv categories (for --list-feeds) ─────────────────────────
ALL_FEEDS = {
    # Computer Science
    "cs.AI":  "Artificial Intelligence",
    "cs.CL":  "Computation and Language (NLP)",
    "cs.CR":  "Cryptography and Security",
    "cs.CV":  "Computer Vision",
    "cs.CY":  "Computers and Society",
    "cs.DC":  "Distributed & Parallel Computing",
    "cs.DS":  "Data Structures and Algorithms",
    "cs.GT":  "Computer Science and Game Theory",
    "cs.HC":  "Human-Computer Interaction",
    "cs.IR":  "Information Retrieval",
    "cs.IT":  "Information Theory",
    "cs.LG":  "Machine Learning",
    "cs.LO":  "Logic in Computer Science",
    "cs.MA":  "Multiagent Systems",
    "cs.MM":  "Multimedia",
    "cs.MS":  "Mathematical Software",
    "cs.NA":  "Numerical Analysis",
    "cs.NE":  "Neural and Evolutionary Computing",
    "cs.NI":  "Networking and Internet Architecture",
    "cs.PF":  "Performance",
    "cs.PL":  "Programming Languages",
    "cs.RO":  "Robotics",
    "cs.SC":  "Symbolic Computation",
    "cs.SD":  "Sound",
    "cs.SE":  "Software Engineering",
    "cs.SI":  "Social and Information Networks",
    "cs.SY":  "Systems and Control",
    # Statistics
    "stat.ML": "Machine Learning (Statistics)",
    "stat.AP": "Applications (Statistics)",
    "stat.CO": "Computation (Statistics)",
    "stat.ME": "Methodology (Statistics)",
    "stat.TH": "Theory (Statistics)",
    # Physics / Math
    "math.OC": "Optimization and Control",
    "math.ST": "Statistics Theory",
    "eess.SP": "Signal Processing",
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "q-bio.NC": "Neurons and Cognition",
}


def _build_feed_urls(categories: List[str]) -> List[str]:
    """Convert category codes to full arXiv RSS URLs."""
    return [f"https://export.arxiv.org/rss/{cat.strip()}" for cat in categories]


class ArxivCrawler(BaseCrawler):
    """Crawls arXiv RSS feeds. Feed list configurable at construction time."""

    def __init__(
        self,
        max_papers_per_feed: int = 100,
        feeds: Optional[List[str]] = None,
    ):
        super().__init__()
        self.max_papers_per_feed = max_papers_per_feed
        # feeds is a list of full URLs; defaults to the module-level ARXIV_FEEDS
        self.feed_urls = feeds if feeds is not None else ARXIV_FEEDS

    async def fetch(self) -> List[Dict]:
        """Fetch papers from configured arXiv RSS feeds."""
        all_papers = []

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for feed_url in self.feed_urls:
                cat = feed_url.split("/")[-1]
                try:
                    response = await client.get(feed_url)
                    response.raise_for_status()

                    feed = feedparser.parse(response.text)

                    skip_days = [d.get("value", "") for d in getattr(feed.feed, "skip_days", [])]
                    if not feed.entries and skip_days:
                        logger.warning(f"Feed empty — arXiv skips {skip_days} (weekend/holiday)")
                        print(f"  ⚠️  {cat}: empty (arXiv skips {skip_days})")
                        continue

                    papers = feed.entries[:self.max_papers_per_feed]
                    all_papers.extend([
                        {"raw": entry, "feed_url": feed_url}
                        for entry in papers
                    ])
                    print(f"  ✅  {cat}: {len(papers)} papers")

                except Exception as e:
                    logger.error(f"Error fetching {feed_url}: {e}")
                    print(f"  ❌  {cat}: {e}")

        return all_papers

    async def parse(self, raw_data: List[Dict]) -> List[Dict]:
        """Parse arXiv feed entries into structured paper dicts."""
        parsed = []
        seen_ids = set()

        for item in raw_data:
            entry = item["raw"]
            try:
                arxiv_id = self._extract_arxiv_id(entry.get("id", ""))
                if not arxiv_id or arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                authors = []
                if hasattr(entry, "authors") and entry.authors:
                    raw_authors = entry.authors[0].get("name", "")
                    authors = [a.strip() for a in raw_authors.split(",") if a.strip()]
                elif hasattr(entry, "author"):
                    authors = [a.strip() for a in entry.author.split(",") if a.strip()]

                abstract = entry.get("summary", "").replace("\n", " ").strip()

                parsed.append({
                    "arxiv_id":       arxiv_id,
                    "title":          entry.get("title", "").replace("\n", " ").strip(),
                    "abstract":       abstract,
                    "authors":        authors,
                    "published_date": self._parse_date(entry.get("published", "")),
                    "url":            entry.get("link", f"https://arxiv.org/abs/{arxiv_id}"),
                    "categories":     [tag.get("term", "") for tag in entry.get("tags", [])],
                })

            except Exception as e:
                logger.error(f"Error parsing entry: {e}")

        logger.info(f"Parsed {len(parsed)} papers")
        return parsed

    async def store(self, parsed_data: List[Dict]) -> None:
        """Store papers and authors into Neo4j."""
        self.db.connect()

        for paper in parsed_data:
            try:
                self.db.create_paper({
                    "arxiv_id":       paper["arxiv_id"],
                    "title":          paper["title"],
                    "abstract":       paper["abstract"],
                    "published_date": paper["published_date"],
                    "url":            paper["url"],
                })
                for author_name in paper["authors"]:
                    if author_name.strip():
                        self.db.create_author(author_name.strip())
                        self.db.link_author_to_paper(author_name.strip(), paper["arxiv_id"])
            except Exception as e:
                logger.error(f"Error storing paper {paper.get('arxiv_id')}: {e}")

        self.db.close()
        logger.info(f"Stored {len(parsed_data)} papers to Neo4j")

    def _extract_arxiv_id(self, id_str: str) -> str:
        if "arXiv.org:" in id_str:
            return id_str.split("arXiv.org:")[-1].split("v")[0]
        if "/abs/" in id_str:
            return id_str.split("/abs/")[-1].split("v")[0]
        return ""

    def _parse_date(self, date_str: str) -> str:
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str).strftime("%Y-%m-%d")
        except Exception:
            return datetime.now().strftime("%Y-%m-%d")

    def _is_relevant(self, paper: dict) -> bool:
        return True


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl arXiv RSS feeds and store papers in Neo4j."
    )
    parser.add_argument(
        "--feeds", type=str, default=None,
        help="Comma-separated arXiv category codes to crawl, e.g. cs.CV,cs.LG,stat.ML"
    )
    parser.add_argument(
        "--max", type=int, default=200,
        help="Max papers per feed (default 200)"
    )
    parser.add_argument(
        "--list-feeds", action="store_true",
        help="Print all available arXiv categories and exit"
    )
    args = parser.parse_args()

    if args.list_feeds:
        print("\nAvailable arXiv categories:\n")
        for code, desc in sorted(ALL_FEEDS.items()):
            default_mark = "  ← default" if f"https://export.arxiv.org/rss/{code}" in ARXIV_FEEDS else ""
            print(f"  {code:<12}  {desc}{default_mark}")
        print()
        exit(0)

    feeds = _build_feed_urls(args.feeds.split(",")) if args.feeds else None

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-8s %(message)s")

    async def main():
        crawler = ArxivCrawler(max_papers_per_feed=args.max, feeds=feeds)
        result  = await crawler.run()
        print(f"\nDone — fetched: {result['fetched']}  stored: {result['stored']}\n")

    asyncio.run(main())