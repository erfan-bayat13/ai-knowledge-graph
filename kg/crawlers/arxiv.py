import asyncio
import httpx
import feedparser
import logging
from typing import List, Dict
from datetime import datetime

from kg.crawlers.base import BaseCrawler

logger = logging.getLogger(__name__)

# arXiv RSS feeds to crawl
ARXIV_FEEDS = [
    "http://export.arxiv.org/rss/cs.LG",
    "http://export.arxiv.org/rss/cs.DC",
    "http://export.arxiv.org/rss/cs.AI",
    "http://export.arxiv.org/rss/cs.NE",
    "http://export.arxiv.org/rss/cs.CV",   # Computer Vision
    "http://export.arxiv.org/rss/cs.CL",   # Computation & Language
    "http://export.arxiv.org/rss/cs.CR",   # Cryptography & Security
    "http://export.arxiv.org/rss/cs.SE",   # Software Engineering
    "http://export.arxiv.org/rss/stat.ML", # Statistics ML
]

# Keywords to filter infrastructure-relevant papers
INFRASTRUCTURE_KEYWORDS = [
    # Training & Optimization
    "distributed", "training", "inference", "optimization",
    "compression", "quantization", "pruning", "federated",
    "gradient", "parallelism", "gpu", "hardware", "efficient",
    "scalable", "deployment", "serving", "pipeline",
    
    # Models & Architectures
    "transformer", "attention", "llm", "foundation model",
    "large language", "mixture of experts", "moe", "lora",
    "fine-tun", "pre-train", "distill",
    
    # Systems & Infrastructure
    "cluster", "hpc", "collective communication", "nccl",
    "multimodal", "agentic", "orchestration", "container",
    "thousand-gpu", "multi-gpu", "multi-node", "neuromorphic",
    "surrogate", "sparse", "routing", "decentralized",
    
    # General ML Systems
    "reinforcement learning", "model editing", "unlearning",
    "benchmark", "evaluation", "agent", "tool use",
]


class ArxivCrawler(BaseCrawler):
    """Crawls arXiv RSS feeds for AI infrastructure papers."""

    def __init__(self, max_papers_per_feed: int = 100):
        super().__init__()
        self.max_papers_per_feed = max_papers_per_feed

    async def fetch(self) -> List[Dict]:
        """Fetch papers from all arXiv RSS feeds."""
        all_papers = []

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for feed_url in ARXIV_FEEDS:
                try:
                    logger.info(f"Fetching feed: {feed_url}")
                    response = await client.get(feed_url)
                    response.raise_for_status()

                    feed = feedparser.parse(response.text)
                    papers = feed.entries[:self.max_papers_per_feed]
                    all_papers.extend([
                        {"raw": entry, "feed_url": feed_url}
                        for entry in papers
                    ])
                    logger.info(f"Got {len(papers)} papers from {feed_url}")

                except Exception as e:
                    logger.error(f"Error fetching {feed_url}: {e}")
                    continue

        return all_papers

    async def parse(self, raw_data: List[Dict]) -> List[Dict]:
        """Parse arXiv feed entries into structured paper dicts."""
        parsed = []
        seen_ids = set()

        for item in raw_data:
            entry = item["raw"]

            try:
                # Extract arxiv ID from URL
                arxiv_id = self._extract_arxiv_id(entry.get("id", ""))
                if not arxiv_id or arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                # Extract authors
                # Extract authors (arXiv returns them as comma-separated string in one dict)
                authors = []
                if hasattr(entry, "authors") and entry.authors:
                    raw_authors = entry.authors[0].get("name", "")
                    authors = [a.strip() for a in raw_authors.split(",") if a.strip()]
                elif hasattr(entry, "author"):
                    authors = [a.strip() for a in entry.author.split(",") if a.strip()]

                # Clean abstract
                abstract = entry.get("summary", "")
                abstract = abstract.replace("\n", " ").strip()

                # Parse date
                published_date = self._parse_date(entry.get("published", ""))

                paper = {
                    "arxiv_id": arxiv_id,
                    "title": entry.get("title", "").replace("\n", " ").strip(),
                    "abstract": abstract,
                    "authors": authors,
                    "published_date": published_date,
                    "url": entry.get("link", f"https://arxiv.org/abs/{arxiv_id}"),
                    "categories": [tag.get("term", "") for tag in entry.get("tags", [])],
                }

                # Filter: only keep infrastructure-relevant papers
                if self._is_relevant(paper):
                    parsed.append(paper)

            except Exception as e:
                logger.error(f"Error parsing entry: {e}")
                continue

        # logger.info(f"Filtered to {len(parsed)} relevant papers")
        logger.info(f"Parsed {len(parsed)} papers")
        return parsed

    async def store(self, parsed_data: List[Dict]) -> None:
        """Store papers and authors into Neo4j."""
        self.db.connect()

        for paper in parsed_data:
            try:
                # Create paper node
                self.db.create_paper({
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "published_date": paper["published_date"],
                    "url": paper["url"],
                })

                # Create author nodes + relationships
                for author_name in paper["authors"]:
                    if author_name.strip():
                        self.db.create_author(author_name.strip())
                        self.db.link_author_to_paper(
                            author_name.strip(),
                            paper["arxiv_id"]
                        )

            except Exception as e:
                logger.error(f"Error storing paper {paper.get('arxiv_id')}: {e}")
                continue

        self.db.close()
        logger.info(f"Stored {len(parsed_data)} papers to Neo4j")

    def _extract_arxiv_id(self, id_str: str) -> str:
        """Extract arxiv ID from oai:arXiv.org:2603.11049v1 or URL format."""
        try:
            # Handle oai format: oai:arXiv.org:2603.11049v1
            if "arXiv.org:" in id_str:
                arxiv_id = id_str.split("arXiv.org:")[-1]
                arxiv_id = arxiv_id.split("v")[0]  # Remove version
                return arxiv_id
            # Handle URL format: https://arxiv.org/abs/2603.11049
            if "/abs/" in id_str:
                arxiv_id = id_str.split("/abs/")[-1]
                arxiv_id = arxiv_id.split("v")[0]
                return arxiv_id
        except Exception:
            pass
        return ""

    def _parse_date(self, date_str: str) -> str:
        """Parse date string to ISO format."""
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return datetime.now().strftime("%Y-%m-%d")

    # def _is_relevant(self, paper: dict) -> bool:
    #     """Filter papers to infrastructure-relevant ones."""
    #     text = (paper["title"] + " " + paper["abstract"]).lower()
    #     return any(keyword in text for keyword in INFRASTRUCTURE_KEYWORDS)

    ## temp just to store the max
    def _is_relevant(self, paper: dict) -> bool:
        """Store all papers (no filtering at crawl stage)."""
        return True


if __name__ == "__main__":
    async def main():
        crawler = ArxivCrawler(max_papers_per_feed=100)
        result = await crawler.run()
        print(f"Done: {result}")

    asyncio.run(main())