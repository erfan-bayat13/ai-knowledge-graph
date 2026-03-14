import asyncio
import httpx
import logging
import re
from typing import List, Dict

from kg.crawlers.base import BaseCrawler
from kg.utils.config import get_settings

logger = logging.getLogger(__name__)

# Search queries for finding AI infrastructure repos
GITHUB_SEARCH_QUERIES = [
    # Original (topic-based, narrow)
    "llm-optimization",
    "distributed-training",
    "ai-infrastructure",
    "ml-inference",
    "transformer-optimization",
    "federated-learning",
    "model-compression",
    "gradient-compression",

    # Broader: key framework/system names
    "llm serving",
    "llm inference engine",
    "model quantization",
    "efficient transformers",
    "training framework",
    "gpu training",
    "deep learning optimization",
    "large language model",

    # Broader: common repo description words
    "vllm",
    "llama inference",
    "speculative decoding",
    "flash attention",
    "mixture of experts",
    "lora fine-tuning",
    "rlhf training",
    "knowledge distillation",
]

GITHUB_API_BASE = "https://api.github.com"


class GitHubCrawler(BaseCrawler):
    """Crawls GitHub for AI infrastructure repositories."""

    def __init__(self, max_repos_per_query: int = 30):
        super().__init__()
        settings = get_settings()
        self.token = settings.github_token
        self.max_repos_per_query = max_repos_per_query
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    async def fetch(self) -> List[Dict]:
        """Search GitHub API for relevant repos."""
        all_repos = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in GITHUB_SEARCH_QUERIES:
                try:
                    logger.info(f"Searching GitHub: {query}")
                    url = f"{GITHUB_API_BASE}/search/repositories"
                    params = {
                        "q": f"{query} language:python",
                        "sort": "stars",
                        "order": "desc",
                        "per_page": self.max_repos_per_query,
                    }

                    response = await client.get(
                        url,
                        headers=self.headers,
                        params=params
                    )
                    response.raise_for_status()
                    data = response.json()

                    repos = data.get("items", [])
                    all_repos.extend(repos)
                    logger.info(f"Found {len(repos)} repos for '{query}'")

                    # Respect GitHub rate limits
                    await asyncio.sleep(1.5)

                except Exception as e:
                    logger.error(f"Error fetching GitHub query '{query}': {e}")
                    continue

        return all_repos

    async def parse(self, raw_data: List[Dict]) -> List[Dict]:
        """Parse GitHub API responses into structured repo dicts."""
        parsed = []
        seen_urls = set()

        for repo in raw_data:
            try:
                url = repo.get("html_url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                # Extract arXiv links from description + README (if available)
                description = repo.get("description", "") or ""
                arxiv_ids = self._extract_arxiv_ids(description)

                parsed.append({
                    "name": repo.get("name", ""),
                    "url": url,
                    "description": description,
                    "stars": repo.get("stargazers_count", 0),
                    "language": repo.get("language", ""),
                    "topics": repo.get("topics", []),
                    "last_updated": repo.get("updated_at", ""),
                    "arxiv_ids": arxiv_ids,  # papers this repo implements
                })

            except Exception as e:
                logger.error(f"Error parsing repo: {e}")
                continue

        # Sort by stars
        parsed.sort(key=lambda x: x["stars"], reverse=True)
        logger.info(f"Parsed {len(parsed)} unique repos")
        return parsed

    async def store(self, parsed_data: List[Dict]) -> None:
        """Store repositories into Neo4j and link to papers."""
        self.db.connect()

        for repo in parsed_data:
            try:
                # Create repository node
                self.db.create_repository({
                    "name": repo["name"],
                    "url": repo["url"],
                    "description": repo["description"],
                    "stars": repo["stars"],
                    "language": repo["language"],
                })

                # Link to papers if arxiv IDs found in description
                for arxiv_id in repo["arxiv_ids"]:
                    # Only link if paper exists in graph
                    existing = self.db.run_query(
                        "MATCH (p:Paper {arxiv_id: $id}) RETURN p.arxiv_id as id",
                        {"id": arxiv_id}
                    )
                    if existing:
                        self.db.link_repo_to_paper(repo["url"], arxiv_id)
                        logger.info(f"Linked {repo['name']} → {arxiv_id}")

            except Exception as e:
                logger.error(f"Error storing repo {repo.get('name')}: {e}")
                continue

        self.db.close()
        logger.info(f"Stored {len(parsed_data)} repos to Neo4j")

    def _extract_arxiv_ids(self, text: str) -> List[str]:
        """Extract arXiv IDs from text using regex."""
        if not text:
            return []
        # Matches patterns like 2106.05233 or arxiv:2106.05233
        pattern = r"(?:arxiv[:\s])?(\d{4}\.\d{4,5})"
        matches = re.findall(pattern, text.lower())
        return list(set(matches))


if __name__ == "__main__":
    async def main():
        crawler = GitHubCrawler(max_repos_per_query=10)
        result = await crawler.run()
        print(f"Done: {result}")

    asyncio.run(main())