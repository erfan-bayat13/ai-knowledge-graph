# kg/crawlers/github.py
#
# Changes from old version:
#   - fetch_readme() added — scans full README, not just description
#   - Repo node uses new schema (Repo not Repository, topics stored as list)
#   - README scan runs as a separate step so it doesn't slow the initial crawl
#   - _extract_arxiv_ids works on any text (description OR readme)

import asyncio
import re
import logging
from typing import List, Dict, Optional

import httpx

from kg.crawlers.base import BaseCrawler
from kg.graph.neo4j_client import Neo4jClient
from kg.utils.config import get_settings

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"

GITHUB_SEARCH_QUERIES = [
    "distributed training deep learning",
    "llm inference optimization",
    "gradient compression federated learning",
    "model parallelism pipeline",
    "vllm llm serving",
    "flashattention transformer",
    "lora fine-tuning",
    "mixture of experts moe",
]


class GitHubCrawler(BaseCrawler):
    def __init__(self, max_repos_per_query: int = 30):
        super().__init__()
        settings = get_settings()
        self.token = settings.github_token
        self.max_repos_per_query = max_repos_per_query
        self.db = Neo4jClient()
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    # ── Fetch ─────────────────────────────────────────────────────────────────

    async def fetch(self) -> List[Dict]:
        """Search GitHub API for relevant repos."""
        all_repos = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in GITHUB_SEARCH_QUERIES:
                try:
                    logger.info(f"GitHub search: {query!r}")
                    response = await client.get(
                        f"{GITHUB_API_BASE}/search/repositories",
                        headers=self.headers,
                        params={
                            "q":        f"{query} language:python",
                            "sort":     "stars",
                            "order":    "desc",
                            "per_page": self.max_repos_per_query,
                        },
                    )
                    response.raise_for_status()
                    repos = response.json().get("items", [])
                    all_repos.extend(repos)
                    logger.info(f"  → {len(repos)} repos")
                    await asyncio.sleep(1.5)   # GitHub rate limit
                except Exception as e:
                    logger.error(f"GitHub fetch error for {query!r}: {e}")
                    continue

        return all_repos

    async def fetch_readme(self, owner: str, repo_name: str) -> Optional[str]:
        """
        Fetch the raw README text for a repo.
        Returns None if the README doesn't exist or the request fails.
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                # GitHub API returns base64-encoded content
                response = await client.get(
                    f"{GITHUB_API_BASE}/repos/{owner}/{repo_name}/readme",
                    headers=self.headers,
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                data = response.json()

                # Decode base64 content
                import base64
                content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
                return content

            except Exception as e:
                logger.debug(f"README fetch failed for {owner}/{repo_name}: {e}")
                return None

    # ── Parse ─────────────────────────────────────────────────────────────────

    async def parse(self, raw_data: List[Dict]) -> List[Dict]:
        """Parse GitHub API response into clean repo dicts."""
        parsed = []
        seen_urls = set()

        for repo in raw_data:
            try:
                url = repo.get("html_url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                description = repo.get("description") or ""
                # Scan description for arXiv IDs (quick, no extra request)
                arxiv_ids = self._extract_arxiv_ids(description)

                parsed.append({
                    "name":         repo.get("name", ""),
                    "url":          url,
                    "owner":        repo.get("owner", {}).get("login", ""),
                    "description":  description,
                    "stars":        repo.get("stargazers_count", 0),
                    "language":     repo.get("language") or "",
                    "topics":       repo.get("topics") or [],
                    "last_updated": repo.get("updated_at") or "",
                    "arxiv_ids":    arxiv_ids,
                })

            except Exception as e:
                logger.error(f"Parse error: {e}")
                continue

        parsed.sort(key=lambda x: x["stars"], reverse=True)
        logger.info(f"Parsed {len(parsed)} unique repos")
        return parsed

    # ── Store ─────────────────────────────────────────────────────────────────

    async def store(self, parsed_data: List[Dict]) -> None:
        """Store repos into Neo4j. Links to papers only when arXiv IDs confirmed."""
        self.db.connect()

        for repo in parsed_data:
            try:
                self.db.create_repo({
                    "name":        repo["name"],
                    "url":         repo["url"],
                    "description": repo["description"],
                    "stars":       repo["stars"],
                    "language":    repo["language"],
                    "topics":      repo["topics"],
                })

                # Link from description arXiv IDs (already parsed)
                for arxiv_id in repo["arxiv_ids"]:
                    existing = self.db.run_query(
                        "MATCH (p:Paper {arxiv_id: $id}) RETURN p.arxiv_id AS id",
                        {"id": arxiv_id},
                    )
                    if existing:
                        self.db.link_repo_implements_paper(repo["url"], arxiv_id)
                        logger.info(f"Linked (desc) {repo['name']} → {arxiv_id}")

            except Exception as e:
                logger.error(f"Store error for {repo.get('name')}: {e}")
                continue

        self.db.close()
        logger.info(f"Stored {len(parsed_data)} repos")

    async def scan_readmes(self, limit: int = 100) -> None:
        """
        Second-pass: fetch README for repos not yet scanned,
        extract arXiv IDs, create IMPLEMENTS edges.

        Run this after store() — it's intentionally separate so the
        initial crawl stays fast and README fetching can be retried
        independently.
        """
        self.db.connect()
        repos = self.db.get_repos_without_readme_scan(limit=limit)
        logger.info(f"README scan: {len(repos)} repos to process")

        for repo in repos:
            repo_url  = repo["url"]
            repo_name = repo["name"]

            # Parse owner/name from URL: https://github.com/owner/name
            parts = repo_url.rstrip("/").split("/")
            if len(parts) < 2:
                self.db.mark_repo_readme_scanned(repo_url)
                continue

            owner     = parts[-2]
            name_only = parts[-1]

            readme = await self.fetch_readme(owner, name_only)
            await asyncio.sleep(0.5)  # be polite

            if readme:
                arxiv_ids = self._extract_arxiv_ids(readme)
                for arxiv_id in arxiv_ids:
                    existing = self.db.run_query(
                        "MATCH (p:Paper {arxiv_id: $id}) RETURN p.arxiv_id AS id",
                        {"id": arxiv_id},
                    )
                    if existing:
                        self.db.link_repo_implements_paper(repo_url, arxiv_id)
                        logger.info(f"Linked (readme) {repo_name} → {arxiv_id}")

            self.db.mark_repo_readme_scanned(repo_url)

        self.db.close()
        logger.info("README scan complete")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_arxiv_ids(self, text: Optional[str]) -> List[str]:
        """
        Extract arXiv IDs from any text.
        Matches:
          - 2106.05233
          - arxiv:2106.05233
          - arxiv.org/abs/2106.05233
          - https://arxiv.org/pdf/2106.05233
        """
        if not text:
            return []
        pattern = r"(?:arxiv\.org/(?:abs|pdf)/|arxiv[:\s])?(\d{4}\.\d{4,5})(?:v\d+)?"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(dict.fromkeys(matches))   # deduplicate, preserve order

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self) -> dict:
        raw    = await self.fetch()
        parsed = await self.parse(raw)
        await self.store(parsed)
        await self.scan_readmes(limit=200)
        return {"repos_stored": len(parsed)}


if __name__ == "__main__":
    async def main():
        crawler = GitHubCrawler(max_repos_per_query=100)
        result  = await crawler.run()
        print(f"Done: {result}")
    asyncio.run(main())