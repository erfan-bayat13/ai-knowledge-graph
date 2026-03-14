from abc import ABC, abstractmethod
from typing import List, Dict
import logging
from kg.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class BaseCrawler(ABC):
    """Base class for all crawlers."""

    def __init__(self):
        self.db = Neo4jClient()
        self.results: List[Dict] = []

    @abstractmethod
    async def fetch(self) -> List[Dict]:
        """Fetch raw data from source."""
        pass

    @abstractmethod
    async def parse(self, raw_data: List[Dict]) -> List[Dict]:
        """Parse raw data into structured format."""
        pass

    @abstractmethod
    async def store(self, parsed_data: List[Dict]) -> None:
        """Store parsed data into Neo4j."""
        pass

    async def run(self) -> Dict:
        """Execute full crawl pipeline."""
        logger.info(f"Starting {self.__class__.__name__}")

        raw = await self.fetch()
        logger.info(f"Fetched {len(raw)} raw items")

        parsed = await self.parse(raw)
        logger.info(f"Parsed {len(parsed)} items")

        await self.store(parsed)
        logger.info(f"Stored {len(parsed)} items to Neo4j")

        return {
            "crawler": self.__class__.__name__,
            "fetched": len(raw),
            "stored": len(parsed),
        }