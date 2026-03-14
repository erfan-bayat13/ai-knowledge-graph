from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from typing import Optional
import logging
from kg.utils.config import get_settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self):
        settings = get_settings()
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self._driver = None

    def connect(self):
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully")
        except ServiceUnavailable as e:
            logger.error(f"Could not connect to Neo4j: {e}")
            raise

    def close(self):
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def run_query(self, query: str, parameters: dict = None):
        if not self._driver:
            self.connect()
        with self._driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    # ─── Schema Setup ────────────────────────────────────────────

    def setup_schema(self):
        """Create constraints and indexes."""
        constraints = [
            # Uniqueness constraints
            "CREATE CONSTRAINT paper_arxiv_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT methodology_name IF NOT EXISTS FOR (m:Methodology) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT repo_url IF NOT EXISTS FOR (r:Repository) REQUIRE r.url IS UNIQUE",
        ]
        indexes = [
            # Indexes for fast lookup
            "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX paper_published IF NOT EXISTS FOR (p:Paper) ON (p.published_date)",
            "CREATE INDEX author_affiliation IF NOT EXISTS FOR (a:Author) ON (a.affiliation)",
        ]
        for query in constraints + indexes:
            self.run_query(query)
        logger.info("Schema setup complete")

    # ─── Paper Operations ─────────────────────────────────────────

    def create_paper(self, paper: dict) -> bool:
        """Create or update a Paper node."""
        query = """
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        SET p.title = $title,
            p.abstract = $abstract,
            p.published_date = $published_date,
            p.url = $url,
            p.updated_at = timestamp()
        RETURN p.arxiv_id as id
        """
        result = self.run_query(query, paper)
        return len(result) > 0

    def create_author(self, name: str, affiliation: str = "") -> bool:
        """Create or update an Author node."""
        query = """
        MERGE (a:Author {name: $name})
        SET a.affiliation = $affiliation,
            a.updated_at = timestamp()
        RETURN a.name as name
        """
        result = self.run_query(query, {"name": name, "affiliation": affiliation})
        return len(result) > 0

    def link_author_to_paper(self, author_name: str, arxiv_id: str) -> bool:
        """Create WRITTEN_BY relationship."""
        query = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MATCH (a:Author {name: $author_name})
        MERGE (p)-[:WRITTEN_BY]->(a)
        RETURN p.arxiv_id as id
        """
        result = self.run_query(query, {
            "arxiv_id": arxiv_id,
            "author_name": author_name
        })
        return len(result) > 0

    def create_topic(self, name: str, category: str = "") -> bool:
        """Create or update a Topic node."""
        query = """
        MERGE (t:Topic {name: $name})
        SET t.category = $category,
            t.updated_at = timestamp()
        RETURN t.name as name
        """
        result = self.run_query(query, {"name": name, "category": category})
        return len(result) > 0

    def link_topic_to_paper(self, topic_name: str, arxiv_id: str) -> bool:
        """Create MENTIONS_TOPIC relationship."""
        query = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MATCH (t:Topic {name: $topic_name})
        MERGE (p)-[:MENTIONS_TOPIC]->(t)
        RETURN p.arxiv_id as id
        """
        result = self.run_query(query, {
            "arxiv_id": arxiv_id,
            "topic_name": topic_name
        })
        return len(result) > 0

    def create_methodology(self, name: str, description: str = "") -> bool:
        """Create or update a Methodology node."""
        query = """
        MERGE (m:Methodology {name: $name})
        SET m.description = $description,
            m.updated_at = timestamp()
        RETURN m.name as name
        """
        result = self.run_query(query, {"name": name, "description": description})
        return len(result) > 0

    def link_methodology_to_paper(self, methodology_name: str, arxiv_id: str) -> bool:
        """Create USES_METHODOLOGY relationship."""
        query = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MATCH (m:Methodology {name: $methodology_name})
        MERGE (p)-[:USES_METHODOLOGY]->(m)
        RETURN p.arxiv_id as id
        """
        result = self.run_query(query, {
            "arxiv_id": arxiv_id,
            "methodology_name": methodology_name
        })
        return len(result) > 0

    # ─── Repository Operations ────────────────────────────────────

    def create_repository(self, repo: dict) -> bool:
        """Create or update a Repository node."""
        query = """
        MERGE (r:Repository {url: $url})
        SET r.name = $name,
            r.description = $description,
            r.stars = $stars,
            r.language = $language,
            r.updated_at = timestamp()
        RETURN r.url as url
        """
        result = self.run_query(query, repo)
        return len(result) > 0

    def link_repo_to_paper(self, repo_url: str, arxiv_id: str) -> bool:
        """Create IMPLEMENTS_PAPER relationship."""
        query = """
        MATCH (r:Repository {url: $repo_url})
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MERGE (r)-[:IMPLEMENTS_PAPER]->(p)
        RETURN r.url as url
        """
        result = self.run_query(query, {
            "repo_url": repo_url,
            "arxiv_id": arxiv_id
        })
        return len(result) > 0

    # ─── Query Operations ─────────────────────────────────────────

    def search_papers(self, query_text: str, limit: int = 10) -> list:
        """Basic paper search by title keyword."""
        query = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($query_text)
           OR toLower(p.abstract) CONTAINS toLower($query_text)
        RETURN p.arxiv_id as arxiv_id,
               p.title as title,
               p.published_date as published_date,
               p.url as url
        ORDER BY p.published_date DESC
        LIMIT $limit
        """
        return self.run_query(query, {"query_text": query_text, "limit": limit})

    def get_paper_count(self) -> int:
        """Return total number of papers in graph."""
        result = self.run_query("MATCH (p:Paper) RETURN count(p) as count")
        return result[0]["count"] if result else 0