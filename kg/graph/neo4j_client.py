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
    
    # ─── Method / Dataset / Task Node Operations ──────────────────────────────

    def create_method(self, name: str) -> bool:
        """Create or update a Method node."""
        query = """
        MERGE (m:Method {name: $name})
        SET m.updated_at = timestamp()
        RETURN m.name as name
        """
        result = self.run_query(query, {"name": name})
        return len(result) > 0

    def create_dataset(self, name: str) -> bool:
        """Create or update a Dataset node."""
        query = """
        MERGE (d:Dataset {name: $name})
        SET d.updated_at = timestamp()
        RETURN d.name as name
        """
        result = self.run_query(query, {"name": name})
        return len(result) > 0

    def create_task(self, name: str) -> bool:
        """Create or update a Task node."""
        query = """
        MERGE (t:Task {name: $name})
        SET t.updated_at = timestamp()
        RETURN t.name as name
        """
        result = self.run_query(query, {"name": name})
        return len(result) > 0

    # ─── Enrichment Relationship Operations ───────────────────────────────────

    def link_paper_proposes_method(self, arxiv_id: str, method_name: str) -> bool:
        """Create PROPOSES relationship: Paper → Method."""
        query = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MATCH (m:Method {name: $method_name})
        MERGE (p)-[:PROPOSES]->(m)
        RETURN p.arxiv_id as id
        """
        result = self.run_query(query, {
            "arxiv_id": arxiv_id,
            "method_name": method_name,
        })
        return len(result) > 0

    def link_paper_uses_method(self, arxiv_id: str, method_name: str) -> bool:
        """Create USES relationship: Paper → Method."""
        query = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MATCH (m:Method {name: $method_name})
        MERGE (p)-[:USES]->(m)
        RETURN p.arxiv_id as id
        """
        result = self.run_query(query, {
            "arxiv_id": arxiv_id,
            "method_name": method_name,
        })
        return len(result) > 0

    def link_paper_evaluated_on(self, arxiv_id: str, dataset_name: str) -> bool:
        """Create EVALUATED_ON relationship: Paper → Dataset."""
        query = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MATCH (d:Dataset {name: $dataset_name})
        MERGE (p)-[:EVALUATED_ON]->(d)
        RETURN p.arxiv_id as id
        """
        result = self.run_query(query, {
            "arxiv_id": arxiv_id,
            "dataset_name": dataset_name,
        })
        return len(result) > 0

    def link_paper_addresses_task(self, arxiv_id: str, task_name: str) -> bool:
        """Create ADDRESSES relationship: Paper → Task."""
        query = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MATCH (t:Task {name: $task_name})
        MERGE (p)-[:ADDRESSES]->(t)
        RETURN p.arxiv_id as id
        """
        result = self.run_query(query, {
            "arxiv_id": arxiv_id,
            "task_name": task_name,
        })
        return len(result) > 0

    def link_method_improves_method(
        self, method_name: str, improves_on: str
    ) -> bool:
        """Create IMPROVES relationship: Method → Method."""
        query = """
        MATCH (m1:Method {name: $method_name})
        MATCH (m2:Method {name: $improves_on})
        MERGE (m1)-[:IMPROVES]->(m2)
        RETURN m1.name as name
        """
        result = self.run_query(query, {
            "method_name": method_name,
            "improves_on": improves_on,
        })
        return len(result) > 0

    # ─── Enrichment Query Helpers ─────────────────────────────────────────────

    def get_all_papers(self, batch_size: int = 100, skip: int = 0) -> list:
        """
        Fetch papers in batches for enrichment runner.
        Returns arxiv_id + abstract only — that's all enrichment needs.
        """
        query = """
        MATCH (p:Paper)
        WHERE p.abstract IS NOT NULL AND p.abstract <> ''
        RETURN p.arxiv_id as arxiv_id,
               p.abstract  as abstract,
               p.title     as title
        ORDER BY p.arxiv_id
        SKIP $skip
        LIMIT $batch_size
        """
        return self.run_query(query, {
            "skip": skip,
            "batch_size": batch_size,
        })

    def get_enriched_paper_ids(self) -> set:
        """
        Return set of arxiv_ids that already have at least one
        PROPOSES, USES, EVALUATED_ON, or ADDRESSES relationship.
        Used by enrichment runner to skip already-processed papers.
        """
        query = """
        MATCH (p:Paper)
        WHERE (p)-[:PROPOSES]->() OR
              (p)-[:USES]->()     OR
              (p)-[:EVALUATED_ON]->() OR
              (p)-[:ADDRESSES]->()
        RETURN p.arxiv_id as arxiv_id
        """
        results = self.run_query(query)
        return {r["arxiv_id"] for r in results}

    def get_paper_count(self) -> int:
        """Return total number of papers in graph."""
        result = self.run_query("MATCH (p:Paper) RETURN count(p) as count")
        return result[0]["count"] if result else 0

    def setup_enrichment_schema(self):
        """Add constraints for new node types."""
        constraints = [
            "CREATE CONSTRAINT method_name  IF NOT EXISTS FOR (m:Method)  REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT task_name    IF NOT EXISTS FOR (t:Task)    REQUIRE t.name IS UNIQUE",
        ]
        for query in constraints:
            self.run_query(query)
        logger.info("Enrichment schema setup complete")