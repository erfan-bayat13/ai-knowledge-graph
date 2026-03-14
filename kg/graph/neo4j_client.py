# kg/graph/neo4j_client.py
#
# Clean schema — replaces the old Method/Dataset/Task/enrichment mess.
#
# Nodes:    Paper, Repo, Author, Chunk
# Edges:    WRITTEN_BY, IMPLEMENTS, SIMILAR_TO, USES, EVALUATED_ON, PROPOSES

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from typing import List, Optional
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
                self.uri, auth=(self.user, self.password)
            )
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j")
        except ServiceUnavailable as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise

    def close(self):
        if self._driver:
            self._driver.close()

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

    # ── Schema ────────────────────────────────────────────────────────────────

    def setup_schema(self):
        """Create constraints and indexes for the clean schema."""
        constraints = [
            "CREATE CONSTRAINT paper_arxiv_id  IF NOT EXISTS FOR (p:Paper)  REQUIRE p.arxiv_id IS UNIQUE",
            "CREATE CONSTRAINT author_name      IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT repo_url         IF NOT EXISTS FOR (r:Repo)   REQUIRE r.url IS UNIQUE",
            "CREATE CONSTRAINT method_name      IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT dataset_name     IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX paper_title     IF NOT EXISTS FOR (p:Paper)  ON (p.title)",
            "CREATE INDEX paper_published IF NOT EXISTS FOR (p:Paper)  ON (p.published_date)",
            "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
        ]
        for q in constraints + indexes:
            try:
                self.run_query(q)
            except Exception as e:
                logger.debug(f"Schema query skipped (likely exists): {e}")
        logger.info("Schema setup complete")

    def nuke_and_reset(self):
        """Delete everything and rebuild schema. Use during development only."""
        self.run_query("MATCH (n) DETACH DELETE n")
        logger.warning("All nodes and edges deleted")
        self.setup_schema()

    # ── Paper ─────────────────────────────────────────────────────────────────

    def create_paper(self, paper: dict) -> bool:
        query = """
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        SET p.title          = $title,
            p.abstract       = $abstract,
            p.published_date = $published_date,
            p.url            = $url,
            p.updated_at     = timestamp()
        RETURN p.arxiv_id AS id
        """
        return len(self.run_query(query, paper)) > 0

    def get_all_papers(self, batch_size: int = 100, skip: int = 0) -> List[dict]:
        query = """
        MATCH (p:Paper)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.abstract AS abstract
        ORDER BY p.arxiv_id
        SKIP $skip LIMIT $batch_size
        """
        return self.run_query(query, {"skip": skip, "batch_size": batch_size})

    def get_paper_count(self) -> int:
        result = self.run_query("MATCH (p:Paper) RETURN count(p) AS count")
        return result[0]["count"] if result else 0

    def get_papers_without_chunks(self, limit: int = 500) -> List[dict]:
        """Return papers that haven't been embedded yet."""
        query = """
        MATCH (p:Paper)
        WHERE NOT (p)-[:HAS_CHUNK]->(:Chunk)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.abstract AS abstract
        LIMIT $limit
        """
        return self.run_query(query, {"limit": limit})

    # ── Author ────────────────────────────────────────────────────────────────

    def create_author(self, name: str) -> bool:
        query = """
        MERGE (a:Author {name: $name})
        SET a.updated_at = timestamp()
        RETURN a.name AS name
        """
        return len(self.run_query(query, {"name": name})) > 0

    def link_author_to_paper(self, author_name: str, arxiv_id: str) -> bool:
        query = """
        MATCH (p:Paper  {arxiv_id: $arxiv_id})
        MATCH (a:Author {name: $author_name})
        MERGE (p)-[:WRITTEN_BY]->(a)
        RETURN p.arxiv_id AS id
        """
        return len(self.run_query(query, {"arxiv_id": arxiv_id, "author_name": author_name})) > 0

    # ── Repo ──────────────────────────────────────────────────────────────────

    def create_repo(self, repo: dict) -> bool:
        query = """
        MERGE (r:Repo {url: $url})
        SET r.name        = $name,
            r.description = $description,
            r.stars       = $stars,
            r.language    = $language,
            r.topics      = $topics,
            r.updated_at  = timestamp()
        RETURN r.url AS url
        """
        return len(self.run_query(query, repo)) > 0

    def link_repo_implements_paper(self, repo_url: str, arxiv_id: str) -> bool:
        """Repo → IMPLEMENTS → Paper (high confidence: arXiv ID found in README)."""
        query = """
        MATCH (r:Repo  {url: $repo_url})
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MERGE (r)-[:IMPLEMENTS]->(p)
        RETURN r.url AS url
        """
        return len(self.run_query(query, {"repo_url": repo_url, "arxiv_id": arxiv_id})) > 0

    def get_repos_without_readme_scan(self, limit: int = 200) -> List[dict]:
        """Return repos where we haven't yet fetched the README."""
        query = """
        MATCH (r:Repo)
        WHERE r.readme_scanned IS NULL
        RETURN r.url AS url, r.name AS name
        LIMIT $limit
        """
        return self.run_query(query, {"limit": limit})

    def mark_repo_readme_scanned(self, repo_url: str) -> bool:
        query = """
        MATCH (r:Repo {url: $url})
        SET r.readme_scanned = true
        RETURN r.url AS url
        """
        return len(self.run_query(query, {"url": repo_url})) > 0

    # ── Method / Dataset (spaCy extracted, high confidence only) ─────────────

    def create_method(self, name: str) -> bool:
        query = """
        MERGE (m:Method {name: $name})
        SET m.updated_at = timestamp()
        RETURN m.name AS name
        """
        return len(self.run_query(query, {"name": name})) > 0

    def create_dataset(self, name: str) -> bool:
        query = """
        MERGE (d:Dataset {name: $name})
        SET d.updated_at = timestamp()
        RETURN d.name AS name
        """
        return len(self.run_query(query, {"name": name})) > 0

    def link_paper_proposes(self, arxiv_id: str, method_name: str) -> bool:
        query = """
        MATCH (p:Paper  {arxiv_id: $arxiv_id})
        MATCH (m:Method {name: $method_name})
        MERGE (p)-[:PROPOSES]->(m)
        RETURN p.arxiv_id AS id
        """
        return len(self.run_query(query, {"arxiv_id": arxiv_id, "method_name": method_name})) > 0

    def link_paper_uses(self, arxiv_id: str, method_name: str) -> bool:
        query = """
        MATCH (p:Paper  {arxiv_id: $arxiv_id})
        MATCH (m:Method {name: $method_name})
        MERGE (p)-[:USES]->(m)
        RETURN p.arxiv_id AS id
        """
        return len(self.run_query(query, {"arxiv_id": arxiv_id, "method_name": method_name})) > 0

    def link_paper_evaluated_on(self, arxiv_id: str, dataset_name: str) -> bool:
        query = """
        MATCH (p:Paper   {arxiv_id: $arxiv_id})
        MATCH (d:Dataset {name: $dataset_name})
        MERGE (p)-[:EVALUATED_ON]->(d)
        RETURN p.arxiv_id AS id
        """
        return len(self.run_query(query, {"arxiv_id": arxiv_id, "dataset_name": dataset_name})) > 0

    # ── Chunks + Embeddings ───────────────────────────────────────────────────

    def create_chunk(self, arxiv_id: str, chunk_index: int, text: str, embedding: List[float]) -> bool:
        chunk_id = f"{arxiv_id}_{chunk_index}"
        
        # Step 1: create the chunk node
        self.run_query("""
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text        = $text,
                c.chunk_index = $chunk_index,
                c.embedding   = $embedding,
                c.updated_at  = timestamp()
        """, {
            "chunk_id":    chunk_id,
            "text":        text,
            "chunk_index": chunk_index,
            "embedding":   embedding,
        })

        # Step 2: link to paper separately
        result = self.run_query("""
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (p)-[:HAS_CHUNK]->(c)
            RETURN c.id AS id
        """, {
            "arxiv_id": arxiv_id,
            "chunk_id": chunk_id,
        })
        
        return len(result) > 0

    def link_similar_chunks(self, chunk_id_a: str, chunk_id_b: str, score: float) -> bool:
        """Create SIMILAR_TO edge between two chunks."""
        query = """
        MATCH (a:Chunk {id: $chunk_id_a})
        MATCH (b:Chunk {id: $chunk_id_b})
        MERGE (a)-[s:SIMILAR_TO]->(b)
        SET s.score = $score
        RETURN a.id AS id
        """
        return len(self.run_query(query, {
            "chunk_id_a": chunk_id_a,
            "chunk_id_b": chunk_id_b,
            "score":      score,
        })) > 0

    def get_all_chunks_with_embeddings(self) -> List[dict]:
        """Return all chunks that have embeddings — used for similarity computation."""
        query = """
        MATCH (p:Paper)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.embedding IS NOT NULL
        RETURN c.id AS chunk_id, c.embedding AS embedding, p.arxiv_id AS arxiv_id
        """
        return self.run_query(query)

    # ── Search ────────────────────────────────────────────────────────────────

    def search_papers(self, query_text: str, limit: int = 10) -> List[dict]:
        query = """
        MATCH (p:Paper)
        WHERE toLower(p.title)    CONTAINS toLower($query_text)
           OR toLower(p.abstract) CONTAINS toLower($query_text)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.published_date AS published_date, p.url AS url
        ORDER BY p.published_date DESC
        LIMIT $limit
        """
        return self.run_query(query, {"query_text": query_text, "limit": limit})

    def get_paper_neighbourhood(self, arxiv_id: str, depth: int = 2) -> List[dict]:
        """
        Return all nodes reachable from a paper within `depth` hops.
        Used for the research path visualization.
        """
        query = """
        MATCH path = (start:Paper {arxiv_id: $arxiv_id})-[*1..$depth]-(neighbour)
        RETURN [node IN nodes(path) | {labels: labels(node), props: properties(node)}] AS nodes,
               [rel  IN relationships(path) | type(rel)] AS rel_types
        LIMIT 200
        """
        return self.run_query(query, {"arxiv_id": arxiv_id, "depth": depth})