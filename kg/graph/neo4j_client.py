# kg/graph/neo4j_client.py
# CHANGED (Phase 1): added Institution node, CITES edge, rank_score property on Paper
# CHANGED (Phase 2): replaced Chunk-based vector index with Paper-level embedding index (768-dim cosine)
# CHANGED (Phase 3): added Topic node, BELONGS_TO edge, trend_score on Topic
# CHANGED: removed Chunk/HAS_CHUNK/SIMILAR_TO — embeddings now live on Paper, not Chunk nodes
#
# Target schema (from design doc):
#   Nodes:    Paper, Author, Topic, Institution
#   Edges:    WRITTEN_BY, CITES, BELONGS_TO, AFFILIATED_WITH

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from typing import List, Optional
import logging
from kg.utils.config import get_settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self):
        settings = get_settings()
        self.uri      = settings.neo4j_uri
        self.user     = settings.neo4j_user
        self.password = settings.neo4j_password
        self._driver  = None

    def connect(self):
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
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

    # ── Schema ──────────────────────────────────────────────────────────────────

    def setup_schema(self):
        """Create all constraints and indexes for the unified design schema."""
        constraints = [
            # Paper uniqueness
            "CREATE CONSTRAINT paper_arxiv_id   IF NOT EXISTS FOR (p:Paper)       REQUIRE p.arxiv_id IS UNIQUE",
            # Author uniqueness
            "CREATE CONSTRAINT author_name       IF NOT EXISTS FOR (a:Author)      REQUIRE a.name IS UNIQUE",
            # Topic uniqueness (Phase 3)
            "CREATE CONSTRAINT topic_name        IF NOT EXISTS FOR (t:Topic)       REQUIRE t.name IS UNIQUE",
            # Institution uniqueness (Phase 1)
            "CREATE CONSTRAINT institution_name  IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE",
        ]
        indexes = [
            # Property indexes for fast lookups
            "CREATE INDEX paper_published  IF NOT EXISTS FOR (p:Paper) ON (p.published_date)",
            "CREATE INDEX paper_rank       IF NOT EXISTS FOR (p:Paper) ON (p.rank_score)",
            # Paper-level vector index (Phase 2) — replaces old Chunk chunk_embedding index
            "CREATE VECTOR INDEX paper_embedding IF NOT EXISTS FOR (p:Paper) ON (p.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}",
        ]
        for q in constraints + indexes:
            try:
                self.run_query(q)
            except Exception as e:
                logger.debug(f"Schema query skipped (likely exists): {e}")
        logger.info("Schema setup complete")

    def nuke_and_reset(self):
        """Delete everything and rebuild schema. Development use only."""
        self.run_query("MATCH (n) DETACH DELETE n")
        logger.warning("All nodes and edges deleted")
        self.setup_schema()

    # ── Paper ───────────────────────────────────────────────────────────────────

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
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.abstract AS abstract,
               p.published_date AS published_date
        ORDER BY p.arxiv_id
        SKIP $skip LIMIT $batch_size
        """
        return self.run_query(query, {"skip": skip, "batch_size": batch_size})

    def get_paper_count(self) -> int:
        result = self.run_query("MATCH (p:Paper) RETURN count(p) AS count")
        return result[0]["count"] if result else 0

    def get_papers_without_embedding(self, limit: int = 500) -> List[dict]:
        """Return papers that don't yet have a paper-level embedding (Phase 2)."""
        query = """
        MATCH (p:Paper)
        WHERE p.embedding IS NULL
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.abstract AS abstract
        LIMIT $limit
        """
        return self.run_query(query, {"limit": limit})

    def set_paper_embedding(self, arxiv_id: str, embedding: List[float]) -> bool:
        """Store a 768-dim SPECTER2 embedding on a Paper node (Phase 2)."""
        result = self.run_query("""
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            SET p.embedding = $embedding
            RETURN p.arxiv_id AS id
        """, {"arxiv_id": arxiv_id, "embedding": embedding})
        return len(result) > 0

    def get_all_papers_with_embeddings(self) -> List[dict]:
        """Return all papers that have embeddings — used for clustering (Phase 3)."""
        query = """
        MATCH (p:Paper)
        WHERE p.embedding IS NOT NULL
        RETURN p.arxiv_id AS arxiv_id, p.title AS title,
               p.embedding AS embedding, p.cluster_id AS cluster_id,
               p.published_date AS published_date, p.rank_score AS rank_score
        """
        return self.run_query(query)

    def vector_search_papers(self, query_vector: List[float], top_k: int = 10) -> List[dict]:
        """Semantic search: find papers nearest to a query vector (Phase 2)."""
        query = """
        CALL db.index.vector.queryNodes('paper_embedding', $top_k, $query_vector)
        YIELD node, score
        RETURN node.arxiv_id AS arxiv_id, node.title AS title,
               node.published_date AS published_date, score
        ORDER BY score DESC
        """
        return self.run_query(query, {"top_k": top_k, "query_vector": query_vector})

    # ── Author ──────────────────────────────────────────────────────────────────

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

    # ── Topic (Phase 3) ─────────────────────────────────────────────────────────

    def create_or_update_topic(self, name: str, trend_score: float = 0.0, paper_count: int = 0) -> bool:
        """Upsert a Topic node with trend_score and paper_count."""
        result = self.run_query("""
            MERGE (t:Topic {name: $name})
            SET t.trend_score = $trend_score,
                t.paper_count = $paper_count,
                t.updated_at  = timestamp()
            RETURN t.name AS name
        """, {"name": name, "trend_score": trend_score, "paper_count": paper_count})
        return len(result) > 0

    def link_paper_to_topic(self, arxiv_id: str, topic_name: str, cluster_id: int) -> bool:
        """BELONGS_TO edge from Paper to Topic, also set cluster_id on Paper."""
        result = self.run_query("""
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MATCH (t:Topic {name: $topic_name})
            MERGE (p)-[:BELONGS_TO]->(t)
            SET p.cluster_id = $cluster_id
            RETURN p.arxiv_id AS id
        """, {"arxiv_id": arxiv_id, "topic_name": topic_name, "cluster_id": cluster_id})
        return len(result) > 0

    # ── Institution (Phase 1) ───────────────────────────────────────────────────

    def create_institution(self, name: str, ror_id: str = "") -> bool:
        result = self.run_query("""
            MERGE (i:Institution {name: $name})
            SET i.ror_id = $ror_id
            RETURN i.name AS name
        """, {"name": name, "ror_id": ror_id})
        return len(result) > 0

    # ── Search ──────────────────────────────────────────────────────────────────

    def search_papers(self, query_text: str, limit: int = 10) -> List[dict]:
        query = """
        MATCH (p:Paper)
        WHERE toLower(p.title)    CONTAINS toLower($query_text)
           OR toLower(p.abstract) CONTAINS toLower($query_text)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title,
               p.published_date AS published_date, p.url AS url,
               p.rank_score AS rank_score
        ORDER BY p.published_date DESC
        LIMIT $limit
        """
        return self.run_query(query, {"query_text": query_text, "limit": limit})

    def get_top_papers(self, limit: int = 20) -> List[dict]:
        """Return papers ranked by rank_score (Phase 1)."""
        query = """
        MATCH (p:Paper)
        WHERE p.rank_score IS NOT NULL
        RETURN p.arxiv_id AS arxiv_id, p.title AS title,
               p.published_date AS published_date,
               p.rank_score AS rank_score,
               p.citation_count AS citation_count,
               p.citation_velocity AS citation_velocity
        ORDER BY p.rank_score DESC
        LIMIT $limit
        """
        return self.run_query(query, {"limit": limit})

    def get_cited_by(self, arxiv_id: str, limit: int = 20) -> List[dict]:
        """Return papers in the graph that cite the given paper (Phase 1)."""
        query = """
        MATCH (citing:Paper)-[:CITES]->(p:Paper {arxiv_id: $arxiv_id})
        RETURN citing.arxiv_id AS arxiv_id, citing.title AS title,
               citing.published_date AS published_date,
               citing.rank_score AS rank_score
        ORDER BY citing.rank_score DESC
        LIMIT $limit
        """
        return self.run_query(query, {"arxiv_id": arxiv_id, "limit": limit})

    def get_citation_flow(self, arxiv_id: str, depth: int = 2) -> List[dict]:
        """
        Reverse citation traversal: find the intellectual ancestry of a paper (Phase 4).
        Returns path nodes and depths for building the citation flow tree.
        """
        query = """
        MATCH path = (start:Paper {arxiv_id: $arxiv_id})-[:CITES*1..$depth]->(ancestor:Paper)
        RETURN [node IN nodes(path) | {
            arxiv_id:       node.arxiv_id,
            title:          node.title,
            published_date: node.published_date,
            rank_score:     node.rank_score,
            cluster_id:     node.cluster_id,
            embedding:      node.embedding
        }] AS path_nodes,
        length(path) AS depth
        ORDER BY depth
        LIMIT 500
        """
        return self.run_query(query, {"arxiv_id": arxiv_id, "depth": depth})

    def get_paper_neighbourhood(self, arxiv_id: str, depth: int = 2) -> List[dict]:
        """Return all nodes reachable from a paper within depth hops (legacy, kept for compatibility)."""
        query = """
        MATCH path = (start:Paper {arxiv_id: $arxiv_id})-[*1..$depth]-(neighbour)
        RETURN [node IN nodes(path) | {labels: labels(node), props: properties(node)}] AS nodes,
               [rel  IN relationships(path) | type(rel)] AS rel_types
        LIMIT 200
        """
        return self.run_query(query, {"arxiv_id": arxiv_id, "depth": depth})

    # ── Repo (GitHub, post-MVP) ─────────────────────────────────────────────────

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
        query = """
        MATCH (r:Repo  {url: $repo_url})
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MERGE (r)-[:IMPLEMENTS]->(p)
        RETURN r.url AS url
        """
        return len(self.run_query(query, {"repo_url": repo_url, "arxiv_id": arxiv_id})) > 0
