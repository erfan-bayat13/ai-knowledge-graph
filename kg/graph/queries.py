# kg/graph/queries.py
# CHANGED (Phase 1): rewrote all queries for unified design schema
# CHANGED (Phase 1): added CITATION_FLOW, TOP_PAPERS, CITED_BY, CITATION_ANCESTORS queries
# CHANGED (Phase 3): added TRENDING_TOPICS based on Topic nodes (not Method/Dataset)
# Old queries referencing Method, Dataset, Methodology, Chunk nodes are removed

# ── Paper lookups ───────────────────────────────────────────────────────────────

PAPER_BY_ARXIV_ID = """
MATCH (p:Paper {arxiv_id: $arxiv_id})
OPTIONAL MATCH (p)-[:WRITTEN_BY]->(a:Author)
OPTIONAL MATCH (p)-[:BELONGS_TO]->(t:Topic)
RETURN p,
       collect(distinct a.name)  AS authors,
       collect(distinct t.name)  AS topics
"""

PAPERS_BY_AUTHOR = """
MATCH (p:Paper)-[:WRITTEN_BY]->(a:Author {name: $author_name})
RETURN p.arxiv_id AS arxiv_id, p.title AS title,
       p.published_date AS published_date, p.rank_score AS rank_score
ORDER BY p.rank_score DESC NULLS LAST
LIMIT $limit
"""

AUTHOR_COLLABORATIONS = """
MATCH (a1:Author)<-[:WRITTEN_BY]-(p:Paper)-[:WRITTEN_BY]->(a2:Author)
WHERE a1.name = $author_name AND a1 <> a2
RETURN a2.name AS collaborator, count(p) AS shared_papers
ORDER BY shared_papers DESC
LIMIT $limit
"""

# ── Ranking ─────────────────────────────────────────────────────────────────────

TOP_PAPERS = """
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

# ── Citation graph ──────────────────────────────────────────────────────────────

CITED_BY = """
MATCH (citing:Paper)-[:CITES]->(p:Paper {arxiv_id: $arxiv_id})
RETURN citing.arxiv_id AS arxiv_id, citing.title AS title,
       citing.published_date AS published_date,
       citing.rank_score AS rank_score
ORDER BY citing.rank_score DESC NULLS LAST
LIMIT $limit
"""

# Reverse traversal for citation flow / Research River (Phase 4)
CITATION_ANCESTORS = """
MATCH path = (start:Paper {arxiv_id: $arxiv_id})-[:CITES*1..$depth]->(ancestor:Paper)
RETURN [node IN nodes(path) | {
    arxiv_id:       node.arxiv_id,
    title:          node.title,
    published_date: node.published_date,
    rank_score:     node.rank_score,
    cluster_id:     node.cluster_id
}] AS path_nodes,
length(path) AS depth
ORDER BY depth
LIMIT 500
"""

# ── Topics + Trends (Phase 3) ────────────────────────────────────────────────────

TRENDING_TOPICS = """
MATCH (p:Paper)-[:BELONGS_TO]->(t:Topic)
WHERE p.published_date >= $since_date
RETURN t.name AS topic, count(p) AS paper_count, t.trend_score AS trend_score
ORDER BY trend_score DESC NULLS LAST, paper_count DESC
LIMIT $limit
"""

PAPERS_IN_TOPIC = """
MATCH (p:Paper)-[:BELONGS_TO]->(t:Topic {name: $topic_name})
RETURN p.arxiv_id AS arxiv_id, p.title AS title,
       p.published_date AS published_date, p.rank_score AS rank_score
ORDER BY p.rank_score DESC NULLS LAST
LIMIT $limit
"""

# ── Repo (post-MVP) ──────────────────────────────────────────────────────────────

REPO_IMPLEMENTS_PAPER = """
MATCH (r:Repo)-[:IMPLEMENTS]->(p:Paper)
RETURN r.name AS repo, r.url AS repo_url, r.stars AS stars,
       p.title AS paper_title, p.arxiv_id AS arxiv_id
ORDER BY r.stars DESC
LIMIT $limit
"""
