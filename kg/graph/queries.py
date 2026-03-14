# Cypher query constants for reuse across the app

PAPER_BY_ARXIV_ID = """
MATCH (p:Paper {arxiv_id: $arxiv_id})
OPTIONAL MATCH (p)-[:WRITTEN_BY]->(a:Author)
OPTIONAL MATCH (p)-[:MENTIONS_TOPIC]->(t:Topic)
OPTIONAL MATCH (p)-[:USES_METHODOLOGY]->(m:Methodology)
RETURN p,
       collect(distinct a.name) as authors,
       collect(distinct t.name) as topics,
       collect(distinct m.name) as methodologies
"""

PAPERS_BY_AUTHOR = """
MATCH (p:Paper)-[:WRITTEN_BY]->(a:Author {name: $author_name})
RETURN p.arxiv_id as arxiv_id,
       p.title as title,
       p.published_date as published_date
ORDER BY p.published_date DESC
LIMIT $limit
"""

PAPERS_BY_METHODOLOGY = """
MATCH (p:Paper)-[:USES_METHODOLOGY]->(m:Methodology {name: $methodology_name})
RETURN p.arxiv_id as arxiv_id,
       p.title as title,
       p.published_date as published_date
ORDER BY p.published_date DESC
LIMIT $limit
"""

TRENDING_TOPICS = """
MATCH (p:Paper)-[:MENTIONS_TOPIC]->(t:Topic)
WHERE p.published_date >= $since_date
RETURN t.name as topic,
       count(p) as paper_count
ORDER BY paper_count DESC
LIMIT $limit
"""

AUTHOR_COLLABORATIONS = """
MATCH (a1:Author)<-[:WRITTEN_BY]-(p:Paper)-[:WRITTEN_BY]->(a2:Author)
WHERE a1.name = $author_name AND a1 <> a2
RETURN a2.name as collaborator,
       count(p) as shared_papers
ORDER BY shared_papers DESC
LIMIT $limit
"""

REPO_IMPLEMENTS_PAPER = """
MATCH (r:Repository)-[:IMPLEMENTS_PAPER]->(p:Paper)
RETURN r.name as repo,
       r.url as repo_url,
       r.stars as stars,
       p.title as paper_title,
       p.arxiv_id as arxiv_id
ORDER BY r.stars DESC
LIMIT $limit
"""