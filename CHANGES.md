# CHANGES — Research Intelligence Platform Refactor

One-line summary of every file added or modified. Use this to review what changed.

---

## Phase 1 — Replace LLM-from-abstract enrichment with citation APIs

- `kg/enrichment/__init__.py` — new module init
- `kg/enrichment/semantic_scholar.py` — fetches citation_count, influential_citation_count, references per paper from Semantic Scholar API
- `kg/enrichment/openalex.py` — fetches citation_velocity, author h_index, institution data per paper from OpenAlex API
- `kg/enrichment/llm_judge.py` — LLM fallback (batch of 20) triggered only when both APIs return nothing or citation_count=0 but paper is cited
- `kg/enrichment/runner.py` — orchestrates S2 + OpenAlex, computes rank_score, writes CITES edges, Institution nodes, AFFILIATED_WITH edges back to Neo4j
- `kg/graph/neo4j_client.py` — added Institution node, CITES edge, rank_score/citation_count/citation_velocity on Paper; added paper-level vector index; removed Chunk/Method/Dataset nodes; added Topic node and BELONGS_TO edge
- `kg/graph/queries.py` — rewrote all Cypher queries for unified schema; removed Method/Dataset/Methodology refs; added TOP_PAPERS, CITED_BY, CITATION_ANCESTORS, TRENDING_TOPICS (topic-based), PAPERS_IN_TOPIC
- `kg/utils/config.py` — added semantic_scholar_api_key and openalex_email; removed unused postgres and twitter settings
- `kg/commands/enrich.py` — new CLI command: kg enrich --limit 100
- `kg/commands/top.py` — new CLI command: kg top --n 20 (top papers by rank_score)
- `kg/commands/cited_by.py` — new CLI command: kg cited-by "2106.09685"

## Phase 2 — Paper-level SPECTER2 embeddings + vector search

- `kg/nlp/embedder.py` — rewritten: title+[SEP]+abstract whole-paper input; stores embedding on Paper node; removed chunk_abstract, find_similar_pairs, create_chunk calls; added embed_query for semantic search
- `kg/commands/embed.py` — new CLI commands: kg embed run, kg embed search "query", kg embed similar "arxiv_id"

## Phase 3 — Topic clustering + trend detection

- `kg/clustering/__init__.py` — new module init
- `kg/clustering/cluster.py` — UMAP (768d→15d for HDBSCAN, 768d→2d for viz) + HDBSCAN; writes Topic nodes, BELONGS_TO edges, umap_x/umap_y on Paper; computes trend_score = new_papers × avg_velocity
- `kg/clustering/naming.py` — LLM topic naming: sends 10-20 titles per cluster to Together AI Llama, returns 2-5 word topic name
- `kg/commands/cluster.py` — new CLI commands: kg cluster run, kg cluster topic "LLM Agents"
- `kg/commands/trends.py` — rewritten: queries Topic nodes + trend_score instead of Method/Dataset; shows new papers count and trend arrows

## Phase 4 — Visualization + Citation Flow

- `kg/flow/__init__.py` — new module init
- `kg/flow/citation_flow.py` — on-demand reverse traversal of CITES edges (depth-configurable, default=2); FlowNode tree builder; divergence detection via embedding cosine similarity; terminal ASCII renderer; JSON export for visualization
- `kg/visualization/graph.html` — D3.js scatter plot (landscape view): UMAP 2D coordinates, topic color, rank_score bubble size, date slider, click to highlight citation/semantic neighbours
- `kg/visualization/river.html` — D3.js Research River (lineage view): horizontal tree layout, divergence nodes highlighted in orange, cluster color-coded, click node to open arXiv
- `kg/commands/trace.py` — new CLI: kg trace "2106.09685" (terminal tree), supports title search + --depth flag
- `kg/commands/flow.py` — new CLI: kg flow "2106.09685" (writes river_data.json, opens river.html in browser)
- `kg/commands/visualize.py` — new CLI: kg visualize (writes graph_data.json, opens graph.html in browser)

## Final wiring

- `kg/cli.py` — rewired: all Phase 1-4 commands registered; status command updated for new node types (Topic, Institution, CITES, enriched/embedded counts)
- `requirements.txt` — added: umap-learn, hdbscan, scikit-learn, adapters, together; removed: fastapi, uvicorn, psycopg2, sqlalchemy, prompt_toolkit (not used)
