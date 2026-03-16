# AI Research Intelligence Platform

A local-first CLI for navigating the AI research landscape. Crawls arXiv daily, builds a knowledge graph in Neo4j, enriches papers with citation data, generates SPECTER2 semantic embeddings, clusters topics automatically with two-level granularity, and surfaces what's trending — all from your terminal.

```
kg search "mechanistic interpretability"
kg trends
kg embed search "sparse autoencoders in transformers"
kg cluster run --sub-cluster
kg visualize
```

---

## What it does

| Layer | What happens |
|---|---|
| **Crawl** | Pulls new papers from arXiv RSS feeds — configurable categories, runs daily |
| **Enrich** | Fetches citation counts, author h-index, and institution data via OpenAlex |
| **Embed** | Generates 768-dim SPECTER2 embeddings per paper (title + abstract) |
| **Cluster** | UMAP + HDBSCAN in two passes — broad topics, then specific sub-topics named by LLM |
| **Visualize** | D3.js landscape view in the browser — zoom, filter by date, click papers |

Everything lives in a local Neo4j graph. No cloud, no SaaS, no API quota anxiety.

---

## Requirements

- Python 3.11+
- Docker (for Neo4j)
- A [Together AI](https://together.ai) API key (free tier works — used for LLM topic naming)
- An [OpenAlex](https://openalex.org) email (free, just for polite-pool rate limits)

---

## Installation

```bash
git clone https://github.com/your-username/ai-research-kg
cd ai-research-kg

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the kg CLI in editable mode
pip install -e .
```

---

## Setup

**1. Start Neo4j**

```bash
docker compose up -d
```

Neo4j browser available at [http://localhost:7474](http://localhost:7474) — default credentials: `neo4j / password123`

**2. Create your `.env` file**

```bash
cp .env.example .env
```

Then edit `.env`:

```env
# Neo4j (matches docker-compose defaults)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# Together AI — for LLM topic naming (get a free key at together.ai)
TOGETHER_API_KEY=your_key_here

# OpenAlex — just your email, no signup needed
OPENALEX_EMAIL=you@example.com

# Optional: Semantic Scholar API key (unlocks citation flow feature)
# Request at: https://www.semanticscholar.org/product/api
SEMANTIC_SCHOLAR_API_KEY=
```

**3. Initialise the schema**

```bash
python -c "from kg.graph.neo4j_client import Neo4jClient; db = Neo4jClient(); db.connect(); db.setup_schema(); db.close(); print('Schema ready')"
```

---

## Daily workflow

```bash
# 1. Crawl today's arXiv papers (run on weekdays — arXiv is quiet on weekends)
python -m kg.crawlers.arxiv

# 2. Enrich with citation data from OpenAlex
kg enrich --limit 200

# 3. Generate SPECTER2 embeddings
kg embed run --limit 200

# 4. Cluster into topics
kg cluster run                   # broad topics only
kg cluster run --sub-cluster     # broad + detailed sub-topics (recommended)
```

---

## CLI reference

### Discovery

```bash
kg search "your query"                  # Keyword search across titles and abstracts
kg embed search "your query"            # Semantic search using SPECTER2 embeddings
kg embed similar 2106.09685             # Papers semantically similar to a given arXiv ID
kg top --n 20                           # Top papers by rank score
kg trends                               # Trending topics (last 7 days)
kg trends --days 30                     # Trending topics over the last 30 days
```

### Exploration

```bash
kg cluster topic "LLM Agents"                        # Papers in a broad topic
kg cluster topic "Reinforcement Learning > Offline"  # Papers in a sub-topic (partial match)
kg status                                            # Graph node and edge counts
```

### Visualization

```bash
kg visualize                            # Opens the Research Landscape (D3.js scatter plot)
```

The landscape view shows all papers as bubbles — colored by topic cluster, sized by rank score, filterable by date. Click any paper to see its metadata and highlight its semantic neighbours. The server runs locally at `http://127.0.0.1:8765` and stays up until you press `Ctrl+C`.

### Pipeline

```bash
# Crawling
python -m kg.crawlers.arxiv                          # default feeds
python -m kg.crawlers.arxiv --feeds cs.CV,cs.RO      # custom categories
python -m kg.crawlers.arxiv --list-feeds             # see all available arXiv categories
python -m kg.crawlers.arxiv --max 500                # papers per feed

# Enrichment
kg enrich --limit 100                   # run OpenAlex enrichment
kg enrich --limit 100 --dry-run         # preview without writing

# Embeddings
kg embed run --limit 500                # run SPECTER2 embeddings

# Clustering
kg cluster run                          # broad topics, UMAP + HDBSCAN + LLM naming
kg cluster run --sub-cluster            # adds a second fine-grained pass per topic
kg cluster run --min-size 5             # smaller min cluster size = more clusters
kg cluster run --sub-cluster \
               --min-size 5 \
               --sub-min-size 2         # maximum granularity
kg cluster run --dry-run                # preview without writing
```

---

## Customising your arXiv feeds

By default the crawler pulls from 9 CS/ML categories. You can scope it to your domain without touching any code:

```bash
# See everything available
python -m kg.crawlers.arxiv --list-feeds

# Computer vision + robotics focus
python -m kg.crawlers.arxiv --feeds cs.CV,cs.RO,cs.LG

# NLP focus
python -m kg.crawlers.arxiv --feeds cs.CL,cs.LG,cs.AI

# Security + systems
python -m kg.crawlers.arxiv --feeds cs.CR,cs.DC,cs.NI,cs.SE
```

Everything downstream — clusters, trends, search — reflects the feeds you choose.

---

## Two-level clustering

Running `kg cluster run --sub-cluster` produces both broad topics and specific sub-topics:

```
Reinforcement Learning
  ↳ Reinforcement Learning > Offline RL Methods
  ↳ Reinforcement Learning > RLHF & Alignment
  ↳ Reinforcement Learning > Multi-Agent Systems

LLM Agents
  ↳ LLM Agents > Tool Use & Function Calling
  ↳ LLM Agents > Reasoning & Planning

Computer Vision
  ↳ Computer Vision > Diffusion-Based Generation
  ↳ Computer Vision > 3D Scene Understanding
```

The LLM naming prompt for sub-topics is given the parent topic as context, so names are specific rather than generic. Use `kg cluster topic` with a partial match to explore either level:

```bash
kg cluster topic "RL"                   # matches anything containing "RL"
kg cluster topic "offline"              # matches "Reinforcement Learning > Offline RL Methods"
kg cluster topic "LLM Agents"           # matches the broad topic and all its sub-topics
```

---

## Citation data transparency

OpenAlex enrichment tracks how each paper's citation count was obtained:

| Source | Meaning |
|---|---|
| `doi` | Exact DOI match — citations are accurate |
| `title_match` | Matched via title search — treat as an estimate |
| `unindexed` | Paper too new (<30 days) — OpenAlex hasn't indexed it yet, count is 0 |

Papers published today get `unindexed` with 0 citations rather than a false match against an older paper with a similar title. Re-running `kg enrich` after a few weeks will pick them up properly via DOI.

---

## How rank score works

Each paper gets a `rank_score` computed from:

```
rank_score = log(citations + 1)
           + 1.5 × citation_velocity
           + 0.8 × recency
           + 0.3 × author_h_index
```

- **citation_velocity** — citations per year (from OpenAlex)
- **recency** — exponential decay from publication date (~0.37 after 1 year)
- **author_h_index** — max h-index across the paper's authors

New preprints score on recency. Foundational papers score on citations. Both surface naturally.

---

## Graph schema

```
(Paper)-[:WRITTEN_BY]->(Author)
(Paper)-[:BELONGS_TO]->(Topic)
(Author)-[:AFFILIATED_WITH]->(Institution)
(Paper)-[:CITES]->(Paper)          ← available once S2 API key is configured
```

Node properties of note: `Paper.rank_score`, `Paper.embedding` (768-dim), `Paper.umap_x/y`, `Paper.citation_source`, `Topic.trend_score`

---

## Coming soon

- **Citation flow** — `kg flow <arxiv_id>` traces the intellectual ancestry of a paper through a D3.js river visualization. Ready to activate once a Semantic Scholar API key is in place — [request one here](https://www.semanticscholar.org/product/api).
- **GitHub crawler** — link papers to their implementation repos
- **AWS deployment** — scheduled crawling, persistent graph, shared access

---

## Project structure

```
kg/
├── crawlers/         arXiv RSS crawler (configurable feeds)
├── enrichment/       OpenAlex enrichment pipeline + LLM fallback judge
├── nlp/              SPECTER2 embeddings
├── clustering/       UMAP + HDBSCAN two-level clustering + LLM topic naming
├── flow/             Citation flow traversal (pending S2 key)
├── graph/            Neo4j client + schema + Cypher queries
├── commands/         CLI command implementations
├── visualization/    D3.js graph.html + river.html
└── utils/            Config, settings
```

---

## Contributing

Issues and PRs welcome. If you're using this and hitting problems, open an issue with your `kg status` output and which command failed.

---