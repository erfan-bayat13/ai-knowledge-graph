# kg/enrichment/runner.py
# Phase 1: orchestrates OpenAlex enrichment for all unenriched papers
# Computes rank_score = log(citations+1) + 1.5*velocity + 0.8*recency + 0.3*author_influence
# Writes rank_score, citation_count, citation_velocity back to Paper nodes
# Creates Institution nodes + AFFILIATED_WITH edges
#
# NOTE: Semantic Scholar removed as primary source — S2 API key pending approval.
# citation_count is sourced from OpenAlex (cited_by_count field) which is reliable.
# TODO: restore CITES edges once S2 API key is approved:
#   1. re-add:  from kg.enrichment import semantic_scholar as s2
#   2. re-add:  s2_data = s2.fetch_paper(arxiv_id)  inside the loop
#   3. restore: the CITES edge block below the rank_score write

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from kg.enrichment import openalex as oa
from kg.enrichment.llm_judge import judge_batch, should_judge
from kg.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Stats ──────────────────────────────────────────────────────────────────────

@dataclass
class EnrichStats:
    total: int = 0
    enriched: int = 0
    oa_found: int = 0
    llm_judged: int = 0
    cites_edges: int = 0        # always 0 until S2 key restored
    institution_nodes: int = 0
    failed: int = 0

    def print_summary(self):
        print(f"\n{'─'*42}")
        print(f"  Papers processed:   {self.enriched}/{self.total}")
        print(f"  OpenAlex:           {self.oa_found} found")
        print(f"  LLM judged:         {self.llm_judged}")
        print(f"  Institution nodes:  {self.institution_nodes}")
        print(f"  CITES edges:        {self.cites_edges}  (pending S2 key)")
        print(f"  Failed:             {self.failed}")
        print("─"*42)


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _recency_score(published_date: Optional[str]) -> float:
    """Exponential decay from published_date — full score today, ~0.37 after 1 year."""
    if not published_date:
        return 0.0
    try:
        pub = datetime.strptime(published_date[:10], "%Y-%m-%d")
        age_days = max((datetime.now() - pub).days, 0)
        return math.exp(-age_days / 365.0)
    except Exception:
        return 0.0


def _rank_score(
    citation_count: int,
    citation_velocity: float,
    recency: float,
    author_influence: float,
) -> float:
    """
    rank_score = log(citations+1) + 1.5*velocity + 0.8*recency + 0.3*author_influence
    Weights are initial guesses — tune once real enrichment data is available.
    citation_count and citation_velocity both sourced from OpenAlex.
    """
    return (
        math.log(citation_count + 1)
        + 1.5 * citation_velocity
        + 0.8 * recency
        + 0.3 * author_influence
    )


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_enrichment(limit: int = 100, dry_run: bool = False, rate_limit_s: float = 0.5) -> EnrichStats:
    """
    Enrichment pipeline — OpenAlex only.

    Args:
        limit:        max papers to process per run
        dry_run:      print what would be written, no DB writes
        rate_limit_s: seconds to sleep between papers (OpenAlex polite pool)
    """
    db = Neo4jClient()
    db.connect()
    stats = EnrichStats()

    # Fetch papers not yet enriched (no rank_score set), newest first
    papers = db.run_query("""
        MATCH (p:Paper)
        WHERE p.rank_score IS NULL
        RETURN p.arxiv_id      AS arxiv_id,
               p.abstract      AS abstract,
               p.title         AS title,
               p.published_date AS published_date
        ORDER BY p.published_date DESC
        LIMIT $limit
    """, {"limit": limit})

    stats.total = len(papers)
    logger.info(f"Papers to enrich: {stats.total}")

    # Papers that get no data from OpenAlex queue up for LLM fallback
    llm_queue = []

    for i, paper in enumerate(papers):
        arxiv_id       = paper["arxiv_id"]
        published_date = paper.get("published_date")
        abstract       = paper.get("abstract") or ""

        logger.debug(f"[{i+1}/{stats.total}] {arxiv_id}")
        print(f"\r  [{i+1}/{stats.total}] enriching...", end="", flush=True)


        # ── OpenAlex ──────────────────────────────────────────────────────────
        oa_data = oa.fetch_paper(arxiv_id, published_date=published_date)

        if oa_data:
            stats.oa_found += 1

        # ── LLM fallback: queue if OpenAlex returned nothing ──────────────────
        # Pass None as s2_data — should_judge treats (None, None) as needs-judge
        cited_in_graph = bool(db.run_query(
            "MATCH ()-[:CITES]->(p:Paper {arxiv_id: $id}) RETURN count(*) AS n",
            {"id": arxiv_id}
        )[0]["n"] > 0)

        if should_judge(None, oa_data, cited_in_graph):
            llm_queue.append({"arxiv_id": arxiv_id, "abstract": abstract})

        # ── Aggregate signals ─────────────────────────────────────────────────
        # citation_count: use OpenAlex cited_by_count (reliable, updated regularly)
        citation_count    = (oa_data or {}).get("citation_count", 0)
        citation_velocity = (oa_data or {}).get("citation_velocity", 0.0)
        citation_source   = (oa_data or {}).get("citation_source", "unknown")
        recency           = _recency_score(published_date)

        authors          = (oa_data or {}).get("authors", [])
        author_influence = max((a.get("h_index", 0) or 0 for a in authors), default=0)

        rank = _rank_score(citation_count, citation_velocity, recency, author_influence)

        if not dry_run:
            # ── Write enrichment data back to Paper node ──────────────────────
            db.run_query("""
                MATCH (p:Paper {arxiv_id: $arxiv_id})
                SET p.citation_count    = $citation_count,
                    p.citation_velocity = $citation_velocity,
                    p.rank_score        = $rank_score,
                    p.citation_source   = $citation_source,
                    p.enriched_at       = timestamp()
            """, {
                "arxiv_id":          arxiv_id,
                "citation_count":    citation_count,
                "citation_velocity": citation_velocity,
                "rank_score":        round(rank, 6),
                "citation_source":   citation_source,
            })

            # TODO: CITES edges — restore this block once S2 API key is approved
            # for ref_arxiv_id in (s2_data or {}).get("references", []):
            #     existing = db.run_query(
            #         "MATCH (p:Paper {arxiv_id: $id}) RETURN p.arxiv_id AS id",
            #         {"id": ref_arxiv_id}
            #     )
            #     if existing:
            #         db.run_query("""
            #             MATCH (a:Paper {arxiv_id: $from_id})
            #             MATCH (b:Paper {arxiv_id: $to_id})
            #             MERGE (a)-[:CITES]->(b)
            #         """, {"from_id": arxiv_id, "to_id": ref_arxiv_id})
            #         stats.cites_edges += 1

            # ── Institution nodes + AFFILIATED_WITH edges ─────────────────────
            for author_data in authors:
                inst_name   = author_data.get("institution_name", "")
                ror_id      = author_data.get("ror_id", "")
                author_name = author_data.get("name", "")

                if inst_name and author_name:
                    db.run_query("""
                        MERGE (i:Institution {name: $name})
                        SET i.ror_id = $ror_id
                    """, {"name": inst_name, "ror_id": ror_id})
                    db.run_query("""
                        MATCH (a:Author {name: $author_name})
                        MATCH (i:Institution {name: $inst_name})
                        MERGE (a)-[:AFFILIATED_WITH]->(i)
                    """, {"author_name": author_name, "inst_name": inst_name})
                    stats.institution_nodes += 1

                # Update author h_index from OpenAlex
                h = author_data.get("h_index", 0) or 0
                if h and author_name:
                    db.run_query("""
                        MATCH (a:Author {name: $name})
                        SET a.h_index = $h_index, a.influence_score = $h_index
                    """, {"name": author_name, "h_index": h})

        else:
            print(f"  [DRY] {arxiv_id}: rank={rank:.3f}  cit={citation_count}  vel={citation_velocity:.3f}")

        stats.enriched += 1
        time.sleep(rate_limit_s)

    # ── LLM batch fallback ─────────────────────────────────────────────────────
    if llm_queue:
        logger.info(f"LLM judge queue: {len(llm_queue)} papers")
        from kg.enrichment.llm_judge import BATCH_SIZE
        for batch_start in range(0, len(llm_queue), BATCH_SIZE):
            batch   = llm_queue[batch_start : batch_start + BATCH_SIZE]
            results = judge_batch(batch)
            stats.llm_judged += len(results)

            if not dry_run:
                for r in results:
                    approx = r.get("approx_citations", 0) or 0
                    db.run_query("""
                        MATCH (p:Paper {arxiv_id: $arxiv_id})
                        SET p.citation_count = CASE WHEN p.citation_count IS NULL
                                                    THEN $approx
                                                    ELSE p.citation_count END,
                            p.rank_score     = CASE WHEN p.rank_score IS NULL
                                                    THEN $rank
                                                    ELSE p.rank_score END
                    """, {
                        "arxiv_id": r.get("arxiv_id", ""),
                        "approx":   approx,
                        "rank":     round(math.log(approx + 1), 6),
                        "citation_source": "llm_judge",
                    })

    db.close()
    stats.print_summary()
    return stats