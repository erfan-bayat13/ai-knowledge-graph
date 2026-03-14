# kg/nlp/enrichment_runner.py
import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from kg.graph.neo4j_client import Neo4jClient
from kg.nlp.extractor import extract

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_NOISE = {
    "null", "none", "n/a", "unknown", "not mentioned",
    "the model", "our model", "the method", "our method",
    "the approach", "baseline",
}

def _is_valid(name: str) -> bool:
    if not name or not name.strip():
        return False
    name = name.strip()
    if name.lower() in _NOISE:
        return False
    if len(name) < 2 or len(name.split()) > 6:
        return False
    return True

@dataclass
class RunStats:
    total: int = 0
    processed: int = 0
    failed: int = 0
    methods_created: int = 0
    datasets_created: int = 0
    edges_created: int = 0

    def print_summary(self):
        print(f"\n{'─'*40}")
        print(f"  Papers processed:  {self.processed}/{self.total}")
        print(f"  Methods created:   {self.methods_created}")
        print(f"  Datasets created:  {self.datasets_created}")
        print(f"  Edges created:     {self.edges_created}")
        print(f"  Failed:            {self.failed}")
        print("─"*40)

def run_enrichment(dry_run: bool = False, limit: Optional[int] = None, batch_size: int = 50):
    db = Neo4jClient()
    stats = RunStats()
    db.connect()

    # Fetch only papers not yet enriched
    query = """
    MATCH (p:Paper)
    WHERE NOT (p)-[:PROPOSES]->()
    RETURN p.arxiv_id AS arxiv_id, p.abstract AS abstract, p.title AS title
    ORDER BY p.published_date DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    papers = db.run_query(query)
    stats.total = len(papers)
    logger.info(f"Papers to enrich: {stats.total}")

    for i, paper in enumerate(papers):
        arxiv_id = paper["arxiv_id"]
        abstract = paper.get("abstract", "")

        if not abstract:
            stats.failed += 1
            continue

        result = extract(abstract)

        if result["source"] == "failed":
            stats.failed += 1
            continue

        if dry_run:
            print(f"\n[DRY] {arxiv_id}")
            print(f"  proposed: {result['proposed_methods']}")
            print(f"  datasets: {result['datasets']}")
            print(f"  builds_on: {result['builds_on']}")
            stats.processed += 1
            continue

        for name in result.get("proposed_methods", []):
            if not _is_valid(name):
                continue
            db.create_method(name)
            stats.methods_created += 1
            if db.link_paper_proposes(arxiv_id, name):
                stats.edges_created += 1

        for name in result.get("datasets", []):
            if not _is_valid(name):
                continue
            db.create_dataset(name)
            stats.datasets_created += 1
            if db.link_paper_evaluated_on(arxiv_id, name):
                stats.edges_created += 1

        stats.processed += 1

        # Rate limit — Together AI free tier is generous but let's be safe
        if (i + 1) % batch_size == 0:
            logger.info(f"Progress: {i+1}/{stats.total}")
            time.sleep(1)

    db.close()
    stats.print_summary()
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    run_enrichment(dry_run=args.dry_run, limit=args.limit, batch_size=args.batch_size)