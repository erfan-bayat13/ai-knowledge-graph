# kg/nlp/enrichment_runner.py
#
# Reads all papers from Neo4j, runs the spaCy extractor on each abstract,
# writes Method/Dataset nodes + relationships back.
#
# This replaces the old hybrid extractor entirely.
# No LLM calls here — that's reserved for on-demand use only.
#
# Usage:
#   python -m kg.nlp.enrichment_runner
#   python -m kg.nlp.enrichment_runner --dry-run
#   python -m kg.nlp.enrichment_runner --limit 50

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from kg.graph.neo4j_client import Neo4jClient
from kg.nlp.extractor import extract

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class RunStats:
    total:            int = 0
    processed:        int = 0
    skipped:          int = 0
    failed:           int = 0
    methods_created:  int = 0
    datasets_created: int = 0
    edges_created:    int = 0
    errors:           List[str] = field(default_factory=list)

    def print_summary(self):
        print("\n" + "─" * 50)
        print(f"  Papers total:      {self.total}")
        print(f"  Processed:         {self.processed}")
        print(f"  Skipped:           {self.skipped}")
        print(f"  Failed:            {self.failed}")
        print(f"  Methods created:   {self.methods_created}")
        print(f"  Datasets created:  {self.datasets_created}")
        print(f"  Edges created:     {self.edges_created}")
        if self.errors:
            print(f"  Errors ({len(self.errors)}):")
            for e in self.errors[:5]:
                print(f"    {e}")
        print("─" * 50)


# ── Noise filter ──────────────────────────────────────────────────────────────

_NOISE = {
    "null", "none", "n/a", "unknown", "not mentioned",
    "the model", "our model", "the method", "our method",
    "the approach", "our approach", "the system", "our system",
}

def _is_valid(name: str) -> bool:
    if not name or not name.strip():
        return False
    name = name.strip()
    if name.lower() in _NOISE:
        return False
    if len(name) < 2 or len(name.split()) > 6:
        return False
    # All-lowercase multi-word → likely a sentence fragment
    if name.islower() and len(name.split()) > 2:
        return False
    return True


# ── Write to graph ────────────────────────────────────────────────────────────

def write_to_neo4j(
    db: Neo4jClient,
    arxiv_id: str,
    result: dict,
    dry_run: bool,
    stats: RunStats,
) -> None:

    for method_name in result.get("proposed_methods", []):
        if not _is_valid(method_name):
            continue
        if dry_run:
            print(f"    [DRY] PROPOSES: {arxiv_id} → {method_name!r}")
            continue
        db.create_method(method_name)
        stats.methods_created += 1
        if db.link_paper_proposes(arxiv_id, method_name):
            stats.edges_created += 1

    for dataset_name in result.get("datasets", []):
        if not _is_valid(dataset_name):
            continue
        if dry_run:
            print(f"    [DRY] EVALUATED_ON: {arxiv_id} → {dataset_name!r}")
            continue
        db.create_dataset(dataset_name)
        stats.datasets_created += 1
        if db.link_paper_evaluated_on(arxiv_id, dataset_name):
            stats.edges_created += 1


# ── Main runner ───────────────────────────────────────────────────────────────

def run_enrichment(
    dry_run:    bool = False,
    limit:      Optional[int] = None,
    batch_size: int = 100,
) -> RunStats:

    db    = Neo4jClient()
    stats = RunStats()

    db.connect()
    stats.total = db.get_paper_count()
    logger.info(f"Total papers: {stats.total}")

    skip              = 0
    processed_this_run = 0

    while True:
        batch = db.get_all_papers(batch_size=batch_size, skip=skip)
        if not batch:
            break

        for paper in batch:
            if limit is not None and processed_this_run >= limit:
                break

            arxiv_id = paper["arxiv_id"]
            abstract = paper.get("abstract") or ""
            title    = paper.get("title", "")

            logger.info(f"[{processed_this_run + 1}] {arxiv_id} — {title[:55]}")

            try:
                result = extract(abstract)

                if dry_run:
                    print(f"\n  Paper: {title[:60]}")
                    print(f"  Proposed:  {result['proposed_methods']}")
                    print(f"  Datasets:  {result['datasets']}")

                write_to_neo4j(db, arxiv_id, result, dry_run, stats)
                stats.processed += 1
                processed_this_run += 1

            except Exception as e:
                logger.error(f"Failed on {arxiv_id}: {e}")
                stats.failed += 1
                stats.errors.append(f"{arxiv_id}: {e}")
                continue

        skip += batch_size

        if limit is not None and processed_this_run >= limit:
            break

    db.close()
    stats.print_summary()
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run spaCy enrichment on all papers")
    parser.add_argument("--dry-run",    action="store_true",  help="Print only, no writes")
    parser.add_argument("--limit",      type=int, default=None, help="Process at most N papers")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    run_enrichment(
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size,
    )