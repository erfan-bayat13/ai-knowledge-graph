# kg/nlp/enrichment_runner.py
#
# Reads all papers from Neo4j, runs extraction on each abstract,
# writes Method/Dataset/Task nodes + relationships back.
#
# Usage:
#   python -m kg.nlp.enrichment_runner
#   python -m kg.nlp.enrichment_runner --dry-run     (no writes, just print)
#   python -m kg.nlp.enrichment_runner --limit 50    (process first N papers)

import argparse
import logging
import time
from typing import Optional
from dataclasses import dataclass, field

import together

from kg.graph.neo4j_client import Neo4jClient
from kg.nlp.extractor import extract
from kg.utils.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

NOISE_NAMES = {
    "null", "none", "n/a", "unknown", "not mentioned",
    "not applicable", "not specified", "the model", "our model",
    "the method", "our method", "the approach", "our approach",
}

def _is_valid_name(name: str) -> bool:
    """
    Reject names that are clearly noise before writing to Neo4j.
    This is the last line of defence after extractor sanitisation.
    """
    if not name or not name.strip():
        return False
    name = name.strip()
    # Reject noise strings
    if name.lower() in NOISE_NAMES:
        return False
    # Reject if too short (single letter) or too long (sentence fragment)
    if len(name) < 2 or len(name.split()) > 6:
        return False
    # Reject if it's all lowercase common words (likely a sentence fragment)
    if name.islower() and len(name.split()) > 2:
        return False
    return True

# ─── Stats tracker ────────────────────────────────────────────────────────────

@dataclass
class RunStats:
    total:          int = 0
    skipped:        int = 0   # already enriched
    processed:      int = 0
    rule_based:     int = 0
    llm_used:       int = 0
    failed:         int = 0
    methods_created:  int = 0
    datasets_created: int = 0
    tasks_created:    int = 0
    edges_created:    int = 0
    errors:         list = field(default_factory=list)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("ENRICHMENT RUN SUMMARY")
        print("=" * 60)
        print(f"  Papers total in graph:  {self.total}")
        print(f"  Already enriched:       {self.skipped}  (skipped)")
        print(f"  Processed this run:     {self.processed}")
        print(f"    └─ rule_based:        {self.rule_based}")
        print(f"    └─ llm fallback:      {self.llm_used}")
        print(f"    └─ failed:            {self.failed}")
        print(f"  Nodes created:")
        print(f"    └─ Method:            {self.methods_created}")
        print(f"    └─ Dataset:           {self.datasets_created}")
        print(f"    └─ Task:              {self.tasks_created}")
        print(f"  Edges created:          {self.edges_created}")
        if self.errors:
            print(f"\n  Errors ({len(self.errors)}):")
            for e in self.errors[:5]:   # show first 5 only
                print(f"    - {e}")
        print("=" * 60)


# ─── Core writer ──────────────────────────────────────────────────────────────

def _is_valid_name(name: str) -> bool:
    """
    Reject names that are clearly noise before writing to Neo4j.
    This is the last line of defence after extractor sanitisation.
    """
    if not name or not name.strip():
        return False
    name = name.strip()
    # Reject noise strings
    if name.lower() in NOISE_NAMES:
        return False
    # Reject if too short (single letter) or too long (sentence fragment)
    if len(name) < 2 or len(name.split()) > 6:
        return False
    # Reject if it's all lowercase common words (likely a sentence fragment)
    if name.islower() and len(name.split()) > 2:
        return False
    return True


def write_extraction_to_neo4j(
    db: Neo4jClient,
    arxiv_id: str,
    result: dict,
    dry_run: bool,
    stats: RunStats,
) -> None:

    # ── Proposed methods ──────────────────────────────────────────────────────
    for method_name in result.get("proposed_methods", []):
        if not _is_valid_name(method_name):
            logger.debug(f"Skipping invalid proposed method: {method_name!r}")
            continue

        if dry_run:
            print(f"    [DRY RUN] PROPOSES: {arxiv_id} → {method_name!r}")
            continue

        db.create_method(method_name)
        stats.methods_created += 1
        if db.link_paper_proposes_method(arxiv_id, method_name):
            stats.edges_created += 1

        # IMPROVES edge — only if valid AND different from proposed
        improves_on = result.get("improves_on")
        if (improves_on
                and _is_valid_name(improves_on)
                and improves_on.lower() != method_name.lower()):
            db.create_method(improves_on)
            if db.link_method_improves_method(method_name, improves_on):
                stats.edges_created += 1

    # ── Used methods ──────────────────────────────────────────────────────────
    for method_name in result.get("used_methods", []):
        if not _is_valid_name(method_name):
            continue
        if dry_run:
            print(f"    [DRY RUN] USES: {arxiv_id} → {method_name!r}")
            continue
        db.create_method(method_name)
        stats.methods_created += 1
        if db.link_paper_uses_method(arxiv_id, method_name):
            stats.edges_created += 1

    # ── Datasets ──────────────────────────────────────────────────────────────
    for dataset_name in result.get("datasets", []):
        if not _is_valid_name(dataset_name):
            continue
        if dry_run:
            print(f"    [DRY RUN] EVALUATED_ON: {arxiv_id} → {dataset_name!r}")
            continue
        db.create_dataset(dataset_name)
        stats.datasets_created += 1
        if db.link_paper_evaluated_on(arxiv_id, dataset_name):
            stats.edges_created += 1

    # ── Tasks ─────────────────────────────────────────────────────────────────
    for task_name in result.get("tasks", []):
        if not _is_valid_name(task_name):
            continue
        if dry_run:
            print(f"    [DRY RUN] ADDRESSES: {arxiv_id} → {task_name!r}")
            continue
        db.create_task(task_name)
        stats.tasks_created += 1
        if db.link_paper_addresses_task(arxiv_id, task_name):
            stats.edges_created += 1


# ─── Main runner ──────────────────────────────────────────────────────────────

def run_enrichment(
    dry_run: bool = False,
    limit: Optional[int] = None,
    batch_size: int = 100,
    delay_between_llm: float = 0.5,
) -> RunStats:
    """
    Main enrichment pipeline.

    Args:
        dry_run:            If True, print what would be written but don't write
        limit:              Process at most this many papers (None = all)
        batch_size:         How many papers to fetch from Neo4j at once
        delay_between_llm:  Seconds to wait between LLM calls (rate limiting)
    """
    settings = get_settings()
    stats = RunStats()

    # ── Initialise clients ────────────────────────────────────────────────────
    db = Neo4jClient()
    db.connect()

    if not dry_run:
        db.setup_enrichment_schema()

    llm_client = None
    if settings.together_api_key:
        llm_client = together.Together(api_key=settings.together_api_key)
        logger.info("LLM client initialised — layer 3 enabled")
    else:
        logger.warning("No TOGETHER_API_KEY — running rule-based only")

    # ── Get already-enriched paper IDs (for skip logic) ───────────────────────
    if not dry_run:
        already_done = db.get_enriched_paper_ids()
        logger.info(f"Already enriched: {len(already_done)} papers (will skip)")
    else:
        already_done = set()

    # ── Count total papers ────────────────────────────────────────────────────
    stats.total = db.get_paper_count()
    logger.info(f"Total papers in graph: {stats.total}")

    # ── Process in batches ────────────────────────────────────────────────────
    skip = 0
    processed_this_run = 0

    while True:
        batch = db.get_all_papers(batch_size=batch_size, skip=skip)

        if not batch:
            break   # no more papers

        for paper in batch:
            arxiv_id = paper["arxiv_id"]
            abstract = paper["abstract"]
            title    = paper["title"]

            # Skip already enriched
            if arxiv_id in already_done:
                stats.skipped += 1
                continue

            # Respect limit
            if limit is not None and processed_this_run >= limit:
                break

            logger.info(f"[{processed_this_run + 1}] {arxiv_id} — {title[:55]}")

            try:
                result = extract(abstract, llm_client)

                if dry_run:
                    print(f"\n  Paper: {title[:60]}")
                    print(f"  Source: {result['source']}")
                    print(f"  Proposed:  {result['proposed_methods']}")
                    print(f"  Used:      {result['used_methods'][:4]}")
                    print(f"  Datasets:  {result['datasets']}")
                    print(f"  Tasks:     {result['tasks']}")
                    print(f"  Improves:  {result['improves_on']}")

                write_extraction_to_neo4j(db, arxiv_id, result, dry_run, stats)

                # Track source stats
                if result["source"] == "rule_based":
                    stats.rule_based += 1
                else:
                    stats.llm_used += 1
                    # Polite delay between LLM calls
                    time.sleep(delay_between_llm)

                stats.processed += 1
                processed_this_run += 1

            except Exception as e:
                logger.error(f"Failed on {arxiv_id}: {e}")
                stats.failed += 1
                stats.errors.append(f"{arxiv_id}: {e}")
                continue

        skip += batch_size

        # Break outer loop if limit reached
        if limit is not None and processed_this_run >= limit:
            break

    db.close()
    stats.print_summary()
    return stats


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run semantic enrichment on all papers in Neo4j"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without touching Neo4j",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N papers (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Papers to fetch from Neo4j per batch (default: 100)",
    )
    args = parser.parse_args()

    run_enrichment(
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size,
    )