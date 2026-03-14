# kg/nlp/llm_enrichment.py
#
# On-demand LLM enrichment via Together AI.
# Use this only for specific papers where you want richer relationships:
#   BUILDS_ON, IMPROVES, COMPARES_WITH
#
# NOT run at ingestion time. NOT run on all papers.
# Call manually for ~50 papers that are central to your research path.
#
# Usage:
#   python -m kg.nlp.llm_enrichment --arxiv-id 2205.14135
#   python -m kg.nlp.llm_enrichment --arxiv-ids 2205.14135 2106.09685 2309.06180
#   python -m kg.nlp.llm_enrichment --dry-run --arxiv-id 2205.14135

import argparse
import json
import logging
import time
from typing import Dict, List, Optional

import together

from kg.graph.neo4j_client import Neo4jClient
from kg.utils.config import get_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

TOGETHER_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"

PROMPT = """\
You are extracting structured relationships from an AI research paper abstract.
Answer ONLY with valid JSON. No explanation, no markdown, no extra text.

Extract:
- builds_on: list of specific named prior methods/models this work explicitly builds on (e.g. ["LoRA", "FlashAttention"]). Empty list if none stated.
- improves_over: list of specific named prior methods/models this work explicitly claims to outperform. Empty list if none stated.
- compares_with: list of specific named baselines used for comparison. Empty list if none stated.

Rules:
- Only include names explicitly mentioned in the abstract.
- Do NOT infer or guess. If unsure, use empty list.
- Names should be short (1-4 words). No sentence fragments.
- Do NOT return generic words like "baseline", "prior work", "existing methods".

Abstract:
{abstract}

JSON response:"""


def _call_llm(abstract: str, client) -> Optional[Dict]:
    """Call Together AI and parse JSON response."""
    try:
        response = client.chat.completions.create(
            model=TOGETHER_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": PROMPT.format(abstract=abstract[:1500]),
            }],
        )
        text = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if "```" in text:
            import re
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        return json.loads(text)

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"LLM call failed: {e}")
        return None


def _clean_list(raw) -> List[str]:
    """Clean and validate a list of names from LLM output."""
    if not raw or not isinstance(raw, list):
        return []
    noise = {"baseline", "prior work", "existing methods", "previous work",
             "the model", "our model", "null", "none", "n/a"}
    result = []
    for item in raw:
        if not isinstance(item, str):
            continue
        item = item.strip().rstrip(".,")
        if not item or len(item) < 2 or len(item.split()) > 5:
            continue
        if item.lower() in noise:
            continue
        result.append(item)
    return result[:5]   # hard cap — LLMs love to over-list


def enrich_paper(arxiv_id: str, dry_run: bool = False) -> Optional[Dict]:
    """
    Run LLM enrichment on a single paper.
    Writes BUILDS_ON, IMPROVES, COMPARES_WITH edges to Neo4j.
    """
    settings = get_settings()
    if not settings.together_api_key:
        raise RuntimeError("TOGETHER_API_KEY not set")

    client = together.Together(api_key=settings.together_api_key)
    db     = Neo4jClient()
    db.connect()

    # Fetch abstract from Neo4j
    result = db.run_query(
        "MATCH (p:Paper {arxiv_id: $id}) RETURN p.abstract AS abstract, p.title AS title",
        {"id": arxiv_id},
    )
    if not result:
        logger.error(f"Paper {arxiv_id} not found in Neo4j")
        db.close()
        return None

    abstract = result[0]["abstract"] or ""
    title    = result[0]["title"] or ""
    logger.info(f"Enriching: {arxiv_id} — {title[:60]}")

    llm_result = _call_llm(abstract, client)
    if not llm_result:
        db.close()
        return None

    builds_on     = _clean_list(llm_result.get("builds_on"))
    improves_over = _clean_list(llm_result.get("improves_over"))
    compares_with = _clean_list(llm_result.get("compares_with"))

    logger.info(f"  builds_on:     {builds_on}")
    logger.info(f"  improves_over: {improves_over}")
    logger.info(f"  compares_with: {compares_with}")

    if not dry_run:
        for name in builds_on:
            db.create_method(name)
            db.run_query(
                "MATCH (p:Paper {arxiv_id: $arxiv_id}) MATCH (m:Method {name: $name}) MERGE (p)-[:BUILDS_ON]->(m)",
                {"arxiv_id": arxiv_id, "name": name},
            )
        for name in improves_over:
            db.create_method(name)
            db.run_query(
                "MATCH (p:Paper {arxiv_id: $arxiv_id}) MATCH (m:Method {name: $name}) MERGE (p)-[:IMPROVES]->(m)",
                {"arxiv_id": arxiv_id, "name": name},
            )
        for name in compares_with:
            db.create_method(name)
            db.run_query(
                "MATCH (p:Paper {arxiv_id: $arxiv_id}) MATCH (m:Method {name: $name}) MERGE (p)-[:COMPARES_WITH]->(m)",
                {"arxiv_id": arxiv_id, "name": name},
            )

    db.close()
    return {
        "arxiv_id":     arxiv_id,
        "builds_on":    builds_on,
        "improves_over": improves_over,
        "compares_with": compares_with,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="On-demand LLM enrichment for specific papers")
    parser.add_argument("--arxiv-id",  type=str,             help="Single arXiv ID")
    parser.add_argument("--arxiv-ids", type=str, nargs="+",  help="Multiple arXiv IDs")
    parser.add_argument("--dry-run",   action="store_true",  help="Print only, no writes")
    args = parser.parse_args()

    ids = []
    if args.arxiv_id:
        ids.append(args.arxiv_id)
    if args.arxiv_ids:
        ids.extend(args.arxiv_ids)

    if not ids:
        print("Provide --arxiv-id or --arxiv-ids")
        exit(1)

    for arxiv_id in ids:
        result = enrich_paper(arxiv_id, dry_run=args.dry_run)
        if result:
            print(f"\n✅ {arxiv_id}: {result}")
        else:
            print(f"\n❌ {arxiv_id}: failed")
        time.sleep(1.0)