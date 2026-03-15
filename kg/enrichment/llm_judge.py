# kg/enrichment/llm_judge.py
# Phase 1: LLM fallback judge — NOT a primary enrichment step
# Triggered only when: (a) both APIs return nothing, or (b) citation_count=0 but paper
# appears in other papers' reference lists. Sends batches of ~20 abstracts to flag anomalies
# or fill missing fields. Keeps LLM usage cheap and targeted, not routine.

import json
import logging
import re
from typing import List

import together

from kg.utils.config import get_settings

logger = logging.getLogger(__name__)

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
BATCH_SIZE = 20  # papers per LLM call

PROMPT = """\
You are reviewing a batch of AI research paper abstracts where citation API data is missing or inconsistent.
For each paper, estimate:
- approx_citations: rough integer estimate of citations (0 if very new/obscure, >0 if you recognize it)
- is_influential: true if this looks like a widely-cited foundational paper, false otherwise
- note: one sentence explaining any data anomaly or reason for missing data

Respond ONLY with a JSON array, one object per paper, in the same order.
No markdown, no explanation outside the JSON.

Papers:
{papers_json}

JSON:"""


def _get_client():
    settings = get_settings()
    if not settings.together_api_key:
        raise RuntimeError("TOGETHER_API_KEY not set — LLM judge unavailable")
    return together.Together(api_key=settings.together_api_key)


def judge_batch(papers: List[dict]) -> List[dict]:
    """
    Send a batch of papers (each with arxiv_id + abstract) to the LLM for fallback judgment.

    Returns list of dicts with keys: arxiv_id, approx_citations, is_influential, note.
    Returns empty list on any failure — callers should treat this as non-critical.
    """
    if not papers:
        return []

    client = _get_client()

    # Build compact input (arxiv_id + first 300 chars of abstract)
    compact = [
        {"arxiv_id": p["arxiv_id"], "abstract": (p.get("abstract") or "")[:300]}
        for p in papers[:BATCH_SIZE]
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": PROMPT.format(
                papers_json=json.dumps(compact, indent=2)
            )}],
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

        results = json.loads(text)
        if not isinstance(results, list):
            raise ValueError("LLM judge response is not a list")

        # Re-attach arxiv_id if the LLM didn't include it
        for i, r in enumerate(results):
            if "arxiv_id" not in r and i < len(compact):
                r["arxiv_id"] = compact[i]["arxiv_id"]

        return results

    except Exception as e:
        logger.warning(f"LLM judge batch failed: {e}")
        return []


def should_judge(s2_result, oa_result, cited_in_graph: bool) -> bool:
    """
    Decide whether to trigger the LLM judge for a paper.
    Triggered when both APIs return nothing OR citation_count=0 but paper is cited by others.
    """
    api_missing = s2_result is None and oa_result is None
    zero_but_cited = (
        s2_result is not None
        and s2_result.get("citation_count", 0) == 0
        and cited_in_graph
    )
    return api_missing or zero_but_cited
