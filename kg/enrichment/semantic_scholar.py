# kg/enrichment/semantic_scholar.py
# Phase 1: fetch citation_count, influential_citation_count, references, year, venue per paper
# Uses the Semantic Scholar public API (no key required, but key unlocks higher rate limits)
# References list used to build CITES edges in Neo4j

import logging
import time
from typing import Optional

import httpx

from kg.utils.config import get_settings

logger = logging.getLogger(__name__)

# Public API base — append your key via header for 100 req/s vs 1 req/s unauthenticated
_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "citationCount,influentialCitationCount,references,year,venue,externalIds"
_REF_FIELDS = "externalIds,title"
_TIMEOUT = 15.0
_RETRY_WAIT = 5.0   # seconds to wait after a 429


def _headers() -> dict:
    """Return auth header if SEMANTIC_SCHOLAR_API_KEY is set, else empty dict."""
    settings = get_settings()
    key = getattr(settings, "semantic_scholar_api_key", "")
    return {"x-api-key": key} if key else {}


def fetch_paper(arxiv_id: str) -> Optional[dict]:
    """
    Fetch metadata for a single paper by arXiv ID from Semantic Scholar.

    Returns dict with keys:
        citation_count, influential_citation_count, year, venue,
        references  — list of arXiv IDs this paper cites (strings, may be empty)
    Returns None on any failure.
    """
    url = f"{_BASE}/paper/arXiv:{arxiv_id}"
    params = {"fields": _FIELDS}

    for attempt in range(3):
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(url, params=params, headers=_headers())

            if resp.status_code == 404:
                logger.debug(f"S2: paper not found — {arxiv_id}")
                return None

            if resp.status_code == 429:
                logger.warning(f"S2 rate limit hit — sleeping {_RETRY_WAIT}s")
                time.sleep(_RETRY_WAIT)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Extract arXiv IDs of references (papers this paper cites)
            refs = []
            for ref in data.get("references", []):
                ext = ref.get("externalIds") or {}
                ref_arxiv = ext.get("ArXiv")
                if ref_arxiv:
                    refs.append(ref_arxiv.split("v")[0])   # strip version suffix

            return {
                "citation_count":              data.get("citationCount") or 0,
                "influential_citation_count":  data.get("influentialCitationCount") or 0,
                "year":                        data.get("year"),
                "venue":                       data.get("venue") or "",
                "references":                  refs,
            }

        except httpx.HTTPStatusError as e:
            logger.warning(f"S2 HTTP error for {arxiv_id}: {e}")
            break
        except Exception as e:
            logger.warning(f"S2 request failed for {arxiv_id} (attempt {attempt+1}): {e}")
            time.sleep(1.0)

    return None
