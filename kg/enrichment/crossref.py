# kg/enrichment/crossref.py
# Drop-in replacement for semantic_scholar.py
# Uses the CrossRef API (api.crossref.org) — free, no key required.
# Polite pool: include your email in CROSSREF_EMAIL env var / config for better rate limits.
#
# Returns the same dict shape as semantic_scholar.fetch_paper() so runner.py needs zero changes:
#   citation_count              int
#   influential_citation_count  int   (CrossRef has no equivalent — always 0, kept for compat)
#   year                        int | None
#   venue                       str
#   references                  list[str]   arXiv IDs this paper cites
#
# Swap instructions (2 lines in runner.py):
#   BEFORE:  from kg.enrichment import semantic_scholar as s2
#   AFTER:   from kg.enrichment import crossref as s2
#   (the call site  s2.fetch_paper(arxiv_id)  stays identical)
#
# Rate limits:
#   Polite pool (email set)  → ~50 req/s sustained, no hard cap
#   Anonymous                → shared pool, much slower — always set the email
#
# CrossRef coverage notes:
#   - citation_count   comes from "is-referenced-by-count" (Crossref Cited-By data)
#   - references list  comes from "reference" array; only DOIs/arXiv IDs are resolved
#   - ~85-90% of arXiv CS/ML papers are indexed; older or preprint-only papers may 404

import logging
import os
import re
import time
from typing import Optional
import feedparser


import httpx

from kg.utils.config import get_settings

logger = logging.getLogger(__name__)

_BASE      = "https://api.crossref.org/works"
_TIMEOUT   = 20.0
_RETRY_WAIT = 10.0   # seconds to wait after a 429 / 503
_MAX_RETRIES = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mailto() -> str:
    """
    CrossRef polite pool requires a contact email in the User-Agent.
    Set CROSSREF_EMAIL in your .env (or config) for priority routing.
    Falls back to the OpenAlex email if already set, then a placeholder.
    """
    settings = get_settings()
    email = (
        os.getenv("CROSSREF_EMAIL")
        or getattr(settings, "crossref_email", None)
        or getattr(settings, "openalex_email", None)
        or "research@example.com"
    )
    return email


def _headers() -> dict:
    return {"User-Agent": f"ResearchKG/1.0 (mailto:{_mailto()})"}


def _arxiv_doi(arxiv_id: str) -> str:
    """Return the canonical arXiv DOI for a given arXiv ID."""
    return f"10.48550/arXiv.{arxiv_id}"


_ARXIV_RE = re.compile(
    r"(?:arxiv[.:\s/]|10\.48550/arxiv\.)(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)


def _fetch_arxiv_title(arxiv_id: str) -> Optional[str]:
    """Fetch paper title from arXiv."""
    try:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        feed = feedparser.parse(url)
        if feed.entries:
            return feed.entries[0].title
    except Exception:
        pass
    return None


def _extract_arxiv_id(ref: dict) -> Optional[str]:
    """
    Try to pull an arXiv ID from a CrossRef reference object.
    Checks: DOI field (10.48550/arXiv.XXXX), unstructured text, and arxiv-id field.
    Returns a clean arXiv ID string (no version suffix) or None.
    """
    # 1. DOI field — fastest path for arXiv preprints
    doi = (ref.get("DOI") or "").lower()
    if "10.48550" in doi:
        # doi looks like 10.48550/arxiv.2106.05233
        match = re.search(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", doi, re.IGNORECASE)
        if match:
            return match.group(1)

    # 2. Explicit arxiv-id field (some records carry this)
    arxiv_field = ref.get("arxiv-id") or ref.get("arxivId") or ""
    if arxiv_field:
        clean = arxiv_field.split("v")[0].strip()
        if re.match(r"\d{4}\.\d{4,5}", clean):
            return clean

    # 3. Unstructured citation text — look for patterns like arXiv:2106.05233
    unstructured = ref.get("unstructured") or ""
    if unstructured:
        match = _ARXIV_RE.search(unstructured)
        if match:
            return match.group(1).split("v")[0]

    return None


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_paper(arxiv_id: str) -> Optional[dict]:
    """
    Fetch citation metadata for a single paper by arXiv ID via CrossRef.

    Returns dict with keys matching semantic_scholar.fetch_paper():
        citation_count              — int   (CrossRef is-referenced-by-count)
        influential_citation_count  — int   (always 0, no CrossRef equivalent)
        year                        — int | None
        venue                       — str   (journal/conference name)
        references                  — list[str]  arXiv IDs this paper cites

    Returns None if the paper is not found or all retries fail.
    """
    doi = _arxiv_doi(arxiv_id)
    url = f"{_BASE}/{doi}"

    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(url, headers=_headers())

    # ── fallback if DOI not found ─────────────────────────────
    if resp.status_code == 404:
        logger.debug(f"CrossRef DOI lookup failed for {arxiv_id}, trying title search")

        title = _fetch_arxiv_title(arxiv_id)
        if not title:
            return None

        search_url = _BASE
        params = {"query.title": title, "rows": 1}

        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(search_url, headers=_headers(), params=params)

        resp.raise_for_status()

        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return None

        data = items[0]

    else:
        resp.raise_for_status()
        data = resp.json().get("message", {})

    for attempt in range(_MAX_RETRIES):
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(url, headers=_headers())

            if resp.status_code == 404:
                logger.debug(f"CrossRef: paper not found — {arxiv_id} (DOI: {doi})")
                return None

            if resp.status_code in (429, 503):
                wait = _RETRY_WAIT * (attempt + 1)   # simple linear backoff
                logger.warning(f"CrossRef rate limit ({resp.status_code}) — sleeping {wait}s")
                time.sleep(wait)
                continue

            if resp.status_code == 400:
                logger.debug(f"CrossRef: bad request for {arxiv_id}")
                return None

            resp.raise_for_status()
            data = resp.json().get("message", {})

            # ── citation count ────────────────────────────────────────────────
            citation_count = data.get("is-referenced-by-count") or 0

            # ── year ─────────────────────────────────────────────────────────
            year = None
            # published-print > published-online > created — prefer the earliest real date
            for date_field in ("published-print", "published-online", "created"):
                date_parts = (data.get(date_field) or {}).get("date-parts", [[]])[0]
                if date_parts and date_parts[0]:
                    year = int(date_parts[0])
                    break

            # ── venue ─────────────────────────────────────────────────────────
            # container-title is a list; take the first non-empty entry
            container = data.get("container-title") or []
            venue = next((t for t in container if t), "")
            if not venue:
                # fall back to event name (conference proceedings)
                venue = (data.get("event") or {}).get("name", "")

            # ── references → arXiv IDs ────────────────────────────────────────
            raw_refs = data.get("reference") or []
            refs: list[str] = []
            seen: set[str] = set()
            for ref in raw_refs:
                arxiv_ref = _extract_arxiv_id(ref)
                if arxiv_ref and arxiv_ref not in seen:
                    seen.add(arxiv_ref)
                    refs.append(arxiv_ref)

            logger.debug(
                f"CrossRef: {arxiv_id} → citations={citation_count}, "
                f"year={year}, venue='{venue}', refs={len(refs)}"
            )

            return {
                "citation_count":             citation_count,
                "influential_citation_count": 0,   # no CrossRef equivalent
                "year":                       year,
                "venue":                      venue,
                "references":                 refs,
            }

        except httpx.HTTPStatusError as e:
            logger.warning(f"CrossRef HTTP error for {arxiv_id}: {e}")
            break
        except httpx.TimeoutException:
            logger.warning(f"CrossRef timeout for {arxiv_id} (attempt {attempt + 1})")
            time.sleep(2.0)
        except Exception as e:
            logger.warning(f"CrossRef request failed for {arxiv_id} (attempt {attempt + 1}): {e}")
            time.sleep(1.0)

    return None