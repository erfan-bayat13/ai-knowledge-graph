# kg/enrichment/openalex.py
# Phase 1: fetch citation_velocity, author influence (h-index), institution data per paper
# OpenAlex is fully free, no API key needed (polite pool: use email in User-Agent)
# Returns per-paper: citation_velocity, authors with h_index and institution

import logging
import os
import time
from typing import Optional
import feedparser


import httpx

from kg.utils.config import get_settings

logger = logging.getLogger(__name__)

_BASE = "https://api.openalex.org"
_TIMEOUT = 15.0
_RETRY_WAIT = 5.0


def _user_agent() -> str:
    """OpenAlex polite pool: include contact email to get better rate limits."""
    settings = get_settings()
    email = getattr(settings, "openalex_email", "") or "research@example.com"
    return f"ResearchKG/1.0 (mailto:{email})"


def _openalex_api_key() -> Optional[str]:
    settings = get_settings()
    return getattr(settings, "openalex_api_key", None) or os.getenv("OPENALEX_API_KEY")


def _openalex_headers() -> dict:
    headers = {"User-Agent": _user_agent()}
    api_key = _openalex_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _fetch_arxiv_title(arxiv_id: str) -> Optional[str]:
    """Fetch paper title from arXiv API."""
    try:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        feed = feedparser.parse(url)
        if feed.entries:
            return feed.entries[0].title
    except Exception:
        pass
    return None


def fetch_paper(arxiv_id: str) -> Optional[dict]:
    """
    Fetch OpenAlex data for a paper by arXiv ID.

    Returns dict with keys:
        citation_velocity   — citations per year (float)
        authors             — list of {name, h_index, institution_name, ror_id}
    Returns None on any failure.
    """
    # Try DOI first (fastest and deterministic)
    doi = f"10.48550/arXiv.{arxiv_id}"
    url = f"{_BASE}/works/https://doi.org/{doi}"

    headers = _openalex_headers()
    params = {}

    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(url, headers=headers)

    # If DOI lookup fails → fallback to title search
    if resp.status_code == 404:
        logger.debug(f"OpenAlex DOI lookup failed for {arxiv_id}, trying title search")

        title = _fetch_arxiv_title(arxiv_id)
        if not title:
            return None

        search_url = f"{_BASE}/works"
        params = {"search": title}

        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(search_url, headers=headers, params=params)

        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None

        data = results[0]

    else:
        resp.raise_for_status()
        data = resp.json()
    headers = _openalex_headers()
    params = {}
    api_key = _openalex_api_key()
    if api_key:
        params["api_key"] = api_key

    for attempt in range(3):
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(url, headers=headers, params=params)

            if resp.status_code == 404:
                logger.debug(f"OpenAlex: paper not found — {arxiv_id}")
                return None

            if resp.status_code == 429:
                logger.warning(f"OpenAlex rate limit — sleeping {_RETRY_WAIT}s")
                time.sleep(_RETRY_WAIT)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Citation velocity = cited_by_count / max(age_in_years, 1)
            pub_year = (data.get("publication_year") or 0)
            from datetime import datetime
            current_year = datetime.now().year
            age = max(current_year - pub_year, 1) if pub_year else 1
            citation_velocity = round((data.get("cited_by_count") or 0) / age, 4)

            # Author details
            authors = []
            for authorship in data.get("authorships", []):
                author_obj = authorship.get("author") or {}
                name = author_obj.get("display_name", "")
                if not name:
                    continue

                # h-index from author summary endpoint (cached if already fetched)
                h_index = _fetch_author_hindex(author_obj.get("id", ""))

                # Institution
                institution_name = ""
                ror_id = ""
                insts = authorship.get("institutions") or []
                if insts:
                    institution_name = insts[0].get("display_name", "")
                    ror_id = insts[0].get("ror", "") or ""

                authors.append({
                    "name":             name,
                    "h_index":          h_index,
                    "institution_name": institution_name,
                    "ror_id":           ror_id,
                })

            return {
                "citation_velocity": citation_velocity,
                "authors":           authors,
            }

        except httpx.HTTPStatusError as e:
            logger.warning(f"OpenAlex HTTP error for {arxiv_id}: {e}")
            break
        except Exception as e:
            logger.warning(f"OpenAlex request failed for {arxiv_id} (attempt {attempt+1}): {e}")
            time.sleep(1.0)

    return None

# Simple in-process cache to avoid re-fetching the same author within a run
_author_cache: dict = {}


def _fetch_author_hindex(author_id_url: str) -> int:
    """Fetch h-index for an author via their OpenAlex ID URL. Returns 0 on failure."""
    if not author_id_url:
        return 0

    # Normalize to OpenAlex API author endpoint (works returns openalex.org page URLs)
    api_url = author_id_url
    if author_id_url.startswith("https://openalex.org/"):
        api_url = author_id_url.replace("https://openalex.org/", f"{_BASE}/authors/")
    elif author_id_url.startswith("http://openalex.org/"):
        api_url = author_id_url.replace("http://openalex.org/", f"{_BASE}/authors/")

    if api_url in _author_cache:
        return _author_cache[api_url]

    headers = _openalex_headers()
    params = {}
    api_key = _openalex_api_key()
    if api_key:
        params["api_key"] = api_key

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(api_url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        h = data.get("summary_stats", {}).get("h_index", 0) or 0
        _author_cache[api_url] = h
        return h
    except Exception:
        pass

    _author_cache[api_url] = 0
    return 0
