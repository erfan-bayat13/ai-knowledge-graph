# kg/enrichment/openalex.py
# Phase 1: fetch citation data, author influence (h-index), institution data per paper
# OpenAlex is fully free, no API key needed (polite pool: use email in User-Agent)
#
# Lookup strategy (in order):
#   1. DOI lookup  — exact, fast, trustworthy. Used when OpenAlex has indexed the paper.
#   2. Title search — fuzzy fallback for older papers (>30 days) not found via DOI.
#                     Skipped for papers <30 days old — too new to be indexed, and title
#                     search returns unrelated well-cited papers (false high citation counts).
#
# citation_source field on return dict:
#   "doi"         — DOI matched exactly, citations are accurate
#   "title_match" — matched via title search, citations are an estimate
#   "unindexed"   — paper is too new (<30 days), citations set to 0

import logging
import os
import time
from datetime import datetime
from typing import Optional

import feedparser
import httpx

from kg.utils.config import get_settings

logger = logging.getLogger(__name__)

# Silence httpx transport logs — they expose URLs and API keys in output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_BASE         = "https://api.openalex.org"
_TIMEOUT      = 15.0
_RETRY_WAIT   = 5.0
_MAX_RETRIES  = 3
_NEW_PAPER_DAYS = 30   # papers younger than this skip title search


# ── Auth / headers ─────────────────────────────────────────────────────────────

def _user_agent() -> str:
    settings = get_settings()
    email = getattr(settings, "openalex_email", "") or "research@example.com"
    return f"ResearchKG/1.0 (mailto:{email})"


def _openalex_api_key() -> Optional[str]:
    settings = get_settings()
    return getattr(settings, "openalex_api_key", None) or os.getenv("OPENALEX_API_KEY")


def _headers() -> dict:
    h = {"User-Agent": _user_agent()}
    key = _openalex_api_key()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


# ── Helpers ────────────────────────────────────────────────────────────────────

def _paper_age_days(published_date: Optional[str]) -> Optional[int]:
    """Return age in days from published_date string (YYYY-MM-DD). None if unparseable."""
    if not published_date:
        return None
    try:
        pub = datetime.strptime(published_date[:10], "%Y-%m-%d")
        return max((datetime.now() - pub).days, 0)
    except Exception:
        return None


def _fetch_arxiv_title(arxiv_id: str) -> Optional[str]:
    """Fetch paper title from arXiv API (used as title search fallback query)."""
    try:
        url  = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        feed = feedparser.parse(url)
        if feed.entries:
            return feed.entries[0].title
    except Exception:
        pass
    return None


def _parse_work(data: dict) -> dict:
    """
    Extract citation_velocity and authors from an OpenAlex work object.
    Shared between DOI path and title search path.
    """
    pub_year = data.get("publication_year") or 0
    age_years = max(datetime.now().year - pub_year, 1) if pub_year else 1
    citation_velocity = round((data.get("cited_by_count") or 0) / age_years, 4)

    authors = []
    for authorship in data.get("authorships", []):
        author_obj = authorship.get("author") or {}
        name = author_obj.get("display_name", "")
        if not name:
            continue

        h_index = _fetch_author_hindex(author_obj.get("id", ""))

        insts            = authorship.get("institutions") or []
        institution_name = insts[0].get("display_name", "") if insts else ""
        ror_id           = insts[0].get("ror", "")           if insts else ""

        authors.append({
            "name":             name,
            "h_index":          h_index,
            "institution_name": institution_name,
            "ror_id":           ror_id,
        })

    return {
        "citation_count":    data.get("cited_by_count") or 0,
        "citation_velocity": citation_velocity,
        "authors":           authors,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_paper(arxiv_id: str, published_date: Optional[str] = None) -> Optional[dict]:
    """
    Fetch OpenAlex data for a single paper by arXiv ID.

    Returns dict with keys:
        citation_count    — int
        citation_velocity — float  (citations per year)
        citation_source   — str    "doi" | "title_match" | "unindexed"
        authors           — list[{name, h_index, institution_name, ror_id}]

    Returns None on network failure / unrecoverable error.
    """
    doi_url = f"{_BASE}/works/https://doi.org/10.48550/arXiv.{arxiv_id}"
    hdrs    = _headers()

    for attempt in range(_MAX_RETRIES):
        try:
            # ── 1. DOI lookup ─────────────────────────────────────────────────
            resp = httpx.get(doi_url, headers=hdrs, timeout=_TIMEOUT)

            if resp.status_code == 429:
                wait = _RETRY_WAIT * (attempt + 1)
                logger.warning(f"OpenAlex rate limit — sleeping {wait}s")
                time.sleep(wait)
                continue

            if resp.status_code == 404:
                # ── 2. Age gate — skip title search for very new papers ───────
                age = _paper_age_days(published_date)

                if age is not None and age < _NEW_PAPER_DAYS:
                    logger.debug(
                        f"OpenAlex: {arxiv_id} too new ({age}d old) — "
                        f"skipping title search to avoid false citation matches"
                    )
                    return {
                        "citation_count":    0,
                        "citation_velocity": 0.0,
                        "citation_source":   "unindexed",
                        "authors":           [],
                    }

                # ── 3. Title search fallback (older papers only) ──────────────
                logger.debug(
                    f"OpenAlex: DOI 404 for {arxiv_id} "
                    f"(age={age}d) — trying title search"
                )
                title = _fetch_arxiv_title(arxiv_id)
                if not title:
                    logger.debug(f"OpenAlex: could not fetch arXiv title for {arxiv_id}")
                    return None

                search_resp = httpx.get(
                    f"{_BASE}/works",
                    headers=hdrs,
                    params={"search": title, "per-page": 1},
                    timeout=_TIMEOUT,
                )

                if search_resp.status_code == 429:
                    wait = _RETRY_WAIT * (attempt + 1)
                    logger.warning(f"OpenAlex rate limit on title search — sleeping {wait}s")
                    time.sleep(wait)
                    continue

                if not search_resp.is_success:
                    logger.warning(
                        f"OpenAlex title search failed for {arxiv_id}: "
                        f"{search_resp.status_code}"
                    )
                    return None

                results = search_resp.json().get("results", [])
                if not results:
                    logger.debug(f"OpenAlex: no title search results for {arxiv_id}")
                    return None

                parsed = _parse_work(results[0])
                parsed["citation_source"] = "title_match"
                logger.debug(
                    f"OpenAlex title_match: {arxiv_id} → "
                    f"cit={parsed['citation_count']} "
                    f"(matched: '{results[0].get('title', '')[:60]}')"
                )
                return parsed

            if not resp.is_success:
                logger.warning(
                    f"OpenAlex unexpected {resp.status_code} for {arxiv_id}"
                )
                return None

            # ── DOI hit — exact match ─────────────────────────────────────────
            parsed = _parse_work(resp.json())
            parsed["citation_source"] = "doi"
            return parsed

        except httpx.TimeoutException:
            logger.warning(f"OpenAlex timeout for {arxiv_id} (attempt {attempt + 1})")
            time.sleep(2.0)
        except Exception as e:
            logger.warning(
                f"OpenAlex request failed for {arxiv_id} "
                f"(attempt {attempt + 1}): {e}"
            )
            time.sleep(1.0)

    return None


# ── Author h-index (cached) ────────────────────────────────────────────────────

_author_cache: dict = {}


def _fetch_author_hindex(author_id_url: str) -> int:
    """Fetch h-index for an author via their OpenAlex ID URL. Returns 0 on failure."""
    if not author_id_url:
        return 0

    api_url = author_id_url
    if author_id_url.startswith("https://openalex.org/"):
        api_url = author_id_url.replace("https://openalex.org/", f"{_BASE}/authors/")
    elif author_id_url.startswith("http://openalex.org/"):
        api_url = author_id_url.replace("http://openalex.org/", f"{_BASE}/authors/")

    if api_url in _author_cache:
        return _author_cache[api_url]

    try:
        resp = httpx.get(api_url, headers=_headers(), timeout=10.0)
        resp.raise_for_status()
        h = resp.json().get("summary_stats", {}).get("h_index", 0) or 0
        _author_cache[api_url] = h
        return h
    except Exception:
        pass

    _author_cache[api_url] = 0
    return 0