# kg/nlp/extractor.py
#
# Replaces the old regex + LLM hybrid entirely.
#
# Strategy:
#   - Use spaCy dependency parsing to find method/dataset names from
#     sentence structure, not vocabulary lists.
#   - "we propose X"  → X is a PROPOSES candidate
#   - "evaluated on X" / "we benchmark on X" → X is an EVALUATED_ON candidate
#   - No .txt vocab files. No LLM at ingestion time.
#   - Vocabulary grows dynamically from what papers actually say.
#
# Requirements:
#   pip install spacy
#   python -m spacy download en_core_web_sm

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    _nlp = None
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")


# ── Verb lists ────────────────────────────────────────────────────────────────
# These match sentence intent, not surface vocabulary.

PROPOSAL_VERBS = {
    "propose", "present", "introduce", "develop", "call",
    "release", "build", "design", "create", "describe", "name",
}

EVAL_PHRASES = [
    r"evaluated?\s+on\s+([\w][\w\s\-]{1,40}?)(?:\s+dataset|\s+benchmark)?(?:\.|,|;|\n|$)",
    r"benchmarked?\s+on\s+([\w][\w\s\-]{1,40}?)(?:\s+dataset|\s+benchmark)?(?:\.|,|;|\n|$)",
    r"tested?\s+on\s+([\w][\w\s\-]{1,40}?)(?:\s+dataset|\s+benchmark)?(?:\.|,|;|\n|$)",
    r"(?:using|with)\s+the\s+([\w][\w\s\-]{1,30}?)\s+(?:dataset|benchmark|corpus)(?:\.|,|;|\n|$)",
]

# Names shorter than this are likely noise (acronyms of 1 char, stray words)
MIN_NAME_LEN = 2
# Names longer than this are likely sentence fragments
MAX_NAME_WORDS = 6

# Common words that slip through dependency parsing — discard them
NOISE = {
    "a", "an", "the", "our", "this", "that", "we", "it", "which",
    "new", "novel", "simple", "efficient", "effective", "large", "small",
    "deep", "neural", "network", "model", "method", "approach",
    "framework", "system", "algorithm", "technique", "architecture",
    "solution", "strategy", "paper", "work", "study", "analysis",
    "null", "none", "unknown", "not", "several", "various",
}


# ── Core extraction ───────────────────────────────────────────────────────────

def _clean(name: str) -> Optional[str]:
    """Normalise and validate a candidate name."""
    name = name.strip().rstrip(".,;:()")
    words = name.split()

    # Strip leading noise words
    while words and words[0].lower() in NOISE:
        words = words[1:]
    if not words:
        return None

    name = " ".join(words)

    if len(name) < MIN_NAME_LEN:
        return None
    if len(words) > MAX_NAME_WORDS:
        return None
    if name.lower() in NOISE:
        return None

    return name


def _extract_proposed_spacy(doc) -> List[str]:
    """
    Walk the dependency tree looking for:
      subject=we/this paper  +  verb=PROPOSAL_VERB  +  object=<name>

    This catches "we propose X", "this paper introduces X",
    "we call our method X", regardless of surrounding words.
    """
    candidates = []

    for token in doc:
        if token.lemma_.lower() not in PROPOSAL_VERBS:
            continue

        # Check subject is first-person or "this paper/work"
        subj_ok = False
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                if child.text.lower() in ("we", "our", "i"):
                    subj_ok = True
                elif child.text.lower() in ("paper", "work", "study"):
                    subj_ok = True
                # "this paper proposes X"
                elif child.text.lower() == "this":
                    subj_ok = True

        if not subj_ok:
            continue

        # Collect direct objects and their noun phrase spans
        for child in token.children:
            if child.dep_ in ("dobj", "obj", "attr", "oprd"):
                # Get the full noun phrase
                end_i = child.right_edge.i + 1
                for token in child.sent[child.i: child.right_edge.i + 1]:
                    if token.dep_ in ("prep", "relcl", "advcl") and token.i > child.i:
                        end_i = token.i
                        break
                span = child.sent[child.left_edge.i: end_i]
                name = _clean(span.text)
                if name:
                    candidates.append(name)

    return list(dict.fromkeys(candidates))


def _extract_proposed_regex(text: str) -> List[str]:
    """
    Fallback regex for proposal patterns — catches parenthetical acronyms
    like "we propose Long Name (ACRONYM)" which spaCy often misses.
    """
    patterns = [
        # "we propose/introduce/present Long Name (ACRONYM)"
        r"[Ww]e\s+(?:propose|present|introduce|develop|describe|build|release)\s+"
        r"[\w][\w\s\-]{1,60}?\(([A-Z][A-Za-z0-9\-]{1,20})\)",

        # "we propose X," or "we propose X which/that"
        r"[Ww]e\s+(?:propose|present|introduce|develop|describe|build|release)\s+"
        r"([\w][\w\s\-]{1,40}?)(?:,|\s+which|\s+that|\s+—|\.|$)",

        # "this paper proposes X"
        r"[Tt]his\s+(?:paper|work)\s+(?:proposes|presents|introduces|develops)\s+"
        r"([\w][\w\s\-]{1,40}?)(?:,|\s+which|\s+that|\.|$)",
    ]
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            name = _clean(match.group(1))
            if name:
                candidates.append(name)
    return list(dict.fromkeys(candidates))


def _extract_datasets_regex(text: str) -> List[str]:
    """
    Extract dataset names from evaluation sentences.
    Uses sentence patterns, not a vocabulary list.
    """
    candidates = []
    for pattern in EVAL_PHRASES:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            name = _clean(match.group(1))
            if name:
                candidates.append(name)
    return list(dict.fromkeys(candidates))


# ── Public API ────────────────────────────────────────────────────────────────

def extract(abstract: str) -> Dict:
    """
    Extract structured information from a paper abstract.

    Returns:
        {
            "proposed_methods": [...],   # names of methods/models proposed
            "datasets":         [...],   # datasets used for evaluation
            "source":           "spacy"  # always — no LLM fallback here
        }
    """
    empty = {"proposed_methods": [], "datasets": [], "source": "spacy"}

    if not abstract or not abstract.strip():
        return empty

    text = " ".join(abstract.split())   # normalise whitespace

    # ── spaCy pass ────────────────────────────────────────────────────────────
    proposed = []
    if _nlp is not None:
        try:
            doc = _nlp(text[:1000])   # cap at 1000 chars to keep it fast
            proposed = _extract_proposed_spacy(doc)
        except Exception as e:
            logger.warning(f"spaCy parse failed: {e}")

    # ── Regex fallback for proposals (catches acronyms spaCy misses) ──────────
    regex_proposed = _extract_proposed_regex(text)

    # Merge, dedup, keep spaCy results first (higher confidence)
    all_proposed = list(dict.fromkeys(proposed + regex_proposed))

    # ── Dataset extraction (regex only — dependency parse adds little here) ───
    datasets = _extract_datasets_regex(text)

    return {
        "proposed_methods": all_proposed,
        "datasets":         datasets,
        "source":           "spacy",
    }