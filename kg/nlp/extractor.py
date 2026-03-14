# kg/nlp/extractor.py

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
import together

logger = logging.getLogger(__name__)


# ─── Load Vocabularies ────────────────────────────────────────────────────────

def _load_vocab(filename: str) -> List[str]:
    path = Path(__file__).parent / "vocabulary" / filename
    terms = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return sorted(terms, key=len, reverse=True)

METHODS  = _load_vocab("methods.txt")
DATASETS = _load_vocab("datasets.txt")
TASKS    = _load_vocab("tasks.txt")


# ─── Layer 1: Rule-Based ─────────────────────────────────────────────────────

PROPOSAL_PATTERNS = [
    # Parenthetical acronym FIRST: "we propose Long Name (ACRONYM)"
    r"[Ww]e (?:propose|present|introduce|develop|call|release|train|build|design|create|describe)\s+[\w][\w\s\-]{1,60}?\(([A-Z][A-Za-z0-9\-]{1,20})\)",

    # "we propose/introduce X"
    r"[Ww]e (?:propose|present|introduce|develop|call|release|train|build|design|create|describe)\s+([\w][\w\s\-]{1,40}?)(?:,|\.|which|that| a | an |\n|$)",

    # "this paper proposes X"
    r"[Tt]his (?:paper|work) (?:proposes|presents|introduces|develops|releases|describes)\s+([\w][\w\s\-]{1,40}?)(?:,|\.|which|that|\n|$)",

    # "we call our method X"
    r"[Ww]e (?:name|call) our (?:method|model|approach|framework|system|algorithm)\s+([\w][\w\s\-]{1,40}?)(?:,|\.|which|that|\n|$)",

    # "X, our proposed method"
    r"([\w][\w\s\-]{1,40}?),\s+our (?:proposed|novel|new)\s+(?:method|model|approach|framework|system|algorithm)",

    # "dubbed/termed/named X"
    r"(?:dubbed|termed|named)\s+([\w][\w\s\-]{1,40}?)(?:,|\.|which|that|\n|$)",

    # "called X" mid-sentence
    r"(?:called|named)\s+([A-Z][A-Za-z0-9][A-Za-z0-9\s\-]{0,30}?)(?:,|\.|which|that|\n|$)",
]

PROPOSAL_BLOCKLIST = {
    "a", "an", "the", "our", "this", "that", "we", "it",
    "new", "novel", "simple", "efficient", "effective",
    "large", "small", "deep", "neural", "network", "model",
    "method", "approach", "framework", "system", "algorithm",
    "technique", "strategy", "solution", "architecture",
    "null", "none", "unknown",   # ← catch LLM bleed-through
}


def _clean_proposals(matches: List[str]) -> List[str]:
    cleaned = []
    for m in matches:
        m = m.strip().rstrip(".,")
        words = m.split()

        if len(m) < 2 or len(words) > 6:
            continue

        # Strip leading conjunction/verb leakage
        while words and words[0].lower() in {
            "and", "or", "the", "a", "an", "also", "here",
            "release", "develop", "present", "propose", "introduce",
        }:
            words = words[1:]
        if not words:
            continue

        m = " ".join(words)

        if m.lower() in PROPOSAL_BLOCKLIST:
            continue
        if words[0].lower() in PROPOSAL_BLOCKLIST:
            continue

        cleaned.append(m)

    return list(dict.fromkeys(cleaned))


def extract_layer1(abstract: str) -> Dict:
    if not abstract or not abstract.strip():
        return {
            "proposed_methods": [],
            "used_methods": [],
            "datasets": [],
            "tasks": [],
            "needs_llm": False,   # empty abstract → don't waste LLM call
        }

    text_normalised = " ".join(abstract.split())
    text_lower = text_normalised.lower()

    found_methods  = [m for m in METHODS  if m.lower() in text_lower]
    found_datasets = [d for d in DATASETS if d.lower() in text_lower]
    found_tasks    = [t for t in TASKS    if t.lower() in text_lower]

    proposed = []
    for pattern in PROPOSAL_PATTERNS:
        matches = re.findall(pattern, text_normalised, re.IGNORECASE)
        proposed.extend(matches)
    proposed = _clean_proposals(proposed)

    word_count = len(text_normalised.split())
    needs_llm = len(proposed) == 0 and word_count > 30

    return {
        "proposed_methods": proposed,
        "used_methods":     found_methods,
        "datasets":         found_datasets,
        "tasks":            found_tasks,
        "needs_llm":        needs_llm,
    }


# ─── Layer 3: LLM Extraction ─────────────────────────────────────────────────

# Tighter prompt — explicitly penalises guessing
EXTRACTION_PROMPT = """\
You are extracting structured data from an AI research paper abstract.
Respond ONLY with valid JSON. No markdown, no explanation, no preamble.

{{
  "proposed_method": null,
  "baseline_methods": [],
  "datasets": [],
  "task": null,
  "improves_on": null
}}

Rules (read carefully):
- proposed_method: ONLY if the abstract explicitly says the paper proposes/introduces/presents a named method or model. Must be a SHORT identifier (e.g. "FlashAttention", "GPT-3"). If no named method is clearly proposed, return null.
- baseline_methods: ONLY method names explicitly mentioned in the abstract. Max 3. Empty list if none.
- datasets: ONLY dataset names explicitly mentioned (e.g. "ImageNet", "GLUE"). Empty list if none.
- task: ONE short phrase for the problem being solved. null if unclear.
- improves_on: ONLY if the abstract explicitly says this work improves on a specific named prior method. null otherwise.

DO NOT guess or infer. If unsure → null or empty list.
DO NOT return the string "null" — use JSON null (no quotes).

Abstract:
{abstract}"""


def _sanitise_llm_output(raw: dict) -> dict:
    """
    Clean LLM output before using it.
    Handles: string "null", empty strings, self-referential values.
    """
    NULL_STRINGS = {"null", "none", "n/a", "unknown", "", "not mentioned",
                    "not applicable", "not specified"}

    def clean_str(v) -> Optional[str]:
        if v is None:
            return None
        v = str(v).strip()
        if v.lower() in NULL_STRINGS:
            return None
        # Reject if it looks like a sentence (too long to be a name)
        if len(v.split()) > 6:
            return None
        return v

    def clean_list(v) -> List[str]:
        if not v or not isinstance(v, list):
            return []
        result = []
        for item in v:
            cleaned = clean_str(item)
            if cleaned:
                result.append(cleaned)
        return result[:3]  # hard cap at 3

    proposed  = clean_str(raw.get("proposed_method"))
    improves  = clean_str(raw.get("improves_on"))
    baselines = clean_list(raw.get("baseline_methods"))
    datasets  = clean_list(raw.get("datasets"))
    task      = clean_str(raw.get("task"))

    # Prevent self-loop: improves_on must differ from proposed_method
    if improves and proposed and improves.lower() == proposed.lower():
        improves = None

    # Prevent improves_on from being a hallucination that matches proposed
    # If proposed is None there's nothing to improve on
    if proposed is None:
        improves = None

    return {
        "proposed_method":  proposed,
        "baseline_methods": baselines,
        "datasets":         datasets,
        "task":             task,
        "improves_on":      improves,
    }


def extract_layer3(abstract: str, client: together.Together) -> dict:
    empty = {
        "proposed_method":  None,
        "baseline_methods": [],
        "datasets":         [],
        "task":             None,
        "improves_on":      None,
    }

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT.format(
                    abstract=abstract[:1500]
                )
            }]
        )

        text = response.choices[0].message.content.strip()

        # Strip markdown fences
        if "```" in text:
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        raw = json.loads(text)
        return _sanitise_llm_output(raw)

    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.warning(f"LLM extraction failed: {e}")
        return empty


# ─── Combined Pipeline ────────────────────────────────────────────────────────

def extract(abstract: str, client=None) -> Dict:
    l1 = extract_layer1(abstract)

    if l1["needs_llm"] and client is not None:
        l3 = extract_layer3(abstract, client)
        return {
            "proposed_methods": [l3["proposed_method"]] if l3["proposed_method"] else [],
            "used_methods":     list(dict.fromkeys(l3["baseline_methods"] + l1["used_methods"])),
            "datasets":         list(dict.fromkeys(l3["datasets"] + l1["datasets"])),
            "tasks":            [l3["task"]] if l3["task"] else l1["tasks"],
            "improves_on":      l3["improves_on"],
            "source":           "llm",
        }

    return {
        "proposed_methods": l1["proposed_methods"],
        "used_methods":     l1["used_methods"],
        "datasets":         l1["datasets"],
        "tasks":            l1["tasks"],
        "improves_on":      None,
        "source":           "rule_based",
    }