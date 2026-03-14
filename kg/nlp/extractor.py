# kg/nlp/extractor.py
import json
import logging
import re
from typing import Optional
import together
from kg.utils.config import get_settings

logger = logging.getLogger(__name__)

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # better than 11B, still cheap

PROMPT = """\
Extract structured info from this AI research paper abstract.
Respond ONLY with valid JSON. No markdown, no explanation.

Extract:
- proposed_methods: list of method/model names this paper introduces (e.g. ["LoRA", "FlashAttention"]). Empty list if none.
- datasets: list of dataset/benchmark names used for evaluation (e.g. ["ImageNet", "GLUE"]). Empty list if none.
- builds_on: list of prior methods this work explicitly builds on. Empty list if none.

Rules:
- Only names explicitly stated in the abstract. Never infer.
- 1-4 words per name. No sentence fragments.
- No generic terms: "baseline", "prior work", "our model", "the method".

Abstract: {abstract}

JSON:"""

_client = None

def _get_client():
    global _client
    if _client is None:
        settings = get_settings()
        _client = together.Together(api_key=settings.together_api_key)
    return _client

def extract(abstract: str) -> dict:
    """Extract method/dataset/relationship names from an abstract."""
    empty = {"proposed_methods": [], "datasets": [], "builds_on": [], "source": "failed"}
    if not abstract or not abstract.strip():
        return empty
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": PROMPT.format(abstract=abstract[:1500])}],
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown fences if model wraps output
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        result = json.loads(text)
        result["source"] = "together_ai"
        return result
    except Exception as e:
        logger.warning(f"Extraction failed: {e}")
        return empty