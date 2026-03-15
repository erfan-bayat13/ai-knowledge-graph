# kg/clustering/naming.py
# Phase 3: LLM topic naming — send 10-20 paper titles per cluster, get a short topic name
# Uses Together AI (Llama) — same integration as existing LLM calls

import json
import logging
import re
import time
from typing import Dict, List

import together

from kg.utils.config import get_settings

logger = logging.getLogger(__name__)

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

PROMPT = """\
You are naming a research cluster. Below are paper titles from one cluster of AI papers.
Give this cluster a short, precise topic name (2-5 words max). Examples of good names:
"LLM Agents and Tool Use", "Diffusion Transformers", "Mechanistic Interpretability",
"Distributed Training Systems", "Model Compression and Quantization"

Respond with ONLY the topic name. No explanation, no quotes, no punctuation.

Paper titles:
{titles}

Topic name:"""


def _get_client():
    settings = get_settings()
    if not settings.together_api_key:
        raise RuntimeError("TOGETHER_API_KEY not set — topic naming unavailable")
    return together.Together(api_key=settings.together_api_key)


def name_cluster(titles: List[str]) -> str:
    """Ask the LLM to name a cluster given a sample of its paper titles. Returns a short string."""
    client = _get_client()
    sample = titles[:20]  # cap at 20 titles per request

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": PROMPT.format(titles="\n".join(f"- {t}" for t in sample))}],
        )
        name = response.choices[0].message.content.strip()
        # Strip any stray quotes or punctuation
        name = re.sub(r'^["\']|["\']$', "", name).strip()
        return name or f"Cluster {sample[0][:30]}"

    except Exception as e:
        logger.warning(f"Topic naming failed: {e}")
        return f"Unnamed Cluster"


def name_clusters(cluster_papers: Dict[int, List[dict]]) -> Dict[int, str]:
    """
    Name all clusters. cluster_papers: {cluster_id: [paper_dict, ...]}.
    Returns {cluster_id: topic_name_string}.
    """
    names = {}
    for cluster_id, papers in cluster_papers.items():
        titles = [p.get("title", "") for p in papers if p.get("title")]
        if not titles:
            names[cluster_id] = f"Cluster {cluster_id}"
            continue

        name = name_cluster(titles)
        names[cluster_id] = name
        logger.info(f"  Cluster {cluster_id} ({len(papers)} papers) → {name}")
        time.sleep(0.3)  # small delay between LLM calls

    return names
