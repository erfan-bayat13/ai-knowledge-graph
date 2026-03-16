# kg/clustering/naming.py
# LLM topic naming — two functions:
#   name_clusters()     — Level 1 coarse topics (broad, 2-5 words)
#   name_subclusters()  — Level 2 fine sub-topics (specific, given parent context)
#
# Both send paper titles to Together AI Llama and parse a JSON response.

import json
import logging
import os
import time
from typing import Dict, List

logger = logging.getLogger(__name__)

_MAX_TITLES    = 20   # titles to send per cluster
_RETRY_WAIT    = 2.0
_MAX_RETRIES   = 3


def _together_client():
    """Return a Together AI client."""
    try:
        from together import Together
        return Together(api_key=os.getenv("TOGETHER_API_KEY", ""))
    except Exception as e:
        raise RuntimeError(f"Together AI client failed: {e}")


def _call_llm(prompt: str) -> str:
    """Call Together AI and return the response text. Retries on failure."""
    client = _together_client()
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
            time.sleep(_RETRY_WAIT)
    return "{}"


def _parse_json_response(text: str, cluster_ids: List[int]) -> Dict[int, str]:
    """
    Parse LLM JSON response into {cluster_id: name} dict.
    Handles markdown fences and falls back gracefully.
    """
    # Strip markdown fences if present
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        raw = json.loads(clean)
        # LLM may return string keys — normalise to int
        return {int(k): str(v) for k, v in raw.items() if int(k) in cluster_ids}
    except Exception:
        logger.warning(f"Could not parse LLM response as JSON: {text[:200]}")
        return {}


# ── Public API ─────────────────────────────────────────────────────────────────

def name_clusters(cluster_papers: Dict[int, List[dict]]) -> Dict[int, str]:
    """
    Level 1 — name coarse topic clusters.

    Args:
        cluster_papers: {cluster_id: [paper_dict, ...]}

    Returns:
        {cluster_id: topic_name}
    """
    if not cluster_papers:
        return {}

    cluster_ids = sorted(cluster_papers.keys())

    # Build title lists per cluster
    cluster_titles: Dict[int, List[str]] = {}
    for cid in cluster_ids:
        titles = [
            p.get("title", "").strip()
            for p in cluster_papers[cid]
            if p.get("title")
        ][:_MAX_TITLES]
        cluster_titles[cid] = titles

    prompt = f"""You are a research taxonomy expert. Below are groups of academic paper titles.
Name each group with a concise 2-5 word topic label that captures the shared research theme.
Be broad enough to cover all papers in the group, but specific enough to be meaningful.

Groups:
{json.dumps(cluster_titles, indent=2)}

Respond ONLY with a JSON object mapping each cluster ID (as a string key) to its topic name.
Example: {{"0": "Large Language Models", "1": "Computer Vision", "2": "Graph Neural Networks"}}
No explanation, no markdown, just the JSON object."""

    text    = _call_llm(prompt)
    names   = _parse_json_response(text, cluster_ids)

    # Fill any missing clusters with a fallback
    for cid in cluster_ids:
        if cid not in names:
            sample_title = cluster_titles[cid][0] if cluster_titles[cid] else ""
            names[cid] = sample_title[:40] or f"Cluster {cid}"

    logger.info(f"Level 1 names: {names}")
    return names


def name_subclusters(
    sub_cluster_papers: Dict[int, List[dict]],
    parent_topic: str,
) -> Dict[int, str]:
    """
    Level 2 — name fine-grained sub-clusters within a parent topic.

    The LLM is told the parent topic so it names specifically
    rather than repeating the broad label.

    Args:
        sub_cluster_papers: {sub_cluster_id: [paper_dict, ...]}
        parent_topic:       name of the parent coarse cluster (e.g. "Reinforcement Learning")

    Returns:
        {sub_cluster_id: sub_topic_name}  (WITHOUT the parent prefix — caller adds it)
    """
    if not sub_cluster_papers:
        return {}

    sub_ids = sorted(sub_cluster_papers.keys())

    sub_titles: Dict[int, List[str]] = {}
    for sid in sub_ids:
        titles = [
            p.get("title", "").strip()
            for p in sub_cluster_papers[sid]
            if p.get("title")
        ][:_MAX_TITLES]
        sub_titles[sid] = titles

    prompt = f"""You are a research taxonomy expert specialising in "{parent_topic}".
Below are sub-groups of paper titles, all belonging to the broader topic "{parent_topic}".
Name each sub-group with a specific 2-5 word label that captures what makes it distinct
within "{parent_topic}". Do NOT repeat "{parent_topic}" in the name — just the distinguishing aspect.

Sub-groups:
{json.dumps(sub_titles, indent=2)}

Respond ONLY with a JSON object mapping each sub-cluster ID (as a string key) to its name.
Example for parent "Reinforcement Learning":
{{"0": "Offline RL Methods", "1": "RLHF & Alignment", "2": "Multi-Agent Systems"}}
No explanation, no markdown, just the JSON object."""

    text  = _call_llm(prompt)
    names = _parse_json_response(text, sub_ids)

    # Fill missing
    for sid in sub_ids:
        if sid not in names:
            sample_title = sub_titles[sid][0] if sub_titles[sid] else ""
            names[sid] = sample_title[:35] or f"Sub-{sid}"

    logger.info(f"Level 2 names under '{parent_topic}': {names}")
    return names