# kg/flow/citation_flow.py
# Phase 4: on-demand reverse citation traversal, tree construction, and divergence scoring
#
# Design: flows are NOT precomputed — generated at query time via reverse traversal of CITES edges
# Reason: 50k papers × 30 refs = 1.5M edges; all-paths is exponential; depth-2 per paper < 100ms
# Default depth=2 captures direct refs + their refs (core intellectual ancestry)
# Depth is configurable — kg trace "..." --depth 3 for deeper lineage
#
# Divergence detection: shared ancestor → two branches with low embedding cosine = divergence point

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from kg.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

DIVERGENCE_THRESHOLD = 0.6   # cosine similarity below this = divergence between branches


# ── Data structures ────────────────────────────────────────────────────────────

class FlowNode:
    """A node in the citation flow tree."""
    __slots__ = ("arxiv_id", "title", "published_date", "rank_score",
                 "cluster_id", "embedding", "depth", "children", "divergence_score")

    def __init__(self, arxiv_id: str, title: str = "", published_date: str = "",
                 rank_score: float = None, cluster_id: int = None,
                 embedding: List[float] = None, depth: int = 0):
        self.arxiv_id       = arxiv_id
        self.title          = title
        self.published_date = published_date
        self.rank_score     = rank_score
        self.cluster_id     = cluster_id
        self.embedding      = embedding
        self.depth          = depth
        self.children:      List["FlowNode"] = []
        self.divergence_score: Optional[float] = None  # set on divergence detection

    def to_dict(self) -> dict:
        return {
            "arxiv_id":        self.arxiv_id,
            "title":           self.title,
            "published_date":  self.published_date,
            "rank_score":      self.rank_score,
            "cluster_id":      self.cluster_id,
            "depth":           self.depth,
            "divergence_score": self.divergence_score,
            "children":        [c.to_dict() for c in self.children],
        }


# ── Traversal ──────────────────────────────────────────────────────────────────

def build_citation_tree(arxiv_id: str, depth: int = 2) -> Optional[FlowNode]:
    """
    Reverse traversal: given a paper, build a tree of its cited ancestors up to `depth` hops.
    Returns the root FlowNode (the selected paper), or None if paper not found.
    """
    db = Neo4jClient()
    db.connect()

    # Fetch the starting paper
    root_rows = db.run_query(
        "MATCH (p:Paper {arxiv_id: $id}) "
        "RETURN p.arxiv_id AS arxiv_id, p.title AS title, "
        "p.published_date AS published_date, p.rank_score AS rank_score, "
        "p.cluster_id AS cluster_id, p.embedding AS embedding",
        {"id": arxiv_id}
    )

    if not root_rows:
        logger.warning(f"Paper not found: {arxiv_id}")
        db.close()
        return None

    r = root_rows[0]
    root = FlowNode(
        arxiv_id=r["arxiv_id"], title=r["title"] or "",
        published_date=r["published_date"] or "", rank_score=r["rank_score"],
        cluster_id=r["cluster_id"], embedding=r["embedding"], depth=0,
    )

    # Fetch all ancestor paths via Cypher (depth-limited)
    paths = db.get_citation_flow(arxiv_id, depth=depth)
    db.close()

    # Build tree from path data (breadth-first deduplication)
    node_map: Dict[str, FlowNode] = {arxiv_id: root}

    for path in paths:
        path_nodes = path.get("path_nodes", [])
        parent = root

        for i, node_data in enumerate(path_nodes[1:], start=1):
            nid   = node_data.get("arxiv_id", "")
            if not nid:
                continue

            if nid not in node_map:
                n = FlowNode(
                    arxiv_id=nid,
                    title=node_data.get("title") or "",
                    published_date=node_data.get("published_date") or "",
                    rank_score=node_data.get("rank_score"),
                    cluster_id=node_data.get("cluster_id"),
                    embedding=node_data.get("embedding"),
                    depth=i,
                )
                node_map[nid] = n
                parent.children.append(n)

            parent = node_map[nid]

    return root


# ── Divergence detection ───────────────────────────────────────────────────────

def _cosine(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def detect_divergence(root: FlowNode) -> None:
    """
    Walk the tree and annotate divergence_score on child nodes.
    A divergence node is a child that took the idea somewhere new
    (low cosine similarity to its siblings from the same parent).
    Modifies the tree in-place.
    """
    if not root.children:
        return

    # For each node with 2+ children, compare sibling embeddings pairwise
    _annotate(root)


def _annotate(node: FlowNode) -> None:
    children = node.children
    if len(children) >= 2:
        for i, c1 in enumerate(children):
            for j, c2 in enumerate(children):
                if i >= j:
                    continue
                if c1.embedding and c2.embedding:
                    sim = _cosine(c1.embedding, c2.embedding)
                    if sim < DIVERGENCE_THRESHOLD:
                        # Mark both as diverging from each other
                        c1.divergence_score = round(1.0 - sim, 4)
                        c2.divergence_score = round(1.0 - sim, 4)

    for child in children:
        _annotate(child)


# ── Terminal rendering ─────────────────────────────────────────────────────────

def render_tree(node: FlowNode, prefix: str = "", is_last: bool = True) -> str:
    """Render the citation tree as an ASCII tree for terminal output."""
    connector = "└── " if is_last else "├── "
    year = (node.published_date or "")[:4]
    score_str = f"  score={node.rank_score:.2f}" if node.rank_score is not None else ""
    div_str = f"  [divergence {node.divergence_score:.2f}]" if node.divergence_score else ""

    line = f"{prefix}{connector if prefix else ''}{node.arxiv_id} ({year})  {node.title[:55]}{score_str}{div_str}\n"

    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        line += render_tree(child, child_prefix, is_last=(i == len(node.children) - 1))

    return line


# ── Export for visualization ───────────────────────────────────────────────────

def export_flow_json(root: FlowNode) -> dict:
    """Export the flow tree as a JSON-serialisable dict for the Research River HTML."""
    return root.to_dict()
