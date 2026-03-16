# kg/clustering/cluster.py
# Phase 3: UMAP + HDBSCAN topic clustering — two-level pass
#
# Level 1 (coarse): broad topics  e.g. "Reinforcement Learning", "LLM Agents"
# Level 2 (fine):   sub-topics within each coarse cluster
#                   e.g. "Reinforcement Learning > Offline RL",
#                        "Reinforcement Learning > RLHF & Alignment"
#
# The LLM naming prompt for level 2 is given the parent topic name as context
# so it names specifically rather than generically.
#
# CLI:
#   kg cluster run                        # coarse only (default)
#   kg cluster run --sub-cluster          # coarse + fine sub-topics
#   kg cluster run --min-size 3           # smaller clusters = more granular
#   kg cluster run --sub-cluster --sub-min-size 2

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np

from kg.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────
UMAP_N_NEIGHBORS    = 15
UMAP_MIN_DIST       = 0.1
UMAP_N_COMPONENTS   = 15   # intermediate dims for HDBSCAN
UMAP_N_COMPONENTS_2D = 2   # final dims for visualization
HDBSCAN_MIN_SIZE    = 10


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_embeddings(db: Neo4jClient) -> Tuple[List[str], np.ndarray, Dict]:
    """Pull all paper embeddings + metadata from Neo4j."""
    rows = db.get_all_papers_with_embeddings()
    if not rows:
        return [], np.array([]), {}

    arxiv_ids  = [r["arxiv_id"] for r in rows]
    embeddings = np.array([r["embedding"] for r in rows], dtype=np.float32)
    meta       = {r["arxiv_id"]: r for r in rows}
    logger.info(f"Loaded {len(arxiv_ids)} embeddings")
    return arxiv_ids, embeddings, meta


def _safe_umap(embeddings: np.ndarray, n_components: int, n_neighbors: int,
               min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    """Run UMAP with dataset-size-safe parameters."""
    import umap
    n = len(embeddings)

    # Need at least 3 points; return zero coords for anything smaller
    if n < 3:
        logger.debug(f"Skipping UMAP — only {n} points")
        return np.zeros((n, max(min(n_components, 2), 1)), dtype=np.float32)

    # n_components must be strictly < n - 1 to avoid scipy eigsh k >= N error
    # n_neighbors  must be strictly < n
    safe_components = min(n_components, max(n - 2, 1))
    safe_neighbors  = min(n_neighbors,  n - 1)

    reducer = umap.UMAP(
        n_components=safe_components,
        n_neighbors=safe_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def _run_hdbscan(reduced: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Run HDBSCAN and return label array."""
    import hdbscan
    safe_min = min(min_cluster_size, max(2, len(reduced) // 3))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=safe_min,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(reduced)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_clustering(
    min_cluster_size: int = HDBSCAN_MIN_SIZE,
    sub_cluster: bool = False,
    sub_min_size: int = 3,
    dry_run: bool = False,
) -> Dict:
    """
    Two-level clustering pipeline.

    Args:
        min_cluster_size: HDBSCAN min_cluster_size for the coarse pass
        sub_cluster:      if True, run a second fine-grained pass within each coarse cluster
        sub_min_size:     min_cluster_size for the sub-cluster pass
        dry_run:          inspect results without writing to Neo4j

    Returns dict with n_papers, n_clusters, n_noise, cluster_names, coords_2d, labels, arxiv_ids.
    """
    try:
        import umap    # noqa: F401
        import hdbscan # noqa: F401
    except ImportError:
        raise ImportError("Run: pip install umap-learn hdbscan")

    db = Neo4jClient()
    db.connect()

    arxiv_ids, embeddings, meta = _load_embeddings(db)
    n = len(arxiv_ids)
    if n == 0:
        logger.warning("No embeddings found — run: kg embed run")
        db.close()
        return {}

    # ── Level 1: coarse clustering ────────────────────────────────────────────
    logger.info(f"Level 1: UMAP {embeddings.shape[1]}d → {UMAP_N_COMPONENTS}d ...")
    reduced_hd = _safe_umap(embeddings, UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS)

    logger.info("Level 1: HDBSCAN ...")
    labels_l1 = _run_hdbscan(reduced_hd, min_cluster_size)

    n_clusters_l1 = len(set(labels_l1)) - (1 if -1 in labels_l1 else 0)
    n_noise_l1    = int((labels_l1 == -1).sum())
    logger.info(f"Level 1: {n_clusters_l1} clusters, {n_noise_l1} noise points")

    # ── 2D coords for visualization (always based on full embedding set) ──────
    logger.info("UMAP 2D for visualization ...")
    coords_2d = _safe_umap(embeddings, UMAP_N_COMPONENTS_2D, UMAP_N_NEIGHBORS, min_dist=0.1)

    # ── Gather paper metadata per coarse cluster ──────────────────────────────
    cluster_papers_l1: Dict[int, List[dict]] = {}
    for arxiv_id, label in zip(arxiv_ids, labels_l1):
        if label == -1:
            continue
        cluster_papers_l1.setdefault(int(label), []).append(meta.get(arxiv_id, {}))

    # ── Level 1 LLM naming ────────────────────────────────────────────────────
    from kg.clustering.naming import name_clusters, name_subclusters
    logger.info("Level 1: LLM topic naming ...")
    cluster_names_l1 = name_clusters(cluster_papers_l1)

    # ── Level 2: sub-clustering (optional) ───────────────────────────────────
    # Maps arxiv_id → final topic name (e.g. "RL > Offline RL")
    final_topic: Dict[str, str] = {}
    final_cluster_id: Dict[str, int] = {}

    # Global cluster counter so every sub-cluster gets a unique int ID
    global_cluster_idx = 0

    # First assign all noise papers to None
    for arxiv_id, label in zip(arxiv_ids, labels_l1):
        if label == -1:
            final_topic[arxiv_id]      = None
            final_cluster_id[arxiv_id] = -1

    if not sub_cluster:
        # No sub-clustering — use level 1 names directly
        for arxiv_id, label in zip(arxiv_ids, labels_l1):
            if label == -1:
                continue
            name = cluster_names_l1.get(int(label), f"Cluster {label}")
            final_topic[arxiv_id]      = name
            final_cluster_id[arxiv_id] = int(label)
        global_cluster_idx = n_clusters_l1

    else:
        logger.info("Level 2: sub-clustering within each coarse topic ...")

        for coarse_id, coarse_name in cluster_names_l1.items():
            papers_in_cluster = cluster_papers_l1.get(coarse_id, [])
            ids_in_cluster    = [p["arxiv_id"] for p in papers_in_cluster if p.get("arxiv_id")]

            if len(ids_in_cluster) < max(sub_min_size * 2, 6):
                # Too small to sub-cluster — keep as-is
                for arxiv_id in ids_in_cluster:
                    final_topic[arxiv_id]      = coarse_name
                    final_cluster_id[arxiv_id] = global_cluster_idx
                global_cluster_idx += 1
                continue

            # Extract embeddings for this cluster
            idx_map    = {aid: i for i, aid in enumerate(arxiv_ids)}
            sub_idx    = [idx_map[aid] for aid in ids_in_cluster if aid in idx_map]
            sub_embeds = embeddings[sub_idx]

            logger.info(
                f"  Sub-clustering '{coarse_name}' "
                f"({len(ids_in_cluster)} papers, min_size={sub_min_size}) ..."
            )

            # UMAP within the sub-cluster (lower dims since dataset is small)
            sub_components = min(UMAP_N_COMPONENTS, len(sub_embeds) - 1)
            sub_reduced    = _safe_umap(sub_embeds, sub_components, min(UMAP_N_NEIGHBORS, len(sub_embeds) - 1))
            sub_labels     = _run_hdbscan(sub_reduced, sub_min_size)

            n_sub = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
            logger.info(f"  → {n_sub} sub-clusters found")

            if n_sub <= 1:
                # Sub-clustering didn't find meaningful structure — keep coarse name
                for arxiv_id in ids_in_cluster:
                    final_topic[arxiv_id]      = coarse_name
                    final_cluster_id[arxiv_id] = global_cluster_idx
                global_cluster_idx += 1
                continue

            # Gather sub-cluster papers for LLM naming
            sub_cluster_papers: Dict[int, List[dict]] = {}
            for aid, sub_label in zip(ids_in_cluster, sub_labels):
                if sub_label == -1:
                    continue
                sub_cluster_papers.setdefault(int(sub_label), []).append(meta.get(aid, {}))

            # LLM naming — pass the parent topic name so the prompt gives context
            logger.info(f"  Level 2 LLM naming for '{coarse_name}' sub-clusters ...")
            sub_names = name_subclusters(sub_cluster_papers, parent_topic=coarse_name)

            for aid, sub_label in zip(ids_in_cluster, sub_labels):
                if sub_label == -1:
                    # Sub-cluster noise — fall back to coarse name
                    final_topic[arxiv_id]      = coarse_name
                    final_cluster_id[arxiv_id] = global_cluster_idx - 1
                    continue
                sub_name  = sub_names.get(int(sub_label), f"{coarse_name} > Sub-{sub_label}")
                full_name = f"{coarse_name} > {sub_name}"
                final_topic[aid]      = full_name
                final_cluster_id[aid] = global_cluster_idx + int(sub_label)

            global_cluster_idx += n_sub

    # ── Write to Neo4j ────────────────────────────────────────────────────────
    if not dry_run:
        from datetime import datetime, timedelta
        one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Aggregate papers per final topic for trend_score
        topic_papers: Dict[str, List[dict]] = {}
        for arxiv_id in arxiv_ids:
            topic = final_topic.get(arxiv_id)
            if topic:
                topic_papers.setdefault(topic, []).append(meta.get(arxiv_id, {}))

        import numpy as _np
        for topic_name, papers_in_topic in topic_papers.items():
            new_papers  = [p for p in papers_in_topic
                           if (p.get("published_date") or "") >= one_week_ago]
            velocities  = [p.get("citation_velocity", 0) or 0 for p in papers_in_topic]
            avg_velocity = float(_np.mean(velocities)) if velocities else 0.0
            trend_score  = len(new_papers) * avg_velocity

            db.create_or_update_topic(
                name=topic_name,
                trend_score=round(trend_score, 4),
                paper_count=len(papers_in_topic),
            )

            for paper_data in papers_in_topic:
                aid = paper_data.get("arxiv_id")
                if aid:
                    cid = final_cluster_id.get(aid, -1)
                    db.link_paper_to_topic(aid, topic_name, cid)

        # Store 2D coords on Paper nodes
        for arxiv_id, (x, y) in zip(arxiv_ids, coords_2d):
            db.run_query("""
                MATCH (p:Paper {arxiv_id: $id})
                SET p.umap_x = $x, p.umap_y = $y
            """, {"id": arxiv_id, "x": float(x), "y": float(y)})

    db.close()

    # Collect unique final topic names for the summary
    all_topic_names = {v for v in final_topic.values() if v is not None}
    n_final_noise   = sum(1 for v in final_topic.values() if v is None)

    return {
        "n_papers":      n,
        "n_clusters":    len(all_topic_names),
        "n_noise":       n_final_noise,
        "cluster_names": {i: name for i, name in enumerate(sorted(all_topic_names))},
        "coords_2d":     coords_2d.tolist(),
        "arxiv_ids":     arxiv_ids,
        "labels":        [final_cluster_id.get(aid, -1) for aid in arxiv_ids],
    }