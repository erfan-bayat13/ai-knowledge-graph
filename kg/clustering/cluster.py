# kg/clustering/cluster.py
# Phase 3: pull paper embeddings from Neo4j, reduce with UMAP, cluster with HDBSCAN
# HDBSCAN chosen over k-means: handles noise, no fixed k, better for organic research clusters
# UMAP first reduces 768d → ~15d before HDBSCAN (better density estimation in lower dims)
# 2D UMAP coordinates stored separately for visualization (Phase 4)
#
# Requirements: pip install umap-learn hdbscan numpy

import logging
from typing import List, Dict, Tuple

import numpy as np

from kg.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Defaults — tune once we have real embedding data
UMAP_N_COMPONENTS = 15      # intermediate reduction before HDBSCAN
UMAP_N_COMPONENTS_2D = 2    # final 2D reduction for visualization
UMAP_N_NEIGHBORS   = 15
UMAP_MIN_DIST      = 0.1
HDBSCAN_MIN_SIZE   = 10     # min_cluster_size — reasonable for 500-1000 papers


def _load_embeddings(db: Neo4jClient) -> Tuple[List[str], np.ndarray]:
    """Pull all paper embeddings from Neo4j. Returns (arxiv_ids, embeddings_matrix)."""
    rows = db.get_all_papers_with_embeddings()
    if not rows:
        return [], np.array([])

    arxiv_ids  = [r["arxiv_id"] for r in rows]
    embeddings = np.array([r["embedding"] for r in rows], dtype=np.float32)
    logger.info(f"Loaded {len(arxiv_ids)} embeddings from Neo4j")
    return arxiv_ids, embeddings


def run_clustering(
    min_cluster_size: int = HDBSCAN_MIN_SIZE,
    dry_run: bool = False,
) -> Dict:
    """
    Full clustering pipeline:
      1. Load paper embeddings from Neo4j
      2. UMAP 768d → 15d (for HDBSCAN)
      3. HDBSCAN clustering
      4. UMAP 768d → 2d (for visualization)
      5. Name each cluster via LLM (kg/clustering/naming.py)
      6. Write Topic nodes + BELONGS_TO edges to Neo4j

    Returns dict with cluster assignments and 2D coordinates for visualization export.
    """
    try:
        import umap
        import hdbscan
    except ImportError:
        raise ImportError("Run: pip install umap-learn hdbscan")

    db = Neo4jClient()
    db.connect()

    arxiv_ids, embeddings = _load_embeddings(db)
    if len(arxiv_ids) == 0:
        logger.warning("No embeddings found — run: kg embed run")
        db.close()
        return {}

    logger.info("Running UMAP (768d → 15d for clustering)...")
    reducer_hd = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    reduced_hd = reducer_hd.fit_transform(embeddings)

    logger.info("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced_hd)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

    logger.info("Running UMAP (768d → 2d for visualization)...")
    reducer_2d = umap.UMAP(
        n_components=UMAP_N_COMPONENTS_2D,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords_2d = reducer_2d.fit_transform(embeddings)

    # Gather paper titles per cluster for LLM naming
    cluster_papers: Dict[int, List[dict]] = {}
    rows = db.get_all_papers_with_embeddings()
    paper_meta = {r["arxiv_id"]: r for r in rows}

    for arxiv_id, label in zip(arxiv_ids, labels):
        if label == -1:
            continue   # noise — not assigned to any cluster
        cluster_papers.setdefault(int(label), []).append(paper_meta.get(arxiv_id, {}))

    # ── LLM topic naming ──────────────────────────────────────────────────────
    from kg.clustering.naming import name_clusters
    cluster_names = name_clusters(cluster_papers)

    # ── Write back to Neo4j ───────────────────────────────────────────────────
    if not dry_run:
        from datetime import datetime, timedelta
        one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        for cluster_id, topic_name in cluster_names.items():
            papers_in_cluster = cluster_papers.get(cluster_id, [])

            # trend_score = new_papers_in_topic × avg_citation_velocity
            new_papers = [
                p for p in papers_in_cluster
                if (p.get("published_date") or "") >= one_week_ago
            ]
            velocities = [p.get("citation_velocity", 0) or 0 for p in papers_in_cluster]
            avg_velocity = np.mean(velocities) if velocities else 0.0
            trend_score = len(new_papers) * float(avg_velocity)

            db.create_or_update_topic(
                name=topic_name,
                trend_score=round(trend_score, 4),
                paper_count=len(papers_in_cluster),
            )

            for paper_data in papers_in_cluster:
                arxiv_id = paper_data.get("arxiv_id")
                if arxiv_id:
                    db.link_paper_to_topic(arxiv_id, topic_name, cluster_id)

        # Store 2D coordinates on Paper nodes for visualization
        for arxiv_id, (x, y) in zip(arxiv_ids, coords_2d):
            db.run_query("""
                MATCH (p:Paper {arxiv_id: $id})
                SET p.umap_x = $x, p.umap_y = $y
            """, {"id": arxiv_id, "x": float(x), "y": float(y)})

    db.close()

    return {
        "n_papers":   len(arxiv_ids),
        "n_clusters": n_clusters,
        "n_noise":    int(n_noise),
        "cluster_names": cluster_names,
        "coords_2d":  coords_2d.tolist(),
        "arxiv_ids":  arxiv_ids,
        "labels":     labels.tolist(),
    }
