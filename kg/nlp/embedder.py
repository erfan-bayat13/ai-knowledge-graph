# kg/nlp/embedder.py
#
# Chunk-based embedding pipeline.
#
# Each abstract is split into overlapping chunks (3-4 sentences, 1-2 overlap).
# Each chunk is embedded separately via Together AI m2-bert.
# Similarity is computed at chunk level → Paper SIMILAR_TO Paper edges
# are created when any chunk pair exceeds the threshold.
#
# This is more precise than embedding the full abstract because
# "the method part of paper A" can match "the method part of paper B"
# even if their evaluation sections look completely different.
#
# Requirements:
#   pip install together numpy

import logging
import time
from typing import List, Dict, Tuple, Optional

import numpy as np

from kg.graph.neo4j_client import Neo4jClient
from kg.utils.config import get_settings

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.82    # cosine similarity above this → SIMILAR_TO edge
CHUNK_SIZE_SENTENCES = 4       # sentences per chunk
CHUNK_OVERLAP = 2              # overlap between consecutive chunks
DELAY_BETWEEN_CALLS = 0.2      # seconds — respect Together AI rate limits

from sentence_transformers import SentenceTransformer
_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Text chunking ─────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter. Good enough for abstracts.
    Avoids pulling in nltk just for this.
    """
    import re
    # Split on ". " or "? " or "! " but not on "e.g. " or "i.e. " or "et al. "
    text = text.strip()
    sentences = re.split(r'(?<![A-Za-z]{2})\.\s+|[?!]\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_abstract(abstract: str) -> List[str]:
    """
    Split an abstract into overlapping chunks of sentences.

    Example with CHUNK_SIZE=4, OVERLAP=2:
      sentences = [s0, s1, s2, s3, s4, s5, s6]
      chunks    = [s0-s3, s2-s5, s4-s6]

    Returns a list of text strings, one per chunk.
    """
    sentences = _split_sentences(abstract)

    if len(sentences) <= CHUNK_SIZE_SENTENCES:
        # Short abstract — just one chunk
        return [abstract]

    chunks = []
    step   = CHUNK_SIZE_SENTENCES - CHUNK_OVERLAP
    start  = 0

    while start < len(sentences):
        end   = min(start + CHUNK_SIZE_SENTENCES, len(sentences))
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        if end == len(sentences):
            break
        start += step

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], client=None) -> List[Optional[List[float]]]:
    """
    Embed a list of texts locally using sentence-transformers.
    `client` param kept for API compatibility but unused.
    """
    try:
        vectors = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return [None] * len(texts)


def embed_texts(texts: List[str], client=None) -> List[Optional[List[float]]]:
    """
    Embed a list of texts locally using sentence-transformers.
    `client` param kept for API compatibility but unused.
    """
    try:
        vectors = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return [None] * len(texts)


# ── Similarity ────────────────────────────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def find_similar_pairs(
    chunks: List[Dict],
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[Tuple[str, str, float]]:
    """
    Given a list of {chunk_id, arxiv_id, embedding} dicts,
    find all pairs from *different papers* whose cosine similarity
    exceeds the threshold.

    Returns list of (chunk_id_a, chunk_id_b, score).

    O(n²) — fine for a few thousand chunks. If you have 100k+ chunks,
    switch to approximate nearest neighbours (faiss or similar).
    """
    pairs = []
    n     = len(chunks)

    for i in range(n):
        for j in range(i + 1, n):
            a = chunks[i]
            b = chunks[j]

            # Never link chunks from the same paper
            if a["arxiv_id"] == b["arxiv_id"]:
                continue

            score = cosine_similarity(a["embedding"], b["embedding"])
            if score >= threshold:
                pairs.append((a["chunk_id"], b["chunk_id"], score))

    return pairs


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_embedding_pipeline(
    limit: int = 500,
    dry_run: bool = False,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Dict:
    """
    Full embedding pipeline:
      1. Fetch papers without chunks from Neo4j
      2. Chunk each abstract
      3. Embed each chunk via Together AI
      4. Store Chunk nodes + HAS_CHUNK edges
      5. Compute pairwise cosine similarity
      6. Store SIMILAR_TO edges above threshold

    Args:
        limit:                Max papers to embed in this run
        dry_run:              Print what would happen, don't write
        similarity_threshold: Cosine similarity cutoff for SIMILAR_TO edges
    """
    db     = Neo4jClient()
    

    db.connect()
    papers = db.get_papers_without_chunks(limit=limit)
    logger.info(f"Papers to embed: {len(papers)}")

    total_chunks   = 0
    failed_papers  = 0

    for i, paper in enumerate(papers):
        arxiv_id = paper["arxiv_id"]
        abstract = paper["abstract"] or ""

        if not abstract.strip():
            logger.debug(f"Skipping {arxiv_id} — empty abstract")
            continue

        logger.info(f"[{i+1}/{len(papers)}] {arxiv_id} — {paper['title'][:55]}")

        chunks = chunk_abstract(abstract)
        embeddings = embed_texts(chunks)

        success = True
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is None:
                success = False
                continue
            if not dry_run:
                db.create_chunk(arxiv_id, idx, chunk_text, embedding)
            else:
                print(f"  [DRY RUN] chunk {idx}: {chunk_text[:60]}...")
            total_chunks += 1

        if not success:
            failed_papers += 1

    logger.info(f"Chunks stored: {total_chunks} | Failed papers: {failed_papers}")

    # ── Similarity pass ───────────────────────────────────────────────────────
    logger.info("Computing chunk-level similarities...")

    all_chunks = db.get_all_chunks_with_embeddings()
    logger.info(f"Total chunks in graph: {len(all_chunks)}")

    pairs = find_similar_pairs(all_chunks, threshold=similarity_threshold)
    logger.info(f"Similar pairs found: {len(pairs)} (threshold={similarity_threshold})")

    edges_created = 0
    for chunk_id_a, chunk_id_b, score in pairs:
        if not dry_run:
            db.link_similar_chunks(chunk_id_a, chunk_id_b, round(score, 4))
        else:
            print(f"  [DRY RUN] SIMILAR_TO {chunk_id_a} → {chunk_id_b} ({score:.3f})")
        edges_created += 1

    db.close()

    return {
        "papers_processed": len(papers),
        "chunks_stored":    total_chunks,
        "similar_edges":    edges_created,
        "failed_papers":    failed_papers,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run chunk embedding pipeline")
    parser.add_argument("--limit",     type=int,   default=100,              help="Max papers to embed")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Cosine similarity cutoff")
    parser.add_argument("--dry-run",   action="store_true",                  help="Don't write to Neo4j")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    result = run_embedding_pipeline(
        limit=args.limit,
        dry_run=args.dry_run,
        similarity_threshold=args.threshold,
    )
    print(f"\nDone: {result}")