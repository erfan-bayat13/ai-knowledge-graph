# kg/nlp/embedder.py
#
# Chunk-based embedding pipeline using allenai/specter2.
#
# SPECTER2 is trained on scientific paper citations — it understands
# that "transformer attention" and "self-attention in NLP" are related.
# Input format: title + [SEP] + abstract_chunk (as per SPECTER2 docs).
#
# Requirements:
#   pip install transformers adapters torch numpy

import logging
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

from kg.graph.neo4j_client import Neo4jClient
from kg.utils.config import get_settings

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD  = 0.85   # cosine similarity cutoff for SIMILAR_TO edges
CHUNK_SIZE_SENTENCES  = 4      # sentences per chunk
CHUNK_OVERLAP         = 2      # overlap between consecutive chunks
EMBED_BATCH_SIZE      = 8      # chunks per forward pass (memory safety)

# ── Model init (loaded once at import time) ───────────────────────────────────

logger.info("Loading SPECTER2 model and tokenizer...")
_tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
_model     = AutoAdapterModel.from_pretrained("allenai/specter2_base")
_model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
_model.eval()
logger.info("SPECTER2 ready.")

# ── Text chunking ─────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter — good enough for abstracts."""
    text = text.strip()
    sentences = re.split(r'(?<![A-Za-z]{2})\.\s+|[?!]\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_abstract(abstract: str) -> List[str]:
    """
    Split an abstract into overlapping chunks of sentences.

    Example with CHUNK_SIZE=4, OVERLAP=2:
      sentences = [s0, s1, s2, s3, s4, s5, s6]
      chunks    = [s0-s3, s2-s5, s4-s6]
    """
    sentences = _split_sentences(abstract)

    if len(sentences) <= CHUNK_SIZE_SENTENCES:
        return [abstract]

    chunks = []
    step   = CHUNK_SIZE_SENTENCES - CHUNK_OVERLAP
    start  = 0

    while start < len(sentences):
        end   = min(start + CHUNK_SIZE_SENTENCES, len(sentences))
        chunks.append(" ".join(sentences[start:end]))
        if end == len(sentences):
            break
        start += step

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(
    texts: List[str],
    title: str = "",
    client=None,               # kept for API compatibility, unused
) -> List[Optional[List[float]]]:
    """
    Embed a list of text chunks using SPECTER2.

    SPECTER2 expects:  title + [SEP] + text
    We prepend the paper title to every chunk so the model has full context.
    Returns a list of 768-dim float vectors (or None on failure per chunk).
    """
    sep = _tokenizer.sep_token  # "[SEP]"

    # Prepend title to each chunk as per SPECTER2 recommended input format
    formatted = [
        (title + sep + chunk) if title else chunk
        for chunk in texts
    ]

    all_embeddings: List[Optional[List[float]]] = []

    try:
        for i in range(0, len(formatted), EMBED_BATCH_SIZE):
            batch = formatted[i : i + EMBED_BATCH_SIZE]
            inputs = _tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            with torch.no_grad():
                outputs = _model(**inputs)

            # CLS token (first token) = sentence embedding
            embs = outputs.last_hidden_state[:, 0, :]
            # L2-normalise so cosine similarity == dot product
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embeddings.extend(embs.cpu().numpy().tolist())

    except Exception as e:
        logger.warning(f"Embedding batch failed: {e}")
        # Pad remaining with None so zip() stays aligned
        all_embeddings.extend([None] * (len(texts) - len(all_embeddings)))

    return all_embeddings


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

    O(n²) — fine for a few thousand chunks.
    """
    pairs = []
    n     = len(chunks)

    for i in range(n):
        for j in range(i + 1, n):
            a = chunks[i]
            b = chunks[j]

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
      3. Embed each chunk via SPECTER2 (title + SEP + chunk)
      4. Store Chunk nodes + HAS_CHUNK edges
      5. Compute pairwise cosine similarity across all chunks
      6. Store SIMILAR_TO edges above threshold
    """
    db = Neo4jClient()
    db.connect()

    papers = db.get_papers_without_chunks(limit=limit)
    logger.info(f"Papers to embed: {len(papers)}")

    total_chunks     = 0
    failed_papers    = 0
    in_memory_chunks = []   # used by dry-run similarity pass

    for i, paper in enumerate(papers):
        arxiv_id = paper["arxiv_id"]
        abstract = paper["abstract"] or ""
        title    = paper.get("title") or ""

        if not abstract.strip():
            logger.debug(f"Skipping {arxiv_id} — empty abstract")
            continue

        logger.info(f"[{i+1}/{len(papers)}] {arxiv_id} — {title[:55]}")

        chunks     = chunk_abstract(abstract)
        # Pass title so SPECTER2 gets full context per chunk
        embeddings = embed_texts(chunks, title=title)

        success = True
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is None:
                success = False
                continue

            chunk_id = f"{arxiv_id}_{idx}"

            if not dry_run:
                db.create_chunk(arxiv_id, idx, chunk_text, embedding)
            else:
                print(f"  [DRY RUN] chunk {idx}: {chunk_text[:60]}...")

            in_memory_chunks.append({
                "chunk_id":  chunk_id,
                "arxiv_id":  arxiv_id,
                "embedding": embedding,
            })
            total_chunks += 1

        if not success:
            failed_papers += 1

    logger.info(f"Chunks stored: {total_chunks} | Failed papers: {failed_papers}")

    # ── Similarity pass ───────────────────────────────────────────────────────
    logger.info("Computing chunk-level similarities...")

    if dry_run:
        # Nothing was written — use what we built in memory
        all_chunks = in_memory_chunks
    else:
        all_chunks = db.get_all_chunks_with_embeddings()
        # Ensure embeddings come back as plain Python floats (Neo4j may wrap them)
        for row in all_chunks:
            if row["embedding"] is not None:
                row["embedding"] = [float(x) for x in row["embedding"]]

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

    parser = argparse.ArgumentParser(description="Run SPECTER2 chunk embedding pipeline")
    parser.add_argument("--limit",     type=int,   default=100,                  help="Max papers to embed")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Cosine similarity cutoff")
    parser.add_argument("--dry-run",   action="store_true",                      help="Don't write to Neo4j")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    result = run_embedding_pipeline(
        limit=args.limit,
        dry_run=args.dry_run,
        similarity_threshold=args.threshold,
    )
    print(f"\nDone: {result}")