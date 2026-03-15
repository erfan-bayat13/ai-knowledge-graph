# kg/nlp/embedder.py
# CHANGED (Phase 2): rewritten from chunk-level to paper-level embeddings
# CHANGED: SPECTER2 input is now title + [SEP] + abstract (whole paper, not sentence windows)
# CHANGED: stores embedding as p.embedding on Paper node (not Chunk nodes)
# CHANGED: removed chunk_abstract(), find_similar_pairs(), create_chunk() calls
# Reason: SPECTER2 is trained on citation graphs — whole-paper context is the correct input unit
#
# Requirements: pip install transformers adapters torch numpy

import logging
from typing import List, Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

from kg.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 8   # papers per forward pass (memory safety)

# ── Model init (loaded once at import time) ────────────────────────────────────

logger.info("Loading SPECTER2 model and tokenizer...")
_tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
_model     = AutoAdapterModel.from_pretrained("allenai/specter2_base")
_model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
_model.eval()
logger.info("SPECTER2 ready.")


# ── Core embedding function ────────────────────────────────────────────────────

def embed_papers(papers: List[Dict]) -> List[Optional[List[float]]]:
    """
    Embed a list of paper dicts (each with 'title' and 'abstract') using SPECTER2.

    SPECTER2 input format: title + [SEP] + abstract  (whole paper, not chunks)
    Returns a list of 768-dim L2-normalised float vectors (None on failure per paper).
    """
    sep = _tokenizer.sep_token

    inputs_text = [
        (p.get("title") or "") + sep + (p.get("abstract") or "")
        for p in papers
    ]

    all_embeddings: List[Optional[List[float]]] = []

    for i in range(0, len(inputs_text), EMBED_BATCH_SIZE):
        batch = inputs_text[i : i + EMBED_BATCH_SIZE]
        try:
            encoded = _tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            with torch.no_grad():
                outputs = _model(**encoded)

            # CLS token embedding, L2-normalised so cosine == dot product
            embs = outputs.last_hidden_state[:, 0, :]
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embeddings.extend(embs.cpu().numpy().tolist())

        except Exception as e:
            logger.warning(f"Embedding batch {i} failed: {e}")
            all_embeddings.extend([None] * len(batch))

    return all_embeddings


def embed_query(query: str) -> Optional[List[float]]:
    """
    Embed a free-text query for semantic search.
    Input is treated as abstract-only (no title prefix).
    """
    results = embed_papers([{"title": "", "abstract": query}])
    return results[0] if results else None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_embedding_pipeline(limit: int = 500, dry_run: bool = False) -> Dict:
    """
    Paper-level embedding pipeline:
      1. Fetch papers without embeddings from Neo4j
      2. Embed each paper via SPECTER2 (title + SEP + abstract)
      3. Store embedding property on Paper node
      4. (Similarity edges are handled by Neo4j vector index queries, not precomputed)
    """
    db = Neo4jClient()
    db.connect()

    papers = db.get_papers_without_embedding(limit=limit)
    logger.info(f"Papers to embed: {len(papers)}")

    embedded = 0
    failed   = 0

    for i in range(0, len(papers), EMBED_BATCH_SIZE):
        batch = papers[i : i + EMBED_BATCH_SIZE]
        embeddings = embed_papers(batch)

        for paper, embedding in zip(batch, embeddings):
            arxiv_id = paper["arxiv_id"]
            title    = (paper.get("title") or "")[:55]

            if embedding is None:
                logger.warning(f"  Failed: {arxiv_id}")
                failed += 1
                continue

            if not dry_run:
                db.set_paper_embedding(arxiv_id, embedding)
            else:
                print(f"  [DRY] {arxiv_id} — {title}... (dim={len(embedding)})")

            embedded += 1
            logger.info(f"[{embedded}/{len(papers)}] {arxiv_id} — {title}")

    db.close()
    result = {"papers_embedded": embedded, "failed": failed}
    logger.info(f"Done: {result}")
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SPECTER2 paper-level embedding pipeline")
    parser.add_argument("--limit",   type=int,  default=500,   help="Max papers to embed")
    parser.add_argument("--dry-run", action="store_true",      help="Print only, no writes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    result = run_embedding_pipeline(limit=args.limit, dry_run=args.dry_run)
    print(f"\nDone: {result}")
