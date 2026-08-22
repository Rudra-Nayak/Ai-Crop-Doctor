"""
Embedding model management.

Provides a singleton HuggingFace embedding model that loads once at startup
and is reused across all RAG operations. Runs locally — no API calls needed.
"""

from __future__ import annotations

import logging
import os
from threading import Lock

# Memory optimization for 512MB RAM environments (e.g. Render Free Tier)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

from functools import lru_cache
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache the SentenceTransformer model singleton exactly once."""
    logger.info("Loading embedding model: %s ...", model_name)
    return SentenceTransformer(model_name, device="cpu")


class LightweightEmbeddings:
    """Lightweight Embeddings adapter backed by cached SentenceTransformer singleton."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def embed_query(self, text: str) -> list[float]:
        model = get_sentence_transformer(self.model_name)
        emb = model.encode(text, normalize_embeddings=True)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = get_sentence_transformer(self.model_name)
        embs = model.encode(texts, normalize_embeddings=True)
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]


_embeddings_instance: LightweightEmbeddings | None = None
_lock = Lock()


def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> LightweightEmbeddings:
    """Get or create the singleton embedding adapter."""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    with _lock:
        if _embeddings_instance is not None:
            return _embeddings_instance
        _embeddings_instance = LightweightEmbeddings(model_name)
        return _embeddings_instance


def is_loaded() -> bool:
    """Check if the embedding model has been loaded into memory."""
    return _embeddings_instance is not None
