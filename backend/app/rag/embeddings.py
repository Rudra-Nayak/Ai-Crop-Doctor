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

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_embeddings_instance: HuggingFaceEmbeddings | None = None
_lock = Lock()


def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Get or create the singleton embedding model.

    First call downloads the model (~80MB) and loads it.
    Subsequent calls return the cached instance.
    """
    global _embeddings_instance

    if _embeddings_instance is not None:
        return _embeddings_instance

    with _lock:
        # Double-check after acquiring lock
        if _embeddings_instance is not None:
            return _embeddings_instance

        logger.info("Loading embedding model: %s ...", model_name)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully.")
        return _embeddings_instance


def is_loaded() -> bool:
    """Check if the embedding model has been loaded."""
    return _embeddings_instance is not None
