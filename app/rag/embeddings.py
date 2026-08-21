"""
Embedding model management.

Provides a singleton HuggingFace embedding model that loads once at startup
and is reused across all RAG operations. Runs locally — no API calls needed.
"""

from __future__ import annotations

import logging
import os
from threading import Lock

# Force offline mode for HuggingFace to avoid hanging on network checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
