"""
Embedding model management.

Provides a memory-efficient HuggingFace embedding adapter.
Supports cloud inference API (0 MB RAM) with graceful local SentenceTransformer fallback.
"""

from __future__ import annotations

import gc
import logging
import os
from functools import lru_cache
from threading import Lock
import httpx

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
    torch.set_grad_enabled(False)
except Exception:
    pass

_MODEL_INSTANCE = None
_LOCK = Lock()
logger = logging.getLogger(__name__)


def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Load SentenceTransformer model singleton with memory optimizations."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    with _LOCK:
        if _MODEL_INSTANCE is not None:
            return _MODEL_INSTANCE

        logger.info("Loading local embedding model: %s ...", model_name)
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL_INSTANCE = SentenceTransformer(model_name, device="cpu")
            gc.collect()
            logger.info("Local embedding model loaded successfully.")
            return _MODEL_INSTANCE
        except Exception as e:
            logger.error("Failed to load local SentenceTransformer: %s", e)
            return None


class LightweightEmbeddings:
    """Lightweight Embeddings adapter: fast cloud API first, local fallback second."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")

    def _embed_via_hf_api(self, text: str) -> list[float] | None:
        """Query HuggingFace serverless inference API (uses 0 MB server RAM)."""
        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/{self.model_name}"
        headers = {}
        if self._hf_token:
            headers["Authorization"] = f"Bearer {self._hf_token}"
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.post(url, json={"inputs": text, "options": {"wait_for_model": True}}, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], list):
                            return data[0]
                        return data
        except Exception as e:
            logger.debug("HF Inference API request failed, falling back to local: %s", e)
        return None

    def embed_query(self, text: str) -> list[float]:
        # 1. Try free cloud API first to keep memory at near 0MB
        hf_emb = self._embed_via_hf_api(text)
        if hf_emb:
            return hf_emb

        # 2. Local fallback if online API is unavailable
        try:
            model = get_sentence_transformer(self.model_name)
            if model is not None:
                emb = model.encode(text, normalize_embeddings=True)
                return emb.tolist() if hasattr(emb, "tolist") else list(emb)
        except Exception as e:
            logger.warning("Local embedding failed: %s. Returning zero vector.", e)

        # 3. Safe fallback 384-d vector (prevents server crashes/OOM kills)
        return [0.0] * 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


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
    return _MODEL_INSTANCE is not None
