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
    """Lightweight Embeddings adapter: cloud API first, zero-RAM fallback on cloud hosts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        self._is_cloud = bool(os.environ.get("RENDER") or os.environ.get("PORT") or os.environ.get("VERCEL"))

    def _embed_via_hf_api(self, text: str) -> list[float] | None:
        """Query HuggingFace serverless inference API (uses 0 MB server RAM)."""
        endpoints = [
            f"https://api-inference.huggingface.co/models/sentence-transformers/{self.model_name}",
            f"https://router.huggingface.co/hf-inference/models/sentence-transformers/{self.model_name}",
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/{self.model_name}",
        ]
        headers = {}
        if self._hf_token:
            headers["Authorization"] = f"Bearer {self._hf_token}"

        for url in endpoints:
            try:
                with httpx.Client(timeout=4.0) as client:
                    resp = client.post(url, json={"inputs": text[:500], "options": {"wait_for_model": False}}, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, list) and len(data) > 0:
                            if isinstance(data[0], list):
                                return data[0]
                            elif isinstance(data[0], (int, float)):
                                return data
            except Exception:
                continue
        return None

    def embed_query(self, text: str) -> list[float]:
        # 1. Try free cloud API first (0 MB RAM)
        hf_emb = self._embed_via_hf_api(text)
        if hf_emb:
            return hf_emb

        # 2. If running on cloud with 512MB RAM, skip heavy PyTorch load to avoid OOM kills
        if self._is_cloud:
            logger.info("Cloud environment detected: bypassing local PyTorch load to protect 512MB memory.")
            return [0.0] * 384

        # 3. Local fallback when running on local machine with full RAM
        try:
            model = get_sentence_transformer(self.model_name)
            if model is not None:
                emb = model.encode(text, normalize_embeddings=True)
                return emb.tolist() if hasattr(emb, "tolist") else list(emb)
        except Exception as e:
            logger.warning("Local embedding failed: %s", e)

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
