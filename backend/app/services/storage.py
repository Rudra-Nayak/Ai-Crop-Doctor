"""
File storage service abstraction.

Provides file upload capability for crop photos and farmer audio recordings.
Uses local filesystem storage by default, or Supabase Storage buckets
(`crop-images` & `farmer-audio`) when configured.
"""

from __future__ import annotations

import logging
import os
import uuid

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class StorageService:
    """Storage service manager — handles local vs Supabase cloud uploads."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = None

        if self._settings.use_supabase_storage and self._settings.supabase_url and self._settings.supabase_key:
            try:
                from supabase import create_client
                self._client = create_client(self._settings.supabase_url, self._settings.supabase_key)
                logger.info("Supabase Storage initialized successfully.")
            except Exception as e:
                logger.warning("Failed to initialize Supabase Storage: %s. Using local storage.", e)
                self._client = None

    async def upload_image(self, image_bytes: bytes, original_filename: str = "crop.jpg") -> str:
        """
        Upload crop image and return file path or public HTTP URL.
        """
        ext = os.path.splitext(original_filename)[1] or ".jpg"
        filename = f"{uuid.uuid4()}{ext}"

        # 1. Try Supabase Storage upload
        if self._client is not None:
            try:
                res = self._client.storage.from_("crop-images").upload(
                    path=filename,
                    file=image_bytes,
                    file_options={"content-type": "image/jpeg"}
                )
                public_url = self._client.storage.from_("crop-images").get_public_url(filename)
                logger.info("Uploaded crop image to Supabase Storage: %s", public_url)
                return public_url
            except Exception as e:
                logger.warning("Supabase image upload failed: %s. Falling back to local disk.", e)

        # 2. Local disk fallback
        os.makedirs(self._settings.upload_dir, exist_ok=True)
        local_path = os.path.abspath(os.path.join(self._settings.upload_dir, filename))
        with open(local_path, "wb") as f:
            f.write(image_bytes)
        return local_path

    async def upload_audio(self, audio_bytes: bytes, original_filename: str = "audio.wav") -> str:
        """
        Upload audio recording and return file path or public HTTP URL.
        """
        ext = os.path.splitext(original_filename)[1] or ".wav"
        filename = f"{uuid.uuid4()}{ext}"

        if self._client is not None:
            try:
                self._client.storage.from_("farmer-audio").upload(
                    path=filename,
                    file=audio_bytes,
                    file_options={"content-type": "audio/wav"}
                )
                public_url = self._client.storage.from_("farmer-audio").get_public_url(filename)
                logger.info("Uploaded farmer audio to Supabase Storage: %s", public_url)
                return public_url
            except Exception as e:
                logger.warning("Supabase audio upload failed: %s. Falling back to local disk.", e)

        os.makedirs(self._settings.upload_dir, exist_ok=True)
        local_path = os.path.abspath(os.path.join(self._settings.upload_dir, filename))
        with open(local_path, "wb") as f:
            f.write(audio_bytes)
        return local_path
