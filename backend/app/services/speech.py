"""
Speech service — Groq Whisper STT + Groq Orpheus TTS.

Handles the full voice pipeline:
  Farmer voice → transcribe (Whisper) → text
  Agent text → synthesize (Orpheus) → audio for farmer

Both directions degrade gracefully on failure.
"""

from __future__ import annotations

import io
import logging
import re
import struct

from groq import Groq

from app.config import get_settings

logger = logging.getLogger(__name__)


class SpeechService:
    """Groq-powered speech-to-text and text-to-speech."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = Groq(api_key=self._settings.groq_api_key)

    # ── Speech-to-Text ────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str | None = None,
    ) -> dict:
        """
        Transcribe audio to text using Groq Whisper.

        If language is None, Whisper auto-detects the spoken language (e.g. Hindi, English).
        Returns:
            {"text": "...", "language": "hi|en|...", "duration_s": 0.0}

        On failure, returns {"text": "", "error": "..."}.
        The caller can fall back to text input.
        """
        try:
            logger.info(
                "Transcribing audio (%d bytes, model: %s, language: %s)",
                len(audio_bytes),
                self._settings.groq_whisper_model,
                language or "auto-detect",
            )

            create_kwargs = {
                "file": (filename, audio_bytes),
                "model": self._settings.groq_whisper_model,
                "response_format": "verbose_json",
                "temperature": 0.0,
            }
            if language:
                create_kwargs["language"] = language

            transcription = self._client.audio.transcriptions.create(**create_kwargs)

            text = transcription.text if hasattr(transcription, "text") else str(transcription)
            duration = getattr(transcription, "duration", 0.0)
            detected_lang = getattr(transcription, "language", language or "en")

            logger.info("Transcription: '%s' (lang: %s, %.1fs)", text[:100], detected_lang, duration)

            return {
                "text": text.strip(),
                "language": detected_lang,
                "duration_s": float(duration) if duration else 0.0,
            }

        except Exception as e:
            logger.error("Transcription failed: %s", e, exc_info=True)
            return {
                "text": "",
                "language": language or "en",
                "duration_s": 0.0,
                "error": f"Transcription failed: {str(e)}",
            }

    # ── Text-to-Speech ────────────────────────────────────────────────

    def _synthesize_gtts(self, text: str, lang: str = "hi") -> bytes:
        """Synthesize text using Google Text-to-Speech (gTTS) for Hindi and fallback."""
        try:
            from gtts import gTTS

            # Clean markdown formatting characters
            clean_text = re.sub(r'[*_#`~]', '', text).strip()
            if not clean_text:
                return b""

            tts = gTTS(text=clean_text, lang=lang, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_bytes = fp.read()
            logger.info("gTTS synthesized %d bytes for lang '%s'", len(audio_bytes), lang)
            return audio_bytes
        except Exception as e:
            logger.error("gTTS synthesis failed for lang '%s': %s", lang, e, exc_info=True)
            return b""

    async def synthesize(self, text: str, language: str | None = None) -> bytes:
        """
        Convert text to speech.

        For Hindi text (or when language='hi'), uses gTTS for natural Hindi audio.
        For English text, uses Groq Orpheus TTS with gTTS fallback.
        """
        if not text or not text.strip():
            return b""

        # Detect Hindi (Devanagari Unicode block \u0900-\u097F)
        contains_hindi = bool(re.search(r'[\u0900-\u097F]', text))
        if contains_hindi or language == "hi":
            logger.info("Hindi detected in text. Using gTTS (hi).")
            return self._synthesize_gtts(text, lang="hi")

        # Detect Punjabi (Gurmukhi Unicode block \u0A00-\u0A7F)
        contains_punjabi = bool(re.search(r'[\u0A00-\u0A7F]', text))
        if contains_punjabi or language == "pa":
            logger.info("Punjabi detected in text. Using gTTS (pa).")
            return self._synthesize_gtts(text, lang="pa")

        try:
            chunks = self._chunk_text(text, max_chars=190)
            logger.info(
                "Synthesizing %d characters in %d chunks (model: %s, voice: %s)",
                len(text),
                len(chunks),
                self._settings.groq_tts_model,
                self._settings.groq_tts_voice,
            )

            audio_parts: list[bytes] = []

            for i, chunk in enumerate(chunks):
                response = self._client.audio.speech.create(
                    model=self._settings.groq_tts_model,
                    voice=self._settings.groq_tts_voice,
                    input=chunk,
                    response_format="wav",
                )

                # Read the audio content
                audio_data = b""
                for data in response.iter_bytes():
                    audio_data += data

                if audio_data:
                    # Fix WAV headers — Groq streams audio and leaves
                    # RIFF/data chunk sizes as 0xFFFFFFFF which browsers
                    # can't play (shows 0:00 duration).
                    audio_data = self._fix_wav_headers(audio_data)
                    audio_parts.append(audio_data)

                logger.debug("Chunk %d/%d synthesized (%d bytes)", i + 1, len(chunks), len(audio_data))

            if not audio_parts:
                logger.warning("Groq TTS produced no audio data, falling back to gTTS (en)")
                return self._synthesize_gtts(text, lang="en")

            if len(audio_parts) == 1:
                result = audio_parts[0]
            else:
                result = self._concatenate_wav(audio_parts)

            logger.info("TTS complete: %d bytes total", len(result))
            return result

        except Exception as e:
            logger.warning("Groq TTS failed (%s), falling back to gTTS: %s", self._settings.groq_tts_model, e)
            return self._synthesize_gtts(text, lang="en")


    @staticmethod
    def _fix_wav_headers(data: bytes) -> bytes:
        """
        Fix WAV files where RIFF and data chunk sizes are 0xFFFFFFFF.

        Groq Orpheus streams audio and never updates the placeholder
        sizes in the header. Browsers require correct sizes to decode
        the WAV and show proper duration.
        """
        if len(data) < 44 or data[:4] != b"RIFF":
            return data  # Not a WAV file, return as-is

        buf = bytearray(data)

        # Fix RIFF chunk size (bytes 4-7): should be file_size - 8
        riff_size = struct.unpack_from("<I", buf, 4)[0]
        if riff_size == 0xFFFFFFFF:
            struct.pack_into("<I", buf, 4, len(buf) - 8)

        # Scan for the 'data' sub-chunk and fix its size
        offset = 12  # Skip RIFF header (4) + size (4) + WAVE (4)
        while offset < len(buf) - 8:
            chunk_id = bytes(buf[offset:offset + 4])
            chunk_size = struct.unpack_from("<I", buf, offset + 4)[0]

            if chunk_id == b"data":
                if chunk_size == 0xFFFFFFFF:
                    actual_data_size = len(buf) - (offset + 8)
                    struct.pack_into("<I", buf, offset + 4, actual_data_size)
                break

            # Move to next chunk (header is 8 bytes + chunk data)
            offset += 8 + chunk_size
            # WAV chunks are word-aligned
            if chunk_size % 2 != 0:
                offset += 1

        return bytes(buf)

    def _chunk_text(self, text: str, max_chars: int = 190) -> list[str]:
        """
        Split text into chunks that fit within the TTS character limit.
        Splits on sentence boundaries first, then on word boundaries.
        """
        if len(text) <= max_chars:
            return [text]

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks: list[str] = []
        current_chunk = ""

        for sentence in sentences:
            if len(sentence) > max_chars:
                # Sentence too long — split on word boundaries
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                words = sentence.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_chars:
                        current_chunk += (" " + word) if current_chunk else word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word
            elif len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += (" " + sentence) if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [c for c in chunks if c]

    def _concatenate_wav(self, wav_parts: list[bytes]) -> bytes:
        """
        Concatenate multiple WAV files into one.

        Simple approach: use the header from the first file and
        concatenate raw PCM data from all files.
        """
        if not wav_parts:
            return b""

        if len(wav_parts) == 1:
            return wav_parts[0]

        try:
            import wave

            # Parse all WAV files
            output = io.BytesIO()
            params_set = False

            with wave.open(output, "wb") as out_wav:
                for part in wav_parts:
                    part_io = io.BytesIO(part)
                    try:
                        with wave.open(part_io, "rb") as in_wav:
                            if not params_set:
                                out_wav.setparams(in_wav.getparams())
                                params_set = True
                            out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
                    except wave.Error:
                        # If individual part fails, skip it
                        continue

            return output.getvalue()

        except Exception as e:
            logger.warning("WAV concatenation failed: %s. Returning first chunk.", e)
            return wav_parts[0]
