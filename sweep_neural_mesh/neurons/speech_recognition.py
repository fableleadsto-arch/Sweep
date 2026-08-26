"""
Speech Recognition — OpenAI Whisper integration with fallbacks.

Architecture:
    Audio Input (path, bytes, or file-like)
        ↓
    [Primary: whisper tiny.en (72MB, CPU)]
    [Fallback: None (graceful error)]
        ↓
    Transcript (text, segments, language, duration)
        ↓
    Optional: speaker diarization hints

Whisper model lazy-loaded on first use.
"""
from __future__ import annotations

import io
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_whisper_model = None
_whisper_backend = None  # 'whisper', None


def _get_whisper():
    global _whisper_model, _whisper_backend
    if _whisper_model is not None:
        return _whisper_model

    try:
        import whisper
        _whisper_model = whisper.load_model("tiny.en")
        _whisper_backend = "whisper"
        logger.info("Loaded Whisper tiny.en model")
        return _whisper_model
    except Exception as e:
        logger.warning(f"Whisper load failed: {e}")
        _whisper_backend = None
        return None


def _is_audio_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma', '.opus', '.webm')


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    text: str
    segments: list[TranscriptSegment]
    language: str
    duration_seconds: float
    backend: str
    latency_ms: float = 0.0


class SpeechRecognizer:
    def __init__(self, model_name: str = "tiny.en"):
        self._model_name = model_name
        self._backend = None

    @property
    def backend(self) -> str:
        return _whisper_backend or "none"

    def recognize(self, audio_input: str | bytes | io.IOBase) -> TranscriptResult:
        t0 = time.perf_counter()

        model = _get_whisper()
        if model is None:
            return TranscriptResult(
                text="", segments=[], language="unknown",
                duration_seconds=0.0, backend="none",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            import whisper
            if isinstance(audio_input, str):
                result = model.transcribe(audio_input)
            elif isinstance(audio_input, bytes):
                result = whisper.transcribe(model, audio_input)
            else:
                result = model.transcribe(audio_input)

            segments = [
                TranscriptSegment(start=s["start"], end=s["end"], text=s["text"].strip())
                for s in result.get("segments", [])
            ]

            total_duration = segments[-1].end if segments else 0.0

            return TranscriptResult(
                text=result.get("text", "").strip(),
                segments=segments,
                language=result.get("language", "unknown"),
                duration_seconds=total_duration,
                backend="whisper",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return TranscriptResult(
                text="", segments=[], language="unknown",
                duration_seconds=0.0, backend="none",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    def recognize_text_only(self, audio_input: str | bytes | io.IOBase) -> str:
        return self.recognize(audio_input).text

    def transcribe_with_timestamps(self, audio_input: str | bytes | io.IOBase) -> list[dict[str, Any]]:
        result = self.recognize(audio_input)
        return [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in result.segments
        ]


_default_recognizer: SpeechRecognizer | None = None


def get_recognizer() -> SpeechRecognizer:
    global _default_recognizer
    if _default_recognizer is None:
        _default_recognizer = SpeechRecognizer()
    return _default_recognizer
