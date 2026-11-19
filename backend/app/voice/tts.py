"""Text-to-speech. Kokoro server-side when configured+installed, else the browser's
speechSynthesis (the client falls back automatically when `audio` is absent)."""
from __future__ import annotations

import base64
import io
import logging
import wave
from typing import Optional

import numpy as np

from ..config import get_settings
from ..vision.models import get_model

log = logging.getLogger(__name__)


def _pipe():
    def make():
        from kokoro import KPipeline
        return KPipeline(lang_code="a")
    return get_model("kokoro", make)


def engine() -> str:
    if get_settings().lca_tts_engine == "kokoro" and _pipe() is not None:
        return "kokoro"
    return "browser"


def synthesize_wav(text: str) -> Optional[bytes]:
    """24 kHz mono WAV, or None when the browser should speak."""
    if engine() != "kokoro":
        return None
    try:
        chunks = [np.asarray(a) for _, _, a in _pipe()(text, voice="af_heart")]
        audio = np.concatenate(chunks) if chunks else None
        if audio is None:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return buf.getvalue()
    except Exception as e:
        log.warning("kokoro synth failed: %s", e)
        return None


def synthesize_b64(text: str) -> Optional[str]:
    w = synthesize_wav(text)
    return base64.b64encode(w).decode() if w else None
