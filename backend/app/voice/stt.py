"""Speech-to-text: Whisper (faster-whisper, CPU int8). Utterance-level, not streaming.

The browser segments speech with a VAD and sends each utterance; barge-in
(interrupting the agent) is handled by the client + orchestrator, see agent/live.py.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import get_settings
from ..vision.models import get_model

log = logging.getLogger(__name__)
# Biases decoding toward our domain vocabulary; contains no facts about the user's room.
PROMPT = "Home library inventory. Words: first edition, print, original, shelf, skip, mark corner, camera height, centimetres."


def _model():
    def make():
        from faster_whisper import WhisperModel
        return WhisperModel(get_settings().lca_whisper_model, device="cpu", compute_type="int8")
    return get_model("whisper", make)


def available() -> bool:
    return _model() is not None


def transcribe_pcm16(pcm: bytes, sample_rate: int = 16000) -> Optional[str]:
    m = _model()
    if m is None or len(pcm) < sample_rate * 2 * 0.25:  # <250 ms: noise
        return None
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_rate != 16000:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sample_rate)).astype(int)
        audio = audio[idx]
    segs, _ = m.transcribe(audio, language="en", beam_size=1, vad_filter=True, initial_prompt=PROMPT,
                           condition_on_previous_text=False, no_speech_threshold=0.6)
    text = " ".join(s.text.strip() for s in segs).strip()
    return text or None
