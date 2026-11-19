"""Lazy, shared model handles + availability reporting (no import-time heavy loads)."""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

log = logging.getLogger(__name__)
_lock = threading.Lock()
_cache: dict[str, Any] = {}
_errors: dict[str, str] = {}


def device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_model(key: str, factory) -> Optional[Any]:
    with _lock:
        if key in _cache:
            return _cache[key]
        if key in _errors:
            return None
        try:
            _cache[key] = factory()
            log.info("loaded model %s", key)
            return _cache[key]
        except Exception as e:  # missing package, offline weights, etc.
            _errors[key] = f"{type(e).__name__}: {e}"
            log.warning("model %s unavailable: %s", key, _errors[key])
            return None


def status() -> dict:
    return {"loaded": sorted(_cache), "errors": dict(_errors), "device": device()}
