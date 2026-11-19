"""OCR for book spines (PaddleOCR). Spine text is rotated, so we try both 90° orientations."""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from ..config import get_settings
from .models import get_model

log = logging.getLogger(__name__)


def _engine():
    def make():
        from paddleocr import PaddleOCR
        # PaddleOCR 2.x and 3.x have different constructor args and raise different errors; try 3.x first.
        for kw in ({"lang": "en", "use_textline_orientation": True, "use_doc_orientation_classify": False, "use_doc_unwarping": False},
                   {"lang": "en", "use_textline_orientation": True}, {"lang": "en"},
                   {"use_angle_cls": True, "lang": "en", "show_log": False}):
            try:
                return PaddleOCR(**kw)
            except (TypeError, ValueError):
                continue
        raise RuntimeError("no compatible PaddleOCR constructor")
    return get_model("paddleocr", make)


def available() -> bool:
    return get_settings().lca_ocr_enabled and _engine() is not None


def _run(eng, img) -> list[tuple[str, float]]:
    try:
        res = eng.ocr(img) if hasattr(eng, "ocr") else eng.predict(img)
    except TypeError:
        res = eng.ocr(img, cls=True)
    out: list[tuple[str, float]] = []
    if not res:
        return out
    page = res[0] if isinstance(res, list) and res else res
    if page is None:
        return out
    if isinstance(page, dict):  # 3.x result object style
        for t, s in zip(page.get("rec_texts", []), page.get("rec_scores", [])):
            out.append((str(t), float(s)))
        return out
    if hasattr(page, "get") and "rec_texts" in getattr(page, "keys", lambda: [])():
        for t, s in zip(page["rec_texts"], page["rec_scores"]):
            out.append((str(t), float(s)))
        return out
    for line in page:
        try:
            txt, sc = line[1][0], line[1][1]
            out.append((str(txt), float(sc)))
        except Exception:
            continue
    return out


def read_spine(img_bgr: np.ndarray, bbox, orientation: str = "upright") -> dict:
    """Returns {"lines":[{"text","conf"}], "text": joined, "conf": mean, "rotation": deg}."""
    eng = _engine() if get_settings().lca_ocr_enabled else None
    if eng is None:
        return {"lines": [], "text": "", "conf": 0.0, "rotation": None, "available": False}
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = max(2, int(0.06 * (x2 - x1)))
    crop = img_bgr[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
    if crop.size == 0:
        return {"lines": [], "text": "", "conf": 0.0, "rotation": None, "available": True}
    cands = []
    rots = [("none", None)] if orientation != "upright" else [
        ("cw90", cv2.ROTATE_90_CLOCKWISE), ("ccw90", cv2.ROTATE_90_COUNTERCLOCKWISE)]
    if orientation == "horizontal_stack":
        rots = [("none", None)]
    for name, code in rots:
        im = cv2.rotate(crop, code) if code is not None else crop
        h, w = im.shape[:2]
        if h < 48:  # upscale small text for the recogniser
            f = 48 / h
            im = cv2.resize(im, (int(w * f), 48), interpolation=cv2.INTER_CUBIC)
        lines = [(t, c) for t, c in _run(eng, im) if t.strip() and c >= 0.30]
        score = sum(c * len(t) for t, c in lines)
        cands.append((score, name, lines))
    score, name, lines = max(cands, key=lambda c: c[0])
    mean = float(np.mean([c for _, c in lines])) if lines else 0.0
    return {"lines": [{"text": t, "conf": round(c, 3)} for t, c in lines],
            "text": " ".join(t for t, _ in lines), "conf": round(mean, 3), "rotation": name, "available": True}
