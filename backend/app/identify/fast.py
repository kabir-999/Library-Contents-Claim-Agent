"""Fast path for live display: Tesseract on the spine (~0.1-0.25 s) -> one price search.

This gives the *provisional* on-screen name/price within a fraction of a second. It is never used as the
authoritative identification: the slower pipeline (Paddle OCR, catalogue verification) still decides what goes into
the claim packet.
"""
from __future__ import annotations

import concurrent.futures as cf
import re
import shutil
from typing import Optional

import cv2
import numpy as np

_POOL = cf.ThreadPoolExecutor(3)


def available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    return shutil.which("tesseract") is not None


def _good_word(w: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", w)
    if len(letters) < 2 or len(letters) < 0.7 * len(w):
        return False
    v = sum(c in "aeiouyAEIOUY" for c in letters) / len(letters)
    return 0.15 <= v <= 0.7 or len(letters) <= 3


def _one(crop: np.ndarray, code: Optional[int], side: int = 700) -> tuple[float, list[str]]:
    import pytesseract
    im = cv2.rotate(crop, code) if code is not None else crop
    f = side / max(im.shape[:2])
    im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
    g = cv2.createCLAHE(2.0, (4, 4)).apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    d = pytesseract.image_to_data(g, config="--oem 1 --psm 6", output_type=pytesseract.Output.DICT)
    lines: dict[tuple, list[tuple[str, float]]] = {}
    for i, w in enumerate(d["text"]):
        try:
            c = float(d["conf"][i])
        except Exception:
            continue
        w = (w or "").strip()
        if w and c >= 50 and _good_word(w):
            lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append((w, c))
    out, score = [], 0.0
    for ws in lines.values():
        txt = " ".join(w for w, _ in ws)
        letters = len(re.sub(r"[^A-Za-z]", "", txt))
        long_words = [w for w, _ in ws if len(re.sub(r"[^A-Za-z]", "", w)) >= 3]
        if letters >= 7 and (len(long_words) >= 2 or (long_words and max(len(w) for w in long_words) >= 7)):
            out.append(txt)
            score += letters * float(np.mean([c for _, c in ws]))
    return score, out


def read_lines(crop_bgr: np.ndarray) -> list[str]:
    """Clean title-like text lines from a spine/cover crop. Tries both 90° rotations (and none for wide crops) in parallel."""
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    codes = [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE]
    if crop_bgr.shape[1] / max(1, crop_bgr.shape[0]) > 0.45:
        codes.append(None)
    res = list(_POOL.map(lambda c: _one(crop_bgr, c), codes))
    score, lines = max(res, key=lambda r: r[0])
    return [l.title() for l in lines[:3]] if score > 0 else []


async def quick_price(line: str, country: str, currency: str) -> Optional[dict]:
    """ONE search (eBay where the country has a marketplace, else one Google Shopping call) -> median new price."""
    from ..pricing import providers
    from ..pricing.aggregate import central, filter_book_listings
    q = f"{line} book"
    ls = await providers.ebay_search(q, country, "NEW") if providers.ebay_enabled() else []
    if len([l for l in ls if l["currency"] == currency]) < 2:
        ls = ls + await providers.serp_shopping(q, country)
    ls = [l for l in filter_book_listings(ls, line, "", "") if l["currency"] == currency and str(l.get("condition", "new")).lower() == "new"]
    c = central(ls)
    if not c or c["n"] < 2:
        return None
    return {"amount": c["amount"], "url": c["listing"]["url"], "source": c["listing"]["source"], "n": c["n"]}


_TD: dict = {}


def _text_detector():
    if "m" not in _TD:
        try:
            from paddleocr import TextDetection
            _TD["m"] = TextDetection(model_name="PP-OCRv5_mobile_det", limit_side_len=960, limit_type="max")
        except Exception:
            _TD["m"] = None
    return _TD["m"]


def discover(frame_bgr: np.ndarray, max_boxes: int = 4) -> list[list[float]]:
    """Text-driven book discovery for spines the box detectors miss (dark, tilted, held in a hand).
    PaddleOCR's text *detector* (no recognition, ~180 ms, handles tilt) finds text lines; elongated lines are
    clustered into spine/cover-shaped boxes. Candidates still go through the book gate like any detection."""
    td = _text_detector()
    if td is None:
        return []
    H, W = frame_bgr.shape[:2]
    try:
        polys = td.predict(frame_bgr)[0]["dt_polys"]
    except Exception:
        return []
    lines = []
    for p in polys:
        x1, y1, x2, y2 = float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())
        w, h = x2 - x1, y2 - y1
        if min(w, h) < 8 or max(w, h) < 40 or max(w, h) / max(1.0, min(w, h)) < 2.2:
            continue
        lines.append([x1, y1, x2, y2, h > w])
    clusters: list[list] = []
    for x1, y1, x2, y2, vert in sorted(lines, key=lambda l: -(l[2] - l[0]) * (l[3] - l[1])):
        px, py = (0.9 * (x2 - x1), 60) if vert else (60, 0.9 * (y2 - y1))
        for c in clusters:
            if x1 - px <= c[2] and x2 + px >= c[0] and y1 - py <= c[3] and y2 + py >= c[1] and c[4] == vert:
                c[0], c[1], c[2], c[3] = min(c[0], x1), min(c[1], y1), max(c[2], x2), max(c[3], y2); c[5] += 1; break
        else:
            clusters.append([x1, y1, x2, y2, vert, 1])
    out = []
    for x1, y1, x2, y2, vert, n in clusters[:max_boxes]:
        w, h = x2 - x1, y2 - y1
        px, py = (max(0.6 * w, 10), 0.08 * h) if vert else (0.08 * w, max(0.6 * h, 10))
        out.append([max(0.0, x1 - px), max(0.0, y1 - py), min(float(W), x2 + px), min(float(H), y2 + py)])
    return out
