"""Stage 1 of the per-object pipeline: *is this a book at all?*

A detector box is only a candidate. Before OCR, searching or counting, each new track's crop is scored by CLIP
zero-shot (book spine/cover vs. whiteboard/wall/ceiling/person/...). Tracks that fail are kept (so the tracker does not
re-create them every frame) but flagged `not_book` and hidden from every later stage.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .models import device, get_model

log = logging.getLogger(__name__)
POS = ["a photo of a book spine", "a photo of a book cover", "a photo of a hardcover book",
       "a photo of a paperback book", "a photo of several books on a shelf"]
NEG = ["a photo of a whiteboard", "a photo of a ceiling", "a photo of a wall", "a photo of a person's face",
       "a photo of a person", "a photo of a shelf with objects", "a photo of a window", "a photo of a door",
       "a photo of a laptop", "a photo of a plain surface", "a photo of a cardboard box", "a photo of a lamp",
       "a photo of a picture frame", "a photo of a table", "a photo of a hand",
       "a photo of a chair", "a photo of an office chair", "a photo of electronic equipment", "a photo of a circuit board",
       "a photo of a box or package", "a photo of a keyboard", "a photo of a monitor", "a photo of a bag", "a photo of clothing",
       "a photo of a bottle", "a photo of a notebook or papers", "a photo of a poster", "a photo of a ceiling tile"]
THRESHOLD = 0.5


_state: dict = {"model": None, "loading": False, "error": None}


def _build():
    import clip
    import torch
    dev = device()
    model, pre = clip.load("ViT-B/32", device=dev)   # first run downloads ~340 MB (done off the live path)
    model.eval()
    with torch.no_grad():
        tf = model.encode_text(clip.tokenize(POS + NEG).to(dev))
        tf = tf / tf.norm(dim=-1, keepdim=True)
    return model, pre, tf, dev


def warm() -> None:
    """Load CLIP in a background thread. Until it is ready the gate reports 'unavailable' and callers use their
    stricter fallback, so the live loop never blocks on a download or model load."""
    import threading
    if _state["model"] or _state["loading"]:
        return
    _state["loading"] = True

    def run():
        try:
            _state["model"] = _build(); log.info("book gate (CLIP) ready")
        except Exception as e:
            _state["error"] = f"{type(e).__name__}: {e}"; log.warning("book gate unavailable: %s", _state["error"])
        finally:
            _state["loading"] = False
    threading.Thread(target=run, daemon=True, name="clip-warm").start()


def _load():
    return _state["model"]


def available() -> bool:
    return _state["model"] is not None


def book_probs(crops: list[np.ndarray]) -> Optional[list[float]]:
    """P(book) per crop, or None if CLIP isn't available (caller falls back to a stricter heuristic)."""
    h = _load()
    if h is None or not crops:
        return None if h is None else []
    import torch
    from PIL import Image
    model, pre, tf, dev = h
    ims, owner = [], []
    for k, c in enumerate(crops):
        c = c if c is not None and c.size else np.zeros((32, 32, 3), np.uint8)
        for seg in _squares(c):   # CLIP centre-crops to a square, so score a long thin spine in square segments
            ims.append(pre(Image.fromarray(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)))); owner.append(k)
    with torch.no_grad():
        f = model.encode_image(torch.stack(ims).to(dev))
        f = f / f.norm(dim=-1, keepdim=True)
        p = (100.0 * f @ tf.T).softmax(-1).cpu().numpy()
    pb = p[:, :len(POS)].sum(1)
    owner = np.array(owner)
    return [float(pb[owner == k].mean()) for k in range(len(crops))]


def _squares(c: np.ndarray, max_seg: int = 3) -> list[np.ndarray]:
    h, w = c.shape[:2]
    if min(h, w) < 4:
        return [c]
    if h > 1.6 * w:
        n = min(max_seg, max(2, round(h / w)))
        side = w; step = (h - side) / max(1, n - 1)
        return [c[int(i * step):int(i * step) + side] for i in range(n)]
    if w > 1.6 * h:
        n = min(max_seg, max(2, round(w / h)))
        side = h; step = (w - side) / max(1, n - 1)
        return [c[:, int(i * step):int(i * step) + side] for i in range(n)]
    return [c]


def book_probs_multiscale(img: np.ndarray, boxes: list) -> Optional[list[float]]:
    """P(book) per box using three views of the same region: tight, zoomed-in (centre 75%), zoomed-out (+35% context).
    A real book scores well on the tight view; an object that only looks bookish at one scale does not."""
    if not available():
        return None
    H, W = img.shape[:2]
    views: list[np.ndarray] = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = [int(v) for v in (x1, y1, x2, y2)]
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        def crop(a, b, c, d):
            return img[max(0, int(b)):min(H, int(d)), max(0, int(a)):min(W, int(c))]
        views += [crop(x1, y1, x2, y2),
                  crop(x1 + 0.125 * w, y1 + 0.125 * h, x2 - 0.125 * w, y2 - 0.125 * h),
                  crop(x1 - 0.35 * w, y1 - 0.35 * h, x2 + 0.35 * w, y2 + 0.35 * h)]
    p = book_probs(views)
    if p is None:
        return None
    out = []
    for i in range(len(boxes)):
        tight, zin, zout = p[3 * i], p[3 * i + 1], p[3 * i + 2]
        ctx = (zin + zout) / 2
        # tight view is the evidence; the zoomed views veto things that only look bookish up close (chairs, devices)
        out.append(tight if ctx >= 0.2 else min(tight, ctx))
    return out


def status() -> str:
    if _state["model"] is not None:
        return "ready"
    return "loading" if _state["loading"] or not _state["error"] else "failed"
