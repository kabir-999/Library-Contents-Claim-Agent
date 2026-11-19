"""Classical spine segmentation used inside YOLO 'book' regions.

COCO-pretrained YOLO has one 'book' class and usually returns a *blob* for a
tightly packed run. We split a run into spines using vertical-edge energy and
column-colour changes (OpenCV). If custom spine weights are configured the
detector bypasses this module.
"""
from __future__ import annotations

import cv2
import numpy as np


def split_run(img_bgr: np.ndarray, box, min_width_px: int = 9, max_aspect_for_single: float = 0.55) -> list[list[float]]:
    """Split a box containing several upright spines into per-spine boxes.
    Returns boxes in image coordinates. If nothing sensible, returns [box]."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 2 * min_width_px or h < 40:
        return [list(map(float, box))]
    # A single spine is much taller than wide; only try to split wide runs.
    if w / h < max_aspect_for_single * 0.8:
        return [list(map(float, box))]
    crop = img_bgr[y1:y2, x1:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, 1, 0, ksize=3))
    # Only vertical edges that persist along most of the height count (text is short/noisy).
    edge_prof = (gx > 0.5 * gx.max()).astype(np.float32).mean(axis=0) if gx.max() > 0 else np.zeros(w, np.float32)
    colmean = lab.mean(axis=0)  # (w,3)
    cdiff = np.zeros(w, np.float32)
    k = 3
    for i in range(k, w - k):
        cdiff[i] = np.linalg.norm(colmean[i - k:i].mean(0) - colmean[i:i + k].mean(0))
    cdiff = cdiff / (cdiff.max() + 1e-6)
    score = 0.6 * (edge_prof / (edge_prof.max() + 1e-6)) + 0.4 * cdiff
    score = np.convolve(score, np.ones(3) / 3, mode="same")
    thr = max(0.35, float(score.mean() + 1.0 * score.std()))
    cuts = []
    for i in range(min_width_px, w - min_width_px):
        if score[i] >= thr and score[i] == score[max(0, i - min_width_px // 2): i + min_width_px // 2 + 1].max():
            if not cuts or i - cuts[-1] >= min_width_px:
                cuts.append(i)
    if not cuts:
        return [list(map(float, box))]
    edges = [0] + cuts + [w]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a >= min_width_px:
            out.append([float(x1 + a), float(y1), float(x1 + b), float(y2)])
    return out or [list(map(float, box))]
