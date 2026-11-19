"""Book-spine detection (YOLO + classical spine splitting). Never an LLM."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import get_settings
from .models import device, get_model
from .spines import split_run

log = logging.getLogger(__name__)
COCO_BOOK = 73


def orientation_of(box) -> str:
    w, h = box[2] - box[0], box[3] - box[1]
    if h >= 1.5 * w:
        return "upright"
    if w >= 1.5 * h:
        return "horizontal_stack"
    return "flat"


class BookDetector:
    def __init__(self):
        self.s = get_settings()
        self.custom = bool(self.s.lca_yolo_spine_weights)

    def _model(self):
        def make():
            from ultralytics import YOLO
            return YOLO(self.s.lca_yolo_spine_weights or "yolov8s.pt")
        return get_model("yolo_book", make)

    @property
    def available(self) -> bool:
        return self._model() is not None

    def detect(self, img_bgr: np.ndarray, conf: float | None = None) -> list[dict]:
        conf = self.s.lca_book_conf if conf is None else conf
        m = self._model()
        if m is None:
            return []
        kw = dict(imgsz=1280, conf=conf, iou=0.5, verbose=False, device=device())
        if not self.custom:
            kw["classes"] = [COCO_BOOK]
        res = m.predict(img_bgr, **kw)[0]
        out: list[dict] = []
        H, W = img_bgr.shape[:2]
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            c = float(b.conf[0])
            boxes = [[x1, y1, x2, y2]] if self.custom else split_run(img_bgr, [x1, y1, x2, y2])
            for bb in boxes:
                w, h = bb[2] - bb[0], bb[3] - bb[1]
                if w < 6 or h < 25:
                    continue
                if w * h > 0.9 * W * H:   # whole-frame blob, not a book
                    continue
                out.append({"bbox": bb, "conf": c, "label": "book", "orientation": orientation_of(bb),
                            "from_split": len(boxes) > 1})
        # drop near-duplicates produced by overlapping YOLO boxes
        out.sort(key=lambda d: -d["conf"])
        kept: list[dict] = []
        from .tracker import iou
        for d in out:
            if all(iou(d["bbox"], k["bbox"]) < 0.6 for k in kept):
                kept.append(d)
        return kept
