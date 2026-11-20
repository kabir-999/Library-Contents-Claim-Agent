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
            return YOLO(self.s.lca_yolo_spine_weights or ("yolov8n.pt" if self.s.lca_edge_profile else "yolov8s.pt"))
        return get_model("yolo_book", make)

    @property
    def available(self) -> bool:
        return self._model() is not None

    def _world(self):
        """Open-vocabulary 'book' detector: complements COCO YOLO, which misses many clearly visible spines."""
        def make():
            from ultralytics import YOLOWorld
            m = YOLOWorld("yolov8s-worldv2.pt")
            m.set_classes(["book", "book spine"])
            return m
        return get_model("yolo_world_book", make)

    def _raw(self, m, img, conf, imgsz, classes, ox=0, oy=0):
        kw = dict(imgsz=imgsz, conf=conf, iou=0.6, verbose=False, device=device())
        if classes is not None:
            kw["classes"] = classes
        res = m.predict(img, **kw)[0]
        return [([b.xyxy[0][0].item() + ox, b.xyxy[0][1].item() + oy, b.xyxy[0][2].item() + ox, b.xyxy[0][3].item() + oy],
                 float(b.conf[0])) for b in res.boxes]

    def detect(self, img_bgr: np.ndarray, conf: float | None = None) -> list[dict]:
        conf = self.s.lca_book_conf if conf is None else conf
        m = self._model()
        if m is None:
            return []
        H, W = img_bgr.shape[:2]
        cls = None if self.custom else [COCO_BOOK]
        fast = self.s.lca_fast_detect
        raw = self._raw(m, img_bgr, max(conf, 0.1) if fast else conf, 960 if fast else 1280, cls)
        # overlapping tiles recover small/dense spines the full-frame pass misses
        if W >= 900 and not self.custom and not fast:
            tw = int(W * 0.58)
            for ox in (0, W - tw):
                raw += self._raw(m, img_bgr[:, ox:ox + tw], conf, 960, cls, ox=ox)
        wm = self._world()   # open-vocab 'book': finds tilted / held-up spines the COCO class misses (28 ms at 960 px)
        if wm is not None:
            raw += [(b, c * 0.9) for b, c in self._raw(wm, img_bgr, max(conf, 0.08), 960, None)]
        return self._post(img_bgr, raw)

    def _post(self, img_bgr: np.ndarray, raw: list) -> list[dict]:
        H, W = img_bgr.shape[:2]
        out: list[dict] = []
        for (x1, y1, x2, y2), c in raw:
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
            boxes = [[x1, y1, x2, y2]] if self.custom else split_run(img_bgr, [x1, y1, x2, y2])
            for bb in boxes:
                w, h = bb[2] - bb[0], bb[3] - bb[1]
                if w < 6 or h < 25:
                    continue
                if w * h > 0.9 * W * H:   # whole-frame blob, not a book
                    continue
                if _not_a_book(img_bgr, bb):
                    continue
                out.append({"bbox": bb, "conf": c, "label": "book", "orientation": orientation_of(bb),
                            "from_split": len(boxes) > 1})
        # merge duplicates from the full/tiled/open-vocab passes (also containment, not just IoU)
        out.sort(key=lambda d: -d["conf"])
        kept: list[dict] = []
        from .tracker import iou
        for d in out:
            if all(iou(d["bbox"], k["bbox"]) < 0.5 and _contain(d["bbox"], k["bbox"]) < 0.8 for k in kept):
                kept.append(d)
        return kept

    def _raw_batch(self, m, imgs, conf, imgsz, classes):
        kw = dict(imgsz=imgsz, conf=conf, iou=0.6, verbose=False, device=device(), batch=len(imgs))
        if classes is not None:
            kw["classes"] = classes
        return [[([float(v) for v in bx.xyxy[0].tolist()], float(bx.conf[0])) for bx in r.boxes] for r in m.predict(imgs, **kw)]

    def detect_batch(self, imgs: list, chunk: int = 16) -> list[list[dict]]:
        """Many frames at once (offline sweep processing): YOLO runs over lists, so throughput scales with the device."""
        m = self._model()
        if m is None:
            return [[] for _ in imgs]
        wm, cls, conf = self._world(), (None if self.custom else [COCO_BOOK]), self.s.lca_book_conf
        out = []
        for i in range(0, len(imgs), chunk):
            part = imgs[i:i + chunk]
            coco = self._raw_batch(m, part, max(conf, 0.1), 960, cls)
            world = self._raw_batch(wm, part, max(conf, 0.08), 960, None) if wm is not None else [[] for _ in part]
            for img, c1, w1 in zip(part, coco, world):
                out.append(self._post(img, c1 + [(b, c * 0.9) for b, c in w1]))
        return out


def _contain(a, b) -> float:
    """Fraction of the smaller box covered by the other."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    sa, sb = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return ix * iy / max(1e-6, min(sa, sb))


def _not_a_book(img, bb) -> bool:
    """Hands/skin and blank wall are the main false positives for book detectors."""
    import cv2
    x1, y1, x2, y2 = [int(v) for v in bb]
    crop = img[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return True
    small = cv2.resize(crop, (32, 48))
    if float(small.std()) < 10:
        return True
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycc, (0, 135, 85), (255, 175, 125))
    return float((skin > 0).mean()) > 0.55
