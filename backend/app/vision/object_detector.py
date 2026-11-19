"""Non-book object detection.

Primary: YOLO-World (open vocabulary -> bookshelf, cabinet, lamp, rug, framed
painting, ...) because COCO lacks most furniture/decor classes.
Fallback: COCO YOLO with a COCO->category mapping (documented limitation: no
rugs, lamps, cabinets, framed art).
"""
from __future__ import annotations

import logging

import numpy as np

from ..config import get_settings
from .models import device, get_model

log = logging.getLogger(__name__)

VOCAB = [
    "bookshelf", "cabinet", "desk", "table", "chair", "armchair", "sofa", "lamp", "floor lamp", "rug",
    "framed painting", "framed picture", "portrait", "mirror", "coffee machine", "television", "laptop",
    "computer monitor", "speaker", "vase", "sculpture", "potted plant", "clock", "globe", "window",
    "door", "curtains", "storage box", "candle", "decorative object",
]
CATEGORY = {
    "bookshelf": "bookshelf", "cabinet": "cabinet", "desk": "desk", "table": "table", "chair": "chair",
    "armchair": "chair", "sofa": "sofa", "lamp": "lamp", "floor lamp": "lamp", "rug": "rug",
    "framed painting": "artwork", "framed picture": "artwork", "portrait": "portrait", "mirror": "mirror",
    "coffee machine": "coffee_machine", "television": "electronics", "laptop": "electronics",
    "computer monitor": "electronics", "speaker": "electronics", "vase": "decorative", "sculpture": "decorative",
    "potted plant": "decorative", "clock": "decorative", "globe": "decorative", "candle": "decorative",
    "decorative object": "decorative", "storage box": "decorative", "window": "fixture", "door": "fixture",
    "curtains": "fixture",
}
NOT_CLAIMABLE = {"fixture"}  # structure of the room, not contents
COCO_MAP = {
    "chair": "chair", "couch": "sofa", "bed": "furniture", "dining table": "table", "tv": "electronics",
    "laptop": "electronics", "keyboard": "electronics", "mouse": "electronics", "cell phone": "electronics",
    "remote": "electronics", "potted plant": "decorative", "vase": "decorative", "clock": "decorative",
    "bottle": "decorative", "cup": "decorative", "teddy bear": "decorative", "microwave": "appliance",
    "oven": "appliance", "toaster": "appliance", "refrigerator": "appliance", "suitcase": "luggage",
    "backpack": "luggage", "umbrella": "decorative",
}


class ObjectDetector:
    def __init__(self):
        self.s = get_settings()
        self.mode = "none"

    def _model(self):
        def make_world():
            from ultralytics import YOLOWorld
            m = YOLOWorld("yolov8s-worldv2.pt")
            m.set_classes(VOCAB)
            return m
        m = get_model("yolo_world", make_world)
        if m is not None:
            self.mode = "yolo_world"
            return m

        def make_coco():
            from ultralytics import YOLO
            return YOLO(self.s.lca_yolo_object_weights)
        m = get_model("yolo_coco", make_coco)
        if m is not None:
            self.mode = "coco"
        return m

    @property
    def available(self) -> bool:
        return self._model() is not None

    def detect(self, img_bgr: np.ndarray, conf: float = 0.25) -> list[dict]:
        m = self._model()
        if m is None:
            return []
        res = m.predict(img_bgr, imgsz=960, conf=conf, verbose=False, device=device())[0]
        names = res.names
        out = []
        for b in res.boxes:
            name = names[int(b.cls[0])]
            if name == "book":
                continue
            cat = CATEGORY.get(name) if self.mode == "yolo_world" else COCO_MAP.get(name)
            if not cat or cat in NOT_CLAIMABLE:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            crop = img_bgr[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            if crop.size == 0 or float(crop.std()) < 12.0:   # blank wall / uniform patch: not an object
                continue
            out.append({"bbox": [x1, y1, x2, y2], "conf": float(b.conf[0]), "label": name, "category": cat})
        return out
