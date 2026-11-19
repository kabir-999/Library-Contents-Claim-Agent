"""Metric scale. The single place where pixels become centimetres.

We never guess a scale. A `PlaneCalibration` is created only from one of:

  reference_object   A visible A4/Letter sheet or ID-1 card detected by OpenCV; its
                     four corners define a homography pixel->cm for that plane
                     (corrects perspective for objects in the same plane).
  two_point          The user marks two points on a saved frame and supplies the
                     tape-measured distance (also covers "known shelf dimension").
  depth              Metric depth (LiDAR / depth sensor) at the plane + intrinsics.
  pinhole            Camera-to-plane distance + focal length in pixels. Low
                     confidence: intrinsics are usually approximate.

Spine measurements made on a plane inherit its calibration. If a book has no
applicable calibration its metric values stay None (-> review queue).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

# Physical sizes of detectable references (cm). Long side first.
REFERENCE_OBJECTS = {
    "A4": (29.7, 21.0),
    "US_LETTER": (27.94, 21.59),
    "ID1_CARD": (8.56, 5.398),  # credit card
}
# Max confidence per method (how far we are willing to trust it)
METHOD_CONFIDENCE = {"reference_object": 0.80, "two_point": 0.80, "depth": 0.70, "pinhole": 0.35}


@dataclass
class PlaneCalibration:
    id: str
    method: str
    scope: str  # shelf id or "global"
    frame_ref: str
    confidence: float
    px_per_cm: Optional[float] = None  # scalar scale (mean)
    homography: Optional[np.ndarray] = None  # 3x3 pixel -> cm on the plane
    note: str = ""
    reference: str = ""
    created_ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "method": self.method, "scope": self.scope, "frame_ref": self.frame_ref,
            "confidence": round(self.confidence, 3),
            "px_per_cm": None if self.px_per_cm is None else round(self.px_per_cm, 4),
            "has_homography": self.homography is not None, "note": self.note, "reference": self.reference,
        }

    def to_cm(self, pts: Sequence[Sequence[float]]) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.homography is not None:
            return cv2.perspectiveTransform(p.reshape(-1, 1, 2), self.homography).reshape(-1, 2)
        assert self.px_per_cm
        return p / self.px_per_cm

    def length_cm(self, p1, p2) -> float:
        a, b = self.to_cm([p1, p2])
        return float(np.linalg.norm(a - b))


def from_two_points(cal_id: str, p1, p2, real_cm: float, frame_ref: str, scope: str = "global", what: str = "user-measured distance") -> PlaneCalibration:
    if real_cm <= 0:
        raise ValueError("real distance must be positive")
    px = math.dist(p1, p2)
    if px < 20:
        raise ValueError("calibration points too close together to give a reliable scale")
    # Confidence grows with the length of the baseline (longer => smaller relative marking error)
    conf = METHOD_CONFIDENCE["two_point"] * min(1.0, 0.6 + px / 800)
    return PlaneCalibration(cal_id, "two_point", scope, frame_ref, conf, px_per_cm=px / real_cm,
                            note=f"{what}: {real_cm} cm over {px:.0f}px", reference=what)


def from_pinhole(cal_id: str, distance_m: float, focal_px: float, frame_ref: str, scope: str = "global") -> PlaneCalibration:
    if distance_m <= 0 or focal_px <= 0:
        raise ValueError("distance and focal length must be positive")
    return PlaneCalibration(cal_id, "pinhole", scope, frame_ref, METHOD_CONFIDENCE["pinhole"],
                            px_per_cm=focal_px / (distance_m * 100.0),
                            note=f"pinhole: Z={distance_m:.2f} m, f={focal_px:.0f}px (approximate intrinsics)")


def from_depth(cal_id: str, depth_m: float, focal_px: float, frame_ref: str, scope: str = "global") -> PlaneCalibration:
    c = from_pinhole(cal_id, depth_m, focal_px, frame_ref, scope)
    c.method = "depth"
    c.confidence = METHOD_CONFIDENCE["depth"]
    c.note = f"depth sensor: Z={depth_m:.2f} m, f={focal_px:.0f}px"
    return c


def _order_quad(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)  # tl,tr,br,bl


def homography_from_reference(quad_px: np.ndarray, long_cm: float, short_cm: float) -> np.ndarray:
    """Pixel -> cm homography from a detected reference quad (tl,tr,br,bl).
    The quad's longer pixel side is mapped to the object's long side."""
    q = _order_quad(np.asarray(quad_px, dtype=np.float32))
    top = np.linalg.norm(q[1] - q[0]); left = np.linalg.norm(q[3] - q[0])
    w_cm, h_cm = (long_cm, short_cm) if top >= left else (short_cm, long_cm)
    dst = np.array([[0, 0], [w_cm, 0], [w_cm, h_cm], [0, h_cm]], dtype=np.float32)
    return cv2.getPerspectiveTransform(q, dst)


def detect_reference_object(img_bgr: np.ndarray, kinds: Sequence[str] = ("A4", "US_LETTER")) -> Optional[dict]:
    """Find a bright, quadrilateral, correct-aspect sheet. Returns quad + kind or None.

    Deliberately conservative: false negatives just mean we ask for another
    calibration source; a false positive would corrupt every measurement.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Paper is locally bright: adaptive threshold copes with uneven light.
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(thr, bright)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.004 * w * h or area > 0.6 * w * h:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = _order_quad(approx.reshape(4, 2).astype(np.float32))
        sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
        # opposite sides must be near-equal (limits perspective skew)
        if abs(sides[0] - sides[2]) / max(sides[0], sides[2]) > 0.25 or abs(sides[1] - sides[3]) / max(sides[1], sides[3]) > 0.25:
            continue
        ratio = max(sides) / max(1e-6, min(sides))
        # fill ratio: the contour should be a clean rectangle
        if area / (cv2.contourArea(quad) + 1e-6) < 0.85:
            continue
        for kind in kinds:
            lc, sc = REFERENCE_OBJECTS[kind]
            if abs(ratio - lc / sc) / (lc / sc) < 0.12:
                score = area * (1 - abs(ratio - lc / sc))
                if best is None or score > best["score"]:
                    best = {"kind": kind, "quad": quad, "score": score, "ratio": ratio}
    return best


def from_reference(cal_id: str, img_bgr: np.ndarray, frame_ref: str, scope: str = "global") -> Optional[PlaneCalibration]:
    det = detect_reference_object(img_bgr)
    if not det:
        return None
    lc, sc = REFERENCE_OBJECTS[det["kind"]]
    H = homography_from_reference(det["quad"], lc, sc)
    mean_px = np.mean([np.linalg.norm(det["quad"][(i + 1) % 4] - det["quad"][i]) for i in range(4)])
    return PlaneCalibration(cal_id, "reference_object", scope, frame_ref, METHOD_CONFIDENCE["reference_object"],
                            px_per_cm=float(mean_px / ((lc + sc) / 2)), homography=H,
                            note=f"detected {det['kind']} sheet ({lc}x{sc} cm)", reference=det["kind"])


@dataclass
class CalibrationStore:
    items: list[PlaneCalibration] = field(default_factory=list)

    def add(self, c: PlaneCalibration):
        self.items.append(c)

    def best_for(self, scope: str) -> Optional[PlaneCalibration]:
        cands = [c for c in self.items if c.scope == scope] or [c for c in self.items if c.scope == "global"]
        return max(cands, key=lambda c: c.confidence, default=None)

    def all_dicts(self) -> list[dict]:
        return [c.to_dict() for c in self.items]
