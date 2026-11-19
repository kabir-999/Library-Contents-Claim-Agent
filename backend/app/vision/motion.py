"""Frame-to-frame planar registration (ORB + RANSAC homography).

Why: (1) the same physical book seen at different positions/distances must map to
the same mosaic location, so it is counted once; (2) a metric calibration made in
one frame must transfer to frames taken closer/farther, which only works if we
know the frame-to-frame homography (scale + perspective), not just translation.

Each frame gets H_to_ref: full-res pixels of this frame -> reference-frame pixels
of the current *segment*. If registration fails (too few inliers: fast turn, new
wall) a new segment starts rather than guessing. Planar assumption: the shelf
front is approximately a plane. Drift accumulates along long chains; for
measurement we re-estimate a direct homography to the calibration frame
(`direct_homography`) and only fall back to the chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

WORK_W = 640
MIN_INLIERS = 18


def _prep(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    f = WORK_W / w if w > WORK_W else 1.0
    g = cv2.cvtColor(cv2.resize(img_bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA) if f != 1 else img_bgr, cv2.COLOR_BGR2GRAY)
    return g, f


_orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=6, fastThreshold=12)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def features(img_bgr: np.ndarray):
    g, f = _prep(img_bgr)
    kp, des = _orb.detectAndCompute(g, None)
    return {"kp": kp, "des": des, "f": f}


def homography_between(src: dict, dst: dict) -> Optional[tuple[np.ndarray, int]]:
    """H mapping src full-res pixels -> dst full-res pixels, and the inlier count."""
    if src["des"] is None or dst["des"] is None or len(src["kp"]) < 12 or len(dst["kp"]) < 12:
        return None
    m = _bf.knnMatch(src["des"], dst["des"], k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < MIN_INLIERS:
        return None
    p1 = np.float32([src["kp"][g.queryIdx].pt for g in good]) / src["f"]
    p2 = np.float32([dst["kp"][g.trainIdx].pt for g in good]) / dst["f"]
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    if H is None:
        return None
    n = int(mask.sum())
    if n < MIN_INLIERS or n / len(good) < 0.35:
        return None
    # reject wild homographies (flips, extreme perspective)
    sx = np.hypot(H[0, 0], H[1, 0]); sy = np.hypot(H[0, 1], H[1, 1])
    if not (0.25 < sx < 4 and 0.25 < sy < 4) or abs(H[2, 0]) > 5e-3 or abs(H[2, 1]) > 5e-3:
        return None
    return H, n


@dataclass
class MotionEstimator:
    segment: int = 0
    prev: Optional[dict] = None
    H_prev_to_ref: np.ndarray = field(default_factory=lambda: np.eye(3))
    frames_in_segment: int = 0

    def update(self, img_bgr: np.ndarray) -> tuple[int, np.ndarray, bool, dict]:
        """Returns (segment, H_to_ref, new_segment_started, features)."""
        feat = features(img_bgr)
        new_seg = False
        if self.prev is None:
            H = np.eye(3)
        else:
            r = homography_between(feat, self.prev)   # this -> prev
            if r is None:
                self.segment += 1; H = np.eye(3); new_seg = True; self.frames_in_segment = 0
            else:
                H = self.H_prev_to_ref @ r[0]
        self.prev, self.H_prev_to_ref = feat, H
        self.frames_in_segment += 1
        return self.segment, H, new_seg, feat


def map_box(H: np.ndarray, box) -> list[float]:
    x1, y1, x2, y2 = box
    pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def map_points(H: np.ndarray, pts) -> np.ndarray:
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)
