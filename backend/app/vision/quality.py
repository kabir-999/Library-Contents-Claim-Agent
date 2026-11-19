"""Frame-quality analysis (OpenCV only). Drives live guidance."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

BLUR_VAR_MIN = 60.0     # variance of Laplacian on a 640px-wide frame
GLARE_FRAC_MAX = 0.04   # fraction of near-white low-saturation pixels
DARK_LUMA_MIN = 55.0
BRIGHT_LUMA_MAX = 215.0


@dataclass
class FrameQuality:
    sharpness: float
    blurred: bool
    glare_frac: float
    glare: bool
    mean_luma: float
    too_dark: bool
    too_bright: bool
    glare_box: list | None = None  # x1,y1,x2,y2 of largest glare blob (frame px)

    @property
    def usable(self) -> bool:
        return not (self.blurred or self.too_dark)

    @property
    def score(self) -> float:
        s = min(1.0, self.sharpness / (BLUR_VAR_MIN * 3))
        return round(s * (0.6 if self.glare else 1.0) * (0.5 if self.too_dark or self.too_bright else 1.0), 3)

    def issues(self) -> list[str]:
        out = []
        if self.blurred: out.append("blur")
        if self.glare: out.append("glare")
        if self.too_dark: out.append("dark")
        if self.too_bright: out.append("overexposed")
        return out

    def to_dict(self) -> dict:
        d = asdict(self); d["score"] = self.score; d["issues"] = self.issues(); return d


def analyze(img_bgr: np.ndarray) -> FrameQuality:
    h, w = img_bgr.shape[:2]
    scale = 640.0 / w if w > 640 else 1.0
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale != 1 else img_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    glare_mask = ((hsv[..., 2] > 245) & (hsv[..., 1] < 40)).astype(np.uint8)
    glare_frac = float(glare_mask.mean())
    gbox = None
    if glare_frac > 0.005:
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            gbox = [x / scale, y / scale, (x + bw) / scale, (y + bh) / scale]
    luma = float(gray.mean())
    return FrameQuality(
        sharpness=sharp, blurred=sharp < BLUR_VAR_MIN,
        glare_frac=glare_frac, glare=glare_frac > GLARE_FRAC_MAX,
        mean_luma=luma, too_dark=luma < DARK_LUMA_MIN, too_bright=luma > BRIGHT_LUMA_MAX, glare_box=gbox,
    )


def region_sharpness(img_bgr: np.ndarray, bbox) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = img_bgr[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if g.shape[0] < 8 or g.shape[1] < 8:
        return 0.0
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
