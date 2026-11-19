"""Per-book spine height/thickness from a detected box + a plane calibration."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..schemas import Measurement
from .calibration import PlaneCalibration

# Physical sanity bounds for a trade book spine (cm). Outside => reject, don't report.
MAX_HEIGHT_CM = 60.0
MAX_THICKNESS_CM = 15.0


def measure_spine(
    bbox: Sequence[float],
    orientation: str,
    cal: Optional[PlaneCalibration],
    frame_ref: str,
    frame_size: tuple[int, int],
    det_conf: float = 1.0,
    margin_px: int = 3,
    h_to_cal: Optional[np.ndarray] = None,
) -> tuple[Measurement, Measurement, list[str]]:
    """Returns (height, thickness, flags). Missing scale or truncated boxes => empty Measurements."""
    flags: list[str] = []
    empty = Measurement(unit="cm")
    if cal is None:
        return empty, empty, ["no_calibration"]
    x1, y1, x2, y2 = bbox
    W, H = frame_size
    truncated = x1 <= margin_px or y1 <= margin_px or x2 >= W - margin_px or y2 >= H - margin_px
    if truncated:
        flags.append("truncated_by_frame")
        return empty, empty, flags  # a clipped box under-measures: refuse rather than report

    # Corners of the box -> calibration frame (scale/perspective-aware) -> cm.
    pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if h_to_cal is not None:
        from ..vision.motion import map_points
        pts = map_points(h_to_cal, pts)
    tl, tr, br, bl = cal.to_cm(pts)
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    vert, horiz = float((left + right) / 2), float((top + bottom) / 2)
    if orientation == "upright":  # spine runs vertically
        h_cm, t_cm = vert, horiz
    else:                          # horizontal stack / flat: spine runs horizontally
        h_cm, t_cm = horiz, vert

    if not (3 <= h_cm <= MAX_HEIGHT_CM) or not (0.3 <= t_cm <= MAX_THICKNESS_CM):
        flags.append("implausible_dimensions")
        return empty, empty, flags

    conf = round(cal.confidence * min(1.0, 0.5 + det_conf / 2), 3)
    mk = lambda v: Measurement(value=round(v, 1), unit="cm", method=cal.method, confidence=conf, frame_ref=frame_ref, calibration_ref=cal.id)
    return mk(h_cm), mk(t_cm), flags
