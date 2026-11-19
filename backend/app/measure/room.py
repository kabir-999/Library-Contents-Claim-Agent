"""Room geometry from visible-floor-plane ranging.

Method (documented assumption set, no LLM involved):
  A phone held at known height h above the floor, pointed at a floor/wall
  junction with pitch θ below horizontal, sees that point at horizontal
  distance d = h / tan θ. With the compass/yaw heading ψ, the point is
  (d·sin ψ, d·cos ψ) in a room-fixed frame centred on the user. Marking each
  floor corner gives a polygon (shoelace area). Ceiling height follows from
  aiming at the ceiling/wall junction above the same corner:
  H = h + d·tan θ_up.

Limitations (surfaced in output): needs device orientation sensors, camera
height, ≥3 floor corners, and the user standing roughly still while marking.
Pitch noise of ±2° at 45° pitch is ~±7% distance; we propagate that into the
confidence instead of hiding it. Wall area is gross (doors/windows not subtracted).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

PITCH_SIGMA_DEG = 2.0  # assumed consumer-phone orientation noise


@dataclass
class CornerObs:
    heading_deg: float  # compass-like yaw, degrees
    pitch_deg: float  # positive = below horizon (floor)
    frame_ref: str = ""
    ts: float = 0.0
    kind: str = "floor"  # floor | ceiling


def floor_point(cam_h_m: float, o: CornerObs) -> Optional[tuple[float, float, float]]:
    """(x, y, distance) of a floor point, or None if geometry is unusable."""
    if o.pitch_deg < 10 or o.pitch_deg > 85:  # near-horizon rays blow up; near-vertical is the user's feet
        return None
    d = cam_h_m / math.tan(math.radians(o.pitch_deg))
    psi = math.radians(o.heading_deg)
    return d * math.sin(psi), d * math.cos(psi), d


def polygon_area(pts: Sequence[tuple[float, float]]) -> float:
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def _order_ccw(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]


def _is_simple(pts: np.ndarray) -> bool:
    n = len(pts)
    def ccw(a, b, c): return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    def inter(a, b, c, d): return ccw(a,c,d) != ccw(b,c,d) and ccw(a,b,c) != ccw(a,b,d)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) in (1, n - 1):
                continue
            if inter(pts[i], pts[(i+1) % n], pts[j], pts[(j+1) % n]):
                return False
    return True


def estimate_room(
    cam_height_m: Optional[float],
    floor_obs: Sequence[CornerObs],
    ceiling_obs: Sequence[tuple[CornerObs, CornerObs]] = (),  # (floor, ceiling) of same corner
    shelved_units: Sequence[tuple[float, float]] = (),  # (width_m, height_m) of calibrated shelving units
) -> dict:
    out = {
        "length_m": None, "width_m": None, "height_m": None, "floor_area_m2": None, "wall_area_m2": None,
        "shelved_wall_area_m2": None, "scale_method": "", "confidence": None, "shape": "",
        "assumptions": [], "evidence_frames": [], "status": "unavailable", "notes": [],
    }
    if shelved_units:
        out["shelved_wall_area_m2"] = round(sum(w * h for w, h in shelved_units), 2)
    if not cam_height_m or cam_height_m <= 0:
        out["notes"].append("camera height above floor not provided: floor ranging impossible")
        return out
    pts, frames, dists = [], [], []
    for o in floor_obs:
        p = floor_point(cam_height_m, o)
        if p is None:
            out["notes"].append(f"corner at pitch {o.pitch_deg:.0f}° rejected (outside 10–85° usable range)")
            continue
        pts.append((p[0], p[1])); dists.append(p[2]); frames.append(o.frame_ref)
    out["evidence_frames"] = [f for f in frames if f]
    if len(pts) < 3:
        out["notes"].append(f"{len(pts)} usable floor corners; need >=3")
        return out
    arr = _order_ccw(np.array(pts))
    if not _is_simple(arr):
        out["notes"].append("corner polygon self-intersects: marks are inconsistent")
        return out
    area = polygon_area([tuple(p) for p in arr])
    (_, _), (rw, rh), _ = cv2.minAreaRect(arr.astype(np.float32))
    length, width = max(rw, rh), min(rw, rh)
    rect_fill = area / max(1e-6, length * width)
    out.update(
        floor_area_m2=round(area, 2), length_m=round(length, 2), width_m=round(width, 2),
        shape="rectangular" if len(pts) == 4 and rect_fill > 0.9 else "non-rectangular",
        scale_method="floor-plane ranging (camera height + device pitch + heading)", status="partial",
    )
    out["assumptions"] = [
        "Floor is level and planar; camera height constant while marking corners.",
        "Marked points are floor/wall junctions; heading from device compass (drift not corrected).",
        f"Pitch noise assumed ±{PITCH_SIGMA_DEG}° and propagated into confidence.",
        "length/width are the minimum-area bounding rectangle; floor area is the exact polygon area.",
    ]
    # error propagation: dd/d = dθ / (sinθ cosθ)
    rel = [math.radians(PITCH_SIGMA_DEG) / (math.sin(math.radians(o.pitch_deg)) * math.cos(math.radians(o.pitch_deg)))
           for o in floor_obs if floor_point(cam_height_m, o)]
    rel_err = float(np.mean(rel)) if rel else 1.0
    conf = max(0.1, min(0.85, 1.0 - 2 * rel_err)) * min(1.0, len(pts) / 4)
    # ceiling height
    heights = []
    for fo, co in ceiling_obs:
        p = floor_point(cam_height_m, fo)
        if p is None or not (5 <= co.pitch_deg * -1 <= 80 or 5 <= -co.pitch_deg <= 80):
            continue
        up = -co.pitch_deg if co.pitch_deg < 0 else 0
        if up <= 5:
            continue
        heights.append(cam_height_m + p[2] * math.tan(math.radians(up)))
        out["evidence_frames"] += [f for f in (fo.frame_ref, co.frame_ref) if f]
    if heights:
        out["height_m"] = round(float(np.median(heights)), 2)
        perim = sum(math.dist(arr[i], arr[(i + 1) % len(arr)]) for i in range(len(arr)))
        out["wall_area_m2"] = round(perim * out["height_m"], 2)
        out["status"] = "measured"
        out["assumptions"].append("Wall area is gross (openings not subtracted) = perimeter × ceiling height.")
    else:
        out["notes"].append("ceiling height unavailable: no (floor corner, ceiling corner) pair captured")
    out["confidence"] = round(conf, 2)
    return out
