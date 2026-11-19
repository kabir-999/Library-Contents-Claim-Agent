"""Measurement validation: turn calibrations + best frames into per-book/-item cm values and the room estimate.
Separated from identification/pricing so each can be tested and run independently."""
from __future__ import annotations

import logging
import statistics
from typing import Optional

import cv2
import numpy as np

from ..measure.calibration import PlaneCalibration
from ..measure.room import CornerObs, estimate_room
from ..measure.spine import measure_spine
from ..schemas import Measurement
from ..vision.motion import features, homography_between

log = logging.getLogger(__name__)
PLANE_CATEGORIES = {"bookshelf", "cabinet", "artwork", "portrait", "mirror"}  # faces roughly in the calibrated wall plane


class Registrar:
    """Homography from any saved frame into a calibration frame (direct ORB match, else tracked chain)."""

    def __init__(self, session):
        self.s = session
        self._feat: dict[str, dict] = {}

    def _f(self, fid: str):
        if fid in self._feat:
            return self._feat[fid]
        fr = self.s.store.get_frame(fid)
        img = cv2.imread(fr["path"]) if fr else None
        self._feat[fid] = features(img) if img is not None else None
        return self._feat[fid]

    def h_to(self, frame_id: str, cal: PlaneCalibration) -> tuple[Optional[np.ndarray], str]:
        if frame_id == cal.frame_ref:
            return np.eye(3), "same_frame"
        fb, fa = self._f(frame_id), self._f(cal.frame_ref)
        if fb and fa:
            r = homography_between(fb, fa)
            if r:
                return r[0], "direct"
        Hb, Ha = self.s.frame_H.get(frame_id), self.s.frame_H.get(cal.frame_ref)
        if Hb is not None and Ha is not None and self.s.frame_seg.get(frame_id) == self.s.frame_seg.get(cal.frame_ref):
            return np.linalg.inv(Ha) @ Hb, "chain"
        return None, "unregistered"


def measure_books(session, tracks, shelf_of: dict[str, tuple[str, int]]) -> dict[str, dict]:
    reg = Registrar(session)
    out: dict[str, dict] = {}
    for t in tracks:
        cal = session.cals.best_for(f"W{t.segment + 1}")
        scoped = cal is not None and cal.scope == f"W{t.segment + 1}"
        if not t.best_bbox or not t.best_frame_size or cal is None or not scoped:
            out[t.tid] = {"h": Measurement(), "t": Measurement(), "flags": ["no_calibration"]}
            continue
        H, how = reg.h_to(t.best_frame_id, cal)
        if H is None:
            out[t.tid] = {"h": Measurement(), "t": Measurement(), "flags": ["unregistered_to_calibration"]}
            continue
        h, th, fl = measure_spine(t.best_bbox, t.orientation, cal, t.best_frame_id, t.best_frame_size, min(1.0, t.conf), h_to_cal=H)
        if how == "chain":  # accumulated drift: discount, say so
            for m in (h, th):
                if m.value is not None:
                    m.confidence = round((m.confidence or 0) * 0.7, 3); m.method = f"{m.method}+tracked_chain"
        out[t.tid] = {"h": h, "t": th, "flags": fl}
    # outlier check vs shelf median (flag, never silently drop)
    by_shelf: dict[str, list[float]] = {}
    for tid, v in out.items():
        sh = shelf_of.get(tid, ("?", 0))[0]
        if v["h"].value is not None:
            by_shelf.setdefault(sh, []).append(v["h"].value)
    for tid, v in out.items():
        sh = shelf_of.get(tid, ("?", 0))[0]
        vals = by_shelf.get(sh, [])
        if v["h"].value is not None and len(vals) >= 5:
            med = statistics.median(vals)
            if abs(v["h"].value - med) / med > 0.45:
                v["flags"].append("height_outlier_vs_shelf")
    return out


def measure_item(session, track, category: str):
    cal = session.cals.best_for(f"W{track.segment + 1}")
    if category not in PLANE_CATEGORIES or cal is None or cal.scope != f"W{track.segment + 1}" or not track.best_bbox:
        return None
    reg = Registrar(session)
    H, how = reg.h_to(track.best_frame_id, cal)
    if H is None:
        return None
    x1, y1, x2, y2 = track.best_bbox
    W, Hh = track.best_frame_size
    if x1 < 3 or y1 < 3 or x2 > W - 3 or y2 > Hh - 3:
        return None
    from ..vision.motion import map_points
    pts = map_points(H, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    tl, tr, br, bl = cal.to_cm(pts)
    w_cm = float((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2); h_cm = float((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)
    conf = round(cal.confidence * 0.6 * (0.7 if how == "chain" else 1.0), 3)  # detector box edges are approximate
    return {"w": round(w_cm, 0), "h": round(h_cm, 0), "method": f"{cal.method} (plane of detected face){'+tracked_chain' if how == 'chain' else ''}", "confidence": conf,
            "calibration_ref": cal.id}


def room_from_session(session, shelved_units: list[tuple[float, float]]) -> dict:
    cam_h = session.camera_height_cm / 100 if session.camera_height_cm else None
    pairs = []
    for c in session.ceil_obs:
        best = min(session.floor_obs, key=lambda f: min(abs(f.heading_deg - c.heading_deg), 360 - abs(f.heading_deg - c.heading_deg)), default=None)
        if best is not None:
            d = min(abs(best.heading_deg - c.heading_deg), 360 - abs(best.heading_deg - c.heading_deg))
            if d < 15:
                pairs.append((best, c))
    return estimate_room(cam_h, session.floor_obs, pairs, shelved_units)
