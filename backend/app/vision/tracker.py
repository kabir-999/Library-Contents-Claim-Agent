"""Multi-frame track association with duplicate prevention.

Detections are expressed in mosaic coordinates (frame px + camera offset, see
motion.py). A detection updates an existing track if it overlaps it enough
(IoU) or, for thin spines where IoU is brittle, if centres agree along the shelf
and heights overlap. Otherwise it spawns a tentative track. A track becomes a
*confirmed* object after `min_hits` observations (suppresses one-frame ghosts).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    tid: str
    kind: str                 # "book" | object label
    segment: int
    mbox: list                # mosaic-coordinate box (running mean)
    label: str = ""
    hits: int = 1
    first_ts: float = 0.0
    last_ts: float = 0.0
    best_frame_id: str = ""
    best_score: float = -1.0
    best_bbox: Optional[list] = None   # in best frame pixels
    best_frame_size: Optional[tuple] = None
    conf: float = 0.0
    orientation: str = "upright"
    flags: set = field(default_factory=set)
    ocr_obs: list = field(default_factory=list)  # list of dicts {text, conf, score}
    label_votes: dict = field(default_factory=dict)
    ocr_score_done: float = -1.0
    removed: bool = False
    meta: dict = field(default_factory=dict)


class Tracker:
    def __init__(self, prefix: str, min_hits: int = 2, iou_thr: float = 0.30, max_age_s: float = 120.0):
        self.prefix, self.min_hits, self.iou_thr, self.max_age = prefix, min_hits, iou_thr, max_age_s
        self.tracks: dict[str, Track] = {}
        self._n = 0

    def _match(self, det_mbox, segment, kind, label) -> Optional[Track]:
        best, best_s = None, 0.0
        for t in self.tracks.values():
            if t.removed or t.segment != segment:
                continue
            if self.prefix != "book" and t.label and label and t.label != label and t.label_votes.get(label, 0) < 2:
                continue
            s = iou(t.mbox, det_mbox)
            if s < self.iou_thr:
                # spine fallback: centres within half a spine width, heights agree
                tcx, dcx = (t.mbox[0] + t.mbox[2]) / 2, (det_mbox[0] + det_mbox[2]) / 2
                tw, dw = t.mbox[2] - t.mbox[0], det_mbox[2] - det_mbox[0]
                vy = min(t.mbox[3], det_mbox[3]) - max(t.mbox[1], det_mbox[1])
                vh = max(1.0, min(t.mbox[3] - t.mbox[1], det_mbox[3] - det_mbox[1]))
                if self.prefix == "book" and abs(tcx - dcx) < 0.45 * max(tw, dw) and vy / vh > 0.7:
                    s = 0.31
            if s > best_s and s >= self.iou_thr:
                best, best_s = t, s
        return best

    def update(self, dets: list[dict], segment: int, H_to_ref, frame_id: str,
               frame_size: tuple, quality: float, ts: float) -> list[Track]:
        """dets: [{bbox:[x1,y1,x2,y2], conf, label, orientation, flags}]; H_to_ref maps frame px -> segment reference px."""
        from .motion import map_box
        touched = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            m = map_box(H_to_ref, d["bbox"])
            label = d.get("label", "")
            t = self._match(m, segment, self.prefix, label)
            # evidence score: detector conf x frame quality x not clipped
            W, H = frame_size
            clipped = x1 < 3 or y1 < 3 or x2 > W - 3 or y2 > H - 3
            area = (x2 - x1) * (y2 - y1) / (W * H)
            score = d.get("conf", 0.5) * (0.3 + quality) * (0.5 if clipped else 1.0) * (0.5 + min(area * 20, 1.0))
            if t is None:
                self._n += 1
                t = Track(tid=f"{self.prefix}_{self._n:04d}", kind=self.prefix, segment=segment, mbox=m, label=label,
                          first_ts=ts, last_ts=ts, orientation=d.get("orientation", "upright"))
                self.tracks[t.tid] = t
            else:
                a = 0.7  # running mean keeps box stable
                t.mbox = [a * o + (1 - a) * n for o, n in zip(t.mbox, m)]
                t.hits += 1; t.last_ts = ts
            if label:
                t.label_votes[label] = t.label_votes.get(label, 0) + 1
                t.label = max(t.label_votes, key=t.label_votes.get)
            t.conf = max(t.conf, d.get("conf", 0.0))
            t.orientation = d.get("orientation", t.orientation)
            if clipped:
                t.flags.add("occluded_or_clipped")
            if score > t.best_score:
                t.best_score, t.best_frame_id, t.best_bbox, t.best_frame_size = score, frame_id, list(d["bbox"]), frame_size
            touched.append(t)
        return touched

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks.values() if not t.removed and t.hits >= self.min_hits]

    def all_active(self) -> list[Track]:
        return [t for t in self.tracks.values() if not t.removed]
