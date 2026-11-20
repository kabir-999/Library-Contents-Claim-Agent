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
    sig: Optional[object] = None      # appearance signature (HSV histogram) for re-identifying a close-up book


class Tracker:
    def __init__(self, prefix: str, min_hits: int = 2, iou_thr: float = 0.30, max_age_s: float = 120.0):
        self.prefix, self.min_hits, self.iou_thr, self.max_age = prefix, min_hits, iou_thr, max_age_s
        self.strong_conf = 0.35   # a single confident detection counts; weak ones need a second sighting
        self.tracks: dict[str, Track] = {}
        self._n = 0

    def _match(self, det_mbox, segment, kind, label, cat: str = "") -> Optional[Track]:
        best, best_s = None, 0.0
        for t in self.tracks.values():
            if t.removed or t.segment != segment:
                continue
            if self.prefix != "book" and cat and t.meta.get("cat") and t.meta["cat"] != cat:
                continue   # "lamp" vs "floor lamp" are the same claim category: still the same object
            s = iou(t.mbox, det_mbox)
            if self.prefix != "book":   # a detection mostly inside a tracked object (or vice versa) is that object
                ix = max(0.0, min(t.mbox[2], det_mbox[2]) - max(t.mbox[0], det_mbox[0])); iy = max(0.0, min(t.mbox[3], det_mbox[3]) - max(t.mbox[1], det_mbox[1]))
                small = max(1.0, min((t.mbox[2] - t.mbox[0]) * (t.mbox[3] - t.mbox[1]), (det_mbox[2] - det_mbox[0]) * (det_mbox[3] - det_mbox[1])))
                if ix * iy / small > 0.7:
                    s = max(s, self.iou_thr + 0.01)
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
               frame_size: tuple, quality: float, ts: float, img=None) -> list[Track]:
        """dets: [{bbox:[x1,y1,x2,y2], conf, label, orientation, flags}]; H_to_ref maps frame px -> segment reference px."""
        from .motion import map_box
        touched = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            m = map_box(H_to_ref, d["bbox"])
            label = d.get("label", "")
            cat = d.get("category", "")
            t = self._match(m, segment, self.prefix, label, cat)
            # evidence score: detector conf x frame quality x not clipped
            W, H = frame_size
            clipped = x1 < 3 or y1 < 3 or x2 > W - 3 or y2 > H - 3
            area = (x2 - x1) * (y2 - y1) / (W * H)
            score = d.get("conf", 0.5) * (0.3 + quality) * (0.12 if clipped else 1.0) * (0.5 + min(area * 20, 1.0))   # a book cut by the frame edge is a poor sighting
            sig = _signature(img, d["bbox"]) if (img is not None and (self.prefix != "book" or area > 0.04)) else None
            if t is None and sig is not None:
                t = self._revive(sig, (x2 - x1) / max(1.0, y2 - y1), ts, touched, cat)
                if t is not None:
                    t.mbox = list(m); t.segment = segment
            if t is None:
                self._n += 1
                t = Track(tid=f"{self.prefix}_{self._n:04d}", kind=self.prefix, segment=segment, mbox=m, label=label,
                          first_ts=ts, last_ts=ts, orientation=d.get("orientation", "upright"))
                if cat: t.meta["cat"] = cat
                self.tracks[t.tid] = t
            else:
                a = 0.7  # running mean keeps box stable
                t.mbox = [a * o + (1 - a) * n for o, n in zip(t.mbox, m)]
                t.hits += 1; t.last_ts = ts
            if label:
                t.label_votes[label] = t.label_votes.get(label, 0) + 1
                t.label = max(t.label_votes, key=t.label_votes.get)
            if d.get("from_text"):
                t.meta["band"] = True        # one lying spine found from its text line: reads must never include neighbours
            if sig is not None:
                t.sig = sig; t.meta["closeup"] = True
            t.conf = max(t.conf, d.get("conf", 0.0))
            t.orientation = d.get("orientation", t.orientation)
            if clipped:
                t.flags.add("occluded_or_clipped")
            cs = t.meta.setdefault("cands", [])   # top-3 distinct sightings: (score, frame_id, bbox, frame_size)
            if not any(c[1] == frame_id for c in cs) and (len(cs) < 6 or score > min(c[0] for c in cs)):
                cs.append((score, frame_id, list(d["bbox"]), frame_size)); cs.sort(key=lambda c: -c[0]); del cs[6:]
            if score > t.best_score:
                t.best_score, t.best_frame_id, t.best_bbox, t.best_frame_size = score, frame_id, list(d["bbox"]), frame_size
            touched.append(t)
        return touched

    def _revive(self, sig, aspect, ts, touched, cat: str = ""):
        """A large (close-up) book that left the view and came back is the same book: match by appearance, not position."""
        import cv2
        item = self.prefix != "book"
        best, best_d = None, 0.30 if item else 0.25
        for t in self.tracks.values():
            if t.removed or t.sig is None or t in touched or (not item and ts - t.last_ts < 0.8):
                continue
            if item and (t.meta.get("cat") != cat or ts - t.last_ts > 600):
                continue
            ta = (t.best_bbox[2] - t.best_bbox[0]) / max(1.0, t.best_bbox[3] - t.best_bbox[1]) if t.best_bbox else aspect
            if abs(ta - aspect) / max(ta, aspect) > 0.3:
                continue
            dist = cv2.compareHist(t.sig, sig, cv2.HISTCMP_BHATTACHARYYA)
            if dist < best_d:
                best, best_d = t, dist
        return best

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks.values() if not t.removed and (t.hits >= self.min_hits or t.conf >= self.strong_conf) and not t.meta.get("not_book") and (self.prefix != "book" or t.meta.get("gate_ok", True))]

    def all_active(self) -> list[Track]:
        return [t for t in self.tracks.values() if not t.removed]


def _signature(img, bbox):
    import cv2
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = img[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(cv2.resize(crop, (64, 96)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).astype("float32")
