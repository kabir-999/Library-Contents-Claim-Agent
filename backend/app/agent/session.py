"""SweepSession: all live state for one continuous camera sweep.

Hot path (per frame): decode -> quality -> save evidence frame -> register to
mosaic (ORB homography) -> YOLO book + object detection (thread) -> tracker ->
guidance. Slow work (OCR, metadata lookups) goes through a background queue so
the camera loop never blocks. Final identification/pricing happens in
pipeline/postprocess.py after the sweep ends.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any, Awaitable, Callable, Optional

import cv2
import numpy as np

from ..config import get_settings
from ..identify import service as ident
from ..measure import calibration as calmod
from ..measure.room import CornerObs
from ..storage import Storage, storage
from ..vision import ocr, quality as q
from ..vision.book_detector import BookDetector
from ..vision.motion import MotionEstimator, features, map_box
from ..vision.object_detector import ObjectDetector
from ..vision.shelves import assign_shelves, shelf_rows
from ..vision.tracker import Tracker
from .guidance import GuidanceState, evaluate

log = logging.getLogger(__name__)
Emit = Callable[[dict], Awaitable[None]]


class SweepSession:
    def __init__(self, sweep_id: str, country: str, currency: str, device: str, camera_height_cm: Optional[float] = None,
                 emit: Optional[Emit] = None, store: Optional[Storage] = None):
        self.id, self.country, self.currency, self.device = sweep_id, country, currency, device
        self.store = store or storage()
        self.emit_cb = emit
        self.camera_height_cm = camera_height_cm
        self.t0 = time.time()
        self.ended_at: Optional[float] = None
        self.frame_n = 0
        self.dropped_frames = 0
        self.motion = MotionEstimator()
        self.books = Tracker("book", min_hits=2, iou_thr=0.30)
        self.objs = Tracker("item", min_hits=3, iou_thr=0.40)
        self.cals = calmod.CalibrationStore()
        self.floor_obs: list[CornerObs] = []
        self.ceil_obs: list[CornerObs] = []
        self.guid = GuidanceState()
        self.last_q: dict = {}
        self.visible: list[dict] = []            # latest frame: [{kind, tid, bbox}]
        self.frame_H: dict[str, np.ndarray] = {} # frame_id -> H_to_ref
        self.frame_seg: dict[str, int] = {}
        self.frame_feat: dict[str, dict] = {}    # only calibration/evidence frames kept (memory)
        self.last_orientation: Optional[dict] = None
        self.skipped_shelves: set[str] = set()
        self.recapture_requests: list[dict] = []
        self.timeline: list[dict] = []
        self.stats = {"frames": 0, "blurred": 0, "glare": 0, "dark": 0, "det_ms": [], "ocr_ms": []}
        self.detector, self.objdet = BookDetector(), ObjectDetector()
        self.ocr_q: asyncio.Queue = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self.user_speaking = False
        self.last_state_emit = 0.0
        self.pending_ident: set[str] = set()
        self.narrate_cb = None            # set by LiveAgent: async fn(text)
        self._narr_pending: list[str] = []
        self._narr_last = 0.0

    # ---------------- events ----------------
    async def emit(self, kind: str, **body):
        n = self.store.add_event(self.id, kind, body)
        if self.emit_cb:
            try:
                await self.emit_cb({"type": kind, "n": n, **body})
            except Exception:
                log.debug("emit failed", exc_info=True)

    @property
    def elapsed(self) -> float:
        return (self.ended_at or time.time()) - self.t0

    # ---------------- lifecycle ----------------
    def start_worker(self):
        if self._worker is None:
            self._worker = asyncio.create_task(self._ocr_worker())

    async def stop_worker(self, drain_timeout: float = 20.0):
        if self._worker:
            try:
                await asyncio.wait_for(self.ocr_q.join(), drain_timeout)
            except asyncio.TimeoutError:
                log.warning("OCR queue not drained in %.0fs", drain_timeout)
            self._worker.cancel(); self._worker = None

    # ---------------- frame hot path ----------------
    async def on_frame(self, jpeg: bytes, ts: Optional[float] = None, orientation: Optional[dict] = None) -> Optional[dict]:
        ts = ts or time.time()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if max(img.shape[:2]) > 1920:   # bound memory/latency for odd cameras
            f = 1920 / max(img.shape[:2]); img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        H_img, W_img = img.shape[:2]
        self.frame_n += 1
        frame_id = f"{self.id}_f{self.frame_n:05d}"
        qual = q.analyze(img)
        self.last_q = qual.to_dict()
        self.last_orientation = orientation
        self.stats["frames"] += 1
        self.stats["blurred"] += int(qual.blurred); self.stats["glare"] += int(qual.glare); self.stats["dark"] += int(qual.too_dark)
        rel = ts - self.t0
        await asyncio.to_thread(self.store.save_frame, self.id, frame_id, jpeg, ts, rel, W_img, H_img, qual.to_dict(), orientation)

        seg, H, new_seg, feat = self.motion.update(img)
        self.frame_H[frame_id] = H; self.frame_seg[frame_id] = seg
        if new_seg:
            await self.emit("event", text=f"New wall/segment started (W{seg + 1})")

        t1 = time.time()
        bdets, odets = await asyncio.to_thread(self._detect, img)
        self.stats["det_ms"].append((time.time() - t1) * 1000)

        # auto reference-object calibration (A4 sheet visible on the shelf)
        if seg not in {int(c.scope[1:]) - 1 for c in self.cals.items if c.scope.startswith("W") and c.scope[1:].isdigit()}:
            cal = calmod.from_reference(f"cal_{len(self.cals.items) + 1}", img, frame_id, scope=f"W{seg + 1}")
            if cal:
                self.cals.add(cal); self.frame_feat[frame_id] = feat
                self.store.mark_keep(self.id, [frame_id])
                await self.emit("event", text=f"Scale established on W{seg + 1} from a visible {cal.reference} sheet.")

        tq = qual.score
        bt = self.books.update(bdets, seg, H, frame_id, (W_img, H_img), tq, ts)
        ot = self.objs.update(odets, seg, H, frame_id, (W_img, H_img), tq, ts)
        self.visible = [{"kind": "book", "tid": t.tid, "bbox": t.best_bbox if t.best_frame_id == frame_id else None} for t in bt] + \
                       [{"kind": "item", "tid": t.tid, "bbox": t.best_bbox if t.best_frame_id == frame_id else None} for t in ot]
        # keep the best frames as evidence
        for t in bt + ot:
            if t.best_frame_id == frame_id:
                self.store.mark_keep(self.id, [frame_id])
                if t.kind == "book" and t.hits >= 2 and t.best_score > 1.3 * t.ocr_score_done:
                    t.ocr_score_done = t.best_score
                    self.ocr_q.put_nowait(t.tid)

        await self._ask_artwork(ot)
        info = await self._guidance(qual, bdets, W_img, H_img, bt)
        await self._state(force=False)
        self.timeline.append({"frame": frame_id, "t": round(rel, 2), "books": len(self.books.confirmed()), "q": qual.score})
        return info

    def _detect(self, img: np.ndarray):
        b = self.detector.detect(img) if self.detector.available else []
        o = self.objdet.detect(img) if self.objdet.available else []
        # nothing is both a non-book item and a book spine
        return b, [d for d in o if d.get("label") != "book"]

    # ---------------- guidance & state ----------------
    def room_status(self) -> dict:
        return {"need_camera_height": self.camera_height_cm is None,
                "corners_marked": len(self.floor_obs),
                "orientation_available": bool(self.last_orientation and self.last_orientation.get("pitch_down") is not None),
                "ceiling_marked": len(self.ceil_obs)}

    async def _guidance(self, qual, bdets, W, H, bt) -> Optional[dict]:
        unread = [{"shelf": assign_shelves(self.books.confirmed()).get(t.tid, ("?", 0))[0], "track": t.tid, "age_s": time.time() - t.first_ts}
                  for t in self.books.confirmed() if t.ocr_score_done > 0 and not t.ocr_obs and (time.time() - t.first_ts) > 8]
        clipped = sum(1 for t in bt if "occluded_or_clipped" in t.flags)
        focus = ""
        vis_ids = {v["tid"] for v in self.visible if v["kind"] == "book"}
        if vis_ids:
            sh = assign_shelves(self.books.confirmed())
            names = [sh[t][0] for t in vis_ids if t in sh]
            focus = max(set(names), key=names.count) if names else ""
        g = evaluate(self.guid, time.time(), focus_shelf=focus, elapsed_s=self.elapsed, quality=self.last_q,
                     book_boxes=[(d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]) for d in bdets], frame_h=H,
                     clipped_books=clipped, unreadable_tracks=unread, shelf_rows=shelf_rows(self.books.confirmed()),
                     room_status=self.room_status(), speaking_user=self.user_speaking)
        if g:
            if g.recapture:
                self.recapture_requests.append({**g.recapture, "t": round(self.elapsed, 1), "resolved": False})
            await self.emit("guidance", code=g.code, text=g.text)
            return {"code": g.code, "text": g.text}
        return None

    async def _ask_artwork(self, ot):
        from ..vision.object_detector import CATEGORY
        for t in ot:
            if t.hits >= 3 and CATEGORY.get(t.label) in ("artwork", "portrait") and not t.meta.get("asked"):
                t.meta["asked"] = True
                self.pending_artwork = getattr(self, "pending_artwork", []) + [t.tid]
                await self.emit("guidance", code="ask_artwork", text="Is the framed artwork you're showing me an original or a print?")
                return

    def counts(self) -> dict:
        shelves = assign_shelves(self.books.confirmed())
        return {"books": len([t for t in self.books.confirmed() if not t.meta.get("excluded")]),
                "items": len([t for t in self.objs.confirmed() if not t.meta.get("excluded")]),
                "shelves": len({s for s, _ in shelves.values()}), "elapsed_s": round(self.elapsed, 1),
                "identified": len([t for t in self.books.confirmed() if t.meta.get("ident", {}).get("status") == "identified"]),
                "calibrated_segments": len({c.scope for c in self.cals.items}),
                "room": {"floor_corners": len(self.floor_obs), "ceiling_marks": len(self.ceil_obs),
                         "camera_height_cm": self.camera_height_cm},
                "quality": self.last_q.get("issues", []), "dropped_frames": self.dropped_frames}

    async def _state(self, force: bool):
        if force or time.time() - self.last_state_emit > 0.5:
            self.last_state_emit = time.time()
            await self.emit("state", **self.counts())

    # ---------------- OCR + identification worker ----------------
    async def _ocr_worker(self):
        while True:
            tid = await self.ocr_q.get()
            try:
                await self.process_track(tid)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("ocr/ident failed for %s", tid)
            finally:
                self.ocr_q.task_done()

    def _load_best(self, t) -> Optional[np.ndarray]:
        fr = self.store.get_frame(t.best_frame_id)
        if not fr:
            return None
        return cv2.imread(fr["path"])

    async def process_track(self, tid: str, force_llm: bool = False):
        t = self.books.tracks.get(tid)
        if not t or t.removed or not t.best_bbox:
            return
        img = await asyncio.to_thread(self._load_best, t)
        if img is None:
            return
        t1 = time.time()
        r = await asyncio.to_thread(ocr.read_spine, img, t.best_bbox, t.orientation)
        self.stats["ocr_ms"].append((time.time() - t1) * 1000)
        t.ocr_obs.append({"frame": t.best_frame_id, "text": r["text"], "conf": r["conf"], "lines": r["lines"], "rotation": r["rotation"]})
        if not r.get("available", True):
            t.meta["ocr_unavailable"] = True
        best = max(t.ocr_obs, key=lambda o: o["conf"] * len(o["text"]))
        if best["text"]:
            d = await ident.identify(best["text"], best["conf"], use_llm=True)
            prev = t.meta.get("ident")
            if not prev or d["confidence"] >= prev.get("confidence", 0) or d["status"] == "identified":
                t.meta["ident"] = {k: v for k, v in d.items() if k != "chosen" and k != "group"}
                t.meta["ident_fields"] = ident.apply_to_book_fields(d)
            if d["status"] == "identified":
                f = t.meta["ident_fields"]
                await self.emit("event", text=f"Identified: {f['title']} — {f['author']} ({f['id_confidence']:.2f})")
                await self._narrate(f["title"])

    async def _narrate(self, title: str):
        """Say what is being logged, batched so the agent doesn't chatter: at most one line per 15 s."""
        self._narr_pending.append(title)
        if not self.narrate_cb or self.user_speaking or time.time() - self._narr_last < 15:
            return
        n, latest = len(self._narr_pending), self._narr_pending[-1]
        self._narr_pending, self._narr_last = [], time.time()
        short = latest.split(":")[0][:60]
        await self.narrate_cb(f"Logged {short}." if n == 1 else f"Logged {n} more books, the latest is {short}.")

    # ---------------- user actions (voice / UI) ----------------
    def focus_tracks(self, group: bool, kind: str = "book") -> list:
        vis = [v for v in self.visible if v["kind"] == kind]
        trs = [(self.books if kind == "book" else self.objs).tracks[v["tid"]] for v in vis if v["tid"] in (self.books if kind == "book" else self.objs).tracks]
        if not trs:
            return []
        if group:
            shelves = assign_shelves(self.books.confirmed()) if kind == "book" else {}
            if kind == "book" and shelves:
                cnt: dict[str, int] = {}
                for t in trs:
                    s = shelves.get(t.tid, ("?", 0))[0]; cnt[s] = cnt.get(s, 0) + 1
                top = max(cnt, key=cnt.get)
                return [t for t in self.books.confirmed() if shelves.get(t.tid, ("", 0))[0] == top]
            return trs
        # nearest to image centre: use box in best frame if visible now, else most recently hit
        def key(t):
            return -t.last_ts
        cx = None
        bbs = [(t, t.best_bbox) for t in trs if t.best_bbox]
        if bbs and self.frame_n:
            fr = self.store.get_frame(f"{self.id}_f{self.frame_n:05d}")
            if fr:
                cx, cy = fr["w"] / 2, fr["h"] / 2
                return [min(bbs, key=lambda tb: ((tb[1][0] + tb[1][2]) / 2 - cx) ** 2 + ((tb[1][1] + tb[1][3]) / 2 - cy) ** 2)[0]]
        return [sorted(trs, key=key)[0]]

    def current_frame_id(self) -> str:
        return f"{self.id}_f{self.frame_n:05d}"

    def add_floor_mark(self, ceiling: bool = False) -> Optional[str]:
        o = self.last_orientation
        if not o:
            return "no_orientation"
        # The browser computes camera pitch/heading from the W3C Euler angles (frontend/lib/orientation.ts).
        if o.get("pitch_down") is None or o.get("heading") is None:
            return "no_orientation"
        pitch_down, heading = float(o["pitch_down"]), float(o["heading"])
        if not (math.isfinite(pitch_down) and math.isfinite(heading)):
            return "no_orientation"
        obs = CornerObs(heading, -pitch_down if ceiling else pitch_down, self.current_frame_id(), time.time() - self.t0, "ceiling" if ceiling else "floor")
        (self.ceil_obs if ceiling else self.floor_obs).append(obs)
        self.store.mark_keep(self.id, [obs.frame_ref])
        return None

    def calibrate_known_dimension(self, value_cm: float, dim: str) -> tuple[bool, str]:
        """User states the unit's height/width; use the YOLO-World 'bookshelf' box in the current frame (must not be clipped)."""
        cand = [t for t in self.objs.tracks.values() if t.label in ("bookshelf",) and t.best_frame_id == self.current_frame_id() and t.best_bbox]
        if not cand:
            return False, "I can't see a complete bookshelf in view right now."
        t = max(cand, key=lambda t: (t.best_bbox[2] - t.best_bbox[0]) * (t.best_bbox[3] - t.best_bbox[1]))
        x1, y1, x2, y2 = t.best_bbox
        W, H = t.best_frame_size
        if x1 < 5 or y1 < 5 or x2 > W - 5 or y2 > H - 5:
            return False, "The shelf is cut off by the frame. Step back so the whole unit is visible and say it again."
        p1, p2 = ((x1, y1), (x1, y2)) if dim == "height" else ((x1, y1), (x2, y1))
        cal = calmod.from_two_points(f"cal_{len(self.cals.items) + 1}", p1, p2, value_cm, self.current_frame_id(),
                                     scope=f"W{self.motion.segment + 1}", what=f"user-stated shelf {dim} (detector box edge)")
        cal.confidence = min(cal.confidence, 0.5)  # detector box edge is approximate
        self.cals.add(cal); self.frame_feat[self.current_frame_id()] = self.motion.prev
        self.store.mark_keep(self.id, [self.current_frame_id()])
        return True, f"Scale set from the {value_cm:.0f} centimetre shelf {dim}."

    def calibrate_two_point(self, frame_id: str, p1, p2, real_cm: float, scope: Optional[str] = None) -> dict:
        seg = self.frame_seg.get(frame_id)
        scope = scope or (f"W{seg + 1}" if seg is not None else "global")
        cal = calmod.from_two_points(f"cal_{len(self.cals.items) + 1}", p1, p2, real_cm, frame_id, scope=scope)
        self.cals.add(cal)
        if frame_id not in self.frame_feat:
            fr = self.store.get_frame(frame_id)
            if fr:
                self.frame_feat[frame_id] = features(cv2.imread(fr["path"]))
        self.store.mark_keep(self.id, [frame_id])
        return cal.to_dict()

    def snapshot_tracks(self):
        """Persist lightweight track snapshots (for crash recovery / inspection)."""
        rows = {t.tid: {"label": t.label, "hits": t.hits, "segment": t.segment, "mbox": t.mbox, "best_frame": t.best_frame_id,
                        "best_bbox": t.best_bbox, "orientation": t.orientation, "conf": t.conf, "meta": json.loads(json.dumps(t.meta, default=str)),
                        "ocr": t.ocr_obs} for t in [*self.books.all_active(), *self.objs.all_active()]}
        self.store.put_many(self.id, "track", rows)
