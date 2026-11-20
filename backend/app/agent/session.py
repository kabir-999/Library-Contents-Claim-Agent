"""SweepSession: all live state for one continuous camera sweep.

Hot path (per frame): decode -> quality -> save evidence frame -> register to
mosaic (ORB homography) -> YOLO book + object detection (thread) -> tracker ->
guidance. Slow work (OCR, metadata lookups) goes through a background queue so
the camera loop never blocks. Final identification/pricing happens in
pipeline/postprocess.py after the sweep ends.
"""
from __future__ import annotations

import asyncio
import re
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
from ..vision.tracker import Tracker, iou
from ..vision import gate
from ..vision.book_detector import orientation_of
from .guidance import GuidanceState, evaluate

log = logging.getLogger(__name__)
Emit = Callable[[dict], Awaitable[None]]


def _cover_px(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1.0, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


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
        self.batch_mode = not get_settings().lca_live_processing
        self.persist_frames = True            # False for the live PREVIEW session: it must not add frames to the recording
        self.frame_prefix = f"{sweep_id}_f"
        self.mem_frames: dict[str, np.ndarray] = {}
        self.deferred: list[dict] = []     # voice/UI corrections made while recording; applied to what was on screen at that time
        self.last_frame_ts = 0.0
        self._bad_run, self._last_tip = 0, 0.0

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
        frame_id = f"{self.frame_prefix}{self.frame_n:05d}"
        qual = q.analyze(img)
        self.last_q = qual.to_dict()
        self.last_orientation = orientation
        self.stats["frames"] += 1
        self.stats["blurred"] += int(qual.blurred); self.stats["glare"] += int(qual.glare); self.stats["dark"] += int(qual.too_dark)
        rel = ts - self.t0
        if self.persist_frames:
            await asyncio.to_thread(self.store.save_frame, self.id, frame_id, jpeg, ts, rel, W_img, H_img, qual.to_dict(), orientation)
        else:                                   # preview: keep a few frames in memory so OCR can read the best one
            self.mem_frames[frame_id] = img
            for k in list(self.mem_frames)[:-40]:
                self.mem_frames.pop(k, None)

        seg, H, new_seg, feat = self.motion.update(img)
        self.frame_H[frame_id] = H; self.frame_seg[frame_id] = seg
        if new_seg:
            await self.emit("event", text=f"New wall/segment started (W{seg + 1})")

        t1 = time.time()
        bdets, odets = await asyncio.to_thread(self._detect, img)
        if not qual.blurred and time.time() - getattr(self, "_disc_last", 0) > 1.5:
            self._disc_last = time.time()          # text detector proposes spine/cover regions (finds 2nd/3rd books too)
            from ..identify import fast
            for bb in await asyncio.to_thread(fast.discover, img):
                if any(_cover_px(bb, d["bbox"]) > 0.5 for d in bdets):
                    continue                        # already covered by a normal detection
                bdets.append({"bbox": bb, "conf": 0.35, "label": "book", "orientation": orientation_of(bb), "from_text": True})
        self.stats["det_ms"].append((time.time() - t1) * 1000)

        # auto reference-object calibration (A4 sheet visible on the shelf)
        if seg not in {int(c.scope[1:]) - 1 for c in self.cals.items if c.scope.startswith("W") and c.scope[1:].isdigit()}:
            cal = calmod.from_reference(f"cal_{len(self.cals.items) + 1}", img, frame_id, scope=f"W{seg + 1}")
            if cal:
                self.cals.add(cal); self.frame_feat[frame_id] = feat
                self.store.mark_keep(self.id, [frame_id])
                await self.emit("event", text=f"Scale established on W{seg + 1} from a visible {cal.reference} sheet.")

        tq = qual.score
        bt = self.books.update(bdets, seg, H, frame_id, (W_img, H_img), tq, ts, img=img)
        pairs = await self._gate(img, list(zip(bt, bdets)))      # stage 1: is it a book at all? everything below sees only books
        bt, bdets = [p[0] for p in pairs], [p[1] for p in pairs]
        ot = self.objs.update(odets, seg, H, frame_id, (W_img, H_img), tq, ts, img=img)
        # an "item" sitting on a confirmed book is that book (e.g. a cover read as electronics/picture): drop it for good
        for t, d in zip(ot, odets):
            if any(_cover_px(d["bbox"], b["bbox"]) > 0.5 for b in bdets):
                t.removed = True
        ot = [t for t in ot if not t.removed]
        self.visible = [{"kind": "book", "tid": t.tid, "bbox": t.best_bbox if t.best_frame_id == frame_id else None} for t in bt] + \
                       [{"kind": "item", "tid": t.tid, "bbox": t.best_bbox if t.best_frame_id == frame_id else None} for t in ot]
        # keep the best frames as evidence
        for t in bt + ot:
            if t.best_frame_id == frame_id:
                self.store.mark_keep(self.id, [frame_id])
                if t.kind == "book" and (t.hits >= 2 or t.conf >= self.books.strong_conf) and t.best_score > 1.3 * t.ocr_score_done:
                    t.ocr_score_done = t.best_score
                    self.ocr_q.put_nowait(t.tid)

        for t in ot:
            if t.hits >= 3 and not t.meta.get("live_price_started"):
                t.meta["live_price_started"] = True
                asyncio.create_task(self._live_price_item(t))
        if not qual.blurred:
            self._scene_trigger(img, bt)
            self._fast_trigger(img, bt, bdets)
        for t in bt:
            if any(c[1] == frame_id for c in t.meta.get("cands", [])):
                self.store.mark_keep(self.id, [frame_id])
        await self._emit_found()
        await self._ask_artwork(ot)
        info = await self._guidance(qual, bdets, W_img, H_img, bt)
        await self._state(force=False)
        self.timeline.append({"frame": frame_id, "t": round(rel, 2), "books": len(self.books.confirmed()), "q": qual.score})
        return info

    # ---------------- batch mode: record now, analyse everything after Stop ----------------
    async def record_frame(self, jpeg: bytes, ts: Optional[float] = None, orientation: Optional[dict] = None) -> Optional[dict]:
        """Sweep phase: store the frame and do ONLY cheap capture-quality checks. No detection, no models."""
        ts = ts or time.time()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if max(img.shape[:2]) > 1920:
            f = 1920 / max(img.shape[:2]); img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        H_img, W_img = img.shape[:2]
        self.frame_n += 1
        frame_id = f"{self.id}_f{self.frame_n:05d}"
        qual = q.analyze(img)
        self.last_q, self.last_orientation, self.last_frame_ts = qual.to_dict(), orientation, ts
        self.stats["frames"] += 1
        self.stats["blurred"] += int(qual.blurred); self.stats["glare"] += int(qual.glare); self.stats["dark"] += int(qual.too_dark)
        await asyncio.to_thread(self.store.save_frame, self.id, frame_id, jpeg, ts, ts - self.t0, W_img, H_img, qual.to_dict(), orientation)
        self._bad_run = self._bad_run + 1 if (qual.blurred or qual.too_dark or qual.glare) else 0
        if self._bad_run >= 4 and time.time() - self._last_tip > 8:
            self._last_tip, self._bad_run = time.time(), 0
            text = "The picture is blurry. Slow down." if qual.blurred else "It is too dark. Turn on a light." if qual.too_dark else "There is glare. Tilt the camera a little."
            await self.emit("guidance", code="capture_quality", text=text)
            return {"text": text}
        if self.frame_n % 6 == 0 and self.emit_cb:
            try:
                await self.emit_cb({"type": "recording", "frames": self.frame_n, "elapsed_s": round(self.elapsed, 1)})
            except Exception:
                pass
        return None

    async def replay_frame(self, frame_id: str, img: np.ndarray, ts: float, orientation: Optional[dict], qual, bdets: list, odets: list) -> None:
        """Analysis of one stored frame with PRECOMPUTED detections (offline). Same tracking logic as live, no live extras."""
        H_img, W_img = img.shape[:2]
        self.last_orientation = orientation
        seg, H, new_seg, feat = self.motion.update(img)
        self.frame_H[frame_id] = H; self.frame_seg[frame_id] = seg
        if seg not in {int(c.scope[1:]) - 1 for c in self.cals.items if c.scope.startswith("W") and c.scope[1:].isdigit()}:
            cal = calmod.from_reference(f"cal_{len(self.cals.items) + 1}", img, frame_id, scope=f"W{seg + 1}")
            if cal:
                self.cals.add(cal); self.frame_feat[frame_id] = feat
                self.store.mark_keep(self.id, [frame_id])
        tq = qual.score
        bt = self.books.update(bdets, seg, H, frame_id, (W_img, H_img), tq, ts, img=img)
        pairs = await self._gate(img, list(zip(bt, bdets)))
        bt, bdets = [p[0] for p in pairs], [p[1] for p in pairs]
        ot = self.objs.update(odets, seg, H, frame_id, (W_img, H_img), tq, ts, img=img)
        for t, d in zip(ot, odets):
            if any(_cover_px(d["bbox"], b["bbox"]) > 0.5 for b in bdets):
                t.removed = True
        ot = [t for t in ot if not t.removed]
        self.visible = [{"kind": "book", "tid": t.tid, "bbox": None} for t in bt] + [{"kind": "item", "tid": t.tid, "bbox": None} for t in ot]
        for t in bt + ot:
            if t.best_frame_id == frame_id or any(c[1] == frame_id for c in t.meta.get("cands", [])):
                self.store.mark_keep(self.id, [frame_id])
        while self.deferred and self.deferred[0]["ts"] <= ts:      # corrections the user spoke at this moment in the sweep
            await self.apply_deferred(self.deferred.pop(0))
        self.timeline.append({"frame": frame_id, "t": round(ts - self.t0, 2), "books": len(self.books.confirmed()), "q": tq})

    async def apply_deferred(self, c: dict) -> None:
        a = c["action"]
        if a == "set_first_edition":
            for t in self.focus_tracks(c.get("target") == "group"): t.meta["first_edition"] = True
        elif a == "set_not_mine":
            for t in self.focus_tracks(c.get("target") == "group") or self.focus_tracks(False, "item"):
                t.meta["excluded"] = True; t.meta["owner_note"] = "owner states it belongs to someone else"
        elif a == "skip_shelf":
            for t in self.focus_tracks(True): t.meta["excluded"] = True; t.meta["owner_note"] = "shelf skipped by owner"
        elif a == "set_artwork_type":
            for t in self.focus_tracks(False, "item"): t.meta["user_note"] = c.get("value") or ""
        elif a == "known_shelf_dimension" and c.get("number"):
            self.calibrate_known_dimension(c["number"], c.get("value") or "height")
        elif a in ("mark_corner", "mark_ceiling"):
            self.last_orientation = c.get("orientation") or self.last_orientation
            self.add_floor_mark(a == "mark_ceiling")

    def _detect(self, img: np.ndarray):
        b = self.detector.detect(img) if self.detector.available else []
        o = self.objdet.detect(img) if self.objdet.available else []
        # nothing is both a non-book item and a book spine
        def on_book(d):   # a "laptop"/"picture" box that is really a book cover (or a book inside it) is not an item
            a = d["bbox"]
            for k in b:
                c = k["bbox"]
                ix = max(0.0, min(a[2], c[2]) - max(a[0], c[0])); iy = max(0.0, min(a[3], c[3]) - max(a[1], c[1]))
                if ix * iy / max(1.0, min((a[2] - a[0]) * (a[3] - a[1]), (c[2] - c[0]) * (c[3] - c[1]))) > 0.6:
                    return True
            return False
        return b, [d for d in o if d.get("label") != "book" and not on_book(d)]

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

    def found(self) -> dict:
        """What the user sees: one line per DISTINCT named book / item (not per tracker box)."""
        from rapidfuzz import fuzz
        books: list[dict] = []
        for t in self.books.tracks.values():
            if t.removed or not self._is_book(t) or t.meta.get("excluded"):
                continue
            f, ver = t.meta.get("ident_fields") or {}, self._verified(t)
            names = [f.get("title")] if ver else str(t.meta.get("live_name") or "").split(" + ")
            for nm in [n for n in names if n]:
                rec = {"name": nm, "author": (f.get("author") or "") if ver else "", "price": t.meta.get("live_price", ""), "ok": ver}
                norm = lambda x: re.sub(r"[^a-z0-9 ]", "", x.lower()).strip()
                dup = next((b for b in books if fuzz.token_set_ratio(norm(b["name"]), norm(nm)) >= 85), None)
                if dup is None:
                    books.append(rec)
                elif ver and not dup["ok"]:
                    dup.update(rec)
        items: dict[str, dict] = {}
        gated = [b for b in self.books.tracks.values() if not b.removed and self._is_book(b)]
        for t in self.objs.confirmed():
            if any(b.segment == t.segment and _cover_px(t.mbox, b.mbox) > 0.5 for b in gated):
                continue
            if not t.meta.get("excluded"):
                cat = t.meta.get("cat") or t.label
                if cat not in items or (t.meta.get("live_price") and not items[cat]["price"]):
                    items[cat] = {"name": cat, "price": t.meta.get("live_price", "")}
        return {"books": books, "items": list(items.values())}

    async def _emit_found(self) -> None:
        if not self.emit_cb:
            return
        f = self.found(); sig = json.dumps(f, sort_keys=True)
        if sig != getattr(self, "_found_sig", None):
            self._found_sig = sig
            try:
                await self.emit_cb({"type": "found", **f})
            except Exception:
                pass

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
        if t.best_frame_id in self.mem_frames:
            return self.mem_frames[t.best_frame_id]
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
        if getattr(self, "finalizing", False):
            await self._multiscale_reads(t, img)   # final pass: zoom-out context + zoom-in views of the same sighting
        from ..identify import fast
        if fast.available():   # second, independent reader (fast, good on clean print); its lines are pooled with Paddle's
            x1, y1, x2, y2 = [int(v) for v in t.best_bbox]
            tl = await asyncio.to_thread(fast.read_lines, img[max(0, y1):y2, max(0, x1):x2])
            if tl:
                t.ocr_obs.append({"frame": t.best_frame_id, "text": " ".join(tl), "conf": 0.7, "lines": [], "rotation": "tesseract"})
        if not r.get("available", True):
            t.meta["ocr_unavailable"] = True
        best = max(t.ocr_obs, key=lambda o: o["conf"] * len(o["text"]))
        # pool words from every sighting of this book: one frame may miss a word another caught
        seen = set(best["text"].lower().split()); extra = []
        for o in t.ocr_obs:
            for w in o["text"].split():
                if w.lower() not in seen and len(w) >= 3:
                    seen.add(w.lower()); extra.append(w)
        pooled = (best["text"] + " " + " ".join(extra[:12])).strip()
        best = {**best, "text": pooled}
        all_raw = " ".join(o["text"] for o in t.ocr_obs if o["text"])
        d = None
        if t.meta.get("band") and len([o for o in t.ocr_obs if o["text"]]) >= 2:
            d = await self._vote_identify(t)          # each read votes on its own: a neighbour's stray words are outvoted, not pooled in
            if d is not None and d["status"] == "identified":
                await self._accept(t, d); return
        hi = [" ".join(l["text"] for l in o.get("lines", []) if l.get("conf", 0) >= 0.75) for o in t.ocr_obs]   # confident lines only: drops a neighbour's leftover words
        for tx in dict.fromkeys(x for x in (ident.consensus_text([o["text"] for o in t.ocr_obs]), ident.consensus_text([h for h in hi if h]), *[h for h in hi if h][:2], best["text"].split(" ", 0)[0] and best["text"], pooled) if x):
            di = await ident.identify(tx, best["conf"], use_llm=False, evidence=(tx if t.meta.get("band") else all_raw))    # query = clean text; scoring = every raw read
            if d is None or (di["status"] == "identified" and d["status"] != "identified") or (d["status"] != "identified" and di["confidence"] > d["confidence"]):
                d = di
            if d["status"] == "identified":
                break
        stable = len(t.ocr_obs) >= 2 or t.hits >= 8 or getattr(self, "finalizing", False)
        if best["text"] and (d is None or d["status"] != "identified") and stable and not t.meta.get("llm_tried") and self._llm_ok("text"):
            t.meta["llm_tried"] = True          # text LLM only splits noisy OCR; result is still verified by the matcher
            d = await ident.identify(best["text"], best["conf"], use_llm=True)
        if (d is None or d["status"] != "identified") and stable and not t.meta.get("vlm_tried") and self._llm_ok("vlm"):
            t.meta["vlm_tried"] = True
            vt = await self._vlm_read(img, t.best_bbox, best["text"])
            if vt:
                d2 = await ident.identify(vt, max(best["conf"], 0.6), use_llm=False)
                if d is None or d2["confidence"] > d["confidence"]:
                    d = d2; t.meta["vlm_read"] = vt
        if d is not None:
            await self._accept(t, d)

    async def rescue_track(self, t, deadline: float) -> None:
        """End-of-sweep retry for a book that stayed unverified: re-read its saved sightings, each re-cropped with the
        text detector (so a cut-off spine is read in full), pooling the words across sightings."""
        from ..identify import fast
        keep = (t.best_frame_id, t.best_bbox, t.best_frame_size)
        try:
            for score, fid, bbox, fsize in list(t.meta.get("cands", []))[:6]:
                if time.time() > deadline or self._verified(t):
                    break
                fr = self.store.get_frame(fid)
                img = await asyncio.to_thread(cv2.imread, fr["path"]) if fr else None
                if img is None:
                    continue
                boxes = await asyncio.to_thread(fast.discover, img)
                near = [b for b in boxes if _cover_px(b, bbox) > 0.3]
                if near and not t.meta.get("band"):   # union of the detector box and the text-derived spine box(es)
                    bbox = [min([bbox[0]] + [b[0] for b in near]), min([bbox[1]] + [b[1] for b in near]),
                            max([bbox[2]] + [b[2] for b in near]), max([bbox[3]] + [b[3] for b in near])]
                t.best_frame_id, t.best_bbox, t.best_frame_size = fid, bbox, fsize
                await self.process_track(t.tid)
        finally:
            if not self._verified(t):
                t.best_frame_id, t.best_bbox, t.best_frame_size = keep

    async def _vote_identify(self, t) -> Optional[dict]:
        """Identify a lying spine from up to 4 INDEPENDENT reads and let them vote on the work. A read that contains a neighbour's
        words votes for the wrong book once; the correct reads agree with each other."""
        from ..identify.matcher import work_key
        seen, items = set(), []
        for o in sorted((o for o in t.ocr_obs if o.get("text")), key=lambda o: -(o["conf"] * len(o["text"]))):
            tx = " ".join(l["text"] for l in o.get("lines", []) if l.get("conf", 0) >= 0.7) or o["text"]
            if len(tx) >= 6 and tx.lower() not in seen:
                seen.add(tx.lower()); items.append((tx, o))
            if len(items) >= 3:
                break
        res = await asyncio.gather(*[ident.identify(tx, o["conf"], use_llm=False, evidence=tx) for tx, o in items], return_exceptions=True)
        votes: dict[str, dict] = {}
        for d in res:
            if isinstance(d, dict) and d.get("chosen") and d["status"] in ("identified", "needs_review") and d["confidence"] >= 0.6:
                k = work_key(d["chosen"].cand.title)
                v = votes.setdefault(k, {"w": 0.0, "n": 0, "d": d})
                v["w"] += d["confidence"] * (1.0 if d["status"] == "identified" else 0.6); v["n"] += 1
                if d["confidence"] > v["d"]["confidence"]:
                    v["d"] = d
        if not votes:
            return None
        v = max(votes.values(), key=lambda v: v["w"])
        return v["d"] if (v["n"] >= 2 or v["d"]["status"] == "identified") else None

    async def _multiscale_reads(self, t, img) -> None:
        """Zoom out (30% context: a cut-off spine's missing part is often just outside the box) and zoom in (x1.8, sharpened)."""
        from ..identify import fast
        H, W = img.shape[:2]; x1, y1, x2, y2 = [int(v) for v in t.best_bbox]
        w, h = x2 - x1, y2 - y1
        band = bool(t.meta.get("band"))   # a band is one lying spine: no zoom-out (it would pull in the neighbouring books' text)
        k = 0.0 if band else 0.3
        ctx = [max(0, x1 - int(k * w)), max(0, y1 - int(k * h)), min(W, x2 + int(k * w)), min(H, y2 + int(k * h))]
        r2 = await asyncio.to_thread(ocr.read_spine, img, ctx, t.orientation) if not band else {"text": ""}
        if r2["text"]:
            t.ocr_obs.append({"frame": t.best_frame_id, "text": r2["text"], "conf": r2["conf"], "lines": r2["lines"], "rotation": "zoom_out"})
        if fast.available():
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                z = cv2.resize(crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
                z = cv2.addWeighted(z, 1.5, cv2.GaussianBlur(z, (0, 0), 3), -0.5, 0)       # unsharp mask
                zl = await asyncio.to_thread(fast.read_lines, z)
                zc = await asyncio.to_thread(fast.read_lines, img[ctx[1]:ctx[3], ctx[0]:ctx[2]])
                for name, lines in (("zoom_in", zl), ("zoom_out_tess", zc)):
                    if lines:
                        t.ocr_obs.append({"frame": t.best_frame_id, "text": " ".join(lines), "conf": 0.7, "lines": [], "rotation": name})

    async def _accept(self, t, d: dict) -> None:
        prev = t.meta.get("ident")
        if not prev or d["confidence"] >= prev.get("confidence", 0) or d["status"] == "identified":
            t.meta["ident"] = {k: v for k, v in d.items() if k not in ("chosen", "group", "runner_group")}
            t.meta["ident_fields"] = ident.apply_to_book_fields(d)
        if d["status"] == "identified":
            f = t.meta["ident_fields"]
            await self.emit("event", text=f"Identified: {f['title']} — {f['author']} ({f['id_confidence']:.2f})")
            self._add_runner_up(t, d)
            self._merge_same_book(t)
            if not t.removed and not t.meta.get("live_price_final"):
                t.meta["live_price_final"] = True
                t.meta["live_price"] = t.meta.get("live_price") or "pricing…"
                asyncio.create_task(self._live_price_book(t, f, d))
            await self._narrate(f["title"])

    # ---- scene reader: a vision model lists every book it can see; each is verified against the catalogue ----
    def _verified(self, t) -> bool:
        return (t.meta.get("ident") or {}).get("status") == "identified"

    async def scene_read(self, img, tracks: list) -> int:
        """Read ALL books visible in one frame (handles tilted/overlapping spines OCR can't separate).
        Returns the number of newly verified books. Never invents: each title must be corroborated by our own OCR
        text and confirmed by Google Books/Open Library via the matcher."""
        import re as _re
        if time.time() < getattr(self, "_scene_block_until", 0):
            return 0
        H, W = img.shape[:2]
        if max(H, W) > 1280:
            f = 1280 / max(H, W); img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        from ..llm.providers import router
        r = await router().groq_vision(jpg, 'List EVERY book visible in this image (spines or covers, possibly tilted, overlapping or held in a hand), '
            'ordered left to right. For each, transcribe only the title and author text that is actually printed and legible. '
            'Do not complete or guess from memory; use null for unreadable parts. '
            'JSON: {"books":[{"title": str|null, "author": str|null}]}')
        books = r if isinstance(r, list) else (r.get("books") if isinstance(r, dict) else None)
        if not isinstance(books, list):
            self._scene_block_until = time.time() + 20   # rate-limited or failed: back off
            return 0
        r = {"books": [b for b in books if isinstance(b, dict)]}
        pool = " ".join(o["text"] for t in tracks for o in t.ocr_obs).lower()
        tok = lambda x: [w for w in _re.findall(r"[a-z0-9]+", x.lower()) if len(w) >= 3]
        from ..identify.matcher import work_key
        have = {work_key((o.meta.get("ident_fields") or {}).get("title", "")) for o in self.books.tracks.values() if not o.removed and self._verified(o)}
        todo = sorted([t for t in tracks if not self._verified(t)], key=lambda t: (t.mbox[0] + t.mbox[2]) / 2)
        found = 0
        for b in r["books"][:8]:
            title = str((b or {}).get("title") or "").strip()
            if len(title) < 3:
                continue
            tw = tok(title)
            ptoks = tok(pool)
            hit = sum(any(w == p or w.startswith(p) or p.startswith(w) for p in ptoks) for w in tw)
            if tw and pool and hit / len(tw) < 0.34 and not any(a in pool for a in tok(str(b.get("author") or ""))):
                continue   # nothing in our own OCR supports this reading
            d = await ident.identify(f'{title} {b.get("author") or ""}'.strip(), 0.7, use_llm=False)
            if d["status"] != "identified":
                continue
            f = ident.apply_to_book_fields(d)
            if work_key(f.get("title", "")) in have:
                continue
            host = todo.pop(0) if todo else None
            if host is None:
                base = max(tracks, key=lambda t: t.best_score)
                import copy
                self.books._n += 1
                host = copy.copy(base); host.tid = f"book_{self.books._n:04d}"; host.meta = {"ghost": True}
                host.ocr_obs = list(base.ocr_obs); host.flags = set(base.flags) | {"shares_crop_with_another_book"}
                self.books.tracks[host.tid] = host
            host.meta["vlm_scene"] = True
            have.add(work_key(f["title"])); found += 1
            await self._accept(host, d)
        return found

    def _llm_ok(self, kind: str) -> bool:
        """Large models are for reasoning/verification, never per-frame. Live: max 6 calls/min and 30/sweep; the
        end-of-sweep pass (self.finalizing) gets its own larger allowance."""
        now = time.time()
        log_ = self.__dict__.setdefault("_llm_log", [])
        self._llm_log = [t for t in log_ if now - t < 60]
        if not getattr(self, "finalizing", False) and (len(self._llm_log) >= 6 or self.stats.get("llm_calls", 0) >= 30):
            return False
        self._llm_log.append(now); self.stats["llm_calls"] = self.stats.get("llm_calls", 0) + 1
        return True

    @staticmethod
    def _is_book(t) -> bool:
        return bool(t.meta.get("gate_ok"))

    async def _gate(self, img, pairs: list) -> list:
        """CLIP zero-shot book/not-book per track (cached on the track; re-checked once with a second sighting).
        Without CLIP: a detection must be confident or carry readable text."""
        if not get_settings().lca_book_gate:
            for t, d in pairs:
                t.meta.update(gate_ok=True, gate=1.0)
            return pairs
        st = gate.status()
        if st == "loading":
            gate.warm()
            return []   # classifier still loading (a few seconds): nothing counts as a book yet
        todo = [(t, d) for t, d in pairs if not t.meta.get("not_book") and (t.meta.get("gate_n", 0) == 0 or (t.meta.get("gate_n") == 1 and t.hits >= 5))]
        # too small to ever read or price reliably: not a candidate
        small = [(t, d) for t, d in todo if min(d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]) < 14]
        for t, d in small:
            t.meta.update(not_book=True, gate_ok=False, gate=0.0)
        todo = [x for x in todo if x not in small]
        for t, d in [(t, d) for t, d in todo if d.get("stack")]:     # an aligned stack of text bands IS a stack of books: no CLIP needed
            votes = t.meta.setdefault("gate_votes", []); votes.append(0.75)
            t.meta.update(gate_n=len(votes), gate=0.75, gate_ok=True, not_book=False)
        todo = [x for x in todo if not x[1].get("stack")]
        if todo:
            boxes = [d["bbox"] for _, d in todo]
            probs = await asyncio.to_thread(gate.book_probs_multiscale, img, boxes)
            crops = [img[max(0, int(b[1])):int(b[3]), max(0, int(b[0])):int(b[2])].copy() for b in boxes]
            from ..identify import fast
            if probs is None:   # CLIP failed to load at all: strict fallback (confident detector AND readable text)
                probs = []
                for (t, d), c in zip(todo, crops):
                    txt = fast.available() and bool(await asyncio.to_thread(fast.read_lines, c))
                    probs.append(1.0 if (d.get("conf", 0) >= 0.3 and txt) else 0.0)
            elif fast.available():
                for i, ((t, d), pb) in enumerate(zip(todo, probs)):
                    # borderline for CLIP (e.g. a thin white spine): readable title-like text settles it
                    if 0.08 <= pb < gate.THRESHOLD and await asyncio.to_thread(fast.read_lines, crops[i]):
                        probs[i] = 0.55
            for (t, d), pb in zip(todo, probs):
                votes = t.meta.setdefault("gate_votes", []); votes.append(pb)
                t.meta["gate_n"] = len(votes); t.meta["gate"] = round(sum(votes) / len(votes), 3)
                t.meta["gate_ok"] = t.meta["gate"] >= gate.THRESHOLD
                t.meta["not_book"] = not t.meta["gate_ok"]
        return [(t, d) for t, d in pairs if t.meta.get("gate_ok")]

    def _fast_trigger(self, img, bt, bdets) -> None:
        """Instant provisional label: Tesseract on the current frame's crop (no disk, no queue)."""
        from ..identify import fast
        if not fast.available():
            return
        for t, d in zip(bt, bdets):
            if t.meta.get("fast_state") or self._verified(t) or not (t.hits >= 2 or t.conf >= self.books.strong_conf):
                continue
            t.meta["fast_state"] = "running"
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            crop = img[max(0, y1):y2, max(0, x1):x2].copy()
            asyncio.create_task(self._fast_one(t, crop))

    async def _fast_one(self, t, crop) -> None:
        from ..identify import fast
        try:
            lines = await asyncio.to_thread(fast.read_lines, crop)
            if not lines:
                t.meta["fast_state"] = None if t.meta.get("fast_tries", 0) < 3 else "gave_up"
                t.meta["fast_tries"] = t.meta.get("fast_tries", 0) + 1
                return
            t.meta["live_name"] = " + ".join(l for l in lines[:2])
            t.meta["fast_state"] = "named"
            res = await asyncio.gather(*[fast.quick_price(l, self.country, self.currency) for l in lines[:2]], return_exceptions=True)
            ok = [r for r in res if isinstance(r, dict)]
            if ok and not t.meta.get("live_price_final"):
                t.meta["live_price"] = "~ " + " + ".join(f"{self.currency} {r['amount']:,.0f}" for r in ok)
        except Exception:
            log.debug("fast path failed", exc_info=True)
            t.meta["fast_state"] = "gave_up"

    def _scene_trigger(self, img, bt) -> None:
        # selective: only books that stayed in view (hits>=8), were OCR'd at least twice and are still unverified
        pend = [t for t in bt if not self._verified(t) and t.hits >= 8 and len(t.ocr_obs) >= 2 and not t.meta.get("scene_tried")]
        if not pend or getattr(self, "_scene_busy", False) or time.time() - getattr(self, "_scene_last", 0) < 10.0 or not self._llm_ok("scene"):
            return
        for t in pend:
            t.meta["scene_tried"] = True
        self._scene_busy, self._scene_last = True, time.time()
        async def run():
            try:
                await self.scene_read(img.copy(), pend)
            except Exception:
                log.exception("scene read failed")
            finally:
                self._scene_busy = False
        asyncio.create_task(run())

    def _label(self, t, cat: str = "") -> dict:
        f = t.meta.get("ident_fields") or {}
        if t.kind == "book":
            title = f.get("title") if (t.meta.get("ident") or {}).get("status") == "identified" else None
            prov = t.meta.get("live_name")
            return {"name": title or (("~ " + prov) if prov else ("reading…" if t.hits >= 1 else "")), "price": t.meta.get("live_price", ""), "ok": bool(title)}
        return {"name": cat or t.label, "price": t.meta.get("live_price", ""), "ok": True}

    async def _live_price_item(self, t):
        from ..pricing import nonbook_pricing as nbp
        from ..vision.object_detector import CATEGORY
        cat = CATEGORY.get(t.label)
        if not cat or cat in ("artwork", "portrait"):
            return
        try:
            r = await nbp.price_item(cat, "", t.label, "", self.country, self.currency)
            if r["priced"]:
                c = r["cost"]; t.meta["live_price"] = f"{self.currency} {c.low:,.0f}–{c.high:,.0f}"
        except Exception:
            log.debug("live item price failed", exc_info=True)

    async def _live_price_book(self, t, f: dict, d: dict):
        from ..pricing import book_pricing
        try:
            g = f.get("group_isbns", [])
            r = await book_pricing.price_book(title=f["title"], author=f["author"], isbn=f.get("isbn", ""), group_isbns=g,
                                              country=self.country, currency=self.currency, allow_foreign_fallback=True,
                                              home_market="US" if self.country != "US" else "GB")
            if r["over_threshold"]:
                t.meta["live_price"] = "needs appraisal"
            elif r["replacement"].amount:
                t.meta["live_price"] = f"{self.currency} {r['replacement'].amount:,.0f}"
            else:
                t.meta["live_price"] = "no price found"
        except Exception:
            log.debug("live book price failed", exc_info=True)

    def _add_runner_up(self, t, d: dict) -> None:
        """A crop that holds two books (tilted/overlapping spines) yields a second verified entry."""
        rg = d.get("runner_group")
        if not rg or t.meta.get("runner_added"):
            return
        t.meta["runner_added"] = True
        d2 = {**d, "chosen": rg[0], "group": rg, "confidence": rg[0].score, "status": "identified", "runner_group": None}
        f2 = ident.apply_to_book_fields(d2)
        from ..identify.matcher import work_key
        if not f2.get("title") or any(work_key((o.meta.get("ident_fields") or {}).get("title", "")) == work_key(f2["title"]) for o in self.books.tracks.values() if not o.removed and self._verified(o)):
            return
        import copy
        self.books._n += 1
        g = copy.copy(t); g.tid = f"book_{self.books._n:04d}"; g.meta = {"ghost": True, "ident": {k: v for k, v in d2.items() if k not in ("chosen", "group", "runner_group")},
                                                                      "ident_fields": f2, "runner_added": True}
        g.ocr_obs = list(t.ocr_obs); g.flags = set(t.flags) | {"shares_crop_with_another_book"}
        self.books.tracks[g.tid] = g

    def _merge_same_book(self, t) -> None:
        """A book that left the view and came back is one book: fold this track into the earlier one with the
        same verified title, unless both are on screen right now (then they are genuinely two copies)."""
        from ..identify.matcher import work_key
        title = work_key((t.meta.get("ident_fields") or {}).get("title", ""))
        if not title:
            return
        vis = {v["tid"] for v in self.visible if v["kind"] == "book"}
        for o in self.books.tracks.values():
            if o is t or o.removed or work_key((o.meta.get("ident_fields") or {}).get("title", "")) != title:
                continue
            if o.tid in vis and t.tid in vis and iou(o.mbox, t.mbox) < 0.3:
                continue   # two copies on screen at once, in different places
            keep, drop = (o, t) if o.first_ts <= t.first_ts else (t, o)
            drop.removed = True; keep.hits += drop.hits
            if drop.best_score > keep.best_score:
                keep.best_score, keep.best_frame_id, keep.best_bbox, keep.best_frame_size = drop.best_score, drop.best_frame_id, drop.best_bbox, drop.best_frame_size
            keep.meta.update({k: v for k, v in drop.meta.items() if k not in keep.meta})
            return

    async def _vlm_read(self, img, bbox, paddle_text: str) -> str:
        """Second reader (Groq VLM) for covers/spines Paddle garbles. It must only TRANSCRIBE; we accept its text
        only when most of its title words are corroborated by Paddle's own (noisy) output, so it cannot inject a
        title from memory. The result is still verified against Metadata APIs by the matcher."""
        import re as _re
        H, W = img.shape[:2]; x1, y1, x2, y2 = [int(v) for v in bbox]
        px, py = int(0.05 * (x2 - x1)), int(0.05 * (y2 - y1))
        crop = img[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]
        if crop.size == 0:
            return ""
        if max(crop.shape[:2]) > 768:
            f = 768 / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        from ..llm.providers import router
        r = await router().groq_vision(jpg, 'Transcribe ONLY the text printed on this book cover or spine (it may be rotated). '
            'Do not use outside knowledge or complete titles. JSON: {"title": str|null, "author": str|null}.')
        if not isinstance(r, dict) or not r.get("title"):
            return ""
        tok = lambda x: [w for w in _re.findall(r"[a-z0-9]+", x.lower()) if len(w) >= 3]
        pad = tok(paddle_text or "")
        def hit(w): return any(w == p or (len(p) >= 3 and (w.startswith(p) or p.startswith(w))) for p in pad)
        tw = tok(str(r["title"]))
        if not tw or sum(hit(w) for w in tw) / len(tw) < 0.34:
            return ""
        return f'{r["title"]} {r.get("author") or ""}'.strip()

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
