"""Offline sweep analysis: the sweep only RECORDS; on Stop every stored frame is analysed here, in bulk.

  1. quality + near-duplicate filter on all frames (thread pool)     -> keyframes
  2. detection over lists of keyframes (YOLO batches; books + items concurrently)
  3. text-line discovery for frames where the detectors found nothing confident (spines the boxes miss)
  4. in-order tracking: registration, book/not-book gate, unique track ids (appearance + overlap based), corrections

Every physical object ends up as ONE track with ONE id; later stages (OCR, catalogue, price) run per id, not per frame.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import logging
import time

import cv2
import numpy as np

from ..config import get_settings
from ..vision import gate, quality as q
from ..vision.book_detector import orientation_of

log = logging.getLogger(__name__)
CHUNK = 32
MIN_KEYFRAMES = 12
_POOL = cf.ThreadPoolExecutor(6)


def _prep(row: dict):
    img = cv2.imread(row["path"])
    if img is None:
        return None
    g = cv2.cvtColor(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return q.analyze(img), g


def select_keyframes(rows: list[dict], preps: list) -> list[int]:
    """Drop blurry frames and frames that barely differ from the last kept one (but never leave long gaps)."""
    keep, last_g, since = [], None, 0
    for i, p in enumerate(preps):
        if p is None:
            continue
        qual, g = p
        since += 1
        diff = 99.0 if last_g is None else float(np.abs(g.astype(np.int16) - last_g.astype(np.int16)).mean())
        if (not qual.blurred and (diff > 2.5 or since >= 10)) or since >= 18:
            keep.append(i); last_g, since = g, 0
    if len(rows) <= 2 * MIN_KEYFRAMES and len(keep) < min(MIN_KEYFRAMES, len(rows)):   # genuinely short recording: don't over-filter
        keep = [i for i, p in enumerate(preps) if p is not None]
    floor = min(len(rows), 30)                            # short sweeps: never starve the readers of sightings
    if len(keep) < floor:
        rest = sorted((i for i, p in enumerate(preps) if p is not None and i not in set(keep)), key=lambda i: -preps[i][0].score)
        keep = sorted(keep + rest[:floor - len(keep)])
    cap = get_settings().lca_batch_max_frames
    if len(keep) > cap:                                   # evenly thin to the budget (keeps coverage of the whole sweep)
        keep = [keep[int(j * len(keep) / cap)] for j in range(cap)]
    return keep


async def replay(session, progress=None) -> dict:
    t0 = time.time()
    rows = session.store.frames_full(session.id)
    total = len(rows)
    if not total:
        return {"total": 0, "kept": 0, "seconds": 0.0}
    loop = asyncio.get_running_loop()
    preps = await asyncio.gather(*[loop.run_in_executor(_POOL, _prep, r) for r in rows])
    keep = select_keyframes(rows, preps)
    for _ in range(40):                                    # the book gate (CLIP) loads in the background at server start
        if gate.status() != "loading":
            break
        gate.warm(); await asyncio.sleep(0.5)
    from ..identify import fast
    discovered = 0
    tm = {"prep": round(time.time() - t0, 2), "detect": 0.0, "discover": 0.0, "track": 0.0}
    for c0 in range(0, len(keep), CHUNK):
        idx = keep[c0:c0 + CHUNK]
        imgs = await asyncio.gather(*[loop.run_in_executor(_POOL, lambda i=i: cv2.imread(rows[i]["path"])) for i in idx])
        ok = [(i, im) for i, im in zip(idx, imgs) if im is not None]
        if not ok:
            continue
        ims = [im for _, im in ok]
        td = time.time()
        sel = [k for k in range(len(ims)) if (c0 + k) % 2 == 0]   # large static objects: every 2nd keyframe is plenty
        bdets_all, o_sel = await asyncio.gather(asyncio.to_thread(session.detector.detect_batch, ims),
                                                asyncio.to_thread(session.objdet.detect_batch, [ims[k] for k in sel]))
        odets_all = [[] for _ in ims]
        for k, od in zip(sel, o_sel):
            odets_all[k] = od
        tm["detect"] += time.time() - td
        for (i, im), bd, od in zip(ok, bdets_all, odets_all):
            od = [d for d in od if d.get("label") != "book"]
            # text-line discovery only where the box detectors found nothing confident (bounded: ~180 ms each)
            if discovered < 16 and (c0 + ok.index((i, im))) % 3 == 0 and not any(d["conf"] >= 0.3 for d in bd) and not preps[i][0].blurred:
                discovered += 1; tx = time.time()
                for bb in await asyncio.to_thread(fast.discover, im):
                    if not any(_cover(bb, d["bbox"]) > 0.5 for d in bd):
                        bd.append({"bbox": bb, "conf": 0.35, "label": "book", "orientation": orientation_of(bb), "from_text": True})
                tm["discover"] += time.time() - tx
            r = rows[i]; tt = time.time()
            await session.replay_frame(r["id"], im, r["ts"], r["orient"], preps[i][0], bd, od)
            tm["track"] += time.time() - tt
        if progress:
            await progress(min(99, int(100 * (c0 + len(idx)) / max(1, len(keep)))))
    while session.deferred:                                # corrections spoken after the last analysed frame
        await session.apply_deferred(session.deferred.pop(0))
    st = {"total": total, "kept": len(keep), "seconds": round(time.time() - t0, 1),
          "books_tracked": len(session.books.confirmed()), "items_tracked": len(session.objs.confirmed()),
          "stage_seconds": {k: round(v, 1) for k, v in tm.items()}}
    log.info("batch replay: %s", st)
    return st


def _cover(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1.0, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
