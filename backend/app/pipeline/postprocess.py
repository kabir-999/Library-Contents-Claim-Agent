"""Post-sweep processing: independent async stages -> claim_packet.json + PDF.

Stage graph (stages in one row run concurrently):
   [books: identify -> duplicates -> measure -> price] [items: classify -> price] [room]
                     \\---------------- review + totals + packet + pdf ----------------/
Hard deadline (default 270 s of the 300 s budget). A stage that overruns is cut and
recorded as 'timeout'; the packet is still produced, with the gap in the review queue.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Optional

import cv2

from ..claim import calculations as calc
from ..claim.pdf import write_pdf
from ..config import get_settings
from ..llm.providers import router
from ..metrics import COUNTS, Timings, estimate_cost
from ..schemas import ClaimPacket, Room, Sweep, now_iso
from ..vision.shelves import assign_shelves
from . import books_stage, items_stage, measure_stage

log = logging.getLogger(__name__)
BUDGET_S = 270.0


async def _stage(timings: Timings, name: str, coro: Awaitable, notify, budget: float):
    await notify(name, "running")
    t = time.time()
    try:
        res = await asyncio.wait_for(coro, budget)
        timings.mark(name, time.time() - t, "ok"); await notify(name, "done"); return res
    except asyncio.TimeoutError:
        timings.mark(name, time.time() - t, "timeout"); await notify(name, "timeout"); log.error("stage %s timed out", name)
    except Exception as e:
        timings.mark(name, time.time() - t, "error"); await notify(name, f"error: {e}"); log.exception("stage %s failed", name)
    return None


def save_crops(session, books, items):
    sd = session.store.sweep_dir(session.id, "crops")
    cache: dict[str, any] = {}
    for obj in [*books, *items]:
        if not obj.bbox or not obj.frame_ref:
            continue
        fr = session.store.get_frame(obj.frame_ref)
        if not fr:
            continue
        img = cache.get(fr["path"])
        if img is None:
            img = cache[fr["path"]] = cv2.imread(fr["path"])
        if img is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        pad = 8
        crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        if crop.size:
            cv2.imwrite(str(sd / f"{obj.id}.jpg"), crop)


async def run(session, notify: Optional[Callable[[str, str], Awaitable[None]]] = None) -> ClaimPacket:
    t_start = time.time(); deadline = t_start + BUDGET_S
    timings = Timings()
    sid = session.id

    async def _n(stage, status):
        await session.emit("processing", stage=stage, status=status)
        if notify:
            await notify(stage, status)

    session.finalizing = True   # end-of-sweep pass: larger model allowance, per-item
    await session.stop_worker()
    session.ended_at = session.ended_at or time.time()
    if session.batch_mode and not session.books.tracks and not session.objs.tracks:
        from . import batch
        tb = time.time()
        await session.emit("processing", stage="frame_analysis", status="starting")
        async def _p(pct): await session.emit("processing", stage="frame_analysis", status=f"{pct}%")
        st = await batch.replay(session, _p)
        timings.mark("frame_analysis", time.time() - tb, f"ok ({st['kept']} of {st['total']} frames analysed)")
        await session.emit("processing", stage="frame_analysis", status="ok")
    session.snapshot_tracks()
    sem_net = asyncio.Semaphore(6)
    left = lambda: max(5.0, deadline - time.time())

    book_tracks = session.books.confirmed()
    obj_tracks = session.objs.confirmed()
    state: dict = {"books": [], "items": [], "measures": {}, "shelf_of": {}}

    async def books_group():
        await _stage(timings, "book_identification", books_stage.finish_identification(session, book_tracks, deadline), _n, left())
        books_stage.detect_duplicates(session, book_tracks); books_stage.clean_books(book_tracks)   # fold dups/fragments FIRST: less to rescue
        budget = min(deadline, time.time() + 14.0)       # rescue + scene read are bounded: Stop -> packet must stay quick
        await asyncio.gather(books_stage.rescue_pass(session, book_tracks, budget), books_stage.scene_pass(session, book_tracks, budget))
        merged = books_stage.detect_duplicates(session, book_tracks)  # sync, ms
        merged += [None] * books_stage.clean_books(book_tracks)
        timings.mark("duplicate_detection", 0.0, f"ok ({len(merged)} merged)")
        live = [t for t in book_tracks if not t.removed]
        state["shelf_of"] = assign_shelves(live)
        books_stage.flag_occlusion(live, state["shelf_of"])
        t = time.time()
        try:
            state["measures"] = measure_stage.measure_books(session, live, state["shelf_of"])
            timings.mark("measurement_validation", time.time() - t, "ok")
        except Exception:
            log.exception("measurement stage failed; metric values left blank")
            state["measures"] = {}
            timings.mark("measurement_validation", time.time() - t, "error")
        state["books"] = books_stage.build_books(session, live, state["measures"], state["shelf_of"])
        await _stage(timings, "book_pricing", books_stage.price_books(session, state["books"], sem_net), _n, left())

    async def items_group():
        async def safe(t):
            try:
                return await items_stage.classify_item(session, t)
            except Exception:
                log.exception("classification failed for %s", t.tid)
                return None
        books_stage.clean_items(obj_tracks, book_tracks)
        its = await _stage(timings, "nonbook_classification", asyncio.gather(*(safe(t) for t in obj_tracks if not t.removed)), _n, left())
        state["items"] = [i for i in (its or []) if i is not None]
        await _stage(timings, "nonbook_pricing", items_stage.price_items(session, state["items"], sem_net), _n, left())

    await asyncio.gather(books_group(), items_group())

    # room (needs shelf items' calibrated dims)
    shelved = [(i.dimensions_cm.w / 100, i.dimensions_cm.h / 100) for i in state["items"]
               if i.category == "bookshelf" and i.dimensions_cm.w and i.dimensions_cm.h and i.status != "excluded"]
    t = time.time()
    try:
        room_d = measure_stage.room_from_session(session, shelved)
        timings.mark("room_measurement", time.time() - t, "ok")
    except Exception:
        log.exception("room stage failed; room left unavailable")
        room_d = {"notes": ["room measurement failed with an internal error"], "assumptions": []}
        timings.mark("room_measurement", time.time() - t, "error")
    room = Room(**{k: v for k, v in room_d.items() if k in Room.model_fields})

    # packet
    sweep_row = session.store.get_sweep(sid)
    packet = ClaimPacket(
        sweep=Sweep(id=sid, captured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(session.t0)), device=session.device,
                    duration_s=round(session.elapsed, 1), country=session.country, currency=session.currency),
        room=room, books=state["books"], items=state["items"], generated_at=now_iso(),
        calibrations=session.cals.all_dicts(), schema_version="1.0")
    packet.room.assumptions = list(room_d.get("assumptions", [])) + [f"note: {n}" for n in room_d.get("notes", [])]

    # evidence frames: keep referenced frames, then prune the rest
    save_crops(session, packet.books, packet.items)
    session.store.mark_keep(sid, [b.frame_ref for b in packet.books] + [i.frame_ref for i in packet.items] +
                            [c["frame_ref"] for c in packet.calibrations] + packet.room.evidence_frames)
    removed = session.store.prune_frames(sid)
    dangling = calc.validate_frame_refs(packet, session.store.frame_ids(sid))
    if dangling:
        log.error("dangling frame refs: %s", dangling)

    # review + advisory narrative (LLM never touches numbers)
    packet = calc.finalize_packet(packet, get_settings().lca_id_confidence_min)
    shelves = {}
    for b in packet.books:
        shelves.setdefault(b.shelf or "unassigned", {"shelf": b.shelf, "book_count": 0})
        shelves[b.shelf or "unassigned"]["book_count"] += 1
    packet.shelves = [{**v, **packet.totals.per_shelf.get(k, {})} for k, v in shelves.items()]
    packet.evidence_frames = [{"id": f["id"], "t_s": round(f["rel_ts"], 2)} for f in session.store.frames_meta(sid)]

    calc.apply_contract_status(packet)   # after totals/review queue: they need the internal statuses

    gt = round(time.time() - t_start, 2)
    timings.mark("packet_generation_total", gt, "ok" if gt < 300 else "OVER_BUDGET")
    packet.processing = timings.rows
    packet.model_extra["run_stats"] = {
        "frames_received": session.stats["frames"], "frames_blurred": session.stats["blurred"], "frames_glare": session.stats["glare"],
        "frames_dark": session.stats["dark"], "frames_dropped_busy": session.dropped_frames, "frames_pruned": removed,
        "avg_detect_ms": round(sum(session.stats["det_ms"]) / max(1, len(session.stats["det_ms"])), 1),
        "avg_ocr_ms": round(sum(session.stats["ocr_ms"]) / max(1, len(session.stats["ocr_ms"])), 1),
        "llm": router().stats, "api_calls": dict(COUNTS), "recapture_requests": session.recapture_requests,
        "cost": estimate_cost(router().stats["groq_tokens"], COUNTS.get("serpapi_calls", 0)),
        "dangling_frame_refs": dangling, "post_sweep_processing_s": gt,
    }
    export(session, packet)
    session.store.update_sweep(sid, state="complete")
    await _n("packet_generation", "done")
    t_ = packet.totals; cur = packet.sweep.currency
    parts = [f"Your claim packet is ready. I counted {t_.book_count} books, {t_.books_identified} identified and {t_.books_unidentified} I could not read.",
             f"{t_.books_priced_replacement} books have a sourced replacement price, totalling {t_.books_replacement_cost:,.2f} {cur}." if t_.books_priced_replacement else "No book has a sourced price yet, so book totals are zero.",
             f"{len(packet.items)} other items found." if packet.items else "",
             f"Floor area {packet.room.floor_area_m2} square metres." if packet.room.floor_area_m2 is not None else "Room dimensions were not measurable from this sweep.",
             f"{len([r for r in packet.review_queue if not r.resolved])} entries need your review."]
    await session.emit("agent_say", text=" ".join(p for p in parts if p), speech_kind="reply", speak=True, final=True, audio_wav_b64=None)
    await session.emit("packet_ready", sweep_id=sid, seconds=gt)
    return packet


def export(session, packet: ClaimPacket) -> None:
    d = session.store.sweep_dir(session.id, "exports")
    (d / "claim_packet.json").write_text(packet.model_dump_json(indent=2, exclude_none=False))
    try:
        write_pdf(packet, d / "claim_report.pdf", session.store.sweep_dir(session.id, "crops"), session.store.sweep_dir(session.id, "frames"))
    except Exception:
        log.exception("PDF generation failed")
    session.store.put(session.id, "packet", "claim", json.loads(packet.model_dump_json()))
