"""Book stages: OCR+identification completion, duplicate detection, pricing."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import cv2

from ..claim.calculations import book_excluded
from ..config import get_settings
from ..pricing import book_pricing
from ..schemas import Book, Measurement, ReplacementCost, UsedValue
from ..vision.shelves import assign_shelves
from ..vision.tracker import Track, iou

log = logging.getLogger(__name__)


async def finish_identification(session, tracks: list[Track], deadline: float) -> None:
    """Run OCR/identify on tracks the live worker has not reached (post-sweep, bounded by deadline)."""
    import time
    todo = [t for t in tracks if not t.ocr_obs and not t.meta.get("excluded")]
    for t in todo:
        if time.time() > deadline:
            t.meta["ident_timeout"] = True
            continue
        try:
            await session.process_track(t.tid)
        except Exception:
            log.exception("identification failed for %s; left unidentified", t.tid)
            t.meta["ident_error"] = True


def detect_duplicates(session, tracks: list[Track]) -> list[tuple[str, str]]:
    """Two tracks with the same identified title that occupy the *same place* in the mosaic are one physical book
    split by the tracker: merge (keep the better-evidenced one). Same title in *different* places = two real copies, kept."""
    merged = []
    ids = [(t, (t.meta.get("ident_fields") or {})) for t in tracks if not t.removed]
    for i, (a, fa) in enumerate(ids):
        for b, fb in ids[i + 1:]:
            if a.removed or b.removed or a.segment != b.segment:
                continue
            ov = iou(a.mbox, b.mbox)
            same_title = fa.get("title") and fa.get("title") == fb.get("title")
            unreadable_pair = not fa.get("title") and not fb.get("title")
            if ov > 0.45 or (same_title and ov > 0.15):
                keep, drop = (a, b) if a.best_score >= b.best_score else (b, a)
                drop.removed = True
                keep.flags.add("possible_duplicate"); keep.hits += drop.hits
                merged.append((keep.tid, drop.tid))
    seen: dict[str, Track] = {}
    for t, f in ids:
        title = f.get("title")
        if t.removed or not title:
            continue
        if title in seen and seen[title].segment != t.segment:
            seen[title].flags.add("possible_duplicate"); t.flags.add("possible_duplicate")
        seen.setdefault(title, t)
    return merged


def flag_occlusion(tracks: list[Track], shelf_of: dict[str, tuple[str, int]]) -> int:
    """A spine whose visible height is far below its shelf's median (same plane, mosaic coords) is probably
    partly hidden by an object/hand/shelf lip, not a genuinely short book: flag for review, never fix silently."""
    import statistics
    by: dict[str, list[Track]] = {}
    for t in tracks:
        if not t.removed and t.orientation == "upright":
            by.setdefault(shelf_of.get(t.tid, ("?", 0))[0], []).append(t)
    n = 0
    for ts in by.values():
        if len(ts) < 5:
            continue
        med = statistics.median(t.mbox[3] - t.mbox[1] for t in ts)
        for t in ts:
            if (t.mbox[3] - t.mbox[1]) < 0.6 * med:
                t.flags.add("occluded_or_clipped"); n += 1
    return n


def build_books(session, tracks: list[Track], measures: dict[str, dict], shelf_of: dict[str, tuple[str, int]]) -> list[Book]:
    books: list[Book] = []
    for t in sorted(tracks, key=lambda t: (shelf_of.get(t.tid, ("~", 0))[0], shelf_of.get(t.tid, ("~", 0))[1])):
        if t.removed:
            continue
        f = t.meta.get("ident_fields") or {}
        shelf, pos = shelf_of.get(t.tid, ("", 0))
        m = measures.get(t.tid, {})
        h, th = m.get("h", Measurement()), m.get("t", Measurement())
        flags = list(f.get("flags", [])) + list(m.get("flags", [])) + [x.replace("occluded_or_clipped", "occluded") for x in t.flags]
        ocr_text = [o["text"] for o in t.ocr_obs if o["text"]]
        status = f.get("status", "unidentified")
        if status == "partial":
            status = "unidentified" if not ocr_text else "partial"
        b = Book(id=t.tid, shelf=shelf, position=pos, frame_ref=t.best_frame_id, status=status, title=f.get("title", ""),
                 author=f.get("author", ""), edition=f.get("edition", ""), isbn=f.get("isbn", ""), publisher=f.get("publisher", ""),
                 spine_height_cm=h.value, spine_thickness_cm=th.value, id_confidence=f.get("id_confidence", 0.0),
                 orientation=t.orientation, ocr_text=ocr_text, bbox=t.best_bbox, height_measurement=h, thickness_measurement=th,
                 flags=sorted(set(flags)), candidates=f.get("candidates", []), verification=f.get("verification", {}), track_id=t.tid)
        if status in ("identified", "needs_review"):
            b.status = status
        if t.meta.get("first_edition"):
            b.edition = "first edition (stated by owner)"; b.flags.append("user_first_edition")
        if t.meta.get("excluded"):
            b.status = "excluded"; b.owner_note = t.meta.get("owner_note", "excluded by owner"); b.flags.append("user_excluded")
        b._group_isbns = f.get("group_isbns", [])
        books.append(b)
    return books


async def price_books(session, books: list[Book], sem: asyncio.Semaphore) -> None:
    s = get_settings()

    async def one(b: Book):
        try:
            await _one(b)
        except Exception:
            log.exception("pricing failed for %s; left blank", b.id)
            b.pricing_note = "Pricing failed (error); left blank."

    async def _one(b: Book):
        if b.status != "identified" or book_excluded(b):
            return
        reason = book_pricing.appraisal_reason(b.flags, " ".join(b.ocr_text), [y for _, y in b._group_isbns],
                                               "user_first_edition" in b.flags)
        if reason:
            b.needs_appraisal = True; b.status = "needs_appraisal"; b.pricing_note = f"Not priced: {reason}."; return
        async with sem:
            r = await book_pricing.price_book(title=b.title, author=b.author, isbn=b.isbn, group_isbns=b._group_isbns,
                                              country=session.country, currency=session.currency, allow_foreign_fallback=True,
                                              home_market="US" if session.country != "US" else "GB")
        b.replacement_cost, b.used_value, b.priced_isbn, b.pricing_note = r["replacement"], r["used"], r["priced_isbn"], r["note"]
        if r["over_threshold"]:
            b.needs_appraisal = True; b.status = "needs_appraisal"
            b.pricing_note = (b.pricing_note + f" Unusually valuable (> {s.lca_appraisal_price_threshold:g} {session.currency}): human appraisal required; not included in totals.").strip()
    await asyncio.gather(*(one(b) for b in books))
