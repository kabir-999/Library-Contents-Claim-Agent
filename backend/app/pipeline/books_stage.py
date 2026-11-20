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
    todo = [t for t in tracks if not t.ocr_obs and not t.meta.get("excluded") and not t.removed]
    sem = asyncio.Semaphore(4)

    async def one(t):
        if time.time() > deadline:
            t.meta["ident_timeout"] = True
            return
        async with sem:
            try:
                await asyncio.wait_for(session.process_track(t.tid), timeout=45)   # one slow book cannot hold up the rest
            except asyncio.TimeoutError:
                t.meta["ident_timeout"] = True
            except Exception:
                log.exception("identification failed for %s; left unidentified", t.tid)
                t.meta["ident_error"] = True
    await asyncio.gather(*(one(t) for t in todo))

async def resolve_conflicts(session, tracks: list[Track]) -> int:
    """ONE title = ONE physical book. When several tracks claim the same (or a prefix-related) work, the track whose OWN reads
    support the title best keeps it; the others are re-read with that work excluded, and left unidentified if nothing else fits."""
    from rapidfuzz import fuzz
    from ..identify import service as ident
    from ..identify.matcher import norm, work_key
    live = [t for t in tracks if not t.removed and (t.meta.get("ident") or {}).get("status") == "identified"]
    key = lambda t: work_key((t.meta.get("ident_fields") or {}).get("title", ""))
    related = lambda a, b: a == b or (min(len(a), len(b)) >= 8 and (a.startswith(b) or b.startswith(a)))

    def support(t: Track) -> float:
        f = t.meta.get("ident_fields") or {}
        title, sur = norm(f.get("title", "")), [w for w in norm(f.get("author", "")).split() if len(w) > 3][-1:]
        best = 0.0
        for o in t.ocr_obs:
            tx = norm(o.get("text", ""))
            if tx:
                best = max(best, fuzz.partial_ratio(title, tx) + (15 if sur and sur[0] in tx else 0))
        return best + len(title) * 0.01          # tie-break toward the more specific title

    groups: list[list[Track]] = []
    for t in live:
        for g in groups:
            if any(related(key(t), key(m)) for m in g):
                g.append(t); break
        else:
            groups.append([t])
    fixed = 0
    for g in groups:
        if len(g) < 2:
            continue
        owner = max(g, key=support)
        claimed = {key(t) for t in live if t not in g} | {key(owner)}
        for t in g:
            if t is owner:
                continue
            fixed += 1
            texts = [o["text"] for o in sorted(t.ocr_obs, key=lambda o: -(o["conf"] * len(o["text"]))) if o.get("text")][:3]
            new = None
            for tx in texts:
                d = await ident.identify(tx, 0.7, use_llm=False, evidence=tx, exclude=frozenset(claimed | {key(t)}))
                if d["status"] == "identified" and d.get("chosen") and work_key(d["chosen"].cand.title) not in claimed:
                    new = d; break
            if new is not None:
                t.meta["ident"] = {k: v for k, v in new.items() if k not in ("chosen", "group", "runner_group")}
                t.meta["ident_fields"] = ident.apply_to_book_fields(new); t.meta["reassigned"] = True
                claimed.add(work_key(new["chosen"].cand.title))
            else:
                t.meta["ident"] = {"status": "needs_review", "confidence": 0.0}; t.meta["ident_fields"] = {}; t.meta["conflict_demoted"] = True
    # Second chance for lying spines still unidentified: their reads often split the clues (title in one read, author in another).
    # Pool the book's OWN reads, but never allow a work another book already owns, so a neighbour's words cannot win.
    claimed = {key(t) for t in tracks if not t.removed and (t.meta.get("ident") or {}).get("status") == "identified"}
    for t in tracks:
        if t.removed or not t.meta.get("band") or (t.meta.get("ident") or {}).get("status") == "identified" or t.hits < 3 or len(t.ocr_obs) < 2:
            continue
        reads = [o["text"] for o in t.ocr_obs if o.get("text")]
        d = await ident.identify(ident.consensus_text(reads) or reads[0], 0.7, use_llm=False, evidence=" ".join(reads), exclude=frozenset(claimed))
        if d["status"] == "identified" and d.get("chosen"):
            t.meta["ident"] = {k: v for k, v in d.items() if k not in ("chosen", "group", "runner_group")}
            t.meta["ident_fields"] = ident.apply_to_book_fields(d); t.meta["pooled_second_chance"] = True
            claimed.add(work_key(d["chosen"].cand.title)); fixed += 1
    return fixed


async def scene_pass(session, tracks: list[Track], deadline: float) -> int:
    """Last chance for books OCR couldn't verify: let the vision model read each distinct best frame (max 6 calls)."""
    import time
    left = [t for t in tracks if not t.removed and not t.meta.get("excluded") and (t.meta.get("ident") or {}).get("status") != "identified"]
    by_frame: dict[str, list[Track]] = {}
    for t in left:
        by_frame.setdefault(t.best_frame_id, []).append(t)
    n = 0
    for fid, ts in list(by_frame.items())[:6]:
        if time.time() > deadline:
            break
        fr = session.store.get_frame(fid)
        img = cv2.imread(fr["path"]) if fr else None
        if img is not None:
            try:
                n += await session.scene_read(img, ts)
            except Exception:
                log.exception("scene pass failed for %s", fid)
    return n


def _cover(a: list, b: list) -> float:
    """Fraction of the smaller box covered by the other (mosaic coordinates)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def _has_text(t: Track) -> bool:
    import re
    return any(len(re.sub(r"\W", "", o.get("text", ""))) >= 4 for o in t.ocr_obs)


def _is_fragment(t: Track, ident: list) -> bool:
    """An unresolved track whose readable words all belong to an identified book's title/author is just a piece of it."""
    import re
    words = [w for w in re.findall(r"[a-z]{3,}", " ".join(o.get("text", "") for o in t.ocr_obs).lower())]
    if not words:
        return False
    for k in ident:
        f = k.meta.get("ident_fields") or {}
        flat = re.sub(r"[^a-z]", "", (f.get("title", "") + f.get("author", "")).lower())   # squashed: 'A.B.C.' == 'ABC'
        if flat and sum(w in flat for w in words) >= max(1, int(0.8 * len(words))):
            return True
    return False


PRODUCT_WORDS = {"lotion", "serum", "cream", "parfum", "perfume", "shampoo", "conditioner", "spf", "sunscreen", "moisturizer", "moisturiser", "deodorant",
                 "soap", "toner", "cleanser", "ingredients", "tablets", "capsules", "ml", "floz", "protection", "nourishing", "nourish", "sanitizer", "toothpaste", "mascara"}


def _is_product_label(t: Track) -> bool:
    import re
    words = set(re.findall(r"[a-z]{2,}", " ".join(o.get("text", "") for o in t.ocr_obs).lower().replace("fl oz", "floz")))
    prefix_hit = any(len(w) >= 6 and any(pw.startswith(w) or w.startswith(pw) for pw in PRODUCT_WORDS if len(pw) >= 6) for w in words)   # 'protectio' (cut off) still counts
    return (len(words & PRODUCT_WORDS) >= 1 or prefix_hit) and (t.meta.get("ident") or {}).get("status") != "identified"


def move_products_to_items(session, tracks: list[Track]) -> list[Track]:
    """Something that reads like a product label (lotion, serum, parfum, SPF...) is NOT a book: it leaves the Books list and becomes an
    item (personal care), so it is classified and priced like the other contents."""
    from copy import copy
    moved = []
    for t in tracks:
        if t.removed or not _is_product_label(t) or not t.best_bbox:
            continue
        session.objs._n += 1
        it = copy(t); it.tid = f"item_{session.objs._n:04d}"; it.kind = "item"; it.label = "skincare product"
        it.meta = {"cat": "personal_care", "moved_from_books": t.tid}
        it.ocr_obs = []; it.flags = set(t.flags)
        session.objs.tracks[it.tid] = it
        t.removed = True; t.meta["moved_to_items"] = it.tid
        moved.append(it)
    return moved


def drop_containers(tracks: list[Track]) -> int:
    """A tracker box that covers a whole tower reads several books at once. If its reads contain the titles of >=2 OTHER identified
    books, it is a container: those books are already tracked individually, so the container is dropped (never listed as a book)."""
    from rapidfuzz import fuzz
    from ..identify.matcher import norm
    ident = [t for t in tracks if not t.removed and (t.meta.get("ident") or {}).get("status") == "identified"]
    n = 0
    for t in tracks:
        if t.removed or t.meta.get("band") or not t.ocr_obs:
            continue
        blob = norm(" ".join(o.get("text", "") for o in t.ocr_obs))
        hits = [o for o in ident if o is not t and o.meta.get("band") and
                len(norm((o.meta.get("ident_fields") or {}).get("title", ""))) >= 6 and
                fuzz.partial_ratio(norm((o.meta.get("ident_fields") or {}).get("title", "")), blob) >= 88]
        if len(hits) >= 2:
            t.removed = True; t.meta["container"] = True; n += 1
    return n


def _only_junk(t: Track) -> bool:
    """Unresolved track whose reads contain no real word (e.g. 'R', '2', 'P'): nothing to identify, nothing worth listing."""
    import re
    if (t.meta.get("ident") or {}).get("status") == "identified":
        return False
    return bool(t.ocr_obs) and not any(len(w) >= 3 for o in t.ocr_obs for w in re.findall(r"[A-Za-z]+", o.get("text", "")))


def clean_books(tracks: list[Track]) -> int:
    """Final de-duplication so one physical book is one entry: same verified title -> one entry; unreadable
    fragments that sit on an identified book, or never produced any text with little evidence, are dropped."""
    n = 0
    by_title: dict[str, Track] = {}
    for t in sorted(tracks, key=lambda t: -t.best_score):
        from ..identify.matcher import work_key
        title = work_key((t.meta.get("ident_fields") or {}).get("title", "")) if (t.meta.get("ident") or {}).get("status") == "identified" else None
        if t.removed or not title:
            continue
        if title in by_title:
            keep = by_title[title]; t.removed = True; keep.hits += t.hits; n += 1
            if not (keep.meta.get("ident_fields") or {}).get("author") and (t.meta.get("ident_fields") or {}).get("author"):
                keep.meta["ident_fields"] = t.meta["ident_fields"]
        else:
            by_title[title] = t
    ident = [t for t in by_title.values()]
    for t in tracks:
        if t.removed or t in ident:
            continue
        if any(t.segment == k.segment and _cover(t.mbox, k.mbox) > 0.5 for k in ident):
            t.removed = True; n += 1
        elif _is_fragment(t, ident) or _is_product_label(t):
            t.removed = True; n += 1
        elif (not _has_text(t) and t.hits < 3) or (t.hits < 3 and t.conf < 0.2) or _only_junk(t):
            t.removed = True; n += 1
    return n


def clean_items(items: list[Track], books: list[Track]) -> int:
    """Drop item detections that are really a book cover/hand, and merge repeats of the same item."""
    n = 0
    live_books = [b for b in books if not b.removed]
    for t in sorted(items, key=lambda t: -t.conf):
        if t.removed:
            continue
        if any(t.segment == b.segment and _cover(t.mbox, b.mbox) > 0.5 for b in live_books):
            t.removed = True; n += 1; continue
        for o in items:
            if o is t or o.removed or o.conf < t.conf or o.meta.get("cat") != t.meta.get("cat"):
                continue
            same_spot = o.segment == t.segment and (iou(o.mbox, t.mbox) > 0.25 or _cover(o.mbox, t.mbox) > 0.6)
            same_look = False
            if o.sig is not None and t.sig is not None:
                import cv2
                same_look = cv2.compareHist(o.sig, t.sig, cv2.HISTCMP_BHATTACHARYYA) < 0.28   # same object seen again, even in another segment
            if same_spot or same_look:
                o.hits += t.hits; t.removed = True; n += 1; break
    return n


def _sig_tokens(t: Track) -> set:
    import re
    txt = " ".join(o.get("text", "") for o in t.ocr_obs)
    return {w for w in re.findall(r"[a-z]{4,}", txt.lower())}


async def rescue_pass(session, tracks: list[Track], deadline: float) -> int:
    import time
    todo = [t for t in tracks if not (t.removed or t.meta.get("excluded") or (t.meta.get("ident") or {}).get("status") == "identified")]
    sem = asyncio.Semaphore(3)

    async def one(t):
        async with sem:
            try:
                await session.rescue_track(t, deadline)
            except Exception:
                log.exception("rescue failed for %s", t.tid)
    await asyncio.gather(*(one(t) for t in todo[:8]))
    return sum((t.meta.get("ident") or {}).get("status") == "identified" for t in todo)


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
            from ..identify.matcher import work_key
            same_title = fa.get("title") and work_key(fa["title"]) == work_key(fb.get("title", ""))
            unreadable_pair = not fa.get("title") and not fb.get("title")
            ta, tb = _sig_tokens(a), _sig_tokens(b)
            same_text = len(ta & tb) >= 3
            disjoint = a.last_ts < b.first_ts or b.last_ts < a.first_ts
            closeup = a.meta.get("closeup") and b.meta.get("closeup")
            if (ov > 0.45 and not (a.meta.get('ghost') or b.meta.get('ghost')) or (same_title and ov > 0.15)) or (same_title and closeup and disjoint) or (same_text and disjoint):
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
        ocr_text = list(dict.fromkeys(o["text"] for o in t.ocr_obs if o["text"]))   # unique, in order
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
        except asyncio.TimeoutError:
            log.warning("pricing timed out for %s", b.id)
            b.pricing_note = "Pricing timed out (provider too slow); left blank — re-run to retry."
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
            r = await asyncio.wait_for(book_pricing.price_book(title=b.title, author=b.author, isbn=b.isbn, group_isbns=b._group_isbns,
                                              country=session.country, currency=session.currency, allow_foreign_fallback=True,
                                              home_market="US" if session.country != "US" else "GB"), timeout=12)
        b.replacement_cost, b.used_value, b.priced_isbn, b.pricing_note = r["replacement"], r["used"], r["priced_isbn"], r["note"]
        if r["over_threshold"]:
            b.needs_appraisal = True; b.status = "needs_appraisal"
            b.pricing_note = (b.pricing_note + f" Unusually valuable (> {r.get('threshold_local', s.lca_appraisal_price_threshold):g} {session.currency}, ~USD {s.lca_appraisal_price_threshold:g}): human appraisal required; not included in totals.").strip()
    await asyncio.gather(*(one(b) for b in books))
