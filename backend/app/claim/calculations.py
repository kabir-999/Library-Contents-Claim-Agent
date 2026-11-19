"""All claim arithmetic lives here, in plain Python. No LLM ever touches a total.

Rules enforced:
* A book/item contributes to a monetary total only if it has a sourced amount
  (source URL + retrieval date) in the claim currency, is not flagged for
  appraisal, and is not excluded (e.g. "belongs to someone else").
* Unknown (None) is never coerced to 0.
* Shelf run is the sum of *calibrated* spine thicknesses only; if any books on a
  shelf lack a calibrated thickness the run is reported as a lower bound and the
  gap is put in the review queue.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Optional

from ..schemas import Book, ClaimPacket, Item, ReviewEntry, Totals

M2_TO_FT2 = 10.763910416709722


def _r(x: float, places: int = 2) -> float:
    q = Decimal(1).scaleb(-places)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def m2_to_ft2(v: Optional[float]) -> Optional[float]:
    return None if v is None else _r(v * M2_TO_FT2, 2)


def has_source(url: str, retrieved_at: str) -> bool:
    return bool(url and url.startswith(("http://", "https://")) and retrieved_at)


def book_excluded(b: Book) -> bool:
    return b.status == "excluded"


def book_replacement_ok(b: Book, currency: str) -> bool:
    rc = b.replacement_cost
    return (
        rc.amount is not None
        and not b.needs_appraisal
        and not book_excluded(b)
        and has_source(rc.url, rc.retrieved_at)
        and (not rc.currency or rc.currency == currency)
    )


def book_used_ok(b: Book, currency: str) -> bool:
    uv = b.used_value
    return (
        uv.amount is not None
        and not b.needs_appraisal
        and not book_excluded(b)
        and has_source(uv.url, uv.retrieved_at)
        and (not uv.currency or uv.currency == currency)
    )


def item_priced_ok(i: Item, currency: str) -> bool:
    c = i.replacement_cost
    return (
        i.status == "priced"
        and c.low is not None
        and c.high is not None
        and c.low <= c.high
        and has_source(c.url, c.retrieved_at)
        and (not c.currency or c.currency == currency)
    )


def compute_shelf_stats(books: Iterable[Book]) -> dict[str, dict]:
    shelves: dict[str, dict] = defaultdict(
        lambda: {"book_count": 0, "measured_thickness_count": 0, "run_cm": 0.0, "heights_cm": []}
    )
    for b in books:
        if book_excluded(b):
            continue
        s = shelves[b.shelf or "unassigned"]
        s["book_count"] += 1
        t = b.spine_thickness_cm
        if t is not None:
            s["measured_thickness_count"] += 1
            s["run_cm"] += t
        if b.spine_height_cm is not None:
            s["heights_cm"].append(b.spine_height_cm)
    out = {}
    for name, s in shelves.items():
        complete = s["measured_thickness_count"] == s["book_count"] and s["book_count"] > 0
        out[name] = {
            "book_count": s["book_count"],
            "measured_thickness_count": s["measured_thickness_count"],
            "run_m": _r(s["run_cm"] / 100, 3) if s["measured_thickness_count"] else None,
            "run_complete": complete,
            "max_height_cm": _r(max(s["heights_cm"]), 1) if s["heights_cm"] else None,
        }
    return out


def compute_totals(books: list[Book], items: list[Item], currency: str) -> Totals:
    counted = [b for b in books if not book_excluded(b)]
    ident = [b for b in counted if b.status == "identified"]
    unident = [b for b in counted if b.status in ("unidentified", "partial", "needs_review")]

    rep = [b.replacement_cost.amount for b in counted if book_replacement_ok(b, currency)]
    used = [b.used_value.amount for b in counted if book_used_ok(b, currency)]
    priced_items = [i for i in items if item_priced_ok(i, currency)]
    lows = [i.replacement_cost.low for i in priced_items]
    highs = [i.replacement_cost.high for i in priced_items]

    shelves = compute_shelf_stats(books)
    runs = [s["run_m"] for s in shelves.values() if s["run_m"] is not None]

    excluded = 0
    for b in books:
        if not (book_replacement_ok(b, currency) or book_used_ok(b, currency)):
            excluded += 1
    for i in items:
        if not item_priced_ok(i, currency):
            excluded += 1

    return Totals(
        book_count=len(counted),
        books_identified=len(ident),
        books_unidentified=len(unident),
        shelf_run_m=_r(sum(runs), 2) if runs else None,
        books_replacement_cost=_r(sum(rep)),
        books_used_value=_r(sum(used)),
        items_replacement_cost_low=_r(sum(lows)),
        items_replacement_cost_high=_r(sum(highs)),
        excluded_from_totals=excluded,
        currency=currency,
        books_priced_replacement=len(rep),
        books_priced_used=len(used),
        items_priced=len(priced_items),
        per_shelf=shelves,
    )


def build_review_queue(packet: ClaimPacket, id_conf_min: float = 0.8) -> list[ReviewEntry]:
    q: list[ReviewEntry] = []
    cur = packet.sweep.currency

    def add(ref, reason, cat, sev="warn", frame=""):
        q.append(ReviewEntry(ref_id=ref, reason=reason, category=cat, severity=sev, frame_ref=frame))

    for b in packet.books:
        if book_excluded(b):
            continue
        if b.status == "unidentified":
            add(b.id, "Unreadable book spine" + (f" (partial OCR: {' | '.join(b.ocr_text[:3])})" if b.ocr_text else ""), "identification", "warn", b.frame_ref)
        elif b.status in ("partial", "needs_review"):
            add(b.id, f"Low-confidence or ambiguous identification ({b.id_confidence:.2f})", "identification", "warn", b.frame_ref)
        if "multiple_editions" in b.flags:
            add(b.id, "Multiple possible editions; edition not assigned", "edition", "warn", b.frame_ref)
        if "possible_duplicate" in b.flags:
            add(b.id, "Possible duplicate of another detected book", "duplicate", "warn", b.frame_ref)
        if "occluded" in b.flags:
            add(b.id, "Partially occluded book", "occlusion", "info", b.frame_ref)
        if b.needs_appraisal:
            add(b.id, "Signed/rare/antiquarian or high-value: needs human appraisal (not priced)", "appraisal", "warn", b.frame_ref)
        if b.spine_height_cm is None or b.spine_thickness_cm is None:
            add(b.id, "Uncalibrated measurement: scale not established", "measurement", "info", b.frame_ref)
        if b.status == "identified" and not b.needs_appraisal:
            if not book_replacement_ok(b, cur):
                add(b.id, "Missing sourced replacement price", "pricing", "info", b.frame_ref)
            if not book_used_ok(b, cur):
                add(b.id, "Missing sourced used-market price", "pricing", "info", b.frame_ref)

    for i in packet.items:
        if i.status == "excluded":
            continue
        if i.status == "needs_appraisal":
            add(i.id, f"{i.category or 'Item'} requires appraisal (original vs print unknown / not auto-priced)", "appraisal", "warn", i.frame_ref)
        elif i.status == "unpriced":
            add(i.id, "Missing price source", "pricing", "info", i.frame_ref)
        if i.confidence < 0.5:
            add(i.id, f"Low classification confidence ({i.confidence:.2f})", "classification", "info", i.frame_ref)
        if "occluded" in i.flags:
            add(i.id, "Partially occluded object", "occlusion", "info", i.frame_ref)
        if i.dimensions_cm.w is None:
            add(i.id, "Dimensions not calibrated", "measurement", "info", i.frame_ref)

    for shelf, s in packet.totals.per_shelf.items():
        if s["book_count"] and not s["run_complete"]:
            add(f"shelf:{shelf}", f"Shelf run is a lower bound: {s['measured_thickness_count']}/{s['book_count']} books have calibrated thickness", "measurement", "info")

    r = packet.room
    for name, v in (("length", r.length_m), ("width", r.width_m), ("ceiling height", r.height_m), ("wall area", r.wall_area_m2)):
        if v is None:
            add("room", f"Room {name} unavailable: insufficient sweep evidence", "room", "warn")
    return q


def validate_frame_refs(packet: ClaimPacket, existing_frame_ids: set[str]) -> list[str]:
    """Every frame_ref must resolve to a saved frame. Returns the dangling refs."""
    bad = []
    for obj in [*packet.books, *packet.items]:
        if obj.frame_ref and obj.frame_ref not in existing_frame_ids:
            bad.append(f"{obj.id}:{obj.frame_ref}")
    return bad


def apply_contract_status(packet: ClaimPacket) -> ClaimPacket:
    """Map internal statuses onto the contract enums, AFTER totals and review queue are computed.

    books.status  : identified | unidentified | needs_appraisal
      - partial / needs_review  -> unidentified; the low-confidence proposal moves to suggested_title/author
        (the brief forbids asserting a guessed title)
      - excluded (owner: not mine) -> removed from `books`, kept in `excluded_books` for traceability
    items.status  : priced | range | needs_appraisal
      - priced w/ low<high -> range ; low==high -> priced ; unpriced -> needs_appraisal (a human must price it)
      - excluded -> `excluded_items`
    The original value is preserved in detail_status.
    """
    keep_b, excl_b = [], []
    for b in packet.books:
        b.detail_status = b.status
        if b.status == "excluded":
            excl_b.append(b); continue
        if b.status in ("partial", "needs_review", "unidentified"):
            if b.title and b.status != "unidentified":
                b.suggested_title, b.suggested_author = b.title, b.author
            b.title = b.author = b.isbn = b.edition = b.publisher = ""
            b.status = "unidentified"
        keep_b.append(b)
    keep_i, excl_i = [], []
    for i in packet.items:
        i.detail_status = i.status
        if i.status == "excluded":
            excl_i.append(i); continue
        if i.status == "priced":
            c = i.replacement_cost
            i.status = "priced" if (c.low is not None and c.low == c.high) else "range"
        elif i.status in ("unpriced", "needs_review"):
            i.status = "needs_appraisal"
        keep_i.append(i)
    packet.books, packet.items = keep_b, keep_i
    packet.excluded_books, packet.excluded_items = excl_b, excl_i
    return packet


def finalize_packet(packet: ClaimPacket, id_conf_min: float = 0.8) -> ClaimPacket:
    cur = packet.sweep.currency
    totals = compute_totals(packet.books, packet.items, cur)
    packet.totals = totals
    r = packet.room
    r.floor_area_ft2 = m2_to_ft2(r.floor_area_m2)
    r.wall_area_ft2 = m2_to_ft2(r.wall_area_m2)
    r.shelved_wall_area_ft2 = m2_to_ft2(r.shelved_wall_area_m2)
    packet.review_queue = build_review_queue(packet, id_conf_min)
    return packet
