"""Readable PDF claim report (ReportLab). Reads numbers from the validated packet only."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..schemas import ClaimPacket

ss = getSampleStyleSheet()
SMALL = ParagraphStyle("small", parent=ss["BodyText"], fontSize=7.5, leading=9)
TINY = ParagraphStyle("tiny", parent=ss["BodyText"], fontSize=6.5, leading=8)


def money(v, cur):
    return "—" if v is None else f"{cur} {v:,.2f}"


def _tbl(data, widths, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(st)); return t


def _link(url, label="source"):
    return Paragraph(f'<link href="{url}" color="blue">{label}</link>', TINY) if url else Paragraph("—", TINY)


def write_pdf(p: ClaimPacket, path: Path, crops_dir: Path, frames_dir: Path) -> None:
    cur = p.sweep.currency
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"Library Contents Claim {p.sweep.id}")
    S = []
    S.append(Paragraph("Library Contents Claim Packet", ss["Title"]))
    S.append(Paragraph(f"Sweep <b>{p.sweep.id}</b> · captured {p.sweep.captured_at} · device {p.sweep.device or 'n/a'} · "
                       f"duration {p.sweep.duration_s or 0:.0f}s · {p.sweep.country} / {cur}", ss["BodyText"]))
    S.append(Paragraph(p.disclaimer, SMALL)); S.append(Spacer(1, 6))

    t = p.totals
    S.append(Paragraph("Claim summary", ss["Heading2"]))
    S.append(_tbl([["Metric", "Value"],
        ["Books detected", t.book_count], ["Identified / unidentified", f"{t.books_identified} / {t.books_unidentified}"],
        ["Shelf run (calibrated books only)", "unavailable" if t.shelf_run_m is None else f"{t.shelf_run_m} m"],
        [f"Books replacement cost ({t.books_priced_replacement} priced)", money(t.books_replacement_cost, cur)],
        [f"Books used value ({t.books_priced_used} priced)", money(t.books_used_value, cur)],
        [f"Non-book replacement range ({t.items_priced} priced)", f"{money(t.items_replacement_cost_low, cur)} – {money(t.items_replacement_cost_high, cur)}"],
        ["Entries excluded from monetary totals", t.excluded_from_totals]], [90 * mm, 80 * mm]))
    S.append(Paragraph("Totals are computed in Python from sourced prices only. Unpriced, appraisal-required and owner-excluded entries contribute nothing.", SMALL))

    r = p.room
    S.append(Paragraph("Room", ss["Heading2"]))
    f = lambda v, u="": "unavailable" if v is None else f"{v}{u}"
    S.append(_tbl([["Dimension", "Value"], ["Length", f(r.length_m, " m")], ["Width", f(r.width_m, " m")], ["Ceiling height", f(r.height_m, " m")],
        ["Floor area", f"{f(r.floor_area_m2, ' m²')} / {f(r.floor_area_ft2, ' ft²')}"], ["Wall area (gross)", f"{f(r.wall_area_m2, ' m²')} / {f(r.wall_area_ft2, ' ft²')}"],
        ["Shelved wall area", f"{f(r.shelved_wall_area_m2, ' m²')} / {f(r.shelved_wall_area_ft2, ' ft²')}"],
        ["Method", r.scale_method or "none established"], ["Confidence", f(r.confidence)], ["Shape", r.shape or "—"]], [60 * mm, 110 * mm]))
    for a in r.assumptions[:8]:
        S.append(Paragraph("• " + a, SMALL))

    S.append(PageBreak()); S.append(Paragraph("Books by shelf", ss["Heading2"]))
    shelves: dict[str, list] = {}
    for b in p.books:
        shelves.setdefault(b.shelf or "unassigned", []).append(b)
    for sh, bs in sorted(shelves.items()):
        S.append(Paragraph(f"Shelf {sh} — {len(bs)} books", ss["Heading3"]))
        rows = [["#", "Title / author", "Status", "Conf", "H×T cm", "Replacement", "Used", "Frame"]]
        for b in bs:
            title = Paragraph(f"<b>{b.title or '(unidentified)'}</b><br/>{b.author}" + (f"<br/><i>{b.edition}</i>" if b.edition else "") +
                              (f"<br/>OCR: {' | '.join(b.ocr_text[:2])}" if not b.title and b.ocr_text else ""), SMALL)
            dims = "—" if b.spine_height_cm is None else f"{b.spine_height_cm}×{b.spine_thickness_cm if b.spine_thickness_cm is not None else '—'}"
            rows.append([b.position, title, b.status, f"{b.id_confidence:.2f}", dims,
                         Paragraph(money(b.replacement_cost.amount, cur) + ("<br/>" + f'<link href="{b.replacement_cost.url}" color="blue">{b.replacement_cost.source[:24]}</link>' if b.replacement_cost.url else ""), TINY),
                         Paragraph(money(b.used_value.amount, cur) + ("<br/>" + f'<link href="{b.used_value.url}" color="blue">{b.used_value.source[:24]}</link>' if b.used_value.url else ""), TINY),
                         Paragraph(b.frame_ref[-9:], TINY)])
        S.append(_tbl(rows, [8 * mm, 58 * mm, 20 * mm, 10 * mm, 18 * mm, 26 * mm, 26 * mm, 14 * mm]))

    S.append(PageBreak()); S.append(Paragraph("Non-book inventory", ss["Heading2"]))
    rows = [["ID", "Category / description", "Dims W×H cm", "Status", "Replacement range", "Sources"]]
    for i in p.items:
        dims = "—" if i.dimensions_cm.w is None else f"{i.dimensions_cm.w:.0f}×{i.dimensions_cm.h:.0f}"
        rng = "—" if i.replacement_cost.low is None else f"{money(i.replacement_cost.low, cur)} – {money(i.replacement_cost.high, cur)}"
        links = Paragraph(" ".join(f'<link href="{s["url"]}" color="blue">[{n + 1}]</link>' for n, s in enumerate(i.replacement_cost.sources[:5])), TINY)
        rows.append([Paragraph(i.id, TINY), Paragraph(f"<b>{i.category}</b> {i.description}" + (f"<br/>{i.brand_model}" if i.brand_model else ""), SMALL), dims, i.status, rng, links])
    S.append(_tbl(rows, [20 * mm, 55 * mm, 22 * mm, 24 * mm, 35 * mm, 22 * mm]))

    if p.second_country_pricing and p.second_country_pricing.rows:
        sc = p.second_country_pricing
        S.append(Spacer(1, 8)); S.append(Paragraph(f"Second-country pricing comparison — {sc.country} ({sc.currency})", ss["Heading2"]))
        rows = [["Book", "Replacement", "Used", "Source", "Note"]]
        for r_ in sc.rows:
            rows.append([Paragraph(r_.title or r_.book_id, SMALL), money(r_.replacement_cost, sc.currency), money(r_.used_value, sc.currency),
                         _link(r_.url, r_.source[:30] or "source"), Paragraph(r_.note, TINY)])
        S.append(_tbl(rows, [60 * mm, 26 * mm, 26 * mm, 30 * mm, 36 * mm]))

    S.append(PageBreak()); S.append(Paragraph("Review queue", ss["Heading2"]))
    rows = [["Ref", "Category", "Reason"]] + [[Paragraph(q.ref_id, TINY), q.category, Paragraph(q.reason, SMALL)] for q in p.review_queue]
    S.append(_tbl(rows if len(rows) > 1 else [["Ref", "Category", "Reason"], ["—", "—", "Nothing to review"]], [32 * mm, 28 * mm, 112 * mm]))

    S.append(Paragraph("Calibrations", ss["Heading2"]))
    rows = [["ID", "Scope", "Method", "Conf", "Frame", "Note"]] + [[c["id"], c["scope"], c["method"], c["confidence"], c["frame_ref"][-9:], Paragraph(c["note"][:90], TINY)] for c in p.calibrations]
    S.append(_tbl(rows if len(rows) > 1 else [["ID", "Scope", "Method", "Conf", "Frame", "Note"], ["—", "—", "none established", "—", "—", "metric values left blank"]], [14 * mm, 16 * mm, 30 * mm, 12 * mm, 22 * mm, 78 * mm]))

    # evidence thumbnails
    thumbs = []
    for b in [x for x in p.books if x.status in ("unidentified", "needs_review", "needs_appraisal")][:6] + p.items[:6]:
        f = crops_dir / f"{b.id}.jpg"
        if f.exists():
            try:
                thumbs.append([Image(str(f), width=28 * mm, height=28 * mm, kind="proportional"), Paragraph(f"{b.id}<br/>{b.frame_ref[-9:]}", TINY)])
            except Exception:
                pass
    if thumbs:
        S.append(Paragraph("Evidence samples (flagged entries)", ss["Heading2"]))
        rows = [sum(thumbs[i:i + 4], []) for i in range(0, len(thumbs), 4)]
        S.append(Table(rows))

    S.append(Paragraph("Processing timings", ss["Heading2"]))
    S.append(_tbl([["Stage", "Seconds", "Status"]] + [[x.stage, x.seconds, x.status] for x in p.processing], [80 * mm, 30 * mm, 60 * mm]))
    doc.build(S)
