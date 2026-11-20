#!/usr/bin/env python3
"""Build the submission folder:

  submission/
    Demo Video.mp4              (copied from --video; you record it, nothing is edited)
    claim_packet.json           (from --sweep)
    Claim Report.pdf            (from --sweep)
    Ground Truth & Results.xlsx (your hand-filled sheet; system-vs-pass-bar results added if you ran the evaluation)
    Architecture.pdf            (generated: one page)
    Failure Log.pdf             (generated: one page; stage latencies come from --sweep's packet when given)
    Evidence Frames/            (every frame the packet references, plus entity crops)

Usage (from the repo root, with the backend venv active):
    python scripts/assemble_submission.py                       # documents only (no sweep needed)
    python scripts/assemble_submission.py --sweep sw_... --video ~/Desktop/demo.mp4
Nothing here invents data: files that depend on a real sweep are only written when you pass it.
"""
from __future__ import annotations

import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


# ---------------------------------------------------------------- Architecture.pdf
def architecture_pdf(path: Path) -> None:
    W, H = landscape(A4)
    c = canvas.Canvas(str(path), pagesize=(W, H))
    c.setTitle("Architecture note - Library Contents Claim Agent")
    c.setFont("Helvetica-Bold", 14); c.drawString(15 * mm, H - 14 * mm, "Library Contents Claim Agent - architecture (one page)")
    c.setFont("Helvetica", 7.5)
    c.drawString(15 * mm, H - 19 * mm, "Perception and numbers come from CV, geometry and retrieved listings. LLMs only read text/covers as a fallback and never price, count or total.")

    def box(x, y, w, h, title, lines, fill):
        c.setFillColor(fill); c.setStrokeColor(colors.HexColor("#334155")); c.roundRect(x, y, w, h, 3, fill=1, stroke=1)
        c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 8); c.drawString(x + 2.5 * mm, y + h - 4.2 * mm, title)
        c.setFont("Helvetica", 6.6)
        for i, t in enumerate(lines):
            c.drawString(x + 2.5 * mm, y + h - 8.2 * mm - i * 3.1 * mm, t)

    def arrow(x1, y1, x2, y2):
        c.setStrokeColor(colors.HexColor("#475569")); c.setFillColor(colors.HexColor("#475569")); c.line(x1, y1, x2, y2)
        import math
        a = math.atan2(y2 - y1, x2 - x1); L = 2.2 * mm
        pth = c.beginPath(); pth.moveTo(x2, y2)
        pth.lineTo(x2 - L * math.cos(a - .4), y2 - L * math.sin(a - .4)); pth.lineTo(x2 - L * math.cos(a + .4), y2 - L * math.sin(a + .4)); pth.close()
        c.drawPath(pth, fill=1, stroke=0)

    y0, bh = H - 62 * mm, 36 * mm
    bw = (W - 30 * mm - 3 * 6 * mm) / 4
    xs = [15 * mm + i * (bw + 6 * mm) for i in range(4)]
    box(xs[0], y0, bw, bh, "1. SWEEP (phone/laptop browser)", [
        "camera ~6-7 JPEG frames/s + orientation", "mic -> client VAD -> utterance -> Whisper",
        "agent voice: browser TTS (intro, guidance,", "  narration, replies, final summary)", "voice/typed corrections (regex, closed set)",
        "Stop releases camera + mic"], colors.HexColor("#e0f2fe"))
    box(xs[1], y0, bw, bh, "2. RECORD + LIVE PREVIEW (FastAPI)", [
        "every frame stored (SQLite + files)", "preview ~1 frame/s from memory:", "  YOLO + CLIP book gate + Tesseract label",
        "  -> 'Found so far', spoken guidance", "blur/dark/glare tips; asks original-or-print", "preview never feeds the packet"], colors.HexColor("#ecfccb"))
    box(xs[2], y0, bw, bh, "3. BATCH ANALYSIS (after Stop)", [
        "drop blurry/duplicate frames (<=200)", "YOLO COCO + YOLO-World over frame lists", "PaddleOCR text-line discovery for missed spines",
        "ORB+RANSAC registration -> mosaic coords", "CLIP book/not-book gate", "tracks -> ONE unique id per object"], colors.HexColor("#fef9c3"))
    box(xs[3], y0, bw, bh, "4. PER-ID: identify, measure, price", [
        "Tesseract + PaddleOCR, zoom in/out, consensus", "Open Library / Google Books (author-first)",
        "deterministic matcher; LLM split/read only if", "  stable+unresolved, re-verified in catalogue", "measure via calibration -> cm", "price from listings (URL+date) -> Python totals"], colors.HexColor("#fce7f3"))
    for i in range(3):
        arrow(xs[i] + bw, y0 + bh / 2, xs[i + 1], y0 + bh / 2)
    c.setFont("Helvetica", 7); c.setFillColor(colors.HexColor("#475569"))
    c.drawString(xs[2] + 2 * mm, y0 - 4 * mm, "Stop")

    # outputs strip
    ob = 15 * mm
    c.setFillColor(colors.HexColor("#f1f5f9")); c.setStrokeColor(colors.HexColor("#334155")); c.roundRect(15 * mm, y0 - 22 * mm, W - 30 * mm, 14 * mm, 3, fill=1, stroke=1)
    c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 8)
    c.drawString(18 * mm, y0 - 14 * mm, "5. PACKET:  review queue (every low-confidence line + reason)  ->  claim_packet.json + claim_report.pdf  +  evidence frames (every frame_ref opens)")
    c.setFont("Helvetica", 6.8)
    c.drawString(18 * mm, y0 - 18.5 * mm, "Totals are summed in Python from validated lines; an unknown value is null and excluded; priced lines carry source, URL, date, currency, condition.")

    # three tables
    def table(x, y, w, title, rows, widths):
        data = [[Paragraph(f"<b>{a}</b>", ParagraphStyle("c", fontSize=6.4, leading=7.4)), Paragraph(b, ParagraphStyle("d", fontSize=6.4, leading=7.4))] for a, b in rows]
        t = Table(data, colWidths=[w * widths, w * (1 - widths)])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
        c.setFont("Helvetica-Bold", 8.5); c.setFillColor(colors.black); c.drawString(x, y + 2, title)
        tw, th = t.wrapOn(c, w, 500); t.drawOn(c, x, y - th)
    colw = (W - 30 * mm - 2 * 6 * mm) / 3
    ty = y0 - 28 * mm
    table(15 * mm, ty, colw, "Which model does what", [
        ("YOLOv8 (COCO) + YOLO-World", "find book boxes and room objects (no spine-class model exists; we do not train)"),
        ("CLIP ViT-B/32", "is this crop a book at all (rejects whiteboards, ceilings, chairs)"),
        ("PaddleOCR + Tesseract", "read spines/covers; PaddleOCR text detector also proposes missed spines"),
        ("Open Library / Google Books", "resolve OCR text to a work/edition; matcher is evidence-bound"),
        ("Groq gpt-oss-120b / qwen3.8 (budgeted)", "split noisy OCR; read a cover when OCR fails; output re-verified; never prices/counts"),
        ("Whisper (faster-whisper)", "speech to text for corrections"),
        ("OpenCV (ORB+RANSAC, homography)", "frame registration, A4 detection, quality (blur/glare)")], .36)
    table(15 * mm + colw + 6 * mm, ty, colw, "Where metric scale comes from", [
        ("A4/Letter sheet", "placed flat on a shelf edge; 4 corners -> homography -> cm/pixel (preferred)"),
        ("Known dimension", "'this bookcase is 200 cm tall' scales from the detected shelf edge (confidence capped 0.5)"),
        ("Two-point click", "any frame; two points a tape-measured distance apart"),
        ("Transfer", "scale is bound to its frame and carried to other frames by homography; confidence discounted per hop"),
        ("Room", "phone height + pitch at each floor/wall junction -> distance = h / tan(pitch); heading -> polygon (shoelace); ceiling from the ceiling junction"),
        ("No scale", "all cm fields stay blank and the lines go to the review queue")], .30)
    table(15 * mm + 2 * (colw + 6 * mm), ty, colw, "Where prices come from", [
        ("Books", "eBay Browse API (per-country marketplace) first, Google Shopping via Serper as fallback / for India; new and used listings"),
        ("Aggregation", "listings must match the title/ISBN; trimmed median; >=2 listings; used > 3x new is withheld"),
        ("Currency", "local only. If none, converted from another market via Frankfurter (ECB) and labelled converted; India is never converted (left blank)"),
        ("Non-book", "interquartile range of >=3 local listings, all URLs kept; art/portraits: needs appraisal unless the owner says print"),
        ("Appraisal flags", "signed, first edition, pre-1925, or above the value threshold: not priced, excluded from totals"),
        ("Never", "an LLM price from memory")], .22)
    c.setFont("Helvetica-Oblique", 6.5); c.setFillColor(colors.HexColor("#475569"))
    c.drawString(15 * mm, 10 * mm, "Stack: FastAPI + SQLite, Next.js. Stages are separate and unit-tested (81 tests). Accuracy against ground truth: see 'Ground Truth & Results.xlsx' (not claimed here).")
    c.showPage(); c.save()


# ---------------------------------------------------------------- Failure Log.pdf
ERRORS = [
    ("1. One book listed as 2-3 entries",
     "Tracker boxes fragmented (cut-off pieces, tilted spines split into fake columns) and the catalogue returns the same work under different strings ('The A.B.C. Murders' vs 'The ABC Murders'; '... (adaptation)').",
     "Normalised work key + merge of fragments whose words belong to an identified book; splitter only splits runs of >=3 narrow spines.",
     "Test case: 4 entries -> 2 books (unit test + replay of a real sweep)."),
    ("2. Wrong things counted as books, real ones missed",
     "'25 books' for 2: whiteboard panels, ceiling, shelf objects passed a recall-first detector; a dark, tilted spine was never detected (COCO 'book' class misses it); my own speed change once removed the only detector that found it.",
     "CLIP book/not-book gate (multi-zoom, readable-text rule), open-vocabulary detector kept, PaddleOCR text-line discovery for missed spines.",
     "False positives score 0.03-0.30, real books 0.54-0.98; missed spine now found (gate 0.98). Not yet measured on a 60-book room."),
    ("3. Readable book left unidentified",
     "Pooling every noisy OCR read into one catalogue query ('AgatleChustic ... POOT ... Borwat') returned unrelated books, so the right title was never a candidate; the author line was garbled.",
     "Search with consensus words (seen in >=2 reads), score candidates against all raw reads, author-first filtering, multi-zoom + rescue pass.",
     "Same 7 reads: unidentified -> identified at 0.92. Stop->packet 48 s -> 18.6 s (parallel pricing, bounded passes)."),
]


def failure_log_pdf(path: Path, packet: dict | None) -> None:
    ss = lambda n, **k: ParagraphStyle(n, fontSize=k.get("s", 7.4), leading=k.get("l", 8.8), **{x: v for x, v in k.items() if x not in ("s", "l")})
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=11 * mm, bottomMargin=9 * mm, title="Failure log")
    st = [Paragraph("<b>Failure log (one page)</b>", ss("h", s=13, l=15)),
          Paragraph("<i>Honest scope: the three errors below were found and fixed on small development captures (about 2 books). They are NOT results from a 60-book room. "
                    "After the real-room run, replace them with that run's three worst errors (docs/capture_and_eval_checklist.md).</i>", ss("n", s=7, l=8.2)), Spacer(1, 3)]
    data = [[Paragraph("<b>Error</b>", ss("a")), Paragraph("<b>Root cause</b>", ss("a")), Paragraph("<b>Fix</b>", ss("a")), Paragraph("<b>Measured effect</b>", ss("a"))]]
    for e in ERRORS:
        data.append([Paragraph(f"<b>{e[0]}</b>", ss("b", s=7)), Paragraph(e[1], ss("b", s=6.8, l=8)), Paragraph(e[2], ss("b", s=6.8, l=8)), Paragraph(e[3], ss("b", s=6.8, l=8))])
    t = Table(data, colWidths=[30 * mm, 62 * mm, 52 * mm, 38 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#94a3b8")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    st += [t, Spacer(1, 5)]
    st.append(Paragraph("<b>Latency per stage</b>", ss("h2", s=9, l=11)))
    if packet and packet.get("processing"):
        rows = [["Stage (from this sweep's packet)", "Seconds", "Status"]] + [[p["stage"].replace("_", " "), f'{p["seconds"]}', str(p["status"])[:60]] for p in packet["processing"]]
        src = f"Measured on sweep {packet['sweep']['id']} (duration {packet['sweep'].get('duration_s')} s)."
    else:
        rows = [["Stage (development measurements, small captures)", "Latency", ""],
                ["Book + object detection, per analysed frame", "~46 ms + ~20-40 ms", "Apple-silicon laptop"], ["CLIP book gate, per new track", "15-120 ms", ""],
                ["Spine OCR: Tesseract / PaddleOCR, per crop", "0.12 s / 0.8-1.5 s", ""], ["Catalogue identification, per book", "0.9-2.1 s", ""],
                ["Pricing, 4 books in parallel", "4.0 s", "Serper, India"], ["Batch frame analysis, 328 frames -> 135 analysed", "21 s", "detect 10, discover 4.5, track 3.3, load 3.3"],
                ["Stop -> packet, 2-book sweep", "18.6 s", "limit: 300 s"]]
        src = "Placeholder numbers from development captures; run scripts/assemble_submission.py --sweep <id> to replace them with the real sweep's timings."
    lt = Table(rows, colWidths=[85 * mm, 30 * mm, 67 * mm])
    lt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#cbd5e1")), ("FONTSIZE", (0, 0), (-1, -1), 6.8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")), ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
    st += [Paragraph(src, ss("s", s=6.8, l=8)), lt, Spacer(1, 5)]
    cost = "(no sweep supplied)"
    if packet and (packet.get("run_stats") or {}).get("cost_estimate"):
        cost = json.dumps(packet["run_stats"]["cost_estimate"])
    st.append(Paragraph("<b>Cost per sweep</b> (estimate, not a bill)", ss("h2", s=9, l=11)))
    st.append(Paragraph("Local models (YOLO, CLIP, OCR, Whisper) cost nothing per sweep. Paid calls: price searches (Serper, about 2-3 queries per priced book plus one per item category; "
                        "at ~US$1 per 1,000 queries a 60-book room is roughly 150-250 queries, i.e. a few tens of cents; the free tier covers 2,500) and large-model calls (Groq: budgeted to "
                        "6/minute and 30 per sweep live plus a bounded final pass; free tier). eBay Browse is free. Packet's own estimate: " + cost + ".", ss("c", s=7, l=8.4)))
    st += [Spacer(1, 5), Paragraph("<b>What I would do with two more weeks</b>", ss("h2", s=9, l=11)),
           Paragraph("1) Run the 60-book ground-truth evaluation and tune on a held-out shelf. 2) Fine-tune a spine detector (pretrained detectors have no spine class). 3) Replace the planar-shelf homography assumption with depth (LiDAR/stereo) for tilted or deep shelves. "
                     "4) Real-time full-duplex voice. 5) Condition grading per book. 6) An independent checker agent that audits the packet.", ss("c", s=7, l=8.4))]
    doc.build(st)


# ---------------------------------------------------------------- workbook, frames
def ground_truth_xlsx(out: Path) -> str:
    import openpyxl
    src = ROOT / "data/eval/ground_truth.xlsx"
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        import subprocess
        subprocess.run([sys.executable, "-m", "app.eval.ground_truth", str(src)], cwd=ROOT / "backend", check=True)
    wb = openpyxl.load_workbook(src)
    res = ROOT / "data/eval/results/results.xlsx"
    note = "template only: fill it by hand, then run the evaluation"
    if res.exists():
        rw = openpyxl.load_workbook(res)
        for name in rw.sheetnames:
            ws = wb.create_sheet(("SYSTEM_" + name)[:31])
            for row in rw[name].iter_rows(values_only=True):
                ws.append(list(row))
        note = "includes system-vs-pass-bar results"
    wb.save(out)
    return note


def collect_sweep(sweep: str, out: Path) -> list[str]:
    sd = ROOT / "data/sweeps" / sweep
    notes = []
    pk = sd / "exports/claim_packet.json"
    if not pk.exists():
        sys.exit(f"no packet for {sweep}: finish the sweep first ({pk})")
    shutil.copy(pk, out / "claim_packet.json")
    pdf = sd / "exports/claim_report.pdf"
    if pdf.exists():
        shutil.copy(pdf, out / "Claim Report.pdf")
    P = json.loads(pk.read_text())
    ids = {b.get("frame_ref") for b in P["books"]} | {i.get("frame_ref") for i in P["items"]} | {f["id"] for f in P.get("evidence_frames", [])}
    ids |= set(P["room"].get("evidence_frames", [])) | {c.get("frame_ref") for c in P.get("calibrations", [])}
    for b in P["books"] + P["items"]:
        for k in ("height_measurement", "thickness_measurement"):
            ids.add((b.get(k) or {}).get("frame_ref"))
    ids.discard(None); ids.discard("")
    fr = out / "Evidence Frames"; fr.mkdir(exist_ok=True)
    missing = []
    for f in sorted(ids):
        s = sd / "frames" / f"{f}.jpg"
        (shutil.copy(s, fr / s.name) if s.exists() else missing.append(f))
    if (sd / "crops").exists():
        shutil.copytree(sd / "crops", fr / "crops", dirs_exist_ok=True)
    notes.append(f"{len(ids) - len(missing)} referenced frames copied" + (f"; MISSING {len(missing)}: {missing[:5]}" if missing else "; every frame_ref resolves"))
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep"); ap.add_argument("--video"); ap.add_argument("--out", default=str(ROOT / "submission"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    packet = None
    if a.sweep:
        pkp = ROOT / "data/sweeps" / a.sweep / "exports/claim_packet.json"
        packet = json.loads(pkp.read_text()) if pkp.exists() else None
    architecture_pdf(out / "Architecture.pdf"); print("wrote Architecture.pdf")
    failure_log_pdf(out / "Failure Log.pdf", packet); print("wrote Failure Log.pdf", "(timings from the sweep)" if packet else "(placeholder timings)")
    print("wrote Ground Truth & Results.xlsx -", ground_truth_xlsx(out / "Ground Truth & Results.xlsx"))
    if a.sweep:
        for n in collect_sweep(a.sweep, out):
            print(" ", n)
    else:
        print("SKIPPED (need a real sweep): claim_packet.json, Claim Report.pdf, Evidence Frames/  -> rerun with --sweep <id>")
    if a.video:
        shutil.copy(a.video, out / "Demo Video.mp4"); print("copied Demo Video.mp4 (unedited)")
    else:
        print("SKIPPED: Demo Video.mp4 -> rerun with --video <path-to-your-recording>")
    print("\nsubmission folder:", out)


if __name__ == "__main__":
    main()
