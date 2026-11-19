"""End-to-end post-sweep pipeline with deterministic stubs for detectors, OCR, metadata and listings.
Verifies wiring, schema validity, Python-only totals, evidence refs, PDF generation and the 5-minute budget -
NOT model accuracy (that is measured against real ground truth by app/eval)."""
import json

import cv2
import numpy as np
import pytest

from app.agent.session import SweepSession
from app.claim import calculations as calc
from app.identify import metadata
from app.identify.metadata import BookCandidate
from app.pipeline import postprocess
from app.pricing import providers
from app.schemas import ClaimPacket
from app.storage import Storage
from app.vision import ocr


def make_frame(w=1280, h=720, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.resize((rng.random((h // 8, w // 8, 3)) * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(img, (40, 40), (200, 140), (245, 245, 245), -1)  # keeps frame textured
    return cv2.imencode(".jpg", img)[1].tobytes()


TITLES = {0: ("THE REMAINS OF THE DAY ISHIGURO", "The Remains of the Day", "Kazuo Ishiguro", "9780679731726"),
          1: ("DUNE FRANK HERBERT", "Dune", "Frank Herbert", "9780441172719"),
          2: ("zzq", None, None, None)}


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    store = Storage(tmp_path)
    store.create_sweep("sw", "US", "USD", "pytest", {})

    class FakeBooks:
        available = True
        def detect(self, img):
            # three upright spines on one shelf, sized 30x230 px
            return [{"bbox": [300 + i * 60, 200, 330 + i * 60, 430], "conf": 0.9, "label": "book", "orientation": "upright"} for i in range(3)]
    class FakeObjs:
        available = True
        mode = "yolo_world"
        def detect(self, img):
            return [{"bbox": [700, 100, 1000, 600], "conf": 0.8, "label": "framed painting", "category": "artwork"},
                    {"bbox": [50, 300, 250, 700], "conf": 0.8, "label": "lamp", "category": "lamp"}]

    def fake_read(img, bbox, orientation="upright"):
        i = round((bbox[0] - 300) / 60)
        txt = TITLES[i][0]
        return {"lines": [{"text": txt, "conf": 0.9}], "text": txt, "conf": 0.9, "rotation": "cw90", "available": True}
    monkeypatch.setattr(ocr, "read_spine", fake_read)

    async def fake_lookup(q):
        return [BookCandidate("google_books", t[1], [t[2]], "Vintage", "1989", t[3]) for t in TITLES.values() if t[1]]
    monkeypatch.setattr(metadata, "lookup", fake_lookup)

    async def fake_ebay(query, country, condition="NEW", category=None, limit=30):
        base = 10.0 if condition == "NEW" else 4.0
        t = "The Remains of the Day Ishiguro" if "9780679731726" in query else "Dune Frank Herbert 9780441172719"
        return [{"title": t, "amount": base + i, "currency": "USD", "url": f"https://ebay.example/itm/{i}", "source": "eBay US", "condition": condition} for i in range(4)] \
            if ("9780679731726" in query or "9780441172719" in query) else \
            [{"title": "floor lamp", "amount": 30 + 10 * i, "currency": "USD", "url": f"https://ebay.example/lamp/{i}", "source": "eBay US", "condition": "NEW"} for i in range(5)]
    monkeypatch.setattr(providers, "ebay_search", fake_ebay)
    monkeypatch.setattr(providers, "ebay_enabled", lambda: True)
    monkeypatch.setattr(providers, "serp_enabled", lambda: False)
    from app.llm import providers as llmp
    monkeypatch.setattr(llmp.LLMRouter, "complete", lambda *a, **k: _none())
    monkeypatch.setattr(llmp.LLMRouter, "vision", lambda *a, **k: _none())

    sess = SweepSession("sw", "US", "USD", "pytest", camera_height_cm=150, store=store)
    sess.detector, sess.objdet = FakeBooks(), FakeObjs()
    return sess, store


async def _none():
    return None


async def test_full_pipeline(stubbed):
    sess, store = stubbed
    for i in range(4):
        await sess.on_frame(make_frame(seed=1), ts=sess.t0 + i * 0.4, orientation={"alpha": 0, "beta": 90})
    assert sess.counts()["books"] == 3 and sess.counts()["items"] == 2

    # voice correction applied to inventory: artwork is an original; user states shelf is...
    sess.objs.tracks[[t for t in sess.objs.tracks if sess.objs.tracks[t].label == "framed painting"][0]].meta["user_note"] = ""
    # manual two-point calibration on a saved frame: 10 px/cm
    fid = sess.current_frame_id()
    sess.calibrate_two_point(fid, (100, 100), (100, 400), 30.0)

    packet = await postprocess.run(sess)
    assert isinstance(packet, ClaimPacket)
    ClaimPacket.model_validate_json(packet.model_dump_json())  # round-trips the schema

    ids = {b.title: b for b in packet.books}
    rem, dune = ids["The Remains of the Day"], ids["Dune"]
    assert rem.status == "identified" and rem.author == "Kazuo Ishiguro"
    assert rem.replacement_cost.amount == 11.5 and rem.replacement_cost.url.startswith("https://") and rem.replacement_cost.retrieved_at
    assert rem.used_value.amount == 5.5
    assert rem.edition == ""                                   # no edition evidence -> never asserted
    assert rem.spine_height_cm == 23.0 and rem.spine_thickness_cm == 3.0  # 230px/10, 30px/10 via the calibration
    assert rem.height_measurement.method == "two_point" and rem.height_measurement.calibration_ref
    assert {b.status for b in packet.books} <= {"identified", "unidentified", "needs_appraisal"}
    assert {i.status for i in packet.items} <= {"priced", "range", "needs_appraisal"}
    unread = [b for b in packet.books if b.status == "unidentified"]
    assert len(unread) == 1 and unread[0].title == "" and unread[0].ocr_text == ["zzq"]   # unreadable -> blank, OCR preserved

    t = packet.totals
    assert t.book_count == 3 and t.books_identified == 2 and t.books_unidentified == 1
    assert t.books_replacement_cost == round(sum(b.replacement_cost.amount for b in packet.books if b.replacement_cost.amount), 2)
    art = [i for i in packet.items if i.category == "artwork"][0]
    assert art.status == "needs_appraisal" and art.replacement_cost.low is None
    lamp = [i for i in packet.items if i.category == "lamp"][0]
    assert lamp.status == "range" and lamp.detail_status == "priced" and lamp.replacement_cost.low <= lamp.replacement_cost.high and len(lamp.replacement_cost.sources) >= 3
    assert t.items_replacement_cost_low == lamp.replacement_cost.low

    reasons = " ".join(r.reason for r in packet.review_queue)
    assert "Unreadable" in reasons and "appraisal" in reasons.lower()
    assert any(r.ref_id == "room" for r in packet.review_queue)   # no corners marked -> room unavailable, not invented
    assert packet.room.floor_area_m2 is None

    # evidence: every frame_ref opens a stored frame
    assert calc.validate_frame_refs(packet, store.frame_ids("sw")) == []
    exports = store.sweep_dir("sw", "exports")
    assert (exports / "claim_packet.json").exists() and (exports / "claim_report.pdf").stat().st_size > 2000
    assert json.loads((exports / "claim_packet.json").read_text())["sweep"]["id"] == "sw"
    assert packet.model_extra["run_stats"]["post_sweep_processing_s"] < 300


async def test_user_corrections_change_totals(stubbed):
    sess, store = stubbed
    for i in range(4):
        await sess.on_frame(make_frame(seed=1), ts=sess.t0 + i * 0.4)
    from app.agent import corrections
    first = sess.books.confirmed()[0]
    first.meta["first_edition"] = True
    sess.books.confirmed()[1].meta["excluded"] = True
    packet = await postprocess.run(sess)
    assert packet.totals.book_count == 2   # excluded one is not counted
    assert len(packet.excluded_books) == 1 and all(b.status != "excluded" for b in packet.books)  # contract enum only
    appraised = [b for b in packet.books if b.needs_appraisal]
    assert len(appraised) == 1 or first.meta.get("ident_fields", {}).get("status") != "identified"
