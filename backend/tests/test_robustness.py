"""Edge cases: bad input, partial failures, hostile ids. Each asserts the system degrades (blank + review), never crashes or invents."""
import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.pipeline import books_stage
from app.pricing import book_pricing
from app.schemas import Book
from app.storage import Storage
from app.vision.tracker import Track


def test_storage_rejects_path_traversal(tmp_path):
    s = Storage(tmp_path)
    for bad in ["../x", "a/b", "", "x" * 200, "..", "a b"]:
        with pytest.raises(ValueError):
            s.sweep_dir(bad)


def test_one_pricing_failure_does_not_kill_the_stage(monkeypatch):
    async def boom(**kw):
        if kw["title"] == "Bad":
            raise RuntimeError("api exploded")
        return {"replacement": book_pricing.ReplacementCost(amount=5.0, url="https://x", retrieved_at="t", currency="USD"),
                "used": book_pricing.UsedValue(currency="USD"), "priced_isbn": "", "note": "", "over_threshold": False}
    monkeypatch.setattr(book_pricing, "price_book", boom)
    import asyncio
    class S: country, currency = "US", "USD"
    bad = Book(id="b1", title="Bad", status="identified"); good = Book(id="b2", title="Good", status="identified")
    asyncio.run(books_stage.price_books(S(), [bad, good], asyncio.Semaphore(2)))
    assert bad.replacement_cost.amount is None and "failed" in bad.pricing_note
    assert good.replacement_cost.amount == 5.0


def test_identification_error_leaves_book_unidentified():
    import asyncio
    class S:
        async def process_track(self, tid): raise RuntimeError("ocr crashed")
    t = Track(tid="b", kind="book", segment=0, mbox=[0, 0, 1, 1])
    asyncio.run(books_stage.finish_identification(S(), [t], deadline=1e18))
    assert t.meta.get("ident_error")


def test_same_title_on_different_walls_flagged_not_merged():
    a = Track(tid="a", kind="book", segment=0, mbox=[0, 0, 10, 100]); b = Track(tid="b", kind="book", segment=1, mbox=[0, 0, 10, 100])
    for t in (a, b): t.meta["ident_fields"] = {"title": "Dune"}
    merged = books_stage.detect_duplicates(None, [a, b])
    assert merged == [] and not a.removed and not b.removed
    assert "possible_duplicate" in a.flags and "possible_duplicate" in b.flags


def test_overlapping_same_place_tracks_are_merged():
    a = Track(tid="a", kind="book", segment=0, mbox=[0, 0, 10, 100], best_score=2); b = Track(tid="b", kind="book", segment=0, mbox=[1, 0, 11, 100], best_score=1)
    assert books_stage.detect_duplicates(None, [a, b]) == [("a", "b")] and b.removed


@pytest.fixture
def client(tmp_path, monkeypatch):
    st = Storage(tmp_path)
    import app.main as m, app.agent.session as sess
    monkeypatch.setattr(m, "storage", lambda: st); monkeypatch.setattr(sess, "storage", lambda: st)
    return TestClient(m.app), st


def test_ws_survives_garbage_and_bad_fields(client):
    c, st = client
    sid = c.post("/api/sweeps", json={"country": "US", "currency": "USD"}).json()["sweep_id"]
    with c.websocket_connect(f"/ws/sweeps/{sid}") as ws:
        for raw in ["not json", "[]", "{}", json.dumps({"type": "frame"}), json.dumps({"type": "frame", "data": "!!!"}),
                    json.dumps({"type": "set_camera_height", "cm": "abc"}), json.dumps({"type": "set_camera_height", "cm": 5}),
                    json.dumps({"type": "text"}), json.dumps({"type": "utterance", "pcm16_b64": "AAAA", "sample_rate": "x"}),
                    json.dumps({"type": "unknown_type"})]:
            ws.send_text(raw)
        ws.send_text(json.dumps({"type": "text", "text": "how many books so far"}))
        seen = set()
        for _ in range(12):
            seen.add(ws.receive_json()["type"])
            if "correction" in seen: break
        assert "correction" in seen   # still alive and handling valid messages after the garbage


def test_undecodable_frame_ignored_and_oversized_frame_downscaled(client):
    import asyncio
    from app.agent.session import SweepSession
    c, st = client
    st.create_sweep("sw", "US", "USD", "t", {})
    s = SweepSession("sw", "US", "USD", "t", store=st)
    assert asyncio.run(s.on_frame(b"definitely not a jpeg")) is None and s.frame_n == 0
    big = cv2.imencode(".jpg", np.random.default_rng(0).integers(0, 255, (2400, 3200, 3), dtype=np.uint8))[1].tobytes()
    asyncio.run(s.on_frame(big))
    fr = st.get_frame("sw_f00001")
    assert max(fr["w"], fr["h"]) <= 1920


def test_finish_endpoint_idempotent_and_unknown_session(client):
    c, st = client
    assert c.post("/api/sweeps/nope/finish").status_code == 409
    assert c.get("/api/sweeps/..%2Fetc/packet").status_code in (404, 422)


def test_nan_orientation_rejected():
    from app.agent.session import SweepSession
    import tempfile, pathlib
    st = Storage(pathlib.Path(tempfile.mkdtemp())); st.create_sweep("s", "US", "USD", "t", {})
    s = SweepSession("s", "US", "USD", "t", store=st)
    s.last_orientation = {"pitch_down": float("nan"), "heading": 10.0}
    assert s.add_floor_mark() == "no_orientation" and s.floor_obs == []


async def test_narration_is_batched_and_silent_while_user_speaks():
    from app.agent.session import SweepSession
    import tempfile, pathlib
    st = Storage(pathlib.Path(tempfile.mkdtemp())); st.create_sweep("s", "US", "USD", "t", {})
    s = SweepSession("s", "US", "USD", "t", store=st); said = []
    async def cb(t): said.append(t)
    s.narrate_cb = cb
    s.user_speaking = True; await s._narrate("A"); assert said == []          # never talks over the user
    s.user_speaking = False; await s._narrate("Dune: Deluxe"); assert said == ["Logged 2 more books, the latest is Dune."]
    await s._narrate("C"); assert len(said) == 1                              # rate-limited (15 s)
