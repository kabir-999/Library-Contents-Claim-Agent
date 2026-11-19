from app.storage import Storage


def test_frames_entities_events_roundtrip(tmp_path):
    s = Storage(tmp_path)
    s.create_sweep("sw1", "US", "USD", "test", {})
    s.save_frame("sw1", "sw1_f00001", b"\xff\xd8x", 1.0, 0.0, 10, 10, {"score": 1}, None)
    assert "sw1_f00001" in s.frame_ids("sw1")
    s.put("sw1", "book", "b1", {"a": 1}); assert s.get("sw1", "book", "b1") == {"a": 1}
    assert s.list("sw1", "book") == [{"a": 1}]
    n = s.add_event("sw1", "x", {"m": 1}); assert s.events("sw1", n - 1)[0]["m"] == 1


def test_prune_keeps_evidence(tmp_path):
    s = Storage(tmp_path); s.create_sweep("s", "US", "USD", "t", {})
    for i in range(10):
        s.save_frame("s", f"s_f{i}", b"x", float(i), float(i), 1, 1, {}, None)
    s.mark_keep("s", ["s_f3"])
    s.prune_frames("s", keep_every_s=100)
    ids = s.frame_ids("s")
    assert "s_f3" in ids and len(ids) <= 3
