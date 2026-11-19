from app.agent.guidance import GuidanceState, evaluate

BASE = dict(elapsed_s=30, book_boxes=[(30, 200)] * 5, frame_h=720, clipped_books=0, unreadable_tracks=[], shelf_rows=[], room_status={})


def run(st, q, t, **kw):
    a = {**BASE, **kw}
    return evaluate(st, t, quality=q, **a)


def test_blur_guidance_after_sustained_blur():
    st = GuidanceState(); out = None
    for i in range(6):
        out = run(st, {"blurred": True}, 100 + i * 0.5) or out
    assert out and out.code == "blur"


def test_no_guidance_on_good_frames():
    st = GuidanceState()
    assert all(run(st, {}, 100 + i) is None for i in range(8))


def test_rate_limit_and_no_repeat():
    st = GuidanceState(); msgs = []
    for i in range(30):
        g = run(st, {"blurred": True}, 100 + i)
        if g: msgs.append((100 + i, g.code))
    assert len(msgs) >= 1 and all(b[0] - a[0] >= 6 for a, b in zip(msgs, msgs[1:]))
    assert len(msgs) <= 2  # 30 s => repeat gap 25 s


def test_move_closer_and_missed_shelf_and_recapture():
    st = GuidanceState()
    for i in range(4): run(st, {}, 100 + i * 0.1)
    g = run(st, {}, 200, book_boxes=[(10, 60)] * 6)
    assert g.code == "closer"
    rows = [{"segment": 0, "y": 100, "count": 8}, {"segment": 0, "y": 400, "count": 8}, {"segment": 0, "y": 1000, "count": 8}]
    g = run(st, {}, 300, shelf_rows=rows)
    assert g.code == "missed_shelf" and g.recapture
    g = run(st, {}, 400, unreadable_tracks=[{"shelf": "S2"}] * 5)
    assert g.code == "recapture"


def test_silent_while_user_speaks():
    st = GuidanceState()
    assert evaluate(st, 1, quality={"blurred": True}, speaking_user=True, **{k: v for k, v in BASE.items()}) is None


def test_shelf_specific_wording():
    from app.agent.guidance import humanize_shelf
    assert humanize_shelf("W1-S1") == "top shelf" and humanize_shelf("W1-S2") == "second shelf from the top"
    assert humanize_shelf("W2-S3") == "third shelf from the top on wall 2" and humanize_shelf("junk") == "shelf"
    st = GuidanceState()
    for i in range(4): run(st, {}, 100 + i * 0.1)
    g = run(st, {}, 200, book_boxes=[(10, 60)] * 6, focus_shelf="W1-S2")
    assert g.code == "closer" and "second shelf from the top" in g.text
