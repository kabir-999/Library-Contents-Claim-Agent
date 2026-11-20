import numpy as np
from app.identify.service import query_variants
from app.vision.tracker import Tracker


def test_query_variants_has_short_windows():
    qs = query_variants("AGATHA CHRISTIE'S MASTERPIECE' AND THE WORLD'S FAVOURITE AGATHA CHRISTIE THEN THERE WERE NONE")
    assert len(qs) > 3 and any("THEN THERE" in q for q in qs)


def _img():
    im = np.zeros((600, 800, 3), np.uint8)
    im[100:500, 250:550] = (30, 40, 200)       # one big red 'book'
    return im


def test_closeup_book_leaving_and_returning_is_one_track():
    tr, im = Tracker("book", min_hits=1), _img()
    d = [{"bbox": [250, 100, 550, 500], "conf": 0.8, "label": "book"}]
    I = np.eye(3)
    tr.update(d, 0, I, "f1", (800, 600), 1.0, 0.0, img=im)
    # book returns far away in the mosaic (registration moved) a few seconds later
    I2 = np.array([[1, 0, 900], [0, 1, 0], [0, 0, 1]], float)
    tr.update(d, 0, I2, "f2", (800, 600), 1.0, 5.0, img=im)
    assert len([t for t in tr.tracks.values() if not t.removed]) == 1


def test_same_lamp_with_changing_label_is_one_track():
    tr = Tracker("item", min_hits=1, iou_thr=0.4)
    im = np.zeros((600, 800, 3), np.uint8); im[100:500, 300:420] = (20, 200, 220)
    I = np.eye(3)
    for i, (lab, box) in enumerate([("lamp", [300, 100, 420, 500]), ("floor lamp", [310, 110, 430, 490]),
                                    ("lamp", [330, 90, 450, 520]), ("floor lamp", [300, 100, 420, 500])]):
        tr.update([{"bbox": box, "conf": 0.6, "label": lab, "category": "lamp"}], 0, I, f"f{i}", (800, 600), 1.0, i * 0.3, img=im)
    assert len([t for t in tr.tracks.values() if not t.removed]) == 1


def test_llm_budget_limits_live_calls():
    from app.agent.session import SweepSession
    s = SweepSession("sw_budget", "US", "USD", "desktop")
    assert sum(s._llm_ok("x") for _ in range(20)) == 6      # live: max 6 per minute
    s._llm_log.clear(); s.finalizing = True
    assert all(s._llm_ok("x") for _ in range(20))            # end-of-sweep pass has its own allowance


def test_keyframe_selection_drops_blur_and_repeats():
    import numpy as np
    from types import SimpleNamespace as NS
    from app.pipeline.batch import select_keyframes
    still, moved = np.full((36, 64), 100, np.uint8), np.full((36, 64), 140, np.uint8)
    sharp = lambda: NS(blurred=False, score=0.9)
    blur = lambda: NS(blurred=True, score=0.1)
    preps = [(sharp(), still)] * 150 + [(blur(), moved)] * 20 + [(sharp(), moved)] * 150
    keep = select_keyframes([{}] * len(preps), preps)
    assert 30 <= len(keep) < 60                     # 320 frames -> a few dozen distinct keyframes (floor 30)
    assert sum(preps[i][0].blurred for i in keep) <= 2   # a long blurry stretch keeps at most a couple of frames (gap guard)
