from app.vision.shelves import assign_shelves
from app.vision.tracker import Track


def T(i, x, y, seg=0, h=100):
    return Track(tid=f"b{i}", kind="book", segment=seg, mbox=[x, y, x + 30, y + h], orientation="upright")


def test_rows_and_positions():
    ts = [T(1, 100, 0), T(2, 10, 5), T(3, 60, 2), T(4, 20, 300), T(5, 80, 305), T(6, 5, 0, seg=1)]
    a = assign_shelves(ts)
    assert a["b2"] == ("W1-S1", 1) and a["b3"] == ("W1-S1", 2) and a["b1"] == ("W1-S1", 3)
    assert a["b4"][0] == "W1-S2" and a["b5"] == ("W1-S2", 2)
    assert a["b6"] == ("W2-S1", 1)


def test_occlusion_flags_only_much_shorter_spines():
    from app.pipeline.books_stage import flag_occlusion
    ts = [T(i, i * 40, 0, h=200) for i in range(1, 7)] + [T(9, 300, 90, h=90)]   # last one visible 45% of shelf height
    sh = assign_shelves(ts)
    assert flag_occlusion(ts, sh) == 1
    assert "occluded_or_clipped" in ts[-1].flags and not any("occluded_or_clipped" in t.flags for t in ts[:-1])
    assert flag_occlusion(ts[:3], assign_shelves(ts[:3])) == 0     # too few neighbours: no verdict
