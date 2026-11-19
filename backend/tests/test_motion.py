import cv2
import numpy as np

from app.vision.motion import MotionEstimator, homography_between, features, map_box
from app.vision.tracker import Tracker


def textured(w=1280, h=720, seed=1):
    rng = np.random.default_rng(seed)
    img = (rng.random((h // 8, w // 8, 3)) * 255).astype(np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    for i in range(200):
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(3, 20)), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def view(world, dx, scale=1.0, size=(1280, 720)):
    M = np.float32([[scale, 0, -dx * scale], [0, scale, 0]])
    return cv2.warpAffine(world, M, size)


def test_homography_recovers_translation_and_scale():
    world = textured(2400, 720)
    a, b = view(world, 0), view(world, 150, 1.25)
    r = homography_between(features(b), features(a))
    assert r is not None
    H, n = r
    # a point at the centre of b -> in a coords: x_a = x_b/1.25 + 150
    p = map_box(H, [630, 350, 650, 370])
    cx = (p[0] + p[2]) / 2
    assert abs(cx - (640 / 1.25 + 150)) < 6


def test_motion_estimator_same_segment_on_pan_new_segment_on_cut():
    world = textured(3000, 720)
    me = MotionEstimator()
    segs = [me.update(view(world, dx))[0] for dx in (0, 40, 80, 120)]
    assert segs == [0, 0, 0, 0]
    seg_cut = me.update(textured(1280, 720, seed=99))[0]
    assert seg_cut == 1


def test_tracker_counts_a_book_once_across_a_pan():
    world = textured(3000, 720)
    me = MotionEstimator(); tr = Tracker("book", min_hits=2)
    for i, dx in enumerate((0, 30, 60, 90, 120)):
        seg, H, _, _ = me.update(view(world, dx))
        # a physical book at world x=600..640 appears at x-dx in the frame
        tr.update([{"bbox": [600 - dx, 100, 640 - dx, 400], "conf": 0.9, "label": "book"}], seg, H, f"f{i}", (1280, 720), 1.0, float(i))
    assert len(tr.tracks) == 1 and tr.confirmed()[0].hits == 5
