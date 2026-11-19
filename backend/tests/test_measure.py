import math
import cv2
import numpy as np
import pytest

from app.measure import calibration as C
from app.measure.room import CornerObs, estimate_room, floor_point, polygon_area
from app.measure.spine import measure_spine


def test_two_point_scale():
    cal = C.from_two_points("c1", (100, 100), (100, 500), 40.0, "f1")
    assert cal.px_per_cm == pytest.approx(10.0)
    assert cal.length_cm((0, 0), (0, 250)) == pytest.approx(25.0)


def test_two_point_rejects_short_baseline():
    with pytest.raises(ValueError):
        C.from_two_points("c", (0, 0), (5, 0), 10, "f")


def synth_a4(angle=0, scale=8.0):
    img = np.full((720, 1280, 3), 60, np.uint8)
    w, h = int(29.7 * scale), int(21.0 * scale)
    sheet = np.full((h, w, 3), 245, np.uint8)
    cv2.putText(sheet, "x", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img[200:200 + h, 300:300 + w] = sheet
    return img, scale


def test_reference_detection_and_homography_scale():
    img, scale = synth_a4()
    cal = C.from_reference("c", img, "f")
    assert cal is not None and cal.method == "reference_object"
    # 10 cm horizontally should be ~ 10*scale px => length_cm of 80px ≈ 10cm
    assert cal.length_cm((400, 300), (480, 300)) == pytest.approx(10.0, rel=0.03)


def test_reference_detection_rejects_blank():
    assert C.from_reference("c", np.full((480, 640, 3), 80, np.uint8), "f") is None


def test_spine_measure_upright_and_refuses_truncated():
    cal = C.from_two_points("c", (0, 0), (0, 300), 30.0, "f")  # 10 px/cm
    h, t, fl = measure_spine((100, 100, 130, 330), "upright", cal, "f", (1280, 720))
    assert h.value == 23.0 and t.value == 3.0 and h.calibration_ref == "c" and h.method == "two_point"
    h2, t2, fl2 = measure_spine((100, 0, 130, 330), "upright", cal, "f", (1280, 720))
    assert h2.value is None and "truncated_by_frame" in fl2


def test_spine_no_calibration_is_blank():
    h, t, fl = measure_spine((100, 100, 130, 330), "upright", None, "f", (1280, 720))
    assert h.value is None and t.value is None and "no_calibration" in fl


def test_spine_horizontal_swaps_axes():
    cal = C.from_two_points("c", (0, 0), (0, 300), 30.0, "f")
    h, t, _ = measure_spine((100, 100, 330, 130), "horizontal_stack", cal, "f", (1280, 720))
    assert h.value == 23.0 and t.value == 3.0


def test_floor_point_geometry():
    d = floor_point(1.5, CornerObs(0, 45))[2]
    assert d == pytest.approx(1.5)


def make_corners(cam_h, pts):
    obs = []
    for x, y in pts:
        d = math.hypot(x, y)
        obs.append(CornerObs(math.degrees(math.atan2(x, y)) % 360, math.degrees(math.atan(cam_h / d))))
    return obs


def test_room_rectangle_recovery():
    # 5 x 4 m room, user at (2.5, 2.0) offset from corners
    cam_h = 1.5
    corners = [(-2.5, -2.0), (2.5, -2.0), (2.5, 2.0), (-2.5, 2.0)]
    r = estimate_room(cam_h, make_corners(cam_h, corners))
    assert r["floor_area_m2"] == pytest.approx(20.0, rel=0.01)
    assert r["length_m"] == pytest.approx(5.0, rel=0.02) and r["width_m"] == pytest.approx(4.0, rel=0.02)
    assert r["shape"] == "rectangular" and r["height_m"] is None and r["wall_area_m2"] is None


def test_room_ceiling_and_wall_area():
    cam_h = 1.5
    corners = [(-2.5, -2.0), (2.5, -2.0), (2.5, 2.0), (-2.5, 2.0)]
    fl = make_corners(cam_h, corners)
    ceil = []
    for f, (x, y) in zip(fl, corners):
        d = math.hypot(x, y)
        up = math.degrees(math.atan((2.6 - cam_h) / d))
        ceil.append((f, CornerObs(f.heading_deg, -up)))
    r = estimate_room(cam_h, fl, ceil, shelved_units=[(1.2, 2.0), (0.8, 2.0)])
    assert r["height_m"] == pytest.approx(2.6, rel=0.01)
    assert r["wall_area_m2"] == pytest.approx(18 * 2.6, rel=0.02)
    assert r["shelved_wall_area_m2"] == 4.0


def test_room_l_shape_is_nonrect():
    cam_h = 1.5
    L = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
    cx, cy = 1.2, 1.0
    r = estimate_room(cam_h, make_corners(cam_h, [(x - cx, y - cy) for x, y in L]))
    assert r["shape"] == "non-rectangular"
    assert r["floor_area_m2"] == pytest.approx(12.0, rel=0.02)


def test_room_unavailable_without_camera_height_or_corners():
    assert estimate_room(None, [])["status"] == "unavailable"
    r = estimate_room(1.5, make_corners(1.5, [(1, 1), (2, 2)]))
    assert r["floor_area_m2"] is None


def test_shoelace():
    assert polygon_area([(0, 0), (4, 0), (4, 3), (0, 3)]) == 12


def test_measure_transfers_scale_across_frames_via_homography():
    # Calibrated in frame A at 10 px/cm. Frame B is closer: everything 2x bigger (H_B->A = scale 0.5).
    cal = C.from_two_points("c", (0, 0), (0, 300), 30.0, "A")
    H = np.diag([0.5, 0.5, 1.0])
    h, t, _ = measure_spine((200, 200, 260, 660), "upright", cal, "B", (1280, 720), h_to_cal=H)
    assert h.value == 23.0 and t.value == 3.0 and h.calibration_ref == "c"
