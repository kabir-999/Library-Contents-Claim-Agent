"""Tests the scoring LOGIC on a tiny synthetic fixture. These numbers say nothing about system accuracy."""
import json
from openpyxl import Workbook
from app.eval.evaluate import evaluate
from app.eval.ground_truth import build


def make(tmp_path, sys_h, price_sys):
    gt = tmp_path / "gt.xlsx"; build(gt)
    from openpyxl import load_workbook
    wb = load_workbook(gt)
    for r in [["b1", "A", 1, 1, "Y", "Dune", "Frank Herbert", ""], ["b2", "A", 1, 2, "N", "", "", ""]]:
        wb["Books"].append(r)
    wb["Spines_measured"].append(["Dune", 20.0, 3.0, "ruler"])
    wb["Prices_checked"].append(["Dune", "x", "replacement", 10.0, "USD", "new", "https://x", "2026-01-01"])
    wb["Room"].append(["floor_area", 20, "m2", "tape"]); wb["Room"].append(["true_book_count", 2, "books", "x"])
    wb["NonBook"].append(["n1", "lamp", "floor lamp", ""])
    wb.save(gt)
    packet = {"totals": {"book_count": 2}, "books": [
        {"id": "b", "title": "Dune", "status": "identified", "id_confidence": 0.9, "spine_height_cm": sys_h, "spine_thickness_cm": 3.1,
         "replacement_cost": {"amount": price_sys}, "used_value": {"amount": None}},
        {"id": "c", "title": "Totally Wrong Title", "status": "identified", "id_confidence": 0.95, "spine_height_cm": None, "spine_thickness_cm": None,
         "replacement_cost": {}, "used_value": {}}],
        "items": [{"category": "lamp"}], "room": {"floor_area_m2": 21.0, "wall_area_m2": None},
        "processing": [{"stage": "packet_generation_total", "seconds": 42}]}
    pk = tmp_path / "p.json"; pk.write_text(json.dumps(packet)); return gt, pk


def test_scoring_logic(tmp_path):
    gt, pk = make(tmp_path, 21.0, 11.0)
    r = evaluate(gt, pk)
    assert r["book_count"]["pass"] and r["floor_area"]["pass"] and r["packet_generation"]["pass"]
    assert r["confidently_wrong"]["wrong"] == 1 and not r["confidently_wrong"]["pass"]
    assert not r["wall_area"]["pass"]                       # blank system value is a failure
    assert not r["spine_dimensions"]["pass"]                # only 1 measured book (<20) -> cannot claim pass
    assert r["non_book_detection"]["found"] == 1 and not r["non_book_detection"]["pass"]  # <8 objects -> cannot claim pass
    gt, pk = make(tmp_path, 40.0, 11.0)
    assert make and evaluate(gt, pk)["spine_dimensions"]["within"] == 0
