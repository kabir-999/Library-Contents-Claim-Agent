"""Score a claim_packet.json against the hand-made ground truth workbook.

    python -m app.eval.evaluate data/eval/ground_truth.xlsx path/to/claim_packet.json [out_dir]

Honesty rules:
 * Pass labels are strict: PASS only if the stated bar is met; otherwise FAIL (with the partial numbers shown).
 * Blank system values (no scale / no price) count as failures for that row, never as "excluded".
 * "Confidently wrong" = system status 'identified' with id_confidence >= threshold whose title matches no ground-truth spine.
 * Difficult/unreadable books are NOT removed from any denominator.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

from openpyxl import load_workbook
from rapidfuzz import fuzz

from ..identify.matcher import norm

TITLE_MATCH = 85
CONF_WRONG_MIN = 0.80


def rows(ws) -> list[dict]:
    it = ws.iter_rows(values_only=True)
    head = [str(h).split(" (")[0].strip() if h else "" for h in next(it)]
    return [dict(zip(head, r)) for r in it if any(c not in (None, "") for c in r)]


def ape(sys_v: Optional[float], truth: float) -> Optional[float]:
    return None if sys_v is None or not truth else abs(sys_v - truth) / truth


def match_books(gt_titles: list[str], sys_books: list[dict]) -> dict[int, int]:
    """Greedy one-to-one by fuzzy title. Returns {gt_index: sys_index}."""
    pairs = []
    for gi, g in enumerate(gt_titles):
        for si, b in enumerate(sys_books):
            if b.get("title"):
                s = max(fuzz.token_set_ratio(norm(g), norm(b["title"])), fuzz.partial_ratio(norm(g), norm(b["title"].split(":")[0])) if len(norm(g)) > 6 else 0)
                if s >= TITLE_MATCH:
                    pairs.append((s, gi, si))
    pairs.sort(reverse=True); used_g, used_s, out = set(), set(), {}
    for s, gi, si in pairs:
        if gi not in used_g and si not in used_s:
            out[gi] = si; used_g.add(gi); used_s.add(si)
    return out


def evaluate(gt_path: Path, packet_path: Path) -> dict[str, Any]:
    wb = load_workbook(gt_path, data_only=True)
    P = json.loads(packet_path.read_text())
    books = P["books"]
    gt_books = [r for r in rows(wb["Books"]) if r.get("title")]
    legible = [r for r in gt_books if str(r.get("legible", "")).strip().upper().startswith("Y")]
    room_gt = {r["field"]: r["value"] for r in rows(wb["Room"]) if r.get("field") and r.get("value") not in (None, "")}
    res: dict[str, Any] = {}

    true_n = int(room_gt.get("true_book_count") or len(gt_books))
    sys_n = P["totals"]["book_count"]
    err = abs(sys_n - true_n) / true_n if true_n else None
    res["book_count"] = {"bar": "within 5% of true count", "true": true_n, "system": sys_n, "error": err, "pass": err is not None and err <= 0.05}

    m = match_books([r["title"] for r in legible], books)
    ok_ident = [gi for gi, si in m.items() if books[si]["status"] in ("identified", "needs_appraisal")]
    rate = len(m) / len(legible) if legible else None
    res["title_identification"] = {"bar": ">=70% of legible spines correct", "legible_spines": len(legible), "correct": len(m),
                                   "correct_and_marked_identified": len(ok_ident), "rate": rate, "pass": rate is not None and rate >= 0.70}
    gt_all_titles = [r["title"] for r in gt_books]
    wrong = [b for b in books if b["status"] == "identified" and b["id_confidence"] >= CONF_WRONG_MIN and
             not any(fuzz.token_set_ratio(norm(b["title"]), norm(g)) >= TITLE_MATCH for g in gt_all_titles)]
    wr = len(wrong) / len(legible) if legible else None
    res["confidently_wrong"] = {"bar": "<=3% (of legible spines)", "wrong": len(wrong), "rate": wr, "pass": wr is not None and wr <= 0.03,
                                "examples": [(b["id"], b["title"], b["id_confidence"]) for b in wrong[:10]]}

    sp = rows(wb["Spines_measured"]); det = []
    sm = match_books([r["title"] for r in sp], books)
    for i, r in enumerate(sp):
        b = books[sm[i]] if i in sm else None
        eh = ape(b["spine_height_cm"], float(r["height_cm"])) if b else None
        et = ape(b["spine_thickness_cm"], float(r["thickness_cm"])) if b and r.get("thickness_cm") else None
        det.append({"title": r["title"], "matched": b is not None, "height_ape": eh, "thickness_ape": et})
    within = sum(1 for d in det if d["height_ape"] is not None and d["height_ape"] <= 0.15 and (d["thickness_ape"] is None or d["thickness_ape"] <= 0.15))
    hv = [d["height_ape"] for d in det if d["height_ape"] is not None]
    res["spine_dimensions"] = {"bar": "within 15% on 20 measured books (every book, both dims; blank = fail)", "n": len(sp), "within": within,
                               "median_height_ape": statistics.median(hv) if hv else None, "blank_or_unmatched": sum(1 for d in det if d["height_ape"] is None),
                               "pass": len(sp) >= 20 and within == len(sp), "detail": det}

    pr = rows(wb["Prices_checked"]); pdet = []
    pm = match_books([r["title"] for r in pr], books)
    for i, r in enumerate(pr):
        b = books[pm[i]] if i in pm else None
        kind = str(r.get("kind", "replacement")).lower()
        sv = None
        if b:
            sv = (b["replacement_cost"] if kind.startswith("rep") else b["used_value"]).get("amount")
        pdet.append({"title": r["title"], "kind": kind, "true": r["price"], "system": sv, "ape": ape(sv, float(r["price"]))})
    pw = sum(1 for d in pdet if d["ape"] is not None and d["ape"] <= 0.25)
    res["book_prices"] = {"bar": "within 25% on 15 checked prices (every price; blank = fail)", "n": len(pr), "within": pw,
                          "pass": len(pr) >= 15 and pw == len(pr), "detail": pdet}

    nb = rows(wb["NonBook"]); items = list(P["items"]); found = 0; nbd = []
    for r in nb:
        hit = next((i for i in items if i["category"] == str(r["category"]).strip().lower()), None)
        if hit: items.remove(hit); found += 1
        nbd.append({"gt": r.get("description"), "category": r["category"], "found": hit is not None})
    nbr = found / len(nb) if nb else None
    res["non_book_detection"] = {"bar": ">=80% found with correct category (and >=8 objects)", "n": len(nb), "found": found, "rate": nbr,
                                 "pass": nbr is not None and len(nb) >= 8 and nbr >= 0.8, "detail": nbd}

    def area(name, field, tol):
        t = room_gt.get(field); s = P["room"].get({"floor_area": "floor_area_m2", "wall_area": "wall_area_m2"}[field])
        e = ape(s, float(t)) if t else None
        res[name] = {"bar": f"within {int(tol*100)}%", "true": t, "system": s, "error": e, "pass": e is not None and e <= tol}
    area("floor_area", "floor_area", 0.10); area("wall_area", "wall_area", 0.15)
    gen = next((p["seconds"] for p in P["processing"] if p["stage"] == "packet_generation_total"), None)
    res["packet_generation"] = {"bar": "< 300 s after sweep end", "seconds": gen, "pass": gen is not None and gen < 300}
    res["_summary"] = {"passed": [k for k, v in res.items() if not k.startswith("_") and v["pass"]],
                       "failed": [k for k, v in res.items() if not k.startswith("_") and not v["pass"]]}
    return res


def write_outputs(res: dict, out_dir: Path) -> None:
    from openpyxl import Workbook
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=str))
    wb = Workbook(); ws = wb.active; ws.title = "Summary"
    ws.append(["Metric", "Pass bar", "Result", "PASS/FAIL"])
    for k, v in res.items():
        if k.startswith("_"): continue
        shown = {kk: vv for kk, vv in v.items() if kk not in ("bar", "pass", "detail", "examples")}
        ws.append([k, v["bar"], json.dumps(shown, default=str), "PASS" if v["pass"] else "FAIL"])
    for name in ("spine_dimensions", "book_prices", "non_book_detection"):
        w2 = wb.create_sheet(name); d = res[name]["detail"]
        if d:
            w2.append(list(d[0].keys()))
            for r in d: w2.append(list(r.values()))
    wb.save(out_dir / "results.xlsx")
    lines = ["# Evaluation results", ""]
    for k, v in res.items():
        if k.startswith("_"): continue
        lines.append(f"- **{k}** — bar: {v['bar']} — **{'PASS' if v['pass'] else 'FAIL'}** — " + ", ".join(f"{kk}={vv}" for kk, vv in v.items() if kk not in ("bar", "pass", "detail", "examples")))
    (out_dir / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    gt, pk = Path(sys.argv[1]), Path(sys.argv[2]); out = Path(sys.argv[3] if len(sys.argv) > 3 else "data/eval/results")
    r = evaluate(gt, pk); write_outputs(r, out)
    print(json.dumps(r["_summary"], indent=2)); print("wrote", out)
