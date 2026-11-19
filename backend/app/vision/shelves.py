"""Group tracked books into shelves (per wall segment) and order them left->right."""
from __future__ import annotations

import statistics
from typing import Iterable

from .tracker import Track


def assign_shelves(tracks: Iterable[Track]) -> dict[str, tuple[str, int]]:
    """Returns {track_id: (shelf_name, position)}. Shelf names: W<segment+1>-S<row from top>."""
    by_seg: dict[int, list[Track]] = {}
    for t in tracks:
        by_seg.setdefault(t.segment, []).append(t)
    out: dict[str, tuple[str, int]] = {}
    for seg, ts in sorted(by_seg.items()):
        hs = [t.mbox[3] - t.mbox[1] for t in ts if t.orientation == "upright"] or [t.mbox[3] - t.mbox[1] for t in ts]
        gap = 0.5 * statistics.median(hs)
        ts_sorted = sorted(ts, key=lambda t: (t.mbox[1] + t.mbox[3]) / 2)
        rows: list[list[Track]] = []
        for t in ts_sorted:
            yc = (t.mbox[1] + t.mbox[3]) / 2
            if rows:
                mean = sum((r.mbox[1] + r.mbox[3]) / 2 for r in rows[-1]) / len(rows[-1])
                if abs(yc - mean) <= gap:
                    rows[-1].append(t); continue
            rows.append([t])
        for k, row in enumerate(rows, 1):
            for pos, t in enumerate(sorted(row, key=lambda t: (t.mbox[0] + t.mbox[2]) / 2), 1):
                out[t.tid] = (f"W{seg + 1}-S{k}", pos)
    return out


def shelf_rows(tracks: Iterable[Track]) -> list[dict]:
    """Per-shelf mean y (mosaic coords) + count, used for missed-shelf detection."""
    a = assign_shelves(tracks)
    tr = {t.tid: t for t in tracks}
    agg: dict[str, list] = {}
    for tid, (shelf, _) in a.items():
        t = tr[tid]; agg.setdefault(shelf, []).append(((t.mbox[1] + t.mbox[3]) / 2, t.segment))
    return [{"shelf": s, "segment": v[0][1], "y": sum(y for y, _ in v) / len(v), "count": len(v)} for s, v in agg.items()]
