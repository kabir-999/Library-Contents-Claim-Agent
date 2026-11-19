"""Active camera guidance. Pure rules over recent frame quality + inventory state.

Returns at most one prioritised message per call, rate-limited so the agent
doesn't nag or talk over the user. Fully deterministic and unit-tested.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Guidance:
    code: str
    text: str
    priority: int = 5        # lower = more urgent
    recapture: Optional[dict] = None


@dataclass
class GuidanceState:
    quality: deque = field(default_factory=lambda: deque(maxlen=8))
    last_said: dict = field(default_factory=dict)  # code -> ts
    last_any: float = -1e9
    min_gap_s: float = 6.0
    repeat_gap_s: float = 25.0


def _frac(q, key) -> float:
    return sum(1 for x in q if x.get(key)) / max(1, len(q))


ORD = {1: "top", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}


def humanize_shelf(name: str) -> str:
    """'W2-S3' -> 'third shelf from the top on the second wall' (S index is top-down; W is the wall segment)."""
    try:
        w, s = name.split("-"); wi, si = int(w[1:]), int(s[1:])
    except Exception:
        return "shelf"
    shelf = "top shelf" if si == 1 else f"{ORD.get(si, str(si) + 'th')} shelf from the top"
    return shelf + ("" if wi == 1 else f" on wall {wi}")


def evaluate(st: GuidanceState, now: float, *, elapsed_s: float, quality: dict, book_boxes: list[tuple[float, float]],
             frame_h: int, clipped_books: int, unreadable_tracks: list[dict], shelf_rows: list[dict],
             room_status: dict, speaking_user: bool = False, focus_shelf: str = "") -> Optional[Guidance]:
    """book_boxes: [(w_px,h_px)] in the latest frame. unreadable_tracks: [{track,shelf,age_s}].
    shelf_rows: [{segment,y,count}] in mosaic coords for missed-shelf detection."""
    st.quality.append(quality)
    if speaking_user or now - st.last_any < st.min_gap_s:
        return None
    cands: list[Guidance] = []
    if len(st.quality) >= 4:
        if _frac(st.quality, "blurred") >= 0.6:
            cands.append(Guidance("blur", "You're moving too fast. Slow down and hold steady.", 1))
        if _frac(st.quality, "too_dark") >= 0.6:
            cands.append(Guidance("dark", "It's too dark to read the spines. Turn on a light or open the curtains.", 1))
        if _frac(st.quality, "glare") >= 0.6:
            cands.append(Guidance("glare", "There's glare on the shelf. Tilt the camera slightly to avoid the reflection.", 2))
        if _frac(st.quality, "too_bright") >= 0.6:
            cands.append(Guidance("bright", "The image is overexposed. Angle away from the window.", 2))
    if book_boxes and frame_h:
        med_h = sorted(h for _, h in book_boxes)[len(book_boxes) // 2]
        if med_h < 0.14 * frame_h and len(st.quality) >= 4:
            cands.append(Guidance("closer", f"Move closer to the {humanize_shelf(focus_shelf) if focus_shelf else 'shelf'} so the spines are larger.", 3))
    if clipped_books >= 3:
        cands.append(Guidance("clipped", f"Some books on the {humanize_shelf(focus_shelf) if focus_shelf else 'shelf'} are cut off at the edge of the frame. Pan a little further so they're fully in view.", 4))
    # missed shelf: unusually large vertical gap between consecutive shelf rows in the same wall segment
    by_seg: dict[int, list[dict]] = {}
    for r in shelf_rows:
        by_seg.setdefault(r["segment"], []).append(r)
    for seg, rows in by_seg.items():
        rows.sort(key=lambda r: r["y"])
        if len(rows) >= 2:
            gaps = [b["y"] - a["y"] for a, b in zip(rows[:-1], rows[1:])]
            med = min(gaps)  # shelf pitch: a skipped shelf shows up as ~2x the smallest gap
            for g, (a, b) in zip(gaps, zip(rows[:-1], rows[1:])):
                if len(gaps) >= 2 and g > 1.7 * med and med > 0:
                    cands.append(Guidance("missed_shelf", "I think I missed a shelf between those two rows. Could you tilt back and show the one in the middle?", 3,
                                          {"segment": seg, "y": (a["y"] + b["y"]) / 2, "reason": "gap between shelf rows"}))
                    break
    if unreadable_tracks and len(unreadable_tracks) >= 4:
        t = unreadable_tracks[0]
        cands.append(Guidance("recapture", f"I can't read {len(unreadable_tracks)} spines on the {humanize_shelf(t.get('shelf') or '')}. Please go back and hold the camera closer and steadier on it.", 3,
                              {"shelf": t.get("shelf"), "reason": "unreadable spines", "n": len(unreadable_tracks)}))
    if elapsed_s > 20 and room_status.get("need_camera_height"):
        cands.append(Guidance("camera_height", "To measure the room I need your phone's height above the floor. Tell me, for example, 'camera height is 150 centimetres'.", 6))
    elif elapsed_s > 90 and room_status.get("corners_marked", 0) < 3 and room_status.get("orientation_available", True):
        cands.append(Guidance("corners", "For the room measurement, point at the floor where each wall meets the floor in each corner and say 'mark corner'.", 6))
    if elapsed_s >= 165:
        cands.append(Guidance("wrap_up", "We're near the three-minute mark. Finish the shelf you're on, then say 'done' or press Stop.", 2))
    if not cands:
        return None
    cands.sort(key=lambda g: g.priority)
    for g in cands:
        if now - st.last_said.get(g.code, -1e9) >= st.repeat_gap_s * (4 if g.code in ("wrap_up", "corners", "camera_height") else 1):
            st.last_said[g.code] = now; st.last_any = now
            return g
    return None
