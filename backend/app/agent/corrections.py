"""Spoken-command / correction parsing.

Two layers: (1) deterministic regex grammar - fast, offline, testable;
(2) local Qwen as a fallback for phrasings the grammar misses, constrained to the
same closed action vocabulary and validated before use. The LLM cannot create
facts (prices, sizes) - it can only choose one of these actions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..llm.providers import router

ACTIONS = {"set_first_edition", "set_not_mine", "set_artwork_type", "skip_shelf", "set_camera_height",
           "mark_corner", "mark_ceiling", "known_shelf_dimension", "query_count", "query_status", "stop", "none"}


@dataclass
class Action:
    action: str
    target: str = "focus"      # focus (that/this) | group (those/shelf) | all
    value: Optional[str] = None
    number: Optional[float] = None
    unit: str = ""
    raw: str = ""
    source: str = "regex"


def _num_cm(text: str) -> tuple[Optional[float], str]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(cm|centimet(?:er|re)s?|m\b|met(?:er|re)s?|inch(?:es)?|in\b|feet|foot|ft)", text, re.I)
    if not m:
        return None, ""
    v, u = float(m.group(1)), m.group(2).lower()
    if u.startswith("cm") or u.startswith("centi"): return v, "cm"
    if u.startswith("m"): return v * 100, "cm"
    if u.startswith("in"): return v * 2.54, "cm"
    return v * 30.48, "cm"


def parse_regex(text: str) -> list[Action]:
    t = " " + re.sub(r"\s+", " ", text.lower().strip()) + " "
    out: list[Action] = []
    group = bool(re.search(r"\b(those|these|they|all (of )?the|this shelf|that shelf|the whole shelf|everything on)\b", t))
    tgt = "group" if group else "focus"

    if re.search(r"\b(done|finished|that'?s (all|it)|stop (the )?(sweep|recording)|end (the )?sweep|i'?m done)\b", t):
        out.append(Action("stop", raw=text))
    if re.search(r"first edition|1st edition|first printing", t):
        out.append(Action("set_first_edition", tgt, "first edition", raw=text))
    if re.search(r"(belongs? to|belonging to|owned by|is|are|'s) (someone else|somebody else|my \w+'?s?|not mine)|not mine|isn'?t mine|aren'?t mine|don'?t belong to me|"
                 r"(my|his|her|their) (wife|husband|partner|friend|brother|sister|son|daughter|roommate|flatmate|landlord)'?s", t):
        out.append(Action("set_not_mine", tgt, raw=text))
    if re.search(r"\b(print|reproduction|poster|copy)\b", t) and re.search(r"\b(not an? original|print|reproduction|poster)\b", t):
        out.append(Action("set_artwork_type", "focus", "print", raw=text))
    elif re.search(r"\b(original|one of a kind|hand[- ]painted)\b", t) and re.search(r"\b(painting|artwork|art|portrait|picture|drawing|it)\b", t):
        out.append(Action("set_artwork_type", "focus", "original", raw=text))
    if re.search(r"\b(skip|ignore|exclude|don'?t (count|include)) (this|that|the) shelf\b|\bskip (this|that|it)\b", t):
        out.append(Action("skip_shelf", "group", raw=text))
    if re.search(r"camera (height|is at)|phone (height|is at)|holding (it|the phone) at|height of (the |my )?(camera|phone)", t):
        v, u = _num_cm(t)
        if v:
            out.append(Action("set_camera_height", number=v, unit="cm", raw=text))
    m = re.search(r"(shelf|shelving unit|bookcase|bookshelf)( unit)? is (\d+(?:\.\d+)?\s*\w+) (tall|high|wide|long)", t)
    if m:
        v, _ = _num_cm(m.group(3))
        if v:
            out.append(Action("known_shelf_dimension", value="height" if m.group(4) in ("tall", "high") else "width", number=v, unit="cm", raw=text))
    if re.search(r"\bceiling (corner|junction|mark)|mark (the )?ceiling\b", t):
        out.append(Action("mark_ceiling", raw=text))
    elif re.search(r"\bmark (the |this )?(corner|floor)|this is (a |the )corner|corner (here|marked)|^ ?mark ?$", t):
        out.append(Action("mark_corner", raw=text))
    if re.search(r"how many books|book count|how many (have|did) you (count|find)", t):
        out.append(Action("query_count", raw=text))
    if re.search(r"\b(status|what do you see|what have you (got|found)|how'?s it going)\b", t):
        out.append(Action("query_status", raw=text))
    return out


LLM_SYSTEM = (
    "You convert a homeowner's spoken remark during a room inventory into ONE action. "
    "Allowed actions: set_first_edition, set_not_mine, set_artwork_type(value=print|original), skip_shelf, "
    "set_camera_height(number in cm), mark_corner, mark_ceiling, known_shelf_dimension(value=height|width, number in cm), "
    "query_count, query_status, stop, none. target is 'focus' (that/this one), 'group' (those/this shelf) or 'all'. "
    'Reply ONLY JSON: {"action": "...", "target": "...", "value": null, "number": null}. '
    "Use none if the remark is not one of these. Never invent numbers not spoken."
)


async def parse(text: str, use_llm: bool = True) -> list[Action]:
    acts = parse_regex(text)
    if acts or not use_llm:
        return acts
    d = await router().complete_json(LLM_SYSTEM, text, prefer="local")
    if isinstance(d, dict) and d.get("action") in ACTIONS and d["action"] != "none":
        num = d.get("number")
        # Guard: a number must literally be in the utterance
        if num is not None and not re.search(rf"\b{int(float(num)) if float(num).is_integer() else num}\b", text):
            num = None
        return [Action(d["action"], d.get("target") or "focus", d.get("value"), float(num) if num is not None else None,
                       "cm" if num is not None else "", text, "llm")]
    return []
