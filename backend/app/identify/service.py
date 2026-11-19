"""OCR text -> verified book identity. LLMs may *select among* evidence-supported
candidates or *split* OCR text into title/author, but acceptance is always
re-checked deterministically against the OCR text (see matcher.py)."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from ..config import get_settings
from ..llm.providers import router
from . import matcher, metadata

log = logging.getLogger(__name__)
_cache: dict[str, list[metadata.BookCandidate]] = {}
_sem = asyncio.Semaphore(4)

EXTRACT_SYS = ("You are given raw OCR text read from a book spine (may be noisy, words in any order). "
               "Return JSON {\"title\": str|null, \"author\": str|null}. Use ONLY words present in the text. "
               "If the text does not contain a recognisable title, return nulls. Do not guess or complete titles.")
PICK_SYS = ("OCR text from a book spine and numbered candidate books from metadata APIs. Pick the single candidate whose title (and "
            "author if visible) is supported by the OCR text, or none. Return JSON {\"index\": int|null, \"reason\": str}.")


def clean_ocr(text: str) -> str:
    t = re.sub(r"[^\w\s'&:.\-]", " ", text)
    return re.sub(r"\s+", " ", t).strip()


async def _lookup(q: str) -> list[metadata.BookCandidate]:
    if q in _cache:
        return _cache[q]
    async with _sem:
        c = await metadata.lookup(q)
    _cache[q] = c
    return c


async def identify(ocr_text: str, ocr_conf: float, *, use_llm: bool = True) -> dict:
    s = get_settings()
    text = clean_ocr(ocr_text)
    if len(re.sub(r"\W", "", text)) < 4:
        return {"status": "unidentified", "confidence": 0.0, "flags": [], "chosen": None, "alternatives": [], "llm_used": False,
                "ocr_text": ocr_text}
    cands = await _lookup(text)
    d = matcher.decide(text, ocr_conf, cands, s.lca_id_confidence_min)
    llm_used = False
    if use_llm and d["status"] != "identified":
        # 1) LLM splits noisy OCR into title/author -> new targeted query (validated against OCR)
        ex = await router().complete_json(EXTRACT_SYS, text, prefer="groq")
        if isinstance(ex, dict) and ex.get("title") and matcher.verify_llm_extraction(text, ex["title"], ex.get("author") or ""):
            llm_used = True
            q = f'intitle:"{ex["title"]}"' + (f' inauthor:"{ex["author"]}"' if ex.get("author") else "")
            more = await _lookup(q)
            d2 = matcher.decide(text, ocr_conf, cands + more, s.lca_id_confidence_min)
            if d2["confidence"] >= d["confidence"]:
                d, cands = d2, cands + more
        # 2) ambiguous: LLM may pick among candidates, but only if the OCR text supports the pick
        if d["status"] == "needs_review" and cands:
            ms = matcher.rank(text, ocr_conf, cands)[:5]
            listing = "\n".join(f"{i}: {m.cand.title} — {', '.join(m.cand.authors)} ({m.cand.publisher}, {m.cand.year})" for i, m in enumerate(ms))
            pk = await router().complete_json(PICK_SYS, f"OCR: {text}\nCandidates:\n{listing}", prefer="groq")
            if isinstance(pk, dict) and isinstance(pk.get("index"), int) and 0 <= pk["index"] < len(ms):
                m = ms[pk["index"]]
                llm_used = True
                if m.score >= 0.65:  # supported by OCR text -> keep matcher's score; LLM never inflates it
                    d["chosen"] = m
                    d["group"] = next((g for g in matcher.group_works(ms) if m in g), [m])
                    d["confidence"] = m.score
                    d["status"] = "identified" if m.score >= s.lca_id_confidence_min else "needs_review"
    d["llm_used"] = llm_used
    d["ocr_text"] = ocr_text
    return d


def apply_to_book_fields(d: dict) -> dict:
    """Flatten a decision into Book fields. Edition/ISBN only from evidence."""
    ch: Optional[matcher.Match] = d.get("chosen")
    out = {"status": d["status"], "id_confidence": round(float(d.get("confidence", 0.0)), 3), "flags": list(d.get("flags", [])),
           "title": "", "author": "", "publisher": "", "isbn": "", "edition": "", "candidates": d.get("alternatives", []),
           "verification": d.get("verification", {}), "group_isbns": []}
    if ch and d["status"] in ("identified", "needs_review"):
        out.update(title=ch.cand.title, author=", ".join(ch.cand.authors), publisher=ch.cand.publisher if ch.publisher_hit else "",
                   edition=d.get("edition", ""))
        out["isbn"] = matcher.pick_isbn(d.get("group", [ch]), "", d.get("ocr_text", ""))
        out["group_isbns"] = [(m.cand.isbn13, m.cand.year) for m in d.get("group", []) if m.cand.isbn13]
        if not out["publisher"] and out["isbn"]:
            out["publisher"] = next((m.cand.publisher for m in d["group"] if m.cand.isbn13 == out["isbn"]), "")
    elif ch:  # partial: keep nothing asserted, only preserve evidence
        out["candidates"] = d.get("alternatives", [])
    return out
