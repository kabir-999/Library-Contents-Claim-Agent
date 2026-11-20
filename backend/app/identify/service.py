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
_sem = asyncio.Semaphore(8)

EXTRACT_SYS = ("You are given raw OCR text read from a book spine (may be noisy, words in any order). "
               "Return JSON {\"title\": str|null, \"author\": str|null}. Use ONLY words present in the text. "
               "If the text does not contain a recognisable title, return nulls. Do not guess or complete titles.")
PICK_SYS = ("OCR text from a book spine and numbered candidate books from metadata APIs. Pick the single candidate whose title (and "
            "author if visible) is supported by the OCR text, or none. Return JSON {\"index\": int|null, \"reason\": str}.")


def consensus_text(texts: list[str]) -> str:
    """Words that recur (fuzzily) across >=2 independent OCR reads of the same book. Noisy single-read words ('POOT',
    'AgatleChustic') are dropped, so the catalogue query isn't poisoned by garbage."""
    from collections import Counter
    from rapidfuzz import fuzz
    reads = [[w for w in re.findall(r"[A-Za-z']{3,}", t)] for t in texts if t]
    if len(reads) < 2:
        return " ".join(texts[:1])
    clusters: list[dict] = []
    for oi, ws in enumerate(reads):
        for pos, w in enumerate(ws):
            for c in clusters:
                if fuzz.ratio(w.lower(), c["rep"].lower()) >= 75:
                    c["forms"][w] += 1; c["obs"].add(oi); break
            else:
                clusters.append({"rep": w, "forms": Counter({w: 1}), "obs": {oi}, "first": (oi, pos)})
    keep = sorted((c for c in clusters if len(c["obs"]) >= 2), key=lambda c: c["first"])
    return " ".join(max(c["forms"].items(), key=lambda kv: (kv[1], kv[0][0].isupper() or kv[0][0].isalpha()))[0] for c in keep)


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


NOISE = {"the", "and", "of", "a", "an", "new", "edition", "bestseller", "bestselling", "masterpiece", "favourite", "favorite",
         "world", "worlds", "world's", "novel", "classic", "award", "winning", "winner", "penguin", "books", "bbc", "series"}


def query_variants(text: str, max_q: int = 6) -> list[str]:
    """Noisy spine/cover OCR (marketing lines + title + author) returns nothing as one long AND-query.
    Try the full text plus sliding word windows; the matcher still ranks every candidate against the full OCR."""
    toks = [w for w in text.split() if len(re.sub(r"\W", "", w)) >= 2]
    qs = [text]
    core = [w for w in toks if w.lower().strip("'.,") not in NOISE]
    for seq in (toks, core):
        for size in (4, 3):
            for i in range(0, max(1, len(seq) - size + 1)):
                q = " ".join(seq[i:i + size])
                if q not in qs and len(q) > 8:
                    qs.append(q)
    return qs[:max_q + 2]


async def _lookup_variants(text: str) -> list[metadata.BookCandidate]:
    res = await asyncio.gather(*[_lookup(q) for q in query_variants(text)], return_exceptions=True)
    seen, out = set(), []
    for r in res:
        for c in (r if isinstance(r, list) else []):
            k = (c.title.lower(), tuple(a.lower() for a in c.authors[:1]))
            if k not in seen:
                seen.add(k); out.append(c)
    return out


async def identify(ocr_text: str, ocr_conf: float, *, use_llm: bool = True, evidence: str = "") -> dict:
    """`ocr_text` drives the catalogue QUERY (keep it clean); `evidence` (all raw reads of this book, noisy) is what candidates
    are SCORED against, so a garbled author line still counts as an author hit."""
    s = get_settings()
    text = clean_ocr(ocr_text)
    ev = clean_ocr(evidence) if evidence else text
    if len(re.sub(r"\W", "", text)) < 4:
        return {"status": "unidentified", "confidence": 0.0, "flags": [], "chosen": None, "alternatives": [], "llm_used": False,
                "ocr_text": ocr_text}
    cands = await _lookup_variants(text)              # full text + word windows, all in parallel (latency = slowest, not sum)
    # author-first filtering: if an author's name is in the text, only that author's catalogue competes, and the
    # rest of the text is fuzzy-matched against their real titles (handles garbled titles like "BC MURDERS")
    authors: list[str] = []
    for m in matcher.rank(ev, ocr_conf, cands)[:6]:
        if m.author_hit and m.cand.authors and m.cand.authors[0] not in authors:
            authors.append(m.cand.authors[0])
    if authors:
        from rapidfuzz import fuzz
        more = await metadata.works_by_author(authors[0])
        pool = cands + more
        same = [c for c in pool if any(fuzz.token_set_ratio(a.lower(), authors[0].lower()) >= 80 for a in c.authors)]
        cands = same or pool
    d = matcher.decide(ev, ocr_conf, cands, s.lca_id_confidence_min)
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
