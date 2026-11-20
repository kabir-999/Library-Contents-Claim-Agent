"""Verify OCR text against metadata candidates. Deterministic, evidence-bound.

Principle: a book is only 'identified' if the *OCR text itself* supports the
candidate's title. No candidate, LLM output or memory can raise confidence above
what the printed spine text justifies.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz

from .metadata import BookCandidate

STOP = {"the", "a", "an", "of", "and", "to", "in", "on", "for", "by", "with"}
EDITION_RE = re.compile(
    r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|revised|annotated|illustrated|anniversary|"
    r"limited|signed|collector'?s?|special)\s+(edition|ed\.?|printing)\b|\b(\d{1,2})(st|nd|rd|th)\s+ed(ition|\.)\b", re.I)
SIGNED_RE = re.compile(r"\bsigned|autograph|inscribed\b", re.I)
FIRST_ED_RE = re.compile(r"\b(first|1st)\s+(edition|ed\.?|printing)\b", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def toks(s: str) -> list[str]:
    return [t for t in norm(s).split() if t not in STOP and len(t) > 1]


def main_title(t: str) -> str:
    return re.split(r"[:–—-]\s", t, 1)[0]


@dataclass
class Match:
    cand: BookCandidate
    score: float
    title_sim: float
    author_hit: bool
    publisher_hit: bool
    coverage: float
    reasons: list[str] = field(default_factory=list)


def score_candidate(ocr_text: str, ocr_conf: float, c: BookCandidate) -> Match:
    o = norm(ocr_text)
    ot = set(toks(ocr_text))
    ct = toks(main_title(c.title))
    if not ct or not o:
        return Match(c, 0, 0, False, False, 0)
    coverage = sum(1 for t in ct if any(t == x or fuzz.ratio(t, x) >= 85 for x in ot)) / len(ct)
    sq_t, sq_o = norm(main_title(c.title)).replace(' ', ''), o.replace(' ', '')
    sq = fuzz.partial_ratio(sq_t, sq_o) / 100 if len(sq_t) >= 6 else 0.0   # 'A.B.C. Murders' vs OCR 'BC MURDERS'
    sim = max(sq * 0.97, fuzz.partial_ratio(norm(main_title(c.title)), o) / 100, fuzz.token_set_ratio(norm(main_title(c.title)), o) / 100 * coverage)
    surnames = [toks(a)[-1] for a in c.authors if toks(a)]
    author_hit = any(any(s == x or fuzz.ratio(s, x) >= 88 for x in ot) for s in surnames)
    if not author_hit:   # OCR often glues/garbles the author line ("AgalleChristic"): fuzzy-search the surname in the unspaced text
        flat = o.replace(" ", "")
        author_hit = any(len(s) >= 5 and fuzz.partial_ratio(s, flat) >= 85 for s in surnames)
    pub = toks(c.publisher)
    publisher_hit = bool(pub) and any(any(p == x for x in ot) for p in pub[:2])
    # Short/generic titles (<=1 content token) need an author hit, otherwise capped.
    reasons = []
    score = 0.62 * sim * (0.5 + 0.5 * coverage) + 0.28 * (1.0 if author_hit else 0.0) + 0.05 * (1.0 if publisher_hit else 0.0) + 0.05 * ocr_conf
    if author_hit and coverage >= 0.75 and len(ct) >= 3:
        score = max(score, 0.80)   # author found + most of a distinctive multi-word title found
    if not author_hit and coverage >= 0.99 and len(ct) >= 4 and sim >= 0.9:
        score = max(score, 0.80); reasons.append("author not visible; long distinctive title matched exactly")
    if len(ct) <= 1 and not author_hit:
        score = min(score, 0.55); reasons.append("single-word title without author corroboration")
    if coverage < 0.6:
        score = min(score, 0.5); reasons.append("title words not found in OCR text")
    return Match(c, round(min(score, 0.99), 3), round(sim, 3), author_hit, publisher_hit, round(coverage, 3), reasons)


def rank(ocr_text: str, ocr_conf: float, cands: list[BookCandidate]) -> list[Match]:
    ms = [score_candidate(ocr_text, ocr_conf, c) for c in cands]
    ms.sort(key=lambda m: -m.score)
    return ms


def group_works(matches: list[Match]) -> list[list[Match]]:
    """Group candidates that are the same work (similar title+author) i.e. different editions."""
    groups: list[list[Match]] = []
    for m in matches:
        for g in groups:
            g0 = g[0].cand
            a_t, b_t = norm(main_title(g0.title)), norm(main_title(m.cand.title))
            if a_t == b_t:   # same title = same work, whatever the author field says ("Christie, Agatha", adaptations, ...)
                g.append(m); break
            if (fuzz.token_set_ratio(a_t, b_t) >= 92 or (min(len(a_t), len(b_t)) >= 8 and (a_t in b_t or b_t in a_t))) and \
               (not g0.authors or not m.cand.authors or fuzz.token_set_ratio(norm(g0.authors[0]), norm(m.cand.authors[0])) >= 80):
                g.append(m); break
        else:
            groups.append([m])
    return groups


def edition_evidence(ocr_text: str) -> str:
    m = EDITION_RE.search(ocr_text or "")
    return m.group(0).strip() if m else ""


def decide(ocr_text: str, ocr_conf: float, cands: list[BookCandidate], conf_min: float = 0.80) -> dict:
    """Deterministic decision. Returns dict with status, chosen candidate/group, confidence, flags."""
    if not norm(ocr_text):
        return {"status": "unidentified", "confidence": 0.0, "flags": [], "chosen": None, "alternatives": []}
    ms = rank(ocr_text, ocr_conf, cands)
    if not ms or ms[0].score < 0.5:
        return {"status": "unidentified" if len(norm(ocr_text)) < 6 else "partial", "confidence": ms[0].score if ms else 0.0,
                "flags": [], "chosen": None, "alternatives": [m.cand.to_dict() for m in ms[:3]]}
    top = ms[0]
    groups = group_works(ms)
    group = next(g for g in groups if top in g)
    flags = []
    # competing *different works* with a close score => ambiguous
    other_works = [g[0] for g in groups if g is not group]
    ambiguous = bool(other_works) and other_works[0].score >= top.score - 0.08
    distinct_isbn = {m.cand.isbn13 for m in group if m.cand.isbn13}
    ed = edition_evidence(ocr_text)
    if len(distinct_isbn) > 1 and not ed:
        flags.append("multiple_editions")
    if SIGNED_RE.search(ocr_text): flags.append("signed")
    if FIRST_ED_RE.search(ocr_text): flags.append("first_edition_claimed")
    runner = other_works[0] if other_works else None
    runner_group = next((g for g in groups if g[0] is runner), None) if runner else None
    both_strong = bool(runner) and all(m.score >= conf_min and m.author_hit and m.coverage >= 0.75 for m in (top, runner))
    if both_strong:
        # tilted/overlapping spines: one crop legitimately contains two different books. Report both (the second is
        # registered by the caller as its own entry), not an ambiguity.
        flags.append("multiple_books_in_view"); ambiguous = False
    else:
        runner_group = None
    if top.score >= conf_min and not ambiguous:
        status = "identified"
    elif top.score >= 0.5:
        status = "needs_review"
    else:
        status = "partial"
    return {"status": status, "confidence": top.score, "flags": flags, "chosen": top, "group": group, "runner_group": runner_group,
            "edition": ed, "ambiguous": ambiguous, "alternatives": [m.cand.to_dict() for m in ms[:4]],
            "verification": {"title_sim": top.title_sim, "author_hit": top.author_hit, "publisher_hit": top.publisher_hit,
                             "coverage": top.coverage, "sources": sorted({m.cand.source for m in group})}}


def verify_llm_extraction(ocr_text: str, title: str, author: str) -> bool:
    """Reject an LLM title/author split unless the OCR text really contains it (anti-hallucination)."""
    o = norm(ocr_text)
    if not title or fuzz.partial_ratio(norm(title), o) < 82:
        return False
    if author and fuzz.partial_ratio(norm(author), o) < 75:
        return False
    return True


def pick_isbn(group: list[Match], publisher_text: str = "", ocr_text: str = "") -> str:
    """ISBN is edition-specific. Only assign when the evidence pins the edition:
    a single distinct ISBN among matches, or the publisher printed on the spine matches exactly one."""
    distinct = {m.cand.isbn13 for m in group if m.cand.isbn13}
    if len(distinct) == 1:
        return next(iter(distinct))
    o = set(toks(ocr_text))
    hits = {m.cand.isbn13 for m in group if m.cand.isbn13 and toks(m.cand.publisher) and toks(m.cand.publisher)[0] in o}
    return next(iter(hits)) if len(hits) == 1 else ""


def work_key(title: str) -> str:
    """Same book, different catalogue string: 'And Then There Were None (adaptation)' == 'And Then There Were None';
    'The A.B.C. Murders' == 'The ABC Murders'."""
    t = re.sub(r"[\(\[].*?[\)\]]", " ", title or "")
    t = re.split(r"\s*[:–—]\s", t, 1)[0]
    t = re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower())
    return t[3:] if t.startswith("the") and len(t) > 8 else t
