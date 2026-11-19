"""Robust aggregation of real listings into one amount or a range. Pure functions (unit-tested)."""
from __future__ import annotations

import re
import statistics
from typing import Optional

from rapidfuzz import fuzz

from ..identify.matcher import norm

JUNK = re.compile(
    r"\b(lot of|bundle|set of|box set|boxed set|\d+\s*books|collection of|job lot|arc|advance reader|proof|"
    r"audiobook|audio book|\bcd\b|dvd|kindle|ebook|e-book|study guide|summary|workbook|notes|poster|print only|"
    r"signed|autograph|first edition|1st edition|rare|antique|vintage|leather bound|folio|facsimile|reproduction|"
    r"analysis|essay|essays|interpretation|approaches to|companion|casebook|critical|thesis|term paper|seminar|"
    r"film adaptation|cliffs?notes|spark ?notes|literary|criticism|commentary|a study of|student)\b", re.I)
JUNK_SOURCES = ("grin.com", "hausarbeiten", "grin verlag", "studysmarter", "coursehero", "chegg")
ARTICLES = ("the ", "a ", "an ")


def _strip_article(s: str) -> str:
    for a in ARTICLES:
        if s.startswith(a):
            return s[len(a):]
    return s


FOREIGN_WORDS = {"eine", "kurze", "geschichte", "ausgabe", "buch", "historia", "história", "breve", "humanidad", "humanidade",
                 "edicion", "edición", "libro", "livre", "histoire", "edizione", "uma", "una", "des", "les", "der", "und"}


LANG_NAMES = {"farsi", "persian", "hindi", "urdu", "bengali", "tamil", "telugu", "kannada", "malayalam", "marathi", "gujarati",
              "punjabi", "spanish", "french", "german", "italian", "chinese", "japanese", "korean", "arabic", "russian", "translated"}


def looks_foreign(listing_title: str, wanted_title: str) -> bool:
    """Foreign-language editions are not an 'equivalent copy' of an English book. Cheap heuristic: accented/non-ASCII
    letters or foreign function words that are NOT part of the wanted title."""
    letters = [c for c in listing_title if c.isalpha()]
    if letters and sum(1 for c in letters if ord(c) > 127) / len(letters) > 0.04:
        return True
    want = set(norm(wanted_title).split())
    return any((w in FOREIGN_WORDS or w in LANG_NAMES) and w not in want for w in norm(listing_title).split())


def filter_book_listings(listings: list[dict], title: str, isbn: str = "", author: str = "") -> list[dict]:
    """Keep only listings that are clearly the same book, a normal copy (not lots/signed/rare)."""
    keep = []
    nt = norm(title.split(":")[0])
    for l in listings:
        t = l.get("title", "")
        if looks_foreign(t, title):
            continue
        if JUNK.search(t) or any(j in (l.get("source", "") or "").lower() for j in JUNK_SOURCES):
            continue
        has_isbn = bool(isbn) and isbn in re.sub(r"[-\s]", "", t)
        sim = fuzz.token_set_ratio(nt, norm(t)) / 100
        if not (has_isbn or sim >= 0.9):
            continue
        # The listing must START with the book's title. Essays/guides *about* a book put the title mid-string
        # ("Memory in Kazuo Ishiguro's The Remains of the Day"); real listings lead with it.
        lead, want = _strip_article(norm(t)), _strip_article(nt)
        if not has_isbn and fuzz.ratio(" ".join(lead.split()[:len(want.split())]), want) < 88:
            continue
        if author and not has_isbn and fuzz.partial_ratio(norm(author.split()[-1]), norm(t)) < 85:
            # Many retailers (esp. India) list just the title. Accept that ONLY for an exact, word-for-word full title of
            # >=3 words with little trailing text; short/generic titles still need the author.
            lt, wt = lead.split(), want.split()
            if not (len(wt) >= 3 and lt[:len(wt)] == wt and len(lt) - len(wt) <= 3):
                continue
        keep.append(l)
    return keep


def _trim(vals: list[float]) -> list[float]:
    if len(vals) < 5:
        return sorted(vals)
    q1, _, q3 = statistics.quantiles(vals, n=4)
    iqr = q3 - q1
    return sorted(v for v in vals if q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr)


def central(listings: list[dict]) -> Optional[dict]:
    """Median of trimmed amounts + the actual listing closest to it (its URL is the citation)."""
    vals = _trim([l["amount"] for l in listings if l["amount"] > 0])
    if not vals:
        return None
    med = statistics.median(vals)
    rep = min((l for l in listings if l["amount"] in vals), key=lambda l: abs(l["amount"] - med))
    return {"amount": round(med, 2), "listing": rep, "n": len(vals)}


def price_range(listings: list[dict], min_n: int = 3) -> Optional[dict]:
    """Interquartile range for non-book items (needs >= min_n listings, else no range)."""
    vals = _trim([l["amount"] for l in listings if l["amount"] > 0])
    if len(vals) < min_n:
        return None
    q = statistics.quantiles(vals, n=4) if len(vals) >= 4 else [vals[0], statistics.median(vals), vals[-1]]
    low, high = round(q[0], 2), round(q[2], 2)
    return {"low": low, "high": max(low, high), "n": len(vals)}
