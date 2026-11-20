"""Book metadata lookup: Google Books + Open Library. Used to *verify* OCR candidates."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ..config import get_settings
from ..http import request_json
from ..metrics import bump

log = logging.getLogger(__name__)


@dataclass
class BookCandidate:
    source: str                 # google_books | open_library
    title: str
    authors: list[str] = field(default_factory=list)
    publisher: str = ""
    year: str = ""
    isbn13: str = ""
    isbn10: str = ""
    url: str = ""
    language: str = ""
    categories: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _isbns(ids: list[dict]) -> tuple[str, str]:
    i13 = i10 = ""
    for i in ids or []:
        t, v = i.get("type", ""), i.get("identifier", "")
        if t == "ISBN_13": i13 = v
        elif t == "ISBN_10": i10 = v
    return i13, i10


_google_blocked_until = 0.0   # circuit breaker: keyless Google Books 429s constantly; stop hammering it


async def google_books(query: str, max_results: int = 8) -> list[BookCandidate]:
    global _google_blocked_until
    import time
    s = get_settings()
    if time.time() < _google_blocked_until or not s.google_books_api_key:
        return []   # keyless Google Books is rate-limited to uselessness; Open Library is used unless a key is configured
    params = {"q": query, "maxResults": max_results, "printType": "books"}
    if s.google_books_api_key:
        params["key"] = s.google_books_api_key
    bump("google_books_calls")
    try:
        d = await request_json("GET", "https://www.googleapis.com/books/v1/volumes", params=params, retries=0)
    except Exception as e:
        if "429" in str(e):
            _google_blocked_until = time.time() + 300
        return []
    out = []
    for it in d.get("items", []) or []:
        v = it.get("volumeInfo", {})
        i13, i10 = _isbns(v.get("industryIdentifiers", []))
        out.append(BookCandidate("google_books", v.get("title", "") + ((": " + v["subtitle"]) if v.get("subtitle") else ""),
                                 v.get("authors", []) or [], v.get("publisher", ""), (v.get("publishedDate") or "")[:4],
                                 i13, i10, v.get("infoLink", ""), v.get("language", ""), v.get("categories", []) or [],
                                 {"id": it.get("id")}))
    return out


_OL_STATE = {"fails": 0, "open_until": 0.0}


def ol_available() -> bool:
    import time
    return time.time() >= _OL_STATE["open_until"]


async def open_library(query: str, limit: int = 8) -> list[BookCandidate]:
    import time
    if not ol_available():            # circuit open: Open Library is down/throttling, don't wait on it
        return []
    bump("open_library_calls")
    try:
        d = await request_json("GET", "https://openlibrary.org/search.json", params={
            "q": query, "limit": limit,
            "fields": "key,title,subtitle,author_name,publisher,first_publish_year,isbn,edition_count,language"},
            timeout=6, retries=1)
    except Exception:
        OL_FAILS[0] += 1
        _OL_STATE["fails"] += 1
        if _OL_STATE["fails"] >= 3:   # three failures in a row -> stop calling it for 2 minutes; Google search covers meanwhile
            _OL_STATE["open_until"] = time.time() + 120; _OL_STATE["fails"] = 0
            log.warning("Open Library unreachable: using Google search for catalogue lookups for 2 minutes")
        return []
    _OL_STATE["fails"] = 0
    out = []
    for doc in d.get("docs", []) or []:
        isbns = doc.get("isbn", []) or []
        i13 = next((i for i in isbns if len(i) == 13), "")
        i10 = next((i for i in isbns if len(i) == 10), "")
        title = doc.get("title", "") + ((": " + doc["subtitle"]) if doc.get("subtitle") else "")
        pubs = doc.get("publisher", []) or []
        out.append(BookCandidate("open_library", title, doc.get("author_name", []) or [], pubs[0] if pubs else "",
                                 str(doc.get("first_publish_year", "") or ""), i13, i10,
                                 f"https://openlibrary.org{doc.get('key', '')}", (doc.get("language") or [""])[0],
                                 extra={"edition_count": doc.get("edition_count"), "publishers": pubs[:8]}))
    return out


_SUFFIX = re.compile(r"\s*[\|\u2013\u2014]\s*(goodreads|amazon|wikipedia|google books|barnes|penguin|harpercollins|simon|flipkart|abebooks|thriftbooks|bookshop|ebay|worldcat|storygraph|audible)[^|]*$", re.I)


_SERPER_DONE: set = set()
_SERPER_BUDGET = [250]   # catalogue-fallback searches per server run (each costs one Serper credit)


async def serper_books(query: str) -> list[BookCandidate]:
    """Catalogue fallback when Open Library is unreachable: Google results ('Title by Author' pages, the book knowledge panel).
    These are only CANDIDATES; the matcher still requires the OCR text to support the title (and author) before accepting."""
    from ..pricing import providers as _pv
    s = get_settings()
    if not _pv.serper_alive():
        return []
    q = re.sub(r"[^A-Za-z0-9' ]+", " ", query)                       # long noisy OCR (symbols, other alphabets) makes Serper answer 400
    q = " ".join(w for w in q.split() if len(w) <= 18)[:110].strip()
    if len(re.sub(r"[^A-Za-z]", "", q)) < 6 or q.lower() in _SERPER_DONE:
        return []                                                      # hopeless or already asked: don't spend a search credit
    if _SERPER_BUDGET[0] <= 0:
        return []
    _SERPER_BUDGET[0] -= 1; _SERPER_DONE.add(q.lower())
    bump("serper_calls")
    try:
        d = await request_json("POST", "https://google.serper.dev/search", headers={"X-API-KEY": s.serper_api_key, "Content-Type": "application/json"},
                               json={"q": f"{q} book"}, timeout=8, retries=0)
    except Exception as e:
        _pv.mark_serper_dead_if_needed(e)
        return []
    out, seen = [], set()

    def add(title: str, author: str, url: str):
        title = re.sub(r"\s+", " ", title).strip(" -|:")
        if len(title) < 3 or title.lower() in seen:
            return
        seen.add(title.lower())
        out.append(BookCandidate("google_search", title, [author.strip()] if author.strip() else [], "", "", "", "", url, "", [], {}))
    kg = d.get("knowledgeGraph") or {}
    if kg.get("title") and "book" in f"{kg.get('type', '')} {kg.get('description', '')}".lower():
        m = re.search(r"\bby ([A-Z][^,;()]+)", f"{kg.get('type', '')} {kg.get('description', '')}")
        add(kg["title"], (kg.get("attributes") or {}).get("Author") or (m.group(1) if m else ""), kg.get("website", ""))
    for r in d.get("organic", []) or []:
        t = _SUFFIX.sub("", r.get("title", ""))
        m = re.match(r"(.+?)\s+by\s+(.+?)(?:\s*\(|\s*[\|\-\u2013:]|$)", t)
        if m:
            add(m.group(1), m.group(2), r.get("link", ""))
        else:
            m2 = re.match(r"(.+?)\s+[-\u2013]\s+([A-Z][\w.' ]{3,40})$", t)
            add(m2.group(1) if m2 else t.split(" | ")[0], m2.group(2) if m2 else "", r.get("link", ""))
    return out[:10]


async def lookup(query: str) -> list[BookCandidate]:
    g, o = await asyncio.gather(google_books(query), open_library(query))
    res = g + o
    if not res and not ol_available():            # Open Library is down: Google search stands in as the catalogue
        res = await serper_books(query)
    return res


async def by_isbn(isbn: str) -> list[BookCandidate]:
    isbn = re.sub(r"[^0-9Xx]", "", isbn)
    g, o = await asyncio.gather(google_books(f"isbn:{isbn}", 3), open_library(f"isbn:{isbn}", 3))
    return g + o


_AUTHOR_CACHE: dict[str, list] = {}
OL_FAILS = [0]   # counts failed Open Library calls so callers can tell 'no results' from 'the request failed'


async def works_by_author(author: str, limit: int = 80) -> list[BookCandidate]:
    """All of an author's catalogued works (Open Library): lets a garbled title ('BC MURDERS') be fuzzy-matched
    against the author's real titles ('The A.B.C. Murders')."""
    k = author.lower().strip()
    if k not in _AUTHOR_CACHE:
        r = await open_library(f'author:"{author}"', limit) if k else []
        if not r and k and not ol_available():
            r = await serper_books(f"{author} books list")
        if not r:
            return []                       # don't cache a failed/empty fetch
        _AUTHOR_CACHE[k] = r
    return _AUTHOR_CACHE[k]
