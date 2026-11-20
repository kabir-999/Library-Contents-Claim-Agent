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


async def open_library(query: str, limit: int = 8) -> list[BookCandidate]:
    bump("open_library_calls")
    try:
        d = await request_json("GET", "https://openlibrary.org/search.json", params={
            "q": query, "limit": limit,
            "fields": "key,title,subtitle,author_name,publisher,first_publish_year,isbn,edition_count,language"},
            timeout=5, retries=1)
    except Exception:
        return []
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


async def lookup(query: str) -> list[BookCandidate]:
    g, o = await asyncio.gather(google_books(query), open_library(query))
    return g + o


async def by_isbn(isbn: str) -> list[BookCandidate]:
    isbn = re.sub(r"[^0-9Xx]", "", isbn)
    g, o = await asyncio.gather(google_books(f"isbn:{isbn}", 3), open_library(f"isbn:{isbn}", 3))
    return g + o


_AUTHOR_CACHE: dict[str, list] = {}


async def works_by_author(author: str, limit: int = 80) -> list[BookCandidate]:
    """All of an author's catalogued works (Open Library): lets a garbled title ('BC MURDERS') be fuzzy-matched
    against the author's real titles ('The A.B.C. Murders')."""
    k = author.lower().strip()
    if k not in _AUTHOR_CACHE:
        _AUTHOR_CACHE[k] = await open_library(f'author:"{author}"', limit) if k else []
    return _AUTHOR_CACHE[k]
