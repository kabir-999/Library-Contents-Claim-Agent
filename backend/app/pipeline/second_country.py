"""Price the same N books in a second country (local listings first; converted from the home market only as a flagged fallback)."""
from __future__ import annotations

import asyncio
import logging

from ..claim.calculations import book_excluded
from ..pricing import book_pricing
from ..pricing.markets import market
from ..schemas import ClaimPacket, SecondCountryComparison, SecondCountryRow

log = logging.getLogger(__name__)


async def compare(packet: ClaimPacket, country: str, n: int = 10, group_isbns_by_id: dict | None = None) -> SecondCountryComparison:
    mk = market(country)
    home = market(packet.sweep.country)
    cands = [b for b in packet.books if b.status in ("identified", "needs_appraisal") and not b.needs_appraisal and not book_excluded(b) and b.title][:n]
    sem = asyncio.Semaphore(4)

    async def one(b):
        async with sem:
            r = await book_pricing.price_book(title=b.title, author=b.author, isbn=b.isbn,
                                              group_isbns=(group_isbns_by_id or {}).get(b.id, [(b.priced_isbn, "")] if b.priced_isbn else []),
                                              country=mk["code"], currency=mk["currency"], allow_foreign_fallback=True, home_market=home["code"])
        rep, used = r["replacement"], r["used"]
        note = []
        if rep.converted: note.append(f"replacement converted from {rep.original_currency} at {rep.fx_rate} ({rep.fx_source})")
        if rep.amount is None: note.append("no sourced replacement price")
        if used.amount is None: note.append("no sourced used price")
        return SecondCountryRow(book_id=b.id, title=b.title, isbn=r["priced_isbn"], replacement_cost=rep.amount, used_value=used.amount,
                                currency=mk["currency"], source=rep.source or used.source, url=rep.url or used.url,
                                retrieved_at=rep.retrieved_at or used.retrieved_at, note="; ".join(note))
    rows = await asyncio.gather(*(one(b) for b in cands))
    return SecondCountryComparison(country=mk["name"], currency=mk["currency"], rows=list(rows))
