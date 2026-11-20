"""Book pricing: replacement (new/equivalent copy) and used-market value.

Every amount is derived from retrieved listings (URL + date + currency). If no
reliable listing exists the amount stays None. Signed / first-edition / antiquarian /
unusually valuable books are flagged for human appraisal and NOT priced.
"""
from __future__ import annotations

import logging
import asyncio
import re
from datetime import datetime, timezone
from typing import Optional

from ..config import get_settings
from ..schemas import ReplacementCost, UsedValue
from . import fx, providers
from .aggregate import central, filter_book_listings
from .markets import market

log = logging.getLogger(__name__)
BOOKS_CATEGORY = "267"  # eBay "Books & Magazines"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def appraisal_reason(flags: list[str], ocr_text: str, candidate_years: list[str], user_first_edition: bool = False) -> str:
    if "signed" in flags: return "signed copy"
    if user_first_edition or "first_edition_claimed" in flags or "user_first_edition" in flags:
        return "first edition claimed (value depends on printing/condition)"
    years = [int(y) for y in re.findall(r"\b(1[5-9]\d\d)\b", ocr_text or "")]
    if years and min(years) <= 1925: return "antiquarian (printed date on spine <= 1925)"
    ys = [int(y) for y in candidate_years if y and y.isdigit()]
    if ys and max(ys) <= 1925: return "antiquarian (all known editions <= 1925)"
    return ""


_shop_cache: dict[tuple, list[dict]] = {}


async def _shopping(query: str, country: str) -> list[dict]:
    """Several phrasings (Google Shopping recall is patchy), merged and de-duplicated by URL. Cached so the NEW and USED
    passes share the same API calls."""
    key = (query, country)
    if key not in _shop_cache:
        import asyncio as _a
        suffixes = ("book", "paperback buy online", "used book") if market(country).get("no_fx") else ("book", "used book")
        batches = await _a.gather(*(providers.serp_shopping(f"{query} {suffix}".strip(), country) for suffix in suffixes))
        seen, rows = set(), []
        for r in (x for b in batches for x in b):
            if r["url"] not in seen:
                seen.add(r["url"]); rows.append(r)
        _shop_cache[key] = rows
    return _shop_cache[key]


async def _ebay_listings(query: str, country: str, condition: str) -> list[dict]:
    return await providers.ebay_search(query, country, condition, BOOKS_CATEGORY) if providers.ebay_enabled() else []


async def _shopping_listings(query: str, country: str, condition: str) -> list[dict]:
    return [l for l in await _shopping(query, country) if l["condition"] == condition.lower()] if providers.serp_enabled() else []


async def _listings(query: str, country: str, condition: str) -> list[dict]:
    out = []
    if providers.ebay_enabled():
        out += await providers.ebay_search(query, country, condition, BOOKS_CATEGORY)
    if providers.serp_enabled():
        out += [l for l in await _shopping(query, country) if l["condition"] == condition.lower()]
    return out


def available_sources() -> dict:
    s = get_settings()
    return {"ebay": providers.ebay_enabled(), "serper": bool(s.serper_api_key), "serpapi": bool(s.serpapi_api_key)}


_CACHE: dict = {}


async def price_book(**kw) -> dict:
    """Cached wrapper: live (during sweep) and final pricing share one lookup per book."""
    key = (kw.get("isbn") or "", kw["title"].lower(), kw["author"].lower(), kw["country"], bool(kw.get("allow_foreign_fallback")))
    if key in _CACHE:
        return _CACHE[key]
    r = await _price_book(**kw)
    if r["replacement"].amount is not None or r["used"].amount is not None:   # never cache a failed/empty lookup
        _CACHE[key] = r
    return r


async def _price_book(*, title: str, author: str, isbn: str, group_isbns: list[tuple[str, str]], country: str,
                     currency: str, allow_foreign_fallback: bool = False, home_market: str = "US") -> dict:
    """group_isbns: [(isbn13, year)] of matched editions. Returns dict with replacement/used/notes."""
    s = get_settings()
    priced_isbn = isbn
    note = ""
    if not priced_isbn and group_isbns:
        # newest known edition = what a replacement purchase would realistically be
        priced_isbn = sorted(group_isbns, key=lambda t: t[1] or "0", reverse=True)[0][0]
        note = f"Edition unverified; priced against current edition ISBN {priced_isbn} (equivalent copy)."
    # ISBN is a precise query only when the edition was verified. Shopping search is not ISBN-aware, so an
    # unverified "newest edition" ISBN would return unrelated books: search by title+author for an equivalent copy.
    query = isbn or f"{title} {author}".strip()
    rep, used = ReplacementCost(currency=currency), UsedValue(currency=currency)

    async def run(cond: str, ctry: str, expect_cur: str):
        # eBay is the PRIMARY source. Google Shopping (Serper/SerpAPI) is consulted only if eBay does not
        # provide enough matching listings in the right currency.
        ls = filter_book_listings(await _ebay_listings(query, ctry, cond), title, priced_isbn, author)
        if len([l for l in ls if l["currency"] == expect_cur]) < s.lca_min_listings:
            extra = filter_book_listings(await _shopping_listings(query, ctry, cond), title, priced_isbn, author)
            ls = ls + extra
        return [l for l in ls if l["currency"] == expect_cur], ls

    mk = market(country)
    ebay_last_resort = bool(allow_foreign_fallback and mk.get("no_fx"))   # India: local first; converted eBay price ONLY if nothing local exists
    allow_foreign_fallback = allow_foreign_fallback and not mk.get("no_fx")   # no cross-market sanity check / generic conversion there
    cross = allow_foreign_fallback and mk["code"] != home_market
    hm0 = market(home_market)
    # all lookups in parallel: latency = the slowest one, not their sum
    (new_local, _), (used_local, _), home_res = await asyncio.gather(
        run("NEW", country, currency), run("USED", country, currency),
        run("NEW", home_market, hm0["currency"]) if cross else asyncio.sleep(0, result=([], [])))

    c = central(new_local)
    if c and allow_foreign_fallback and mk["code"] != home_market:
        ref = central(home_res[0])
        conv0 = await fx.convert(ref["amount"], hm0["currency"], currency) if ref else None
        if conv0 and c["amount"] > 3 * conv0["amount"]:
            note += (f" Local price {c['amount']:g} {currency} withheld: >3x the converted {hm0['code']} reference "
                     f"({conv0['amount']:g} {currency}); likely a different/collector edition.")
            c = None
    if c and c["n"] < s.lca_min_listings:
        note += f" Replacement price withheld: only {c['n']} matching listing(s) (need {s.lca_min_listings})."
        c = None
    if c:
        l = c["listing"]
        rep = ReplacementCost(amount=c["amount"], source=l["source"], url=l["url"], retrieved_at=now(), converted=False,
                              currency=currency, condition_assumed="new")
    elif allow_foreign_fallback and mk["code"] != home_market:
        hm = market(home_market)
        c = central(home_res[0])
        if c:
            conv = await fx.convert(c["amount"], hm["currency"], currency)
            if conv:
                l = c["listing"]
                rep = ReplacementCost(amount=conv["amount"], source=l["source"] + f" (converted via {conv['source']})", url=l["url"],
                                      retrieved_at=now(), converted=True, currency=currency, condition_assumed="new",
                                      original_amount=c["amount"], original_currency=hm["currency"], fx_rate=conv["rate"],
                                      fx_source=conv["source"] + (f" {conv['rate_date']}" if conv["rate_date"] else ""))
    async def ebay_converted(cond: str):
        """No local (Serper/Google Shopping) price: look at eBay's US, then UK, marketplace and convert with the ECB rate."""
        if not providers.ebay_enabled():
            return None
        for code in ("US", "GB"):
            hm = market(code)
            ls = [l for l in filter_book_listings(await _ebay_listings(query, code, cond), title, priced_isbn, author) if l["currency"] == hm["currency"]]
            cc = central(ls)
            if cc and cc["n"] >= s.lca_min_listings:
                conv = await fx.convert(cc["amount"], hm["currency"], currency)
                if conv:
                    return cc, conv, hm
        return None

    if rep.amount is None and ebay_last_resort:
        r = await ebay_converted("NEW")
        if r:
            cc, conv, hm = r; l = cc["listing"]
            rep = ReplacementCost(amount=conv["amount"], source=f"{l['source']} ({hm['code']} marketplace, converted via {conv['source']})", url=l["url"],
                                  retrieved_at=now(), converted=True, currency=currency, condition_assumed="new",
                                  original_amount=cc["amount"], original_currency=hm["currency"], fx_rate=conv["rate"],
                                  fx_source=conv["source"] + (f" {conv['rate_date']}" if conv["rate_date"] else ""))
            note += f" No local price found; converted from eBay {hm['code']} ({cc['amount']:g} {hm['currency']})."
    cu = central(used_local)
    if cu and rep.amount and cu["amount"] > 3 * rep.amount:
        note += f" Used price {cu['amount']:g} {currency} withheld: >3x the new price ({rep.amount:g}); likely a collectible/different edition."
        cu = None
    if cu and cu["n"] < s.lca_min_listings:
        note += f" Used value withheld: only {cu['n']} matching listing(s) (need {s.lca_min_listings})."
        cu = None
    if cu:
        l = cu["listing"]
        used = UsedValue(amount=cu["amount"], source=l["source"], url=l["url"], retrieved_at=now(),
                         condition_assumed="Used (good/acceptable): median of active used listings",
                         currency=currency, listings_used=cu["n"])
        pass
    if used.amount is None and ebay_last_resort:
        r = await ebay_converted("USED")
        if r:
            cc, conv, hm = r; l = cc["listing"]
            if not (rep.amount and conv["amount"] > 3 * rep.amount):            # same collectible guard as the local path
                used = UsedValue(amount=conv["amount"], source=f"{l['source']} ({hm['code']} marketplace, converted via {conv['source']})", url=l["url"],
                                 retrieved_at=now(), condition_assumed="Used (good/acceptable): median of active used listings",
                                 currency=currency, converted=True, original_amount=cc["amount"], original_currency=hm["currency"],
                                 fx_rate=conv["rate"], listings_used=cc["n"])
                note += f" Used value converted from eBay {hm['code']} ({cc['amount']:g} {hm['currency']})."
    # Unusually valuable => human appraisal, never auto-priced
    thr = s.lca_appraisal_price_threshold   # configured in USD; compare in the claim currency
    if currency != "USD":
        conv = await fx.convert(thr, "USD", currency)
        thr = conv["amount"] if conv else thr * {"INR": 83, "GBP": 0.8, "EUR": 0.93}.get(currency, 1)
    over = [v for v in (rep.amount, used.amount) if v is not None and v > thr]
    return {"threshold_local": round(thr, 2), "currency": currency, "replacement": rep, "used": used, "priced_isbn": priced_isbn, "note": note.strip(), "over_threshold": bool(over)}
