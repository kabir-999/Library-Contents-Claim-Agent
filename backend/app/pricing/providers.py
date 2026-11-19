"""Price listing providers. Each returns real listings: {title, amount, currency, url, source, condition}.

No provider ever fabricates. Missing keys => provider disabled => no price (blank, excluded from totals).
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from typing import Optional

from ..config import get_settings
from ..http import request_json
from ..metrics import bump
from .markets import SYMBOLS, market

log = logging.getLogger(__name__)

_ebay_tok: dict = {"v": None, "exp": 0.0}


async def _ebay_token() -> Optional[str]:
    s = get_settings()
    if not (s.ebay_client_id and s.ebay_client_secret):
        return None
    if _ebay_tok["v"] and _ebay_tok["exp"] > time.time() + 60:
        return _ebay_tok["v"]
    basic = base64.b64encode(f"{s.ebay_client_id}:{s.ebay_client_secret}".encode()).decode()
    d, last = None, None
    for attempt in range(4):   # eBay occasionally answers 401 while key/compliance changes propagate: retry before giving up
        try:
            d = await request_json("POST", "https://api.ebay.com/identity/v1/oauth2/token",
                                   headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
                                   data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"})
            break
        except Exception as e:
            last = e
            await asyncio.sleep(1.5 * (attempt + 1))
    if d is None:
        raise last
    _ebay_tok.update(v=d["access_token"], exp=time.time() + int(d.get("expires_in", 7200)))
    return _ebay_tok["v"]


def ebay_enabled() -> bool:
    s = get_settings()
    return bool(s.ebay_client_id and s.ebay_client_secret)


def serp_enabled() -> bool:
    s = get_settings()
    return bool(s.serper_api_key or s.serpapi_api_key)


async def ebay_search(query: str, country: str, condition: str = "NEW", category: Optional[str] = None, limit: int = 30) -> list[dict]:
    """condition: NEW | USED. Active listings in the country's eBay marketplace."""
    mk = market(country)
    if not mk.get("ebay"):
        return []
    try:
        tok = await _ebay_token()
    except Exception as e:   # bad/inactive keys must degrade to "no eBay listings", never crash pricing
        log.warning("eBay token unavailable (%s); skipping eBay", type(e).__name__)
        return []
    if not tok:
        return []
    params = {"q": query, "limit": limit, "filter": f"conditions:{{{condition}}},buyingOptions:{{FIXED_PRICE}}"}
    if category:
        params["category_ids"] = category
    bump("ebay_calls")
    try:
        d = await request_json("GET", "https://api.ebay.com/buy/browse/v1/item_summary/search", params=params,
                               headers={"Authorization": f"Bearer {tok}", "X-EBAY-C-MARKETPLACE-ID": mk["ebay"]})
    except Exception:
        return []
    out = []
    for it in d.get("itemSummaries", []) or []:
        pr = it.get("price") or {}
        try:
            amt = float(pr.get("value"))
        except Exception:
            continue
        out.append({"title": it.get("title", ""), "amount": amt, "currency": pr.get("currency", ""),
                    "url": it.get("itemWebUrl", ""), "source": f"eBay {mk['code']}",
                    "condition": condition.lower(), "condition_raw": it.get("condition", ""), "seller_country": (it.get("itemLocation") or {}).get("country", "")})
    return [o for o in out if o["url"] and o["currency"]]


def _cur_from_price_str(price: str, default: str) -> str:
    m = re.search(r"\b([A-Z]{3})\b", price or "")
    if m and m.group(1) not in ("USD", "GBP", "EUR", "CAD", "AUD", "INR", "SGD", "NZD", "AED", "ZAR", "CHF", "JPY", "SEK", "NOK", "DKK", "PLN"):
        m = None
    if m:
        return m.group(1)
    for sym, cur in sorted(SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        if sym in (price or "") and cur:
            return cur
    if "$" in (price or ""):
        # A bare "$" is the market's own currency only in dollar markets (US/CA/AU/SG/NZ). Anywhere else it is USD:
        # a mislabelled USD price would otherwise be stored as the local currency.
        return default if default in ("USD", "CAD", "AUD", "SGD", "NZD") else "USD"
    return default


def parse_price(price: str) -> Optional[float]:
    """'$1,299.00' -> 1299.0 ; '1.299,00 €' -> 1299.0 ; '£8.50' -> 8.5 ; ranges / 'from' prices -> None (not a single price)."""
    if not price or re.search(r"\d\s*[-–]\s*[$£€₹]?\s*\d", price) or "from" in price.lower():
        return None
    m = re.search(r"\d[\d.,\s]*\d|\d", price)
    if not m:
        return None
    n = re.sub(r"\s", "", m.group(0))
    if "," in n and "." in n:
        n = n.replace(".", "").replace(",", ".") if n.rfind(",") > n.rfind(".") else n.replace(",", "")
    elif "," in n:
        n = n.replace(",", ".") if re.fullmatch(r"\d+,\d{1,2}", n) else n.replace(",", "")
    try:
        v = float(n)
    except ValueError:
        return None
    return v if v > 0 else None


USED_SOURCES = ("abebooks", "thriftbooks", "better world books", "betterworldbooks", "biblio", "world of books", "wob.com",
                "awesomebooks", "momox", "ebay", "alibris", "half.com", "second hand", "secondhand", "bookfinder", "zvab", "recycle")
USED_TITLE = re.compile(r"\b(used|pre-?owned|second[- ]hand|good condition|acceptable|very good|like new|ex-?library)\b", re.I)


def classify_condition(title: str, source: str, declared: str = "") -> str:
    if declared and declared.lower() not in ("new", ""):
        return "used"
    if USED_TITLE.search(title or "") or any(u in (source or "").lower() for u in USED_SOURCES):
        return "used"
    return "new"


def _shopping_rows(items: list[dict], mk: dict, label: str) -> list[dict]:
    out = []
    for r in items or []:
        raw = r.get("price") or ""
        amt = r.get("extracted_price") if r.get("extracted_price") is not None else parse_price(raw)
        url = r.get("product_link") or r.get("link") or ""
        if amt is None or not url.startswith("http"):
            continue
        title, source = r.get("title", ""), r.get("source", "retailer")
        out.append({"title": title, "amount": float(amt), "currency": _cur_from_price_str(raw, mk["currency"]), "url": url,
                    "source": f"{label} ({source})", "condition": classify_condition(title, source, r.get("second_hand_condition", ""))})
    return out


async def serper_shopping(query: str, country: str, limit: int = 20) -> list[dict]:
    """Google Shopping via Serper.dev (free tier: 2,500 searches, no card)."""
    s = get_settings()
    if not s.serper_api_key:
        return []
    mk = market(country)
    bump("serper_calls")
    try:
        d = await request_json("POST", "https://google.serper.dev/shopping", headers={"X-API-KEY": s.serper_api_key, "Content-Type": "application/json"},
                               json={"q": query, "gl": mk["gl"], "num": limit})
    except Exception:
        return []
    return _shopping_rows(d.get("shopping", []), mk, "Google Shopping")


async def serpapi_shopping(query: str, country: str, limit: int = 20) -> list[dict]:
    """Google Shopping via SerpAPI (free tier: 100 searches/month)."""
    s = get_settings()
    if not s.serpapi_api_key:
        return []
    mk = market(country)
    bump("serpapi_calls")
    try:
        d = await request_json("GET", "https://serpapi.com/search.json",
                               params={"engine": "google_shopping", "q": query, "gl": mk["gl"], "hl": "en", "num": limit, "api_key": s.serpapi_api_key})
    except Exception:
        return []
    return _shopping_rows(d.get("shopping_results", []), mk, "Google Shopping")


async def serp_shopping(query: str, country: str, limit: int = 20) -> list[dict]:
    """Whichever shopping provider has a key (Serper first); each row carries its own condition (new|used)."""
    return await serper_shopping(query, country, limit) or await serpapi_shopping(query, country, limit)
