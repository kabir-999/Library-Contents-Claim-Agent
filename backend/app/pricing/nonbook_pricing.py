"""Non-book pricing: sourced price *ranges* from real listings (never LLM memory).

* artwork / portraits / antiques -> needs_appraisal, never auto-priced.
* known brand/model -> query by model; unknown -> generic descriptive query,
  returned only as an explicit range with all source URLs attached.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..schemas import ItemCost
from . import providers
from .aggregate import price_range
from .markets import market

log = logging.getLogger(__name__)
APPRAISAL_CATEGORIES = {"artwork", "portrait", "antique", "sculpture", "mirror_antique"}
SKIP_CATEGORIES = {"bookshelf_contents", "fixture"}


def needs_appraisal(category: str, user_note: str = "") -> bool:
    if category in APPRAISAL_CATEGORIES:
        return "print" not in user_note.lower() or "original" in user_note.lower()
    return False


def build_query(category: str, brand_model: str, description: str, material: str) -> str:
    if brand_model.strip():
        return brand_model.strip()
    parts = [material, description or category.replace("_", " ")]
    return " ".join(p for p in parts if p).strip()[:120]


async def price_item(category: str, brand_model: str, description: str, material: str, country: str, currency: str) -> dict:
    q = build_query(category, brand_model, description, material)
    def _local(ls):
        return [l for l in ls if l["currency"] == currency and str(l.get("condition", "new")).lower() == "new"]
    listings = await providers.ebay_search(q, country, "NEW") if providers.ebay_enabled() else []   # primary
    if len(_local(listings)) < 3 and providers.serp_enabled():                                   # fallback
        listings += await providers.serp_shopping(q, country)
    local = _local(listings)
    r = price_range(local, min_n=3)
    if not r:
        return {"cost": ItemCost(currency=currency), "priced": False, "query": q,
                "reason": "fewer than 3 sourced local listings" if listings else "no price provider configured or no results"}
    sel = sorted(local, key=lambda l: l["amount"])
    srcs = [{"source": l["source"], "url": l["url"], "amount": l["amount"], "currency": l["currency"], "title": l["title"][:120]}
            for l in sel[:8]]
    cost = ItemCost(low=r["low"], high=r["high"], source=srcs[0]["source"], url=srcs[0]["url"],
                    retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), currency=currency,
                    listings_used=r["n"], sources=srcs)
    return {"cost": cost, "priced": True, "query": q, "reason": ""}
