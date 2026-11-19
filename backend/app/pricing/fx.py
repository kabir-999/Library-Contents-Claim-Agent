"""Exchange rates (Frankfurter, ECB reference rates). Only called when a price is
converted from another market's currency; the rate and date are stored with the price."""
from __future__ import annotations

import logging
from typing import Optional

from ..config import get_settings
from ..http import request_json

log = logging.getLogger(__name__)
_cache: dict[tuple, tuple[float, str]] = {}


async def rate(src: str, dst: str) -> Optional[tuple[float, str]]:
    """Returns (rate, rate_date) or None when unavailable."""
    if src == dst:
        return 1.0, ""
    if (src, dst) in _cache:
        return _cache[(src, dst)]
    try:
        d = await request_json("GET", f"{get_settings().lca_fx_base_url}/latest", params={"from": src, "to": dst})
        r = (float(d["rates"][dst]), d.get("date", ""))
        _cache[(src, dst)] = r
        return r
    except Exception as e:
        log.warning("FX %s->%s unavailable: %s", src, dst, e)
        return None


async def convert(amount: float, src: str, dst: str) -> Optional[dict]:
    r = await rate(src, dst)
    if r is None:
        return None
    return {"amount": round(amount * r[0], 2), "rate": r[0], "rate_date": r[1], "source": "frankfurter.app (ECB)"}
