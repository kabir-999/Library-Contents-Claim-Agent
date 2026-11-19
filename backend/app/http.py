"""Shared async HTTP with timeouts and bounded retries (429/5xx/timeouts)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

from .config import get_settings

log = logging.getLogger(__name__)
_client: Optional[httpx.AsyncClient] = None
UA = "LibraryContentsClaimAgent/0.1 (insurance inventory research tool)"


def client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=get_settings().lca_http_timeout_s, headers={"User-Agent": UA}, follow_redirects=True)
    return _client


async def request_json(method: str, url: str, *, params=None, headers=None, json=None, data=None,
                       auth=None, timeout: Optional[float] = None, retries: Optional[int] = None) -> Any:
    s = get_settings()
    retries = s.lca_http_retries if retries is None else retries
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = await client().request(method, url, params=params, headers=headers, json=json, data=data,
                                       auth=auth, timeout=timeout or s.lca_http_timeout_s)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                await asyncio.sleep(min(2 ** attempt * 0.7, 4))
                continue
            r.raise_for_status()
            return r.json()
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = e
            if attempt < retries:
                await asyncio.sleep(min(2 ** attempt * 0.7, 4)); continue
        except httpx.HTTPStatusError as e:
            last = e
            break
    log.warning("HTTP %s %s failed: %s", method, url, last)
    raise last if last else RuntimeError("request failed")
