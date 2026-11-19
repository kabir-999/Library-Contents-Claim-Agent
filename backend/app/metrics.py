"""Per-sweep cost + latency accounting. Rates are env-configurable inputs (verify against
current provider pricing); *counts* of calls/tokens are measured."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .schemas import ProcessingTiming

COUNTS = {"serper_calls": 0, "serpapi_calls": 0, "ebay_calls": 0, "google_books_calls": 0, "open_library_calls": 0}


def bump(k: str, n: int = 1):
    COUNTS[k] = COUNTS.get(k, 0) + n


@dataclass
class Timings:
    rows: list[ProcessingTiming] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str):
        t = time.time(); status = "ok"
        try:
            yield
        except Exception:
            status = "error"; raise
        finally:
            self.rows.append(ProcessingTiming(stage=name, seconds=round(time.time() - t, 2), status=status))

    def mark(self, name: str, seconds: float, status: str):
        self.rows.append(ProcessingTiming(stage=name, seconds=round(seconds, 2), status=status))


def estimate_cost(groq_tokens: int, serp_calls: int) -> dict:
    # USD per 1M tokens (blended in/out) and per SerpAPI search. Defaults are placeholders: set real rates in .env.
    groq_rate = float(os.getenv("LCA_COST_GROQ_PER_MTOK", "0.45"))
    serp_rate = float(os.getenv("LCA_COST_SERPAPI_PER_SEARCH", "0.015"))
    return {"groq_tokens": groq_tokens, "groq_usd": round(groq_tokens / 1e6 * groq_rate, 4),
            "serpapi_calls": serp_calls, "serpapi_usd": round(serp_calls * serp_rate, 4),
            "local_compute_usd": 0.0, "total_usd": round(groq_tokens / 1e6 * groq_rate + serp_calls * serp_rate, 4),
            "note": "rates from LCA_COST_* env vars (defaults are placeholders, verify); eBay/Google Books/Open Library/Frankfurter are free tiers; local models cost electricity only."}
