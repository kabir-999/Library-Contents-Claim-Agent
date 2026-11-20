"""Country -> market parameters. Data, not logic. Extend freely."""
from __future__ import annotations

MARKETS = {
    "US": {"name": "United States", "currency": "USD", "ebay": "EBAY_US", "gl": "us"},
    "GB": {"name": "United Kingdom", "currency": "GBP", "ebay": "EBAY_GB", "gl": "gb"},
    "CA": {"name": "Canada", "currency": "CAD", "ebay": "EBAY_CA", "gl": "ca"},
    "AU": {"name": "Australia", "currency": "AUD", "ebay": "EBAY_AU", "gl": "au"},
    "DE": {"name": "Germany", "currency": "EUR", "ebay": "EBAY_DE", "gl": "de"},
    "FR": {"name": "France", "currency": "EUR", "ebay": "EBAY_FR", "gl": "fr"},
    "IT": {"name": "Italy", "currency": "EUR", "ebay": "EBAY_IT", "gl": "it"},
    "ES": {"name": "Spain", "currency": "EUR", "ebay": "EBAY_ES", "gl": "es"},
    "IE": {"name": "Ireland", "currency": "EUR", "ebay": "EBAY_IE", "gl": "ie"},
    "NL": {"name": "Netherlands", "currency": "EUR", "ebay": "EBAY_NL", "gl": "nl"},
    "IN": {"name": "India", "currency": "INR", "ebay": None, "gl": "in", "no_fx": True},   # local retailers only (Google Shopping, gl=in); never converted from USD
    "SG": {"name": "Singapore", "currency": "SGD", "ebay": "EBAY_SG", "gl": "sg"},
    "NZ": {"name": "New Zealand", "currency": "NZD", "ebay": None, "gl": "nz"},
    "AE": {"name": "United Arab Emirates", "currency": "AED", "ebay": None, "gl": "ae"},
    "ZA": {"name": "South Africa", "currency": "ZAR", "ebay": None, "gl": "za"},
}
SYMBOLS = {"$": None, "Rs.": "INR", "Rs": "INR", "£": "GBP", "€": "EUR", "₹": "INR", "A$": "AUD", "C$": "CAD", "S$": "SGD", "NZ$": "NZD", "R": "ZAR"}


def market(code: str) -> dict:
    c = (code or "").upper()
    if c in MARKETS:
        return {"code": c, **MARKETS[c]}
    for k, v in MARKETS.items():  # allow country names
        if v["name"].lower() == (code or "").lower():
            return {"code": k, **v}
    return {"code": c, "name": code, "currency": "", "ebay": None, "gl": c.lower()[:2]}
