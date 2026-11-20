import os
os.environ.setdefault("LCA_BOOK_GATE", "false")   # stubbed pipeline tests have no CLIP; the gate has its own checks


import pytest


@pytest.fixture(autouse=True)
def _clear_price_cache():
    from app.pricing import book_pricing
    book_pricing._CACHE.clear()
    yield


@pytest.fixture(autouse=True)
def _no_author_network(monkeypatch):
    from app.identify import metadata

    async def none(author, limit=80):
        return []
    monkeypatch.setattr(metadata, "works_by_author", none)
