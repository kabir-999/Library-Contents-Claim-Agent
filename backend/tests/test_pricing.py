from app.pricing.aggregate import central, filter_book_listings, price_range


def L(title, amt, url="https://x/1"):
    return {"title": title, "amount": amt, "currency": "USD", "url": url}


def test_filter_removes_lots_signed_and_other_books():
    ls = [L("Dune by Frank Herbert paperback", 8), L("Dune lot of 5 books", 20), L("Dune SIGNED first edition", 900),
          L("Children of Dune", 7), L("Dune Frank Herbert 9780441172719", 9)]
    kept = filter_book_listings(ls, "Dune", "9780441172719", "Frank Herbert")
    assert {k["amount"] for k in kept} == {8, 9}


def test_central_median_and_listing_citation():
    c = central([L("a", 5, "u1"), L("b", 6, "u2"), L("c", 7, "u3"), L("d", 500, "u4"), L("e", 6.5, "u5")])
    assert c["amount"] == 6.5 and c["listing"]["url"] == "u5"  # outlier trimmed


def test_none_when_empty():
    assert central([]) is None and price_range([L("a", 5)]) is None


def test_range_needs_three():
    r = price_range([L("a", 10), L("b", 20), L("c", 30), L("d", 40)])
    assert r["low"] <= r["high"] and r["n"] == 4


def test_parse_price_formats():
    from app.pricing.providers import parse_price
    assert parse_price("$1,299.00") == 1299.0 and parse_price("£8.50") == 8.5 and parse_price("1.299,00 €") == 1299.0
    assert parse_price("₹1,299") == 1299.0 and parse_price("12,50 €") == 12.5 and parse_price("Rs. 499") == 499.0
    assert parse_price("$10 - $20") is None and parse_price("from $5") is None and parse_price("") is None and parse_price("free") is None


def test_currency_detection_and_condition():
    from app.pricing.providers import _cur_from_price_str, classify_condition
    assert _cur_from_price_str("Rs. 499", "USD") == "INR" and _cur_from_price_str("£8", "USD") == "GBP"
    assert _cur_from_price_str("$12", "CAD") == "CAD"            # bare $ in a dollar market => that market's currency
    assert _cur_from_price_str("$12", "GBP") == "USD" and _cur_from_price_str("$12", "INR") == "USD"   # never mislabelled as local
    assert classify_condition("Dune paperback", "AbeBooks") == "used"
    assert classify_condition("Dune - Used, Good condition", "Amazon") == "used"
    assert classify_condition("Dune paperback", "Waterstones") == "new"


async def test_serper_rows_become_new_and_used_listings(monkeypatch):
    from app.pricing import providers
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "serper_api_key", "k")
    seen = {}
    async def fake(method, url, **kw):
        seen.update(url=url, headers=kw["headers"], json=kw["json"])
        return {"shopping": [{"title": "Dune Frank Herbert", "source": "Waterstones", "link": "https://w.example/dune", "price": "£9.99"},
                             {"title": "Dune Frank Herbert", "source": "AbeBooks", "link": "https://a.example/dune", "price": "£3.50"},
                             {"title": "Dune", "source": "X", "link": "", "price": "£1"},          # no URL -> dropped
                             {"title": "Dune", "source": "Y", "link": "https://y", "price": "£5 - £9"}]}   # range -> dropped
    monkeypatch.setattr(providers, "request_json", fake)
    rows = await providers.serper_shopping("Dune", "GB")
    assert seen["json"]["gl"] == "gb" and seen["headers"]["X-API-KEY"] == "k"
    assert [(r["amount"], r["currency"], r["condition"], r["url"]) for r in rows] == \
        [(9.99, "GBP", "new", "https://w.example/dune"), (3.5, "GBP", "used", "https://a.example/dune")]
    assert providers.serp_enabled()


async def test_unverified_edition_is_queried_by_title_not_isbn(monkeypatch):
    from app.pricing import book_pricing, providers
    queries = []
    async def fake(query, country, condition="NEW", category=None, limit=30):
        queries.append(query); return []
    monkeypatch.setattr(providers, "ebay_search", fake); monkeypatch.setattr(providers, "ebay_enabled", lambda: True)
    monkeypatch.setattr(providers, "serp_enabled", lambda: False)
    await book_pricing.price_book(title="Dune", author="Frank Herbert", isbn="", group_isbns=[("9780582342989", "2001")], country="US", currency="USD")
    assert queries and all("9780582342989" not in q and "Dune" in q for q in queries)
    queries.clear()
    await book_pricing.price_book(title="Dune", author="Frank Herbert", isbn="9780441172719", group_isbns=[], country="US", currency="USD")
    assert all(q == "9780441172719" for q in queries)


def test_essays_and_study_material_about_a_book_are_not_listings_of_it():
    ls = [L("The Remains of the Day Paperback - January 1, 2010 by Kazuo Ishiguro", 12.35), L("Remains of the Day by Kazuo Ishiguro (Used, Paperback)", 7.0),
          L("Kazuo Ishiguro: The Remains of The Day - Stevens' Philosophy of Life", 15.36), L("Memory in Kazuo Ishiguro's the Remains of the Day - Paperback", 23.0),
          L("The Remains of The Day by Kazuo Ishiguro (Book Analysis)", 14.5), L("About Wasted Opportunities In Kazuo Ishiguro's The Remains Of The Day.", 16.0),
          {**L("The Remains of the Day Ishiguro", 18.0), "source": "Google Shopping (Hausarbeiten.de)"}]
    kept = filter_book_listings(ls, "The Remains of the Day", "", "Kazuo Ishiguro")
    assert sorted(k["amount"] for k in kept) == [7.0, 12.35]


async def test_single_listing_price_is_withheld(monkeypatch):
    from app.pricing import book_pricing, providers
    async def one(query, country, condition="NEW", category=None, limit=30):
        return [{"title": "Dune by Frank Herbert", "amount": 9.0, "currency": "USD", "url": "https://x/1", "source": "eBay US", "condition": condition.lower()}]
    monkeypatch.setattr(providers, "ebay_search", one); monkeypatch.setattr(providers, "ebay_enabled", lambda: True)
    monkeypatch.setattr(providers, "serp_enabled", lambda: False)
    r = await book_pricing.price_book(title="Dune", author="Frank Herbert", isbn="", group_isbns=[], country="US", currency="USD")
    assert r["replacement"].amount is None and r["used"].amount is None and "withheld" in r["note"]


async def test_ebay_auth_failure_degrades_to_empty(monkeypatch):
    from app.pricing import providers
    async def bad(): raise RuntimeError("401")
    monkeypatch.setattr(providers, "_ebay_token", bad)
    assert await providers.ebay_search("Dune", "US", "NEW") == []


def test_foreign_language_editions_rejected():
    ls = [L("Sapiens. A Brief History of Humankind by Yuval Noah Harari", 329), L("SAPIENS - Eine kurze Geschichte der Menschheit", 1880),
          L("Sapiens: História Breve da Humanidade", 932), L("Sapiens by Yuval Noah Harari", 909)]
    kept = filter_book_listings(ls, "Sapiens", "", "Yuval Noah Harari")
    assert sorted(k["amount"] for k in kept) == [329, 909]


def test_author_less_listings_accepted_only_for_exact_long_titles():
    ls = [L("The Remains of The Day", 650), L("Remains Of The Day", 774), L("The Remains of The Dead: A Ghost Dusters Mystery", 509),
          L("Remnants of the Day : the Lost Years", 224), L("The Remains of The Day Novel by Kazuo Ishiguro (Farsi)", 5324),
          L("The Remains of The Day: York Notes Advanced", 1374), L("The Buried Giant", 1336)]
    kept = filter_book_listings(ls, "The Remains of the Day", "", "Kazuo Ishiguro")
    assert sorted(k["amount"] for k in kept) == [650, 774]
    # one-word titles still require the author
    assert filter_book_listings([L("Dune", 8)], "Dune", "", "Frank Herbert") == []
    assert len(filter_book_listings([L("Dune by Frank Herbert", 8)], "Dune", "", "Frank Herbert")) == 1
