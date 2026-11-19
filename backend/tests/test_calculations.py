from app.claim.calculations import compute_totals, finalize_packet, m2_to_ft2, validate_frame_refs
from app.schemas import Book, ClaimPacket, Item, ItemCost, ReplacementCost, UsedValue, Sweep

URL = "https://example.com/x"
TS = "2026-01-01T00:00:00+00:00"


def book(i, amount=None, used=None, **kw):
    return Book(
        id=i, shelf="S1", status=kw.pop("status", "identified"),
        replacement_cost=ReplacementCost(amount=amount, url=URL if amount is not None else "", retrieved_at=TS if amount is not None else "", currency="USD"),
        used_value=UsedValue(amount=used, url=URL if used is not None else "", retrieved_at=TS if used is not None else "", currency="USD"),
        **kw,
    )


def test_totals_sum_only_sourced():
    books = [book("b1", 10.5, 4.25), book("b2", 20, None), book("b3", None, None, status="unidentified")]
    t = compute_totals(books, [], "USD")
    assert t.books_replacement_cost == 30.5
    assert t.books_used_value == 4.25
    assert t.book_count == 3 and t.books_identified == 2 and t.books_unidentified == 1
    assert t.excluded_from_totals == 1


def test_unsourced_amount_is_excluded():
    b = book("b1", 10, None)
    b.replacement_cost.url = ""  # amount without a URL must not count
    assert compute_totals([b], [], "USD").books_replacement_cost == 0


def test_appraisal_and_excluded_not_counted():
    a = book("a", 500, 400, needs_appraisal=True)
    e = book("e", 10, 5, status="excluded")
    t = compute_totals([a, e], [], "USD")
    assert t.books_replacement_cost == 0 and t.books_used_value == 0
    assert t.book_count == 1  # excluded (someone else's) not counted as the policyholder's


def test_currency_mismatch_excluded():
    b = book("b", 10, 5)
    b.replacement_cost.currency = "GBP"
    assert compute_totals([b], [], "USD").books_replacement_cost == 0


def test_items_range():
    it = Item(id="i1", category="lamp", status="priced", replacement_cost=ItemCost(low=20, high=60, url=URL, retrieved_at=TS, currency="USD"))
    art = Item(id="i2", category="artwork", status="needs_appraisal")
    t = compute_totals([], [it, art], "USD")
    assert (t.items_replacement_cost_low, t.items_replacement_cost_high) == (20, 60)
    assert t.excluded_from_totals == 1


def test_shelf_run_lower_bound_flagged():
    b1 = book("b1", spine_thickness_cm=3.0); b2 = book("b2")
    t = compute_totals([b1, b2], [], "USD")
    assert t.shelf_run_m == 0.03
    assert t.per_shelf["S1"]["run_complete"] is False


def test_shelf_run_none_when_uncalibrated():
    assert compute_totals([book("b1")], [], "USD").shelf_run_m is None


def test_zero_vs_unknown_preserved():
    b = book("z", 0.0, None)
    assert b.replacement_cost.amount == 0.0 and b.used_value.amount is None


def test_finalize_builds_review_queue_and_ft2():
    p = ClaimPacket(sweep=Sweep(id="s", currency="USD"), books=[book("u", status="unidentified")])
    p.room.floor_area_m2 = 10
    finalize_packet(p)
    assert p.room.floor_area_ft2 == m2_to_ft2(10) == 107.64
    reasons = " ".join(r.reason for r in p.review_queue)
    assert "Unreadable" in reasons and "Uncalibrated" in reasons and "Room length unavailable" in reasons


def test_frame_refs():
    p = ClaimPacket(books=[Book(id="b", frame_ref="f_1"), Book(id="c", frame_ref="nope")])
    assert validate_frame_refs(p, {"f_1"}) == ["c:nope"]


def test_contract_status_mapping_never_asserts_a_guess():
    from app.claim.calculations import apply_contract_status
    from app.schemas import Item
    guess = Book(id="g", status="needs_review", title="Maybe Dune", author="F. Herbert", isbn="9780441172719", id_confidence=0.6)
    gone = Book(id="x", status="excluded", title="Not Mine")
    ok = Book(id="o", status="identified", title="Dune")
    p = ClaimPacket(books=[guess, gone, ok], items=[Item(id="a", status="unpriced"), Item(id="b", status="excluded"),
                    Item(id="c", status="priced", replacement_cost=ItemCost(low=5, high=9, url=URL, retrieved_at=TS))])
    apply_contract_status(p)
    g = next(b for b in p.books if b.id == "g")
    assert g.status == "unidentified" and g.title == "" and g.isbn == "" and g.suggested_title == "Maybe Dune" and g.detail_status == "needs_review"
    assert [b.id for b in p.excluded_books] == ["x"] and {b.id for b in p.books} == {"g", "o"}
    assert {i.id: i.status for i in p.items} == {"a": "needs_appraisal", "c": "range"} and [i.id for i in p.excluded_items] == ["b"]
