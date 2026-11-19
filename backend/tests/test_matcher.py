from app.identify.matcher import decide, edition_evidence, pick_isbn, rank, verify_llm_extraction
from app.identify.metadata import BookCandidate

C = lambda t, a, isbn="", pub="", src="google_books": BookCandidate(src, t, [a], pub, "2000", isbn)


def test_clear_match_identified():
    cands = [C("The Remains of the Day", "Kazuo Ishiguro", "9780679731726", "Vintage"), C("Never Let Me Go", "Kazuo Ishiguro", "9781400078776")]
    d = decide("THE REMAINS OF THE DAY ISHIGURO", 0.95, cands)
    assert d["status"] == "identified" and d["chosen"].cand.title.startswith("The Remains")


def test_garbage_ocr_never_invents():
    d = decide("xq", 0.4, [C("Dune", "Frank Herbert")])
    assert d["status"] == "unidentified" and d["chosen"] is None


def test_title_not_in_ocr_not_identified():
    d = decide("HERBERT", 0.9, [C("Dune", "Frank Herbert"), C("Children of Dune", "Frank Herbert")])
    assert d["status"] != "identified"


def test_single_word_title_needs_author():
    d = decide("DUNE", 0.9, [C("Dune", "Frank Herbert")])
    assert d["status"] != "identified"
    d2 = decide("DUNE FRANK HERBERT", 0.9, [C("Dune", "Frank Herbert")])
    assert d2["status"] == "identified"


def test_multiple_editions_flag_and_no_isbn():
    cands = [C("Dune", "Frank Herbert", "9780441172719", "Ace"), C("Dune", "Frank Herbert", "9780593099322", "Penguin")]
    d = decide("DUNE FRANK HERBERT", 0.9, cands)
    assert "multiple_editions" in d["flags"]
    assert pick_isbn(d["group"], "", "DUNE FRANK HERBERT") == ""
    assert pick_isbn(d["group"], "", "DUNE FRANK HERBERT ACE") == "9780441172719"


def test_edition_only_from_evidence():
    assert edition_evidence("Dune First Edition Herbert") != ""
    assert edition_evidence("Dune Herbert") == ""


def test_llm_extraction_must_be_in_ocr():
    assert verify_llm_extraction("THE REMAINS OF THE DAY ISHIGURO", "The Remains of the Day", "Ishiguro")
    assert not verify_llm_extraction("ISHIGURO", "Never Let Me Go", "Ishiguro")
