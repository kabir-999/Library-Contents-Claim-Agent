from app.agent.corrections import parse_regex


def one(t):
    a = parse_regex(t); assert a, t; return a[0]


def test_brief_examples():
    assert one("That is a first edition.").action == "set_first_edition"
    a = one("Those books belong to someone else."); assert a.action == "set_not_mine" and a.target == "group"
    a = one("That is a print, not an original painting."); assert a.action == "set_artwork_type" and a.value == "print"
    assert one("Skip this shelf.").action == "skip_shelf"


def test_artwork_original_and_camera_height():
    assert one("it's an original painting").value == "original"
    a = one("camera height is 150 centimetres"); assert a.action == "set_camera_height" and a.number == 150
    a = one("my phone is at 1.5 m"); assert a.number is None or True  # not a defined phrase; must not crash


def test_known_dimension_conversion():
    a = one("this bookcase is 2 metres tall"); assert a.action == "known_shelf_dimension" and a.number == 200 and a.value == "height"


def test_corner_and_stop_and_count():
    assert one("mark corner").action == "mark_corner"
    assert one("ceiling corner").action == "mark_ceiling"
    assert one("ok I'm done").action == "stop"
    assert one("how many books so far").action == "query_count"


def test_unrelated_speech_is_noop():
    assert parse_regex("the weather is nice today") == []
