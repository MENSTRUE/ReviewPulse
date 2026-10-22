from reviewpulse.preprocessing import clean_review


def test_clean_review():
    assert clean_review("  bagus\nsekali  ") == "bagus sekali"
    assert clean_review(None) == ""
    assert clean_review("nan") == ""
