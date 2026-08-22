from spam_detector.preprocess import normalize_corpus, normalize_text


def test_entities_are_replaced_with_placeholders():
    normalized = normalize_text(
        "Claim $500 at http://bit.ly/x or mail me@spam.com, call +1 555 0134"
    )
    assert "__url__" in normalized
    assert "__email__" in normalized
    assert "__money__" in normalized
    assert "bit.ly" not in normalized


def test_shouty_messages_get_a_marker():
    assert "__allcaps__" in normalize_text("URGENT ACCOUNT SUSPENDED VERIFY NOW")
    assert "__allcaps__" not in normalize_text("Are we still on for lunch at noon?")


def test_normalize_corpus_is_elementwise():
    assert normalize_corpus(["A 1", "B 2"]) == [normalize_text("A 1"), normalize_text("B 2")]
