from kb.grader import exact_match, f1, grade, is_unknown, normalize


def test_normalize_lowercases_strips_punct_and_articles():
    assert normalize("The  Quick, Brown Fox!") == "quick brown fox"
    assert normalize("A: an answer") == "answer"


def test_exact_match_over_gold_list():
    assert exact_match("Paris", ["paris"]) == 1.0
    assert exact_match("Paris", ["London", "PARIS."]) == 1.0
    assert exact_match("Paris", ["London"]) == 0.0


def test_f1_token_overlap():
    assert f1("the quick fox", ["quick fox"]) == 1.0
    assert f1("quick", ["quick brown fox"]) > 0.0
    assert f1("zebra", ["quick fox"]) == 0.0
    assert f1("x", []) == 0.0


def test_numeric_variants_digits_and_words():
    assert f1("four", ["4", "four"]) == 1.0
    assert f1("4", ["4", "four"]) == 1.0
    assert exact_match("four", ["4", "four"]) == 1.0


def test_unknown_aliases():
    assert is_unknown("Unknown")
    assert is_unknown("not known")
    assert is_unknown("I don't know.")
    assert not is_unknown("Paris")


def test_grade_answerable_is_f1_em():
    assert grade("quick fox", ["quick fox"]) == (1.0, 1.0)
    f, em = grade("quick", ["quick fox"])
    assert 0 < f < 1 and em == 0.0


def test_grade_unanswerable_all_or_nothing():
    assert grade("unknown", ["unknown"], unanswerable=True) == (1.0, 1.0)
    assert grade("cannot be determined", ["unknown"], unanswerable=True) == (1.0, 1.0)
    # no partial credit for a wrong guess that shares tokens with nothing
    assert grade("some guy", ["unknown"], unanswerable=True) == (0.0, 0.0)
