from ca.grader import exact_match, f1, normalize


def test_normalize():
    assert normalize("The  Answer, is: Blue!") == "answer is blue"


def test_exact_match_with_aliases():
    assert exact_match("W. Somerset Maugham", ["William Somerset Maugham", "W. Somerset Maugham"]) == 1.0
    assert exact_match("Maugham", ["W. Somerset Maugham"]) == 0.0


def test_f1_partial():
    score = f1("Somerset Maugham", ["W. Somerset Maugham"])
    assert 0.0 < score < 1.0
    assert f1("", ["x"]) == 0.0
    assert f1("exact", ["exact"]) == 1.0
