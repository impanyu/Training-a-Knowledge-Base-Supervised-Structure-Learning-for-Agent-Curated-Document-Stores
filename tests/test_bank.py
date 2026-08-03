"""v4 question bank: flat questions with quotas, qid-only addressing."""
import json
from pathlib import Path

import pytest
from ca.bank import BankError, Question, QuestionBank

REAL_BANK = Path(__file__).resolve().parents[1] / "data" / "v4" / "bank.json"


def demo_bank() -> QuestionBank:
    return QuestionBank([
        Question("q0001", "capital of France?", ["Paris"], "2hop", 63000, 2, "k01"),
        Question("q0002", "longest river in France?", ["Loire"], "3hop", 105000, 1, "k01"),
        Question("q0003", "2+2?", ["4", "four"], "4hop", 157500, 3, "k07"),
    ])


def test_questions_are_addressed_by_qid():
    bank = demo_bank()
    assert set(bank.questions) == {"q0001", "q0002", "q0003"}
    q = bank.get("q0002")
    assert q.text == "longest river in France?"
    assert q.price == 105000 and q.quota == 1 and q.topic == "k01"


def test_get_strips_whitespace_but_does_no_fuzzy_matching():
    bank = demo_bank()
    assert bank.get(" q0001 ").qid == "q0001"
    with pytest.raises(BankError):
        bank.get("capital of France?")


def test_unknown_qid_names_a_few_valid_ids():
    bank = demo_bank()
    with pytest.raises(BankError) as e:
        bank.get("q9999")
    msg = str(e.value)
    assert "q9999" in msg
    assert sum(qid in msg for qid in ("q0001", "q0002", "q0003")) >= 2


def test_total_units_is_the_sum_of_quotas():
    assert demo_bank().total_units() == 6


def test_from_json_reads_the_bank_and_ignores_extra_fields(tmp_path):
    p = tmp_path / "bank.json"
    p.write_text(json.dumps({"questions": [
        {"qid": "q0001", "text": "capital of France?", "answers": ["Paris"],
         "difficulty": "2hop", "price": 63000, "quota": 4, "topic": "k20",
         "source": "hotpotqa"}], "total_units": 4, "n_topics": 1}))
    bank = QuestionBank.from_json(str(p))
    q = bank.get("q0001")
    assert q.quota == 4 and q.topic == "k20" and q.answers == ["Paris"]
    assert bank.total_units() == 4
    assert not hasattr(q, "source")


def test_duplicate_qids_are_a_build_error():
    with pytest.raises(BankError):
        QuestionBank([Question("q0001", "a", ["x"], "2hop", 1, 1, "k0"),
                      Question("q0001", "b", ["y"], "2hop", 1, 1, "k0")])


def test_non_positive_quota_is_a_build_error():
    with pytest.raises(BankError):
        QuestionBank([Question("q0001", "a", ["x"], "2hop", 1, 0, "k0")])


def test_real_v4_bank_loads():
    bank = QuestionBank.from_json(str(REAL_BANK))
    assert len(bank.questions) == 500
    assert bank.total_units() == 1478
    assert all(1 <= q.quota <= 5 for q in bank.questions.values())
    assert all(q.topic for q in bank.questions.values())
