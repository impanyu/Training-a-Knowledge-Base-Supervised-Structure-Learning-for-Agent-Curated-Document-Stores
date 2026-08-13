"""The question bank: flat questions, id-only addressing."""
import json
from pathlib import Path

import pytest
from ca.bank import BankError, Question, QuestionBank

REAL_BANK = Path(__file__).resolve().parents[1] / "data" / "v5" / "bank.json"


def demo_bank() -> QuestionBank:
    return QuestionBank([
        Question("q0001", "capital of France?", ["Paris"], "2hop", 18000, "k01"),
        Question("q0002", "longest river in France?", ["Loire"], "3hop", 30000, "k01"),
        Question("q0003", "2+2?", ["4", "four"], "4hop", 45000, "k07"),
    ])


def test_questions_are_addressed_by_qid():
    bank = demo_bank()
    assert set(bank.questions) == {"q0001", "q0002", "q0003"}
    q = bank.get("q0002")
    assert q.text == "longest river in France?"
    assert q.price == 30000 and q.topic == "k01"


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
    assert "list_questions" not in msg         # the board is gone


def test_from_json_reads_questions_and_ignores_extra_fields(tmp_path):
    p = tmp_path / "bank.json"
    p.write_text(json.dumps({
        "questions": [
            {"qid": "q0001", "text": "capital of France?", "answers": ["Paris"],
             "difficulty": "2hop", "price": 18000, "topic": "k20",
             "source": "hotpotqa", "quota": 4},
            {"qid": "q0002", "text": "2+2?", "answers": ["4"],
             "difficulty": "2hop", "price": 1000, "topic": "k20"}],
        "n_topics": 1}))
    bank = QuestionBank.from_json(str(p))
    q = bank.get("q0001")
    assert q.topic == "k20" and q.answers == ["Paris"]
    assert not hasattr(q, "source") and not hasattr(q, "quota")
    assert len(bank.questions) == 2


def test_duplicate_qids_are_a_build_error():
    with pytest.raises(BankError):
        QuestionBank([Question("q0001", "a", ["x"], "2hop", 1, "k0"),
                      Question("q0001", "b", ["y"], "2hop", 1, "k0")])


# ---------------- the real bank ----------------

@pytest.mark.skipif(not REAL_BANK.exists(), reason="v5 bank not built yet")
def test_real_v5_bank_loads():
    bank = QuestionBank.from_json(str(REAL_BANK))
    assert len(bank.questions) == 1000
    assert all(q.topic for q in bank.questions.values())
    assert {q.price for q in bank.questions.values()} <= {18000, 30000, 45000}
    assert {q.difficulty for q in bank.questions.values()} <= {"2hop", "3hop", "4hop"}
