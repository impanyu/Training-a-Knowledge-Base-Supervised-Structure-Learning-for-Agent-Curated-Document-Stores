"""v4.1 question bank: flat questions, posted as jobs, id-only addressing."""
import json
from pathlib import Path

import pytest
from ca.bank import BankError, Job, Question, QuestionBank

REAL_BANK = Path(__file__).resolve().parents[1] / "data" / "v4" / "bank.json"


def demo_bank() -> QuestionBank:
    return QuestionBank([
        Question("q0001", "capital of France?", ["Paris"], "2hop", 63000, "k01"),
        Question("q0002", "longest river in France?", ["Loire"], "3hop", 105000, "k01"),
        Question("q0003", "2+2?", ["4", "four"], "4hop", 157500, "k07"),
    ], [Job("j0001", ["q0001", "q0002"], 168000),
        Job("j0002", ["q0002", "q0003"], 262500)])


def test_questions_are_addressed_by_qid():
    bank = demo_bank()
    assert set(bank.questions) == {"q0001", "q0002", "q0003"}
    q = bank.get("q0002")
    assert q.text == "longest river in France?"
    assert q.price == 105000 and q.topic == "k01"
    assert not hasattr(q, "quota")


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


# ---------------- jobs ----------------

def test_jobs_are_addressed_by_jid():
    bank = demo_bank()
    job = bank.get_job(" j0002 ")
    assert job.qids == ["q0002", "q0003"] and job.price == 262500
    assert bank.job_price("j0001") == 168000


def test_unknown_jid_names_a_few_valid_ids():
    bank = demo_bank()
    with pytest.raises(BankError) as e:
        bank.get_job("j9999")
    msg = str(e.value)
    assert "j9999" in msg and "j0001" in msg and "list_jobs" in msg


def test_total_units_is_the_sum_of_job_sizes():
    assert demo_bank().total_units() == 4          # not 3: q0002 sits in two jobs


def test_from_json_reads_questions_and_jobs_and_ignores_extra_fields(tmp_path):
    p = tmp_path / "bank.json"
    p.write_text(json.dumps({
        "questions": [
            {"qid": "q0001", "text": "capital of France?", "answers": ["Paris"],
             "difficulty": "2hop", "price": 63000, "topic": "k20",
             "source": "hotpotqa", "quota": 4},
            {"qid": "q0002", "text": "2+2?", "answers": ["4"],
             "difficulty": "2hop", "price": 1000, "topic": "k20"}],
        "jobs": [{"jid": "j0001", "qids": ["q0001", "q0002"], "price": 64000}],
        "total_units": 2, "n_topics": 1}))
    bank = QuestionBank.from_json(str(p))
    q = bank.get("q0001")
    assert q.topic == "k20" and q.answers == ["Paris"]
    assert not hasattr(q, "source") and not hasattr(q, "quota")
    assert bank.total_units() == 2
    assert bank.get_job("j0001").price == 64000


def test_a_bank_may_have_no_jobs_at_all():
    bank = QuestionBank([Question("q0001", "a", ["x"], "2hop", 1, "k0")])
    assert bank.jobs == {} and bank.total_units() == 0


# ---------------- build-time invariants ----------------

def test_duplicate_qids_are_a_build_error():
    with pytest.raises(BankError):
        QuestionBank([Question("q0001", "a", ["x"], "2hop", 1, "k0"),
                      Question("q0001", "b", ["y"], "2hop", 1, "k0")])


def test_duplicate_jids_are_a_build_error():
    qs = [Question("q0001", "a", ["x"], "2hop", 1, "k0")]
    with pytest.raises(BankError):
        QuestionBank(qs, [Job("j0001", ["q0001"], 1), Job("j0001", ["q0001"], 1)])


def test_an_empty_job_is_a_build_error():
    with pytest.raises(BankError):
        QuestionBank([Question("q0001", "a", ["x"], "2hop", 1, "k0")],
                     [Job("j0001", [], 0)])


def test_a_job_listing_a_question_twice_is_a_build_error():
    with pytest.raises(BankError):
        QuestionBank([Question("q0001", "a", ["x"], "2hop", 1, "k0")],
                     [Job("j0001", ["q0001", "q0001"], 2)])


def test_a_job_referencing_an_unknown_question_is_a_build_error():
    with pytest.raises(BankError) as e:
        QuestionBank([Question("q0001", "a", ["x"], "2hop", 1, "k0")],
                     [Job("j0001", ["q0001", "q9999"], 2)])
    assert "q9999" in str(e.value)


# ---------------- the real bank ----------------

def test_real_v4_bank_loads():
    bank = QuestionBank.from_json(str(REAL_BANK))
    assert len(bank.questions) == 500
    assert all(q.topic for q in bank.questions.values())
    assert len(bank.jobs) == 185
    assert bank.total_units() == sum(len(j.qids) for j in bank.jobs.values())
    assert 1400 <= bank.total_units() <= 1550          # ~1478 posted units


def test_real_jobs_are_6_to_10_questions_from_one_topic_and_priced_by_sum():
    bank = QuestionBank.from_json(str(REAL_BANK))
    for job in bank.jobs.values():
        assert 6 <= len(job.qids) <= 10
        assert len({bank.get(qid).topic for qid in job.qids}) == 1
        assert job.price == sum(bank.get(qid).price for qid in job.qids)
