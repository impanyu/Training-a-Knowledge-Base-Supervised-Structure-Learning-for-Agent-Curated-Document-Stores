import pytest
from ca.economy import Ledger
from ca.taskboard import TaskBoard, Question, BoardError


def setup():
    qs = [Question("q0001", "capital of France?", ["Paris"], "easy", 100),
          Question("q0002", "2+2?", ["4", "four"], "easy", 100)]
    led = Ledger({"a": 10, "b": 10})
    return led, TaskBoard(qs, led)


def test_claim_hides_from_open_list():
    led, tb = setup()
    tb.claim("a", "q0001")
    assert [q.qid for q in tb.list_open()] == ["q0002"]
    with pytest.raises(BoardError):
        tb.claim("b", "q0001")


def test_deliver_pays_by_f1_and_closes():
    led, tb = setup()
    tb.claim("a", "q0001")
    score, payout = tb.deliver("a", "q0001", "Paris")
    assert score == 1.0 and payout == 100
    assert led.balance("a") == 110
    assert led.conservation_ok()
    with pytest.raises(BoardError):
        tb.deliver("a", "q0001", "Paris")  # one shot


def test_deliver_requires_claimer():
    led, tb = setup()
    tb.claim("a", "q0001")
    with pytest.raises(BoardError):
        tb.deliver("b", "q0001", "Paris")


def test_wrong_answer_pays_zero_and_all_done():
    led, tb = setup()
    tb.claim("a", "q0001")
    tb.deliver("a", "q0001", "London")
    assert not tb.all_done()
    tb.claim("a", "q0002")
    tb.deliver("a", "q0002", "4")
    assert tb.all_done()
