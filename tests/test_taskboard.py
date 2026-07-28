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


def test_expire_claims_reopens_stale_undelivered_claims():
    led, tb = setup()
    tb.claim("a", "q0001", round_no=1)
    assert tb.expire_claims(5, ttl=8) == []          # 5 - 1 <= 8: still working
    assert tb.questions["q0001"].status == "claimed"
    assert tb.expire_claims(10, ttl=8) == ["q0001"]  # 10 - 1 > 8: abandoned
    q = tb.questions["q0001"]
    assert q.status == "open" and q.claimed_by is None and q.claimed_round == 0
    assert {x.qid for x in tb.list_open()} == {"q0001", "q0002"}
    tb.claim("b", "q0001", round_no=11)              # someone else can take it now


def test_expire_claims_never_reopens_answered_questions():
    led, tb = setup()
    tb.claim("a", "q0001", round_no=1)
    tb.deliver("a", "q0001", "Paris")
    assert tb.expire_claims(100, ttl=1) == []
    assert tb.questions["q0001"].status == "closed"


def test_wrong_answer_pays_zero_and_all_done():
    led, tb = setup()
    tb.claim("a", "q0001")
    tb.deliver("a", "q0001", "London")
    assert not tb.all_done()
    tb.claim("a", "q0002")
    tb.deliver("a", "q0002", "4")
    assert tb.all_done()
