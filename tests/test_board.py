"""v5 question board: one claimant per question, short-answer delivery, strikes."""
import json
import zlib

import pytest
from ca.bank import Question, QuestionBank
from ca.board import BoardError, QuestionBoard
from ca.economy import Ledger

AGENTS = ["agent_1", "agent_2", "agent_3"]


def demo_bank() -> QuestionBank:
    return QuestionBank([
        Question("q0001", "capital of France?", ["Paris"], "2hop", 18000, "k01"),
        Question("q0002", "longest river in France?", ["Loire"], "3hop", 30000, "k01"),
        Question("q0003", "2+2?", ["4", "four"], "4hop", 45000, "k07"),
    ])


def make():
    led = Ledger({a: 1000 for a in AGENTS})
    return led, QuestionBoard(demo_bank(), led)


def big_board(n=12):
    qs = [Question(f"q{i:04d}", f"question {i}?", [str(i)], "2hop", 1000 * i, "k00")
          for i in range(1, n + 1)]
    led = Ledger({a: 1000 for a in AGENTS})
    return QuestionBoard(QuestionBank(qs), led)


# ---------------- claiming ----------------

def test_claim_takes_the_question_off_the_board_and_records_a_strike():
    _, b = make()
    q = b.claim("agent_1", "q0001", 1)
    assert q.qid == "q0001" and q.text == "capital of France?"
    assert b.active["q0001"].agent == "agent_1" and b.active["q0001"].round == 1
    assert b.strikes[("q0001", "agent_1")] == 1
    assert [x.qid for x in b.open_questions()] == ["q0002", "q0003"]


def test_a_question_holds_only_one_claimant_at_a_time():
    _, b = make()
    b.claim("agent_1", "q0001", 1)
    with pytest.raises(BoardError) as e:
        b.claim("agent_2", "q0001", 1)
    assert "another agent" in str(e.value)
    with pytest.raises(BoardError) as e:
        b.claim("agent_1", "q0001", 1)
    assert "already hold" in str(e.value)


def test_expiry_returns_the_question_to_the_pool_but_keeps_the_strike():
    _, b = make()
    b.claim("agent_1", "q0001", 1)
    assert b.expire_claims(20, ttl=20) == []          # not old enough yet
    assert b.expire_claims(22, ttl=20) == ["q0001"]
    assert "q0001" in {q.qid for q in b.open_questions()}   # demand restored
    assert b.active == {}
    assert b.strikes[("q0001", "agent_1")] == 1       # attempt still spent


def test_two_strikes_close_the_question_to_that_agent_only():
    _, b = make()
    b.claim("agent_1", "q0001", 1)
    b.expire_claims(22, ttl=20)
    b.claim("agent_1", "q0001", 23)
    b.expire_claims(44, ttl=20)
    assert b.strikes[("q0001", "agent_1")] == 2
    with pytest.raises(BoardError) as e:
        b.claim("agent_1", "q0001", 45)
    assert "twice" in str(e.value)
    b.claim("agent_2", "q0001", 45)                   # other agents unaffected


def test_claim_of_unknown_qid_raises_board_error():
    _, b = make()
    with pytest.raises(BoardError):
        b.claim("agent_1", "q9999", 1)


def test_a_delivered_question_cannot_be_claimed_again():
    _, b = make()
    b.claim("agent_1", "q0001", 1)
    b.deliver("agent_1", "q0001", "Paris", 1)
    with pytest.raises(BoardError) as e:
        b.claim("agent_2", "q0001", 2)
    assert "closed" in str(e.value)


# ---------------- delivery ----------------

def test_delivery_grades_and_mints_once():
    led, b = make()
    b.claim("agent_1", "q0001", 1)
    r = b.deliver("agent_1", "q0001", "Paris", 3)
    assert r.qid == "q0001" and r.agent == "agent_1" and r.round == 3
    assert r.f1 == 1.0 and r.em == 1.0 and r.payout == 18000
    assert led.balance("agent_1") == 1000 + 18000
    assert led.minted == 18000
    assert b.closed == {"q0001"} and b.active == {}
    assert b.results == [r]


def test_partial_credit_is_rounded_price_times_f1():
    led, b = make()
    b.claim("agent_1", "q0002", 1)
    r = b.deliver("agent_1", "q0002", "the Loire river", 1)
    assert 0.0 < r.f1 < 1.0 and r.em == 0.0
    assert r.payout == round(30000 * r.f1)
    assert led.balance("agent_1") - 1000 == r.payout


def test_a_worthless_answer_still_closes_the_question_and_mints_nothing():
    led, b = make()
    b.claim("agent_1", "q0001", 1)
    r = b.deliver("agent_1", "q0001", "zzz", 1)
    assert r.f1 == 0.0 and r.payout == 0
    assert led.minted == 0 and b.closed == {"q0001"}


def test_delivery_needs_an_active_claim_by_that_agent():
    _, b = make()
    with pytest.raises(BoardError):
        b.deliver("agent_1", "q0001", "Paris", 1)
    b.claim("agent_1", "q0001", 1)
    with pytest.raises(BoardError):
        b.deliver("agent_2", "q0001", "Paris", 1)
    b.deliver("agent_1", "q0001", "Paris", 1)
    with pytest.raises(BoardError):                 # claim was consumed
        b.deliver("agent_1", "q0001", "Paris", 2)


def test_delivery_defaults_to_the_last_round_the_board_has_seen():
    _, b = make()
    b.claim("agent_1", "q0001", 7)
    r = b.deliver("agent_1", "q0001", "Paris")
    assert r.round == 7


# ---------------- listing / done ----------------

def test_open_questions_drop_claimed_and_closed_ones():
    _, b = make()
    b.claim("agent_1", "q0001", 1)
    assert [q.qid for q in b.open_questions()] == ["q0002", "q0003"]
    b.deliver("agent_1", "q0001", "Paris", 1)
    assert [q.qid for q in b.open_questions()] == ["q0002", "q0003"]


def test_open_questions_without_viewer_are_in_qid_order():
    b = big_board()
    assert [q.qid for q in b.open_questions()] == sorted(b.bank.questions)


def test_per_viewer_shuffle_is_stable_and_differs_between_viewers():
    b = big_board()
    a1 = [q.qid for q in b.open_questions("agent_1")]
    a2 = [q.qid for q in b.open_questions("agent_2")]
    assert sorted(a1) == sorted(a2) == sorted(b.bank.questions)   # same SET
    assert a1 != a2                                               # different order
    assert a1 == [q.qid for q in b.open_questions("agent_1")]     # stable
    assert a1 == sorted(a1, key=lambda q: zlib.crc32(f"agent_1:{q}".encode()))


def test_all_done_when_every_question_is_closed():
    _, b = make()
    assert not b.all_done()
    b.claim("agent_1", "q0001", 1)
    assert not b.all_done()            # a held claim is not a finished board
    b.deliver("agent_1", "q0001", "Paris", 1)
    b.claim("agent_1", "q0002", 1)
    b.deliver("agent_1", "q0002", "Loire", 1)
    assert not b.all_done()
    b.claim("agent_2", "q0003", 1)
    b.deliver("agent_2", "q0003", "4", 1)
    assert b.all_done()


# ---------------- reporting / checkpoint ----------------

def test_results_json_is_one_row_per_delivered_question():
    _, b = make()
    b.claim("agent_1", "q0001", 1)
    b.deliver("agent_1", "q0001", "Paris", 2)
    b.claim("agent_1", "q0003", 2)
    b.deliver("agent_1", "q0003", "zzz", 2)
    rows = b.results_json()
    assert json.loads(json.dumps(rows)) == rows
    assert len(rows) == 2
    assert rows[0] == {"qid": "q0001", "agent": "agent_1", "submitted": "Paris",
                       "f1": 1.0, "em": 1.0, "payout": 18000, "round": 2,
                       "price": 18000, "difficulty": "2hop", "topic": "k01"}
    assert rows[1]["f1"] == 0.0 and rows[1]["payout"] == 0
    assert "jid" not in rows[0]


def test_state_roundtrips_through_json():
    _, b = make()
    b.claim("agent_1", "q0001", 1)
    b.deliver("agent_1", "q0001", "Paris", 1)
    b.claim("agent_2", "q0002", 4)
    state = json.loads(json.dumps(b.to_state()))

    led2 = Ledger({a: 1000 for a in AGENTS})
    fresh = QuestionBoard(demo_bank(), led2)
    fresh.from_state(state)
    assert fresh.closed == b.closed
    assert fresh.strikes == b.strikes
    assert fresh.results_json() == b.results_json()
    assert (fresh.active["q0002"].agent, fresh.active["q0002"].round) == ("agent_2", 4)
    with pytest.raises(BoardError):
        fresh.claim("agent_3", "q0002", 4)      # restored claim is really active
