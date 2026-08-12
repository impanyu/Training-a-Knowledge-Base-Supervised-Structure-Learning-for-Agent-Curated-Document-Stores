"""v6 question board: one claimant per question, short-answer delivery,
release, strikes."""
import json
import zlib

import pytest
from ca.bank import Question, QuestionBank
from ca.board import BoardError, QuestionBoard

def demo_bank() -> QuestionBank:
    return QuestionBank([
        Question("q0001", "capital of France?", ["Paris"], "2hop", 18000, "k01"),
        Question("q0002", "longest river in France?", ["Loire"], "3hop", 30000, "k01"),
        Question("q0003", "2+2?", ["4", "four"], "4hop", 45000, "k07"),
    ])


def make():
    return QuestionBoard(demo_bank())


def big_board(n=12):
    qs = [Question(f"q{i:04d}", f"question {i}?", [str(i)], "2hop", 1000 * i, "k00")
          for i in range(1, n + 1)]
    return QuestionBoard(QuestionBank(qs))


# ---------------- claiming ----------------

def test_claim_takes_the_question_off_the_board_and_records_a_strike():
    b = make()
    q = b.claim("agent_1", "q0001", 1)
    assert q.qid == "q0001" and q.text == "capital of France?"
    assert b.active["q0001"].agent == "agent_1" and b.active["q0001"].round == 1
    assert b.strikes[("q0001", "agent_1")] == 1
    assert [x.qid for x in b.open_questions()] == ["q0002", "q0003"]


def test_a_question_holds_only_one_claimant_at_a_time():
    b = make()
    b.claim("agent_1", "q0001", 1)
    with pytest.raises(BoardError) as e:
        b.claim("agent_2", "q0001", 1)
    assert "another agent" in str(e.value)
    with pytest.raises(BoardError) as e:
        b.claim("agent_1", "q0001", 1)
    assert "already hold" in str(e.value)


def test_expiry_returns_the_question_to_the_pool_but_keeps_the_strike():
    b = make()
    b.claim("agent_1", "q0001", 1)
    assert b.expire_claims(20, ttl=20) == []          # not old enough yet
    assert b.expire_claims(22, ttl=20) == ["q0001"]
    assert "q0001" in {q.qid for q in b.open_questions()}   # demand restored
    assert b.active == {}
    assert b.strikes[("q0001", "agent_1")] == 1       # attempt still spent


def test_two_strikes_close_the_question_to_that_agent_only():
    b = make()
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
    b = make()
    with pytest.raises(BoardError):
        b.claim("agent_1", "q9999", 1)


def test_a_delivered_question_cannot_be_claimed_again():
    b = make()
    b.claim("agent_1", "q0001", 1)
    b.deliver("agent_1", "q0001", "Paris", 1)
    with pytest.raises(BoardError) as e:
        b.claim("agent_2", "q0001", 2)
    assert "closed" in str(e.value)


# ---------------- delivery ----------------

def test_delivery_grades_and_closes_once():
    b = make()
    b.claim("agent_1", "q0001", 1)
    r = b.deliver("agent_1", "q0001", "Paris", 3)
    assert r.qid == "q0001" and r.agent == "agent_1" and r.round == 3
    assert r.f1 == 1.0 and r.em == 1.0
    assert b.closed == {"q0001"} and b.active == {}
    assert b.results == [r]


def test_a_partial_answer_is_graded_between_zero_and_one():
    b = make()
    b.claim("agent_1", "q0002", 1)
    r = b.deliver("agent_1", "q0002", "the Loire river", 1)
    assert 0.0 < r.f1 < 1.0 and r.em == 0.0


def test_a_worthless_answer_still_closes_the_question():
    b = make()
    b.claim("agent_1", "q0001", 1)
    r = b.deliver("agent_1", "q0001", "zzz", 1)
    assert r.f1 == 0.0 and b.closed == {"q0001"}


def test_a_delivered_result_carries_no_payout_field():
    b = make()
    b.claim("agent_1", "q0001", 1)
    r = b.deliver("agent_1", "q0001", "Paris", 1)
    assert not hasattr(r, "payout")


# ---------------- release ----------------

def test_release_returns_the_question_to_the_open_board():
    b = make()
    b.claim("agent_1", "q0001", 1)
    q = b.release("agent_1", "q0001")
    assert q.qid == "q0001"
    assert b.active == {}
    assert "q0001" in {x.qid for x in b.open_questions()}
    b.claim("agent_2", "q0001", 2)          # anyone may take it now
    assert b.active["q0001"].agent == "agent_2"


def test_release_needs_a_claim_held_by_the_caller():
    b = make()
    with pytest.raises(BoardError) as e:
        b.release("agent_1", "q0001")
    assert "no active claim" in str(e.value)
    b.claim("agent_1", "q0001", 1)
    with pytest.raises(BoardError):
        b.release("agent_2", "q0001")       # not the holder
    assert b.active["q0001"].agent == "agent_1"


def test_release_keeps_the_strike_so_attempts_stay_capped():
    b = make()
    b.claim("agent_1", "q0001", 1)
    b.release("agent_1", "q0001")
    assert b.strikes[("q0001", "agent_1")] == 1
    b.claim("agent_1", "q0001", 2)          # second and last attempt
    b.release("agent_1", "q0001")
    with pytest.raises(BoardError):
        b.claim("agent_1", "q0001", 3)


def test_release_of_an_unknown_qid_raises_board_error():
    b = make()
    with pytest.raises(BoardError):
        b.release("agent_1", "q9999")


def test_delivery_needs_an_active_claim_by_that_agent():
    b = make()
    with pytest.raises(BoardError):
        b.deliver("agent_1", "q0001", "Paris", 1)
    b.claim("agent_1", "q0001", 1)
    with pytest.raises(BoardError):
        b.deliver("agent_2", "q0001", "Paris", 1)
    b.deliver("agent_1", "q0001", "Paris", 1)
    with pytest.raises(BoardError):                 # claim was consumed
        b.deliver("agent_1", "q0001", "Paris", 2)


def test_delivery_defaults_to_the_last_round_the_board_has_seen():
    b = make()
    b.claim("agent_1", "q0001", 7)
    r = b.deliver("agent_1", "q0001", "Paris")
    assert r.round == 7


# ---------------- listing / done ----------------

def test_open_questions_drop_claimed_and_closed_ones():
    b = make()
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
    b = make()
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
    b = make()
    b.claim("agent_1", "q0001", 1)
    b.deliver("agent_1", "q0001", "Paris", 2)
    b.claim("agent_1", "q0003", 2)
    b.deliver("agent_1", "q0003", "zzz", 2)
    rows = b.results_json()
    assert json.loads(json.dumps(rows)) == rows
    assert len(rows) == 2
    assert rows[0] == {"qid": "q0001", "agent": "agent_1", "submitted": "Paris",
                       "f1": 1.0, "em": 1.0, "round": 2,
                       "price": 18000, "difficulty": "2hop", "topic": "k01"}
    assert rows[1]["f1"] == 0.0
    assert "payout" not in rows[0]      # price survives as INERT slicing metadata


def test_state_roundtrips_through_json():
    b = make()
    b.claim("agent_1", "q0001", 1)
    b.deliver("agent_1", "q0001", "Paris", 1)
    b.claim("agent_2", "q0002", 4)
    state = json.loads(json.dumps(b.to_state()))

    fresh = QuestionBoard(demo_bank())
    fresh.from_state(state)
    assert fresh.closed == b.closed
    assert fresh.strikes == b.strikes
    assert fresh.results_json() == b.results_json()
    assert (fresh.active["q0002"].agent, fresh.active["q0002"].round) == ("agent_2", 4)
    with pytest.raises(BoardError):
        fresh.claim("agent_3", "q0002", 4)      # restored claim is really active
