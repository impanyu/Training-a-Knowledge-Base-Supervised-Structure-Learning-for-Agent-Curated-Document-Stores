"""v4 board: quota accounting, strikes, per-viewer shuffle."""
import json
import zlib

import pytest
from ca.bank import Question, QuestionBank
from ca.board import BoardError, QuestionBoard
from ca.economy import Ledger

AGENTS = ["agent_1", "agent_2", "agent_3"]


def demo_bank(quotas=(2, 1, 3)) -> QuestionBank:
    return QuestionBank([
        Question("q0001", "capital of France?", ["Paris"], "2hop", 63000, quotas[0], "k01"),
        Question("q0002", "longest river in France?", ["Loire"], "3hop", 105000, quotas[1], "k01"),
        Question("q0003", "2+2?", ["4", "four"], "4hop", 157500, quotas[2], "k07"),
    ])


def make(quotas=(2, 1, 3)):
    led = Ledger({a: 1000 for a in AGENTS})
    return led, QuestionBoard(demo_bank(quotas), led)


def big_board(n=12):
    qs = [Question(f"q{i:04d}", f"question {i}?", [str(i)], "2hop", 1000 * i, 2, "k00")
          for i in range(1, n + 1)]
    led = Ledger({a: 1000 for a in AGENTS})
    return QuestionBoard(QuestionBank(qs), led)


# ---------------- quota accounting ----------------

def test_remaining_starts_at_quota_and_total_units_matches():
    _, b = make()
    assert b.remaining == {"q0001": 2, "q0002": 1, "q0003": 3}
    assert b.bank.total_units() == 6


def test_claim_decrements_remaining_and_records_a_strike():
    _, b = make()
    q = b.claim("agent_1", "q0001", 1)
    assert q.qid == "q0001" and q.price == 63000
    assert b.remaining["q0001"] == 1
    assert b.strikes[("q0001", "agent_1")] == 1
    assert [c.agent for c in b.active["q0001"]] == ["agent_1"]
    assert b.active["q0001"][0].round == 1


def test_claim_fails_when_no_units_remain():
    _, b = make()
    b.claim("agent_1", "q0002", 1)
    assert b.remaining["q0002"] == 0
    with pytest.raises(BoardError):
        b.claim("agent_2", "q0002", 1)


def test_expiry_returns_the_unit_to_the_pool_but_keeps_the_strike():
    _, b = make()
    b.claim("agent_1", "q0001", 1)
    assert b.expire_claims(5, ttl=8) == []          # not old enough yet
    assert b.expire_claims(10, ttl=8) == ["q0001"]
    assert b.remaining["q0001"] == 2                # demand restored
    assert b.active["q0001"] == []
    assert b.strikes[("q0001", "agent_1")] == 1     # attempt still spent


def test_two_strikes_close_the_question_to_that_agent_only():
    _, b = make()
    b.claim("agent_1", "q0001", 1)
    b.expire_claims(10, ttl=8)
    b.claim("agent_1", "q0001", 11)
    b.expire_claims(20, ttl=8)
    assert b.strikes[("q0001", "agent_1")] == 2
    with pytest.raises(BoardError) as e:
        b.claim("agent_1", "q0001", 21)
    assert "twice" in str(e.value)
    b.claim("agent_2", "q0001", 21)                 # other agents unaffected
    assert b.remaining["q0001"] == 1


def test_an_agent_may_not_hold_two_active_claims_on_one_question():
    _, b = make()
    b.claim("agent_1", "q0003", 1)
    with pytest.raises(BoardError) as e:
        b.claim("agent_1", "q0003", 1)
    assert "already" in str(e.value)
    assert b.remaining["q0003"] == 2


def test_several_agents_hold_the_same_question_concurrently_when_quota_allows():
    _, b = make()
    b.claim("agent_1", "q0003", 1)
    b.claim("agent_2", "q0003", 1)
    b.claim("agent_3", "q0003", 2)
    assert b.remaining["q0003"] == 0
    assert sorted(c.agent for c in b.active["q0003"]) == AGENTS


def test_claim_of_unknown_qid_raises_board_error():
    _, b = make()
    with pytest.raises(BoardError):
        b.claim("agent_1", "q9999", 1)


# ---------------- delivery ----------------

def test_delivery_grades_mints_and_consumes_the_unit():
    led, b = make()
    b.claim("agent_1", "q0001", 1)
    r = b.deliver("agent_1", "q0001", "Paris", 3)
    assert (r.qid, r.agent, r.submitted, r.round) == ("q0001", "agent_1", "Paris", 3)
    assert r.f1 == 1.0 and r.em == 1.0 and r.payout == 63000
    assert led.balance("agent_1") == 1000 + 63000
    assert led.minted == 63000
    assert b.remaining["q0001"] == 1                # NOT restored: consumed
    assert b.active["q0001"] == []
    assert b.results == [r]


def test_partial_credit_is_rounded_price_times_f1():
    led, b = make()
    b.claim("agent_1", "q0002", 1)
    r = b.deliver("agent_1", "q0002", "the Loire river", 1)
    assert 0.0 < r.f1 < 1.0 and r.em == 0.0
    assert r.payout == round(105000 * r.f1) == led.balance("agent_1") - 1000


def test_a_worthless_answer_still_consumes_the_unit_and_mints_nothing():
    led, b = make()
    b.claim("agent_1", "q0001", 1)
    r = b.deliver("agent_1", "q0001", "zzz", 1)
    assert r.f1 == 0.0 and r.payout == 0
    assert led.minted == 0 and b.remaining["q0001"] == 1


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
    assert b.deliver("agent_1", "q0001", "Paris").round == 7


def test_a_question_may_be_delivered_by_several_agents_up_to_its_quota():
    led, b = make()
    for agent in ("agent_1", "agent_2", "agent_3"):
        b.claim(agent, "q0003", 1)
        b.deliver(agent, "q0003", "4", 1)
    assert len(b.results) == 3
    assert b.remaining["q0003"] == 0
    assert led.minted == 3 * 157500


# ---------------- open questions / done ----------------

def test_open_questions_drop_exhausted_ones():
    _, b = make()
    b.claim("agent_1", "q0002", 1)
    assert [q.qid for q in b.open_questions()] == ["q0001", "q0003"]


def test_open_questions_without_viewer_are_in_qid_order():
    b = big_board()
    assert [q.qid for q in b.open_questions()] == sorted(b.remaining)


def test_per_viewer_shuffle_is_stable_and_differs_between_viewers():
    b = big_board()
    a1 = [q.qid for q in b.open_questions("agent_1")]
    a2 = [q.qid for q in b.open_questions("agent_2")]
    assert sorted(a1) == sorted(a2) == sorted(b.remaining)   # same SET
    assert a1 != a2                                          # different order
    assert a1 == [q.qid for q in b.open_questions("agent_1")]  # stable
    assert a1 == sorted(a1, key=lambda q: zlib.crc32(f"agent_1:{q}".encode()))


def test_all_done_when_every_unit_is_delivered():
    _, b = make(quotas=(1, 1, 1))
    assert not b.all_done()
    for qid, ans in (("q0001", "Paris"), ("q0002", "Loire"), ("q0003", "4")):
        b.claim("agent_1", qid, 1)
        assert not b.all_done()          # a held claim is not a finished board
        b.deliver("agent_1", qid, ans, 1)
    assert b.all_done()


def test_all_done_is_false_while_a_unit_is_only_claimed():
    _, b = make(quotas=(1, 1, 1))
    for qid in ("q0001", "q0002", "q0003"):
        b.claim("agent_1", qid, 1)
    assert all(v == 0 for v in b.remaining.values())
    assert not b.all_done()


# ---------------- reporting / checkpoint ----------------

def test_results_json_is_one_row_per_delivered_unit():
    _, b = make()
    b.claim("agent_1", "q0001", 1)
    b.deliver("agent_1", "q0001", "Paris", 2)
    rows = b.results_json()
    assert json.loads(json.dumps(rows)) == rows
    assert rows[0]["qid"] == "q0001" and rows[0]["agent"] == "agent_1"
    assert rows[0]["difficulty"] == "2hop" and rows[0]["topic"] == "k01"
    assert rows[0]["price"] == 63000 and rows[0]["payout"] == 63000
    assert rows[0]["f1"] == 1.0 and rows[0]["em"] == 1.0 and rows[0]["round"] == 2


def test_state_roundtrips_through_json():
    _, b = make()
    b.claim("agent_1", "q0001", 1)
    b.deliver("agent_1", "q0001", "Paris", 1)
    b.claim("agent_2", "q0003", 4)
    state = json.loads(json.dumps(b.to_state()))

    led2 = Ledger({a: 1000 for a in AGENTS})
    fresh = QuestionBoard(demo_bank(), led2)
    fresh.from_state(state)
    assert fresh.remaining == b.remaining
    assert fresh.strikes == b.strikes
    assert fresh.results_json() == b.results_json()
    assert [(c.agent, c.round) for c in fresh.active["q0003"]] == [("agent_2", 4)]
    with pytest.raises(BoardError):
        fresh.claim("agent_2", "q0003", 4)      # restored claim is really active
