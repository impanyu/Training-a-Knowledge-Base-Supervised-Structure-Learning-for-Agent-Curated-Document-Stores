"""v4.1 job board: one claimant per job, all-or-nothing delivery, strikes."""
import json
import zlib

import pytest
from ca.bank import Job, Question, QuestionBank
from ca.board import BoardError, JobBoard
from ca.economy import Ledger

AGENTS = ["agent_1", "agent_2", "agent_3"]


def demo_bank() -> QuestionBank:
    return QuestionBank([
        Question("q0001", "capital of France?", ["Paris"], "2hop", 63000, "k01"),
        Question("q0002", "longest river in France?", ["Loire"], "3hop", 105000, "k01"),
        Question("q0003", "2+2?", ["4", "four"], "4hop", 157500, "k07"),
    ], [Job("j0001", ["q0001", "q0002"], 168000),
        Job("j0002", ["q0002", "q0003"], 262500)])


def make():
    led = Ledger({a: 1000 for a in AGENTS})
    return led, JobBoard(demo_bank(), led)


def big_board(n=12):
    qs = [Question(f"q{i:04d}", f"question {i}?", [str(i)], "2hop", 1000 * i, "k00")
          for i in range(1, n + 1)]
    jobs = [Job(f"j{i:04d}", [f"q{i:04d}"], 1000 * i) for i in range(1, n + 1)]
    led = Ledger({a: 1000 for a in AGENTS})
    return JobBoard(QuestionBank(qs, jobs), led)


# ---------------- claiming ----------------

def test_total_units_counts_job_membership_not_questions():
    _, b = make()
    assert b.bank.total_units() == 4          # q0002 is posted in both jobs


def test_claim_takes_the_job_off_the_board_and_records_a_strike():
    _, b = make()
    job = b.claim("agent_1", "j0001", 1)
    assert job.jid == "j0001" and job.qids == ["q0001", "q0002"]
    assert b.active["j0001"].agent == "agent_1" and b.active["j0001"].round == 1
    assert b.strikes[("j0001", "agent_1")] == 1
    assert [j.jid for j in b.open_jobs()] == ["j0002"]


def test_a_job_holds_only_one_claimant_at_a_time():
    _, b = make()
    b.claim("agent_1", "j0001", 1)
    with pytest.raises(BoardError) as e:
        b.claim("agent_2", "j0001", 1)
    assert "another agent" in str(e.value)
    with pytest.raises(BoardError) as e:
        b.claim("agent_1", "j0001", 1)
    assert "already hold" in str(e.value)


def test_expiry_returns_the_job_to_the_pool_but_keeps_the_strike():
    _, b = make()
    b.claim("agent_1", "j0001", 1)
    assert b.expire_claims(20, ttl=20) == []          # not old enough yet
    assert b.expire_claims(22, ttl=20) == ["j0001"]
    assert "j0001" in {j.jid for j in b.open_jobs()}  # demand restored
    assert b.active == {}
    assert b.strikes[("j0001", "agent_1")] == 1       # attempt still spent


def test_two_strikes_close_the_job_to_that_agent_only():
    _, b = make()
    b.claim("agent_1", "j0001", 1)
    b.expire_claims(22, ttl=20)
    b.claim("agent_1", "j0001", 23)
    b.expire_claims(44, ttl=20)
    assert b.strikes[("j0001", "agent_1")] == 2
    with pytest.raises(BoardError) as e:
        b.claim("agent_1", "j0001", 45)
    assert "twice" in str(e.value)
    b.claim("agent_2", "j0001", 45)                   # other agents unaffected


def test_claim_of_unknown_jid_raises_board_error():
    _, b = make()
    with pytest.raises(BoardError):
        b.claim("agent_1", "j9999", 1)


def test_a_delivered_job_cannot_be_claimed_again():
    _, b = make()
    b.claim("agent_1", "j0001", 1)
    b.deliver("agent_1", "j0001", {"q0001": "Paris", "q0002": "Loire"}, 1)
    with pytest.raises(BoardError) as e:
        b.claim("agent_2", "j0001", 2)
    assert "closed" in str(e.value)


# ---------------- delivery ----------------

def test_delivery_grades_every_member_and_mints_once():
    led, b = make()
    b.claim("agent_1", "j0001", 1)
    rows = b.deliver("agent_1", "j0001", {"q0001": "Paris", "q0002": "Loire"}, 3)
    assert [r.qid for r in rows] == ["q0001", "q0002"]
    assert all(r.jid == "j0001" and r.agent == "agent_1" and r.round == 3 for r in rows)
    assert all(r.f1 == 1.0 and r.em == 1.0 for r in rows)
    assert [r.payout for r in rows] == [63000, 105000]
    assert led.balance("agent_1") == 1000 + 168000
    assert led.minted == 168000                 # ONE settlement for the job
    assert b.closed == {"j0001"} and b.active == {}
    assert b.results == rows


def test_partial_credit_is_rounded_price_times_f1_per_question():
    led, b = make()
    b.claim("agent_1", "j0001", 1)
    rows = b.deliver("agent_1", "j0001",
                     {"q0001": "Paris", "q0002": "the Loire river"}, 1)
    assert rows[0].f1 == 1.0
    assert 0.0 < rows[1].f1 < 1.0 and rows[1].em == 0.0
    assert rows[1].payout == round(105000 * rows[1].f1)
    assert led.balance("agent_1") - 1000 == sum(r.payout for r in rows)


def test_a_worthless_job_still_closes_and_mints_nothing():
    led, b = make()
    b.claim("agent_1", "j0001", 1)
    rows = b.deliver("agent_1", "j0001", {"q0001": "zzz", "q0002": "zzz"}, 1)
    assert all(r.f1 == 0.0 and r.payout == 0 for r in rows)
    assert led.minted == 0 and b.closed == {"j0001"}


def test_an_incomplete_map_is_rejected_and_the_claim_survives():
    led, b = make()
    b.claim("agent_1", "j0001", 1)
    with pytest.raises(BoardError) as e:
        b.deliver("agent_1", "j0001", {"q0001": "Paris"}, 1)
    assert "q0002" in str(e.value) and "STILL YOURS" in str(e.value)
    assert b.active["j0001"].agent == "agent_1"     # attempt NOT consumed
    assert b.results == [] and led.minted == 0
    rows = b.deliver("agent_1", "j0001", {"q0001": "Paris", "q0002": "Loire"}, 2)
    assert len(rows) == 2                           # the retry is the one attempt


def test_a_map_naming_a_question_outside_the_job_is_rejected_free():
    _, b = make()
    b.claim("agent_1", "j0001", 1)
    with pytest.raises(BoardError) as e:
        b.deliver("agent_1", "j0001",
                  {"q0001": "Paris", "q0002": "Loire", "q0003": "4"}, 1)
    assert "q0003" in str(e.value)
    assert b.active["j0001"].agent == "agent_1"
    assert b.strikes[("j0001", "agent_1")] == 1


def test_delivery_needs_an_active_claim_by_that_agent():
    _, b = make()
    full = {"q0001": "Paris", "q0002": "Loire"}
    with pytest.raises(BoardError):
        b.deliver("agent_1", "j0001", full, 1)
    b.claim("agent_1", "j0001", 1)
    with pytest.raises(BoardError):
        b.deliver("agent_2", "j0001", full, 1)
    b.deliver("agent_1", "j0001", full, 1)
    with pytest.raises(BoardError):                 # claim was consumed
        b.deliver("agent_1", "j0001", full, 2)


def test_delivery_defaults_to_the_last_round_the_board_has_seen():
    _, b = make()
    b.claim("agent_1", "j0001", 7)
    rows = b.deliver("agent_1", "j0001", {"q0001": "Paris", "q0002": "Loire"})
    assert all(r.round == 7 for r in rows)


def test_a_question_in_two_jobs_is_paid_under_each_of_them():
    led, b = make()
    b.claim("agent_1", "j0001", 1)
    b.deliver("agent_1", "j0001", {"q0001": "Paris", "q0002": "Loire"}, 1)
    b.claim("agent_2", "j0002", 1)
    b.deliver("agent_2", "j0002", {"q0002": "Loire", "q0003": "4"}, 1)
    paid = [(r.jid, r.qid, r.payout) for r in b.results if r.qid == "q0002"]
    assert paid == [("j0001", "q0002", 105000), ("j0002", "q0002", 105000)]


# ---------------- listing / done ----------------

def test_open_jobs_drop_claimed_and_closed_ones():
    _, b = make()
    b.claim("agent_1", "j0001", 1)
    assert [j.jid for j in b.open_jobs()] == ["j0002"]
    b.deliver("agent_1", "j0001", {"q0001": "Paris", "q0002": "Loire"}, 1)
    assert [j.jid for j in b.open_jobs()] == ["j0002"]


def test_open_jobs_without_viewer_are_in_jid_order():
    b = big_board()
    assert [j.jid for j in b.open_jobs()] == sorted(b.bank.jobs)


def test_per_viewer_shuffle_is_stable_and_differs_between_viewers():
    b = big_board()
    a1 = [j.jid for j in b.open_jobs("agent_1")]
    a2 = [j.jid for j in b.open_jobs("agent_2")]
    assert sorted(a1) == sorted(a2) == sorted(b.bank.jobs)    # same SET
    assert a1 != a2                                           # different order
    assert a1 == [j.jid for j in b.open_jobs("agent_1")]      # stable
    assert a1 == sorted(a1, key=lambda j: zlib.crc32(f"agent_1:{j}".encode()))


def test_all_done_when_every_job_is_closed():
    _, b = make()
    assert not b.all_done()
    b.claim("agent_1", "j0001", 1)
    assert not b.all_done()            # a held claim is not a finished board
    b.deliver("agent_1", "j0001", {"q0001": "Paris", "q0002": "Loire"}, 1)
    assert not b.all_done()
    b.claim("agent_1", "j0002", 1)
    b.deliver("agent_1", "j0002", {"q0002": "Loire", "q0003": "4"}, 1)
    assert b.all_done()


# ---------------- reporting / checkpoint ----------------

def test_results_json_is_one_row_per_delivered_unit():
    _, b = make()
    b.claim("agent_1", "j0001", 1)
    b.deliver("agent_1", "j0001", {"q0001": "Paris", "q0002": "zzz"}, 2)
    rows = b.results_json()
    assert json.loads(json.dumps(rows)) == rows
    assert len(rows) == 2
    assert rows[0]["qid"] == "q0001" and rows[0]["jid"] == "j0001"
    assert rows[0]["agent"] == "agent_1" and rows[0]["round"] == 2
    assert rows[0]["difficulty"] == "2hop" and rows[0]["topic"] == "k01"
    assert rows[0]["price"] == 63000 and rows[0]["payout"] == 63000
    assert rows[1]["f1"] == 0.0 and rows[1]["payout"] == 0


def test_state_roundtrips_through_json():
    _, b = make()
    b.claim("agent_1", "j0001", 1)
    b.deliver("agent_1", "j0001", {"q0001": "Paris", "q0002": "Loire"}, 1)
    b.claim("agent_2", "j0002", 4)
    state = json.loads(json.dumps(b.to_state()))

    led2 = Ledger({a: 1000 for a in AGENTS})
    fresh = JobBoard(demo_bank(), led2)
    fresh.from_state(state)
    assert fresh.closed == b.closed
    assert fresh.strikes == b.strikes
    assert fresh.results_json() == b.results_json()
    assert (fresh.active["j0002"].agent, fresh.active["j0002"].round) == ("agent_2", 4)
    with pytest.raises(BoardError):
        fresh.claim("agent_3", "j0002", 4)      # restored claim is really active
