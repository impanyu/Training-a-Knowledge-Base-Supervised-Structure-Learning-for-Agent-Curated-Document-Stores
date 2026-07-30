import json
import random

import pytest

from ca.actions import dispatch
from ca.agent import Agent, ScriptedPolicy
from ca.config import LEVELS, ExperimentConfig
from ca.infra import Infra
from ca.loans import LoanError, LoanSystem
from ca.economy import Ledger
from ca.recorder import Recorder
from ca.retrieval import KeywordBackend
from ca.scheduler import Scheduler
from fixtures import demo_library, demo_posted

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]


def setup(bal=None):
    bal = bal or {"lender": 1000, "borrower": 1000}
    led = Ledger(bal)
    return led, LoanSystem(led, rate=0.01)


def make_infra(level="L0", capital=1000):
    cfg = ExperimentConfig(level=LEVELS[level], seed=0, seed_capital_total=capital)
    return Infra(cfg, demo_library(), demo_posted(), retriever=KeywordBackend(DOCS))


# ---------------- LoanSystem unit tests ----------------

def test_propose_yields_note_prefixed_id():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 100)
    assert loan.lid == "n0001"
    assert loan.status == "proposed"
    assert loan.lender == "lender" and loan.borrower == "borrower"


def test_accept_transfers_principal_lender_to_borrower():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 200)
    ls.accept("lender", loan.lid)
    assert loan.status == "active"
    assert led.balance("borrower") == 1200
    assert led.balance("lender") == 800
    assert led.conservation_ok()


def test_only_lender_can_accept():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 100)
    with pytest.raises(LoanError):
        ls.accept("borrower", loan.lid)
    assert loan.status == "proposed"


def test_accept_fails_cleanly_if_lender_broke():
    led, ls = setup({"lender": 10, "borrower": 1000})
    loan = ls.propose("borrower", "lender", 500)
    with pytest.raises(LoanError):
        ls.accept("lender", loan.lid)
    assert loan.status == "proposed"
    assert led.balance("lender") == 10
    assert led.balance("borrower") == 1000
    assert led.conservation_ok()


def test_repay_full_closes_loan():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 200)
    ls.accept("lender", loan.lid)
    loan2, paid = ls.repay("borrower", loan.lid, 200)
    assert paid == 200
    assert loan2.status == "repaid"
    assert loan2.principal == 0
    assert led.balance("borrower") == 1000
    assert led.balance("lender") == 1000
    assert led.conservation_ok()


def test_repay_partial_keeps_loan_active():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 200)
    ls.accept("lender", loan.lid)
    loan2, paid = ls.repay("borrower", loan.lid, 50)
    assert paid == 50
    assert loan2.status == "active"
    assert loan2.principal == 150
    assert led.conservation_ok()


def test_repay_caps_at_outstanding_principal():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 200)
    ls.accept("lender", loan.lid)
    loan2, paid = ls.repay("borrower", loan.lid, 9999)
    assert paid == 200
    assert loan2.status == "repaid"
    assert led.conservation_ok()


def test_only_borrower_can_repay():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 200)
    ls.accept("lender", loan.lid)
    with pytest.raises(LoanError):
        ls.repay("lender", loan.lid, 50)


def test_repay_before_accept_fails():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 200)
    with pytest.raises(LoanError):
        ls.repay("borrower", loan.lid, 50)


def test_reject_by_lender_cancels_proposal():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 200)
    ls.reject("lender", loan.lid)
    assert loan.status == "cancelled"
    with pytest.raises(LoanError):
        ls.accept("lender", loan.lid)


def test_reject_by_non_lender_fails():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 200)
    with pytest.raises(LoanError):
        ls.reject("borrower", loan.lid)


def test_cancel_by_borrower():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 200)
    ls.cancel("borrower", loan.lid)
    assert loan.status == "cancelled"


def test_cancel_by_non_borrower_fails():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 200)
    with pytest.raises(LoanError):
        ls.cancel("lender", loan.lid)


def test_cannot_loan_to_self():
    led, ls = setup()
    with pytest.raises(LoanError):
        ls.propose("lender", "lender", 100)


def test_amount_must_be_positive():
    led, ls = setup()
    with pytest.raises(LoanError):
        ls.propose("borrower", "lender", 0)


def test_unknown_loan_id():
    led, ls = setup()
    with pytest.raises(LoanError):
        ls.accept("lender", "n9999")
    with pytest.raises(LoanError):
        ls.repay("borrower", "n9999", 1)


def test_interest_tick_pays_when_solvent():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 1000)
    ls.accept("lender", loan.lid)  # borrower=2000, lender=0
    events = ls.interest_tick()
    assert len(events) == 1
    e = events[0]
    assert e == {"lid": loan.lid, "borrower": "borrower", "lender": "lender",
                 "interest": 10, "paid": True, "principal_after": 1000}
    assert led.balance("borrower") == 1990
    assert led.balance("lender") == 10
    assert led.conservation_ok()


def test_interest_tick_accumulates_total_interest_paid_only_when_paid():
    led, ls = setup({"lender": 1000, "borrower": 1000})
    loan = ls.propose("borrower", "lender", 500)
    ls.accept("lender", loan.lid)  # borrower=1500, lender=500
    ls.interest_tick()  # interest=5, paid
    assert ls.total_interest_paid == 5
    led.burn("borrower", ls.ledger.balance("borrower") - 3)  # leave 3, less than next interest
    ls.interest_tick()  # capitalized this time, not paid
    assert ls.total_interest_paid == 5  # unchanged: capitalization is not a payment


def test_interest_tick_capitalizes_when_borrower_broke():
    led, ls = setup({"lender": 1000, "borrower": 1000})
    loan = ls.propose("borrower", "lender", 500)
    ls.accept("lender", loan.lid)  # borrower=1500, lender=500
    led.burn("borrower", 1497)  # borrower=3, less than interest (5)
    events = ls.interest_tick()
    e = events[0]
    assert e["paid"] is False
    assert e["interest"] == 5
    assert e["principal_after"] == 505
    assert loan.principal == 505
    assert led.balance("lender") == 500  # untouched: no transfer on capitalization
    assert led.balance("borrower") == 3
    assert led.conservation_ok()


def test_interest_tick_minimum_one_token():
    led, ls = setup()
    loan = ls.propose("borrower", "lender", 10)  # 1% of 10 rounds to 0
    ls.accept("lender", loan.lid)
    events = ls.interest_tick()
    assert events[0]["interest"] == 1  # max(1, round(10*0.01))


def test_interest_tick_ignores_non_active_loans():
    led, ls = setup()
    proposed = ls.propose("borrower", "lender", 100)
    active = ls.propose("borrower", "lender", 200)
    ls.accept("lender", active.lid)
    repaid = ls.propose("borrower", "lender", 50)
    ls.accept("lender", repaid.lid)
    ls.repay("borrower", repaid.lid, 50)
    events = ls.interest_tick()
    assert {e["lid"] for e in events} == {active.lid}


def test_pending_for():
    led, ls = setup({"lender": 1000, "borrower": 1000, "other": 1000})
    l1 = ls.propose("borrower", "lender", 100)   # proposed, awaiting lender
    l2 = ls.propose("other", "lender", 50)
    ls.accept("lender", l2.lid)                   # now active
    pend_lender = ls.pending_for("lender")
    assert {p.lid for p in pend_lender} == {l1.lid, l2.lid}
    pend_borrower = ls.pending_for("borrower")
    assert pend_borrower == []
    pend_other = ls.pending_for("other")
    assert {p.lid for p in pend_other} == {l2.lid}


# ---------------- dispatch integration ----------------

def test_dispatch_loan_lifecycle():
    infra = make_infra()
    out = dispatch(infra, "agent_1", "propose_loan", {"to": "agent_2", "amount": 100})
    assert "proposed" in out and "may accept or ignore" in out
    loan = list(infra.loans.loans.values())[0]
    assert loan.lender == "agent_2" and loan.borrower == "agent_1"
    out2 = dispatch(infra, "agent_2", "accept_loan", {"loan_id": loan.lid})
    assert "accepted" in out2
    assert infra.ledger.balance("agent_1") == 125 + 100
    out3 = dispatch(infra, "agent_1", "repay_loan", {"loan_id": loan.lid, "amount": 100})
    assert "repaid" in out3
    assert loan.status == "repaid"
    assert infra.ledger.conservation_ok()


def test_dispatch_loan_wrong_party_returns_error_string():
    infra = make_infra()
    dispatch(infra, "agent_1", "propose_loan", {"to": "agent_2", "amount": 100})
    loan = list(infra.loans.loans.values())[0]
    out = dispatch(infra, "agent_3", "accept_loan", {"loan_id": loan.lid})
    assert out.startswith("ERROR")
    assert loan.status == "proposed"


def test_dispatch_unknown_agent_in_propose_loan():
    infra = make_infra()
    out = dispatch(infra, "agent_1", "propose_loan", {"to": "agent_99", "amount": 100})
    assert out.startswith("ERROR: unknown agent")


# ---------------- scheduler integration ----------------

def test_repay_capped_by_borrower_balance():
    """borrower has less than principal: repay(1000) pays only what they have"""
    # setup a loan of 200 accepted, then drain borrower to 30 via ledger.burn
    led, ls = setup({"lender": 1000, "borrower": 1000})
    loan = ls.propose("borrower", "lender", 200)
    ls.accept("lender", loan.lid)  # borrower=1200, lender=800
    led.burn("borrower", 1170)  # borrower=30, less than principal (200)
    loan2, paid = ls.repay("borrower", loan.lid, 1000)
    # pays only what they have (30), principal becomes 170, no exception
    assert paid == 30
    assert loan2.status == "active"  # not repaid yet
    assert loan2.principal == 170
    assert led.balance("borrower") == 0
    assert led.balance("lender") == 830  # 800 + 30
    assert led.conservation_ok()


def test_repay_broke_borrower_returns_error_via_dispatch():
    """drain borrower to 0; dispatch repay_loan -> result startswith "ERROR"; run does not raise"""
    infra = make_infra()
    dispatch(infra, "agent_1", "propose_loan", {"to": "agent_2", "amount": 100})
    loan = list(infra.loans.loans.values())[0]
    dispatch(infra, "agent_2", "accept_loan", {"loan_id": loan.lid})
    infra.ledger.burn("agent_1", infra.ledger.balance("agent_1"))  # agent_1 = 0
    out = dispatch(infra, "agent_1", "repay_loan", {"loan_id": loan.lid, "amount": 100})
    assert out.startswith("ERROR")
    assert loan.status == "active"  # loan unchanged


def test_scheduler_logs_interest_events_and_conserves(tmp_path):
    cfg = ExperimentConfig(level=LEVELS["L0"], seed=1, seed_capital_total=1000, max_rounds=3)
    infra = Infra(cfg, demo_library(), demo_posted(), retriever=KeywordBackend(DOCS))
    scripts = {
        "agent_1": [("propose_loan", {"to": "agent_2", "amount": 100})],
        "agent_2": [("check_balance", {}), ("accept_loan", {"loan_id": "n0001"})],
    }
    agents = [Agent(a, cfg, infra, ScriptedPolicy(scripts.get(a, []), in_tokens=0, out_tokens=0))
              for a in infra.agent_ids]
    rec = Recorder(str(tmp_path))
    sched = Scheduler(infra, agents, cfg, rec, random.Random(cfg.seed))
    summary = sched.run()
    trace = [json.loads(l) for l in open(tmp_path / "trace.jsonl")]
    interest_events = [e for e in trace if e["action"] == "__interest__"]
    assert len(interest_events) >= 1
    ev = interest_events[0]
    assert ev["agent"] == "agent_1"
    assert ev["category"] == "admin"
    assert ev["tokens_in"] == 0 and ev["tokens_out"] == 0
    assert ev["input"] == {"lid": "n0001", "lender": "agent_2"}
    assert ev["result"] in ("paid 1", "capitalized 1")
    assert ev["balance_after"] == infra.ledger.balance("agent_1")
    assert summary["conservation_ok"] is True
