import pytest
from ca.economy import Ledger
from ca.contracts import ContractSystem, ContractError


def setup():
    led = Ledger({"boss": 100, "worker": 10})
    return led, ContractSystem(led)


def test_propose_accept_deliver_flow():
    led, cs = setup()
    c = cs.propose("boss", "worker", "find X", 30)
    assert c.status == "proposed" and c.awaiting == "worker"
    cs.accept("worker", c.cid)
    assert led.balance("boss") == 70 and led.escrow[c.cid] == 30
    done = cs.deliver("worker", c.cid, "X is 42")
    assert done.status == "delivered" and done.deliverable == "X is 42"
    assert led.balance("worker") == 40 and c.cid not in led.escrow
    assert led.conservation_ok()


def test_counter_offer_flips_awaiting():
    led, cs = setup()
    c = cs.propose("boss", "worker", "t", 10)
    cs.counter("worker", c.cid, 20)
    assert c.price == 20 and c.awaiting == "boss"
    cs.accept("boss", c.cid)          # proposer accepts worker's counter
    assert led.escrow[c.cid] == 20    # escrow always from proposer (payer)
    assert led.conservation_ok()


def test_wrong_party_rejected():
    led, cs = setup()
    c = cs.propose("boss", "worker", "t", 10)
    with pytest.raises(ContractError):
        cs.accept("boss", c.cid)       # boss is not the awaited party
    with pytest.raises(ContractError):
        cs.deliver("boss", c.cid, "z")  # not contractor, not accepted


def test_accept_fails_if_payer_broke():
    led, cs = setup()
    c = cs.propose("worker", "boss", "t", 999)  # worker is payer, has 10
    with pytest.raises(ContractError):
        cs.accept("boss", c.cid)
    assert c.status == "proposed"


def test_cancel_refunds_escrow():
    led, cs = setup()
    c = cs.propose("boss", "worker", "t", 30)
    cs.accept("worker", c.cid)
    cs.cancel("boss", c.cid)
    assert c.status == "cancelled" and led.balance("boss") == 100
    assert led.conservation_ok()


def test_pending_for():
    led, cs = setup()
    c1 = cs.propose("boss", "worker", "t1", 5)
    c2 = cs.propose("boss", "worker", "t2", 5)
    cs.accept("worker", c2.cid)
    pend = cs.pending_for("worker")
    assert {p.cid for p in pend} == {c1.cid, c2.cid}  # one awaiting reply, one to deliver


def test_unpriced_flow_with_set_price():
    led, cs = setup()
    c = cs.propose("boss", "worker", "t")          # no price -> central pricing path
    assert c.status == "unpriced" and c.price == 0
    with pytest.raises(ContractError):
        cs.accept("worker", c.cid)                  # cannot accept before pricing
    with pytest.raises(ContractError):
        cs.set_price(c.cid, 0)                      # price must be positive
    cs.set_price(c.cid, 25)
    assert c.status == "proposed" and c.awaiting == "worker" and c.price == 25
    assert cs.unpriced() == []
    cs.accept("worker", c.cid)
    assert led.escrow[c.cid] == 25
    assert led.conservation_ok()


def test_cancel_unpriced():
    led, cs = setup()
    c = cs.propose("boss", "worker", "t")
    cs.cancel("worker", c.cid)
    assert c.status == "cancelled" and led.conservation_ok()
