import pytest
from ca.economy import Ledger, InsufficientFunds


def make():
    return Ledger({"a": 100, "b": 50})


def test_seed_and_balance():
    led = make()
    assert led.balance("a") == 100
    assert led.conservation_ok()


def test_burn_can_go_negative_then_bankrupt():
    led = make()
    led.burn("b", 60)
    assert led.balance("b") == -10
    assert led.is_bankrupt("b")
    assert not led.is_bankrupt("a")
    assert led.conservation_ok()


def test_mint_and_transfer():
    led = make()
    led.mint("a", 30)          # WORLD payment
    led.transfer("a", "b", 20)
    assert led.balance("a") == 110 and led.balance("b") == 70
    assert led.conservation_ok()


def test_transfer_insufficient_raises():
    led = make()
    with pytest.raises(InsufficientFunds):
        led.transfer("b", "a", 51)


def test_escrow_lifecycle():
    led = make()
    led.lock_escrow("c1", "a", 40)
    assert led.balance("a") == 60 and led.escrow["c1"] == 40
    assert led.conservation_ok()
    led.release_escrow("c1", "b")
    assert led.balance("b") == 90 and "c1" not in led.escrow
    assert led.conservation_ok()


def test_escrow_refund_and_insufficient():
    led = make()
    with pytest.raises(InsufficientFunds):
        led.lock_escrow("c1", "b", 51)
    led.lock_escrow("c2", "a", 10)
    led.refund_escrow("c2", "a")
    assert led.balance("a") == 100
    assert led.conservation_ok()
