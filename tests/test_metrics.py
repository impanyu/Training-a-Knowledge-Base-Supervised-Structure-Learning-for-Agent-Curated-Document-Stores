import pytest
from ca.metrics import compute_metrics, gini


def test_gini():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 10]) == pytest.approx(0.75, abs=0.01)


def test_compute_metrics():
    summary = {
        "questions": [
            {"score": 1.0, "em": 1.0, "status": "closed"},
            {"score": 0.5, "em": 0.0, "status": "closed"},
            {"score": 0.0, "em": 0.0, "status": "open"},
        ],
        "balances": {"a": 100, "b": 0},
        "tokens": {"a": {"billable": 1000, "free": 500}, "b": {"billable": 0, "free": 500}},
        "bankrupt": ["b"],
        "rounds_used": 9,
        "n_contracts": 2,
        "contract_prices": [30, 50],
    }
    m = compute_metrics(summary)
    assert m["total_f1"] == 1.5 and m["total_em"] == 1.0 and m["n_answered"] == 2
    assert m["accuracy_per_ktok_billable"] == pytest.approx(1.5 / 1.0)     # per 1000 billable
    assert m["accuracy_per_ktok_all"] == pytest.approx(1.5 / 2.0)          # per 1000 all
    assert m["coordination_overhead"] == pytest.approx(1000 / 2000)
    assert m["bankrupt_rate"] == 0.5
    assert m["mean_contract_price"] == 40
