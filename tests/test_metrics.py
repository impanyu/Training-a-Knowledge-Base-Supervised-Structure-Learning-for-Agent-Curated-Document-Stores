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
        "tokens": {"a": {"solving": 1000, "admin": 500}, "b": {"solving": 0, "admin": 500}},
        "bankrupt": ["b"],
        "rounds_used": 9,
        "n_contracts": 2,
        "contract_prices": [30, 50],
    }
    m = compute_metrics(summary)
    assert m["total_f1"] == 1.5 and m["total_em"] == 1.0 and m["n_answered"] == 2
    assert m["accuracy_per_ktok_solving"] == pytest.approx(1.5 / 1.0)      # per 1000 solving
    assert m["accuracy_per_ktok_all"] == pytest.approx(1.5 / 2.0)          # per 1000 all
    assert m["coordination_overhead"] == pytest.approx(1000 / 2000)        # admin / (solving+admin)
    assert m["bankrupt_rate"] == 0.5
    assert m["mean_contract_price"] == 40


def test_compute_metrics_empty_balances():
    summary = {"questions": [], "balances": {}, "tokens": {}, "bankrupt": [],
               "rounds_used": 0, "n_contracts": 0, "contract_prices": []}
    m = compute_metrics(summary)
    assert m["bankrupt_rate"] == 0.0 and m["total_f1"] == 0.0
