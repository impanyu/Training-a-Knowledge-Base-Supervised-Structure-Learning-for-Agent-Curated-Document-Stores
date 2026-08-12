import pytest

from ca.metrics import compute_metrics, specialization

ECONOMY_FIELDS = ["gini_final", "bankrupt_rate", "n_contracts",
                  "mean_contract_price", "n_loans", "loan_principal_outstanding",
                  "interest_paid_total", "bad_debt", "coverage"]


def delivery(agent, qid, f1=1.0, em=1.0, topic="k01"):
    return {"qid": qid, "agent": agent, "submitted": "x", "f1": f1,
            "em": em, "round": 1, "price": 100,
            "difficulty": "2hop", "topic": topic}


def base_summary():
    return {
        "deliveries": [
            delivery("a", "q0001", f1=1.0, em=1.0),
            delivery("a", "q0002", f1=0.5, em=0.0),
        ],
        "total_units": 4,
        "tokens": {"a": {"solving": 1000, "admin": 500}, "b": {"solving": 0, "admin": 500}},
        "n_messages": 6,
        "rounds_used": 9,
    }


def test_compute_metrics():
    m = compute_metrics(base_summary())
    assert m["total_f1"] == 1.5 and m["total_em"] == 1.0 and m["n_answered"] == 2
    assert m["mean_f1"] == pytest.approx(0.75) and m["mean_em"] == pytest.approx(0.5)
    assert m["accuracy_per_ktok_solving"] == pytest.approx(1.5 / 1.0)      # per 1000 solving
    assert m["accuracy_per_ktok_all"] == pytest.approx(1.5 / 2.0)          # per 1000 all
    assert m["coordination_overhead"] == pytest.approx(1000 / 2000)        # admin / (solving+admin)
    assert m["admin_solving_ratio"] == pytest.approx(1000 / 1000)
    assert m["rounds_used"] == 9


def test_the_economy_metrics_are_gone():
    m = compute_metrics(base_summary())
    for field in ECONOMY_FIELDS:
        assert field not in m


def test_message_metrics():
    m = compute_metrics(base_summary())
    assert m["n_messages"] == 6
    assert m["messages_per_answer"] == pytest.approx(3.0)


def test_compute_metrics_empty_summary_is_zero_guarded():
    summary = {"tokens": {}, "rounds_used": 0}
    m = compute_metrics(summary)
    assert m["total_f1"] == 0.0 and m["mean_f1"] == 0.0 and m["mean_em"] == 0.0
    assert m["demand_absorbed"] == 0.0 and m["n_answered"] == 0
    assert m["admin_solving_ratio"] == 0.0
    assert m["n_messages"] == 0 and m["messages_per_answer"] == 0.0
    assert m["specialization"] == {} and m["mean_specialization"] == 0.0
    assert m["memory_hit_rate"] == 0.0 and m["improvement_rate"] == 0.0


# ---------------- demand absorbed ----------------

def test_demand_absorbed_is_closed_questions_over_total():
    m = compute_metrics(base_summary())
    assert m["demand_absorbed"] == pytest.approx(2 / 4)


def test_demand_absorbed_zero_guarded_without_a_bank():
    summary = base_summary()
    summary["total_units"] = 0
    assert compute_metrics(summary)["demand_absorbed"] == 0.0


def test_the_job_metrics_are_gone():
    m = compute_metrics(base_summary())
    assert "job_completion_rate" not in m
    assert "delegation_rate" not in m
    assert "task_completion_rate" not in m


# ---------------- specialization over topics ----------------

def test_specialization_single_topic_agent_scores_one():
    summary = {"deliveries": [delivery("A", "q0001", topic="k01"),
                              delivery("A", "q0002", topic="k01")]}
    assert specialization(summary)["A"] == pytest.approx(1.0)


def test_specialization_evenly_split_agent_scores_half():
    summary = {"deliveries": [delivery("B", "q0001", topic="k01"),
                              delivery("B", "q0003", topic="k07")]}
    assert specialization(summary)["B"] == pytest.approx(0.5)


def test_specialization_across_multiple_agents():
    summary = {"deliveries": [
        delivery("A", "q0001", topic="k01"),
        delivery("A", "q0002", topic="k01"),
        delivery("B", "q0003", topic="k07"),
        delivery("B", "q0004", topic="k07", f1=0.0),   # a wrong answer still counts as work
    ]}
    spec = specialization(summary)
    assert spec["A"] == pytest.approx(1.0) and spec["B"] == pytest.approx(1.0)


def test_specialization_three_topics_one_repeat():
    summary = {"deliveries": [delivery("A", "q1", topic="k01"),
                              delivery("A", "q2", topic="k01"),
                              delivery("A", "q3", topic="k07"),
                              delivery("A", "q4", topic="k02")]}
    assert specialization(summary)["A"] == pytest.approx(0.5 ** 2 + 0.25 ** 2 + 0.25 ** 2)


def test_specialization_empty_deliveries_yields_empty_dict():
    assert specialization({"deliveries": []}) == {}


def test_compute_metrics_includes_specialization_without_any_extra_argument():
    """Every delivery row carries its topic, so specialization needs no
    library / bank handed in alongside the summary."""
    summary = base_summary()
    summary["deliveries"] = [delivery("A", "q0001", topic="k01"),
                             delivery("A", "q0002", topic="k01"),
                             delivery("B", "q0003", topic="k07"),
                             delivery("B", "q0004", topic="k02")]
    m = compute_metrics(summary)
    assert m["specialization"] == {"A": pytest.approx(1.0), "B": pytest.approx(0.5)}
    assert m["mean_specialization"] == pytest.approx(0.75)


# ---------------- memory metrics ----------------

def test_memory_metrics_zero_guarded_without_a_memory_block():
    m = compute_metrics(base_summary())      # base_summary carries no "memory" key
    assert m["n_claims"] == 0
    assert m["memory_hit_rate"] == 0.0
    assert m["improvement_rate"] == 0.0
    assert m["answers_in_memory_total"] == 0


def test_memory_hit_rate_is_claims_carrying_a_stored_answer():
    summary = base_summary()
    summary["memory"] = {
        "a": {"answers": 3, "notes": 1, "n_claims": 4, "n_memory_hits": 3,
              "n_repeat_deliveries": 0, "n_improved": 0},
        "b": {"answers": 2, "notes": 0, "n_claims": 1, "n_memory_hits": 0,
              "n_repeat_deliveries": 0, "n_improved": 0},
    }
    m = compute_metrics(summary)
    assert m["n_claims"] == 5
    assert m["memory_hit_rate"] == pytest.approx(3 / 5)
    assert m["answers_in_memory_total"] == 5


def test_improvement_rate_is_over_repeat_deliveries_that_had_a_stored_f1():
    summary = base_summary()
    summary["memory"] = {
        "a": {"answers": 4, "notes": 0, "n_claims": 6, "n_memory_hits": 4,
              "n_repeat_deliveries": 3, "n_improved": 2},
        "b": {"answers": 0, "notes": 0, "n_claims": 1, "n_memory_hits": 0,
              "n_repeat_deliveries": 1, "n_improved": 0},
    }
    m = compute_metrics(summary)
    assert m["improvement_rate"] == pytest.approx(2 / 4)


def test_memory_rates_zero_guarded_when_nothing_happened():
    summary = base_summary()
    summary["memory"] = {"a": {"answers": 3, "notes": 0, "n_claims": 0,
                               "n_memory_hits": 0, "n_repeat_deliveries": 0,
                               "n_improved": 0}}
    m = compute_metrics(summary)
    assert m["memory_hit_rate"] == 0.0 and m["improvement_rate"] == 0.0
    assert m["answers_in_memory_total"] == 3   # answers can be stored with zero claims
