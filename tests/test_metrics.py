import pytest

from ca.metrics import compute_metrics, specialization

DEAD_FIELDS = ["demand_absorbed", "memory_hit_rate", "improvement_rate",
               "n_claims", "answers_in_memory_total", "admin_solving_ratio",
               "accuracy_per_ktok_solving", "accuracy_per_ktok_all",
               "gini_final", "bankrupt_rate", "n_contracts", "n_loans"]


def delivery(agent, qid, f1=1.0, em=1.0, topic="k01", latency=0):
    return {"qid": qid, "agent": agent, "submitted": "x", "f1": f1,
            "em": em, "round_in": 1, "round_out": 1 + latency,
            "latency": latency, "difficulty": "2hop", "topic": topic}


def base_summary():
    return {
        "deliveries": [
            delivery("a", "q0001", f1=1.0, em=1.0, latency=2),
            delivery("a", "q0002", f1=0.5, em=0.0, latency=7),
        ],
        "arrived_total": 4,
        "pending": 2,
        "tokens": {"a": {"solving": 1000, "admin": 500}, "b": {"solving": 0, "admin": 500}},
        "n_messages": 6,
        "turns": {"solving": 10, "selfqa": 4},
        "agents": {"a": {"answered": 2, "f1_sum": 1.5, "em_sum": 1.0,
                         "selfqa": 3, "notes": 1},
                   "b": {"answered": 0, "f1_sum": 0.0, "em_sum": 0.0,
                         "selfqa": 1, "notes": 0}},
        "kb_answers": 2,
        "kb_selfqa": 4,
        "rounds_used": 9,
    }


def test_compute_metrics_headlines():
    m = compute_metrics(base_summary())
    assert m["total_f1"] == 1.5 and m["total_em"] == 1.0 and m["n_answered"] == 2
    assert m["mean_f1"] == pytest.approx(0.75) and m["mean_em"] == pytest.approx(0.5)
    assert m["coverage"] == pytest.approx(2 / 4)
    assert m["mean_latency"] == pytest.approx(4.5)
    assert m["median_latency"] == pytest.approx(4.5)
    assert m["tokens_per_answer"] == pytest.approx(2000 / 2)
    assert m["coordination_overhead"] == pytest.approx(1000 / 2000)
    assert m["rounds_used"] == 9


def test_the_dead_metrics_are_gone():
    m = compute_metrics(base_summary())
    for field in DEAD_FIELDS:
        assert field not in m, field


def test_proactive_metrics():
    m = compute_metrics(base_summary())
    assert m["selfqa_total"] == 4
    assert m["selfqa_per_agent"] == {"a": 3, "b": 1}
    assert m["proactive_ratio"] == pytest.approx(4 / 10)


def test_message_metrics():
    m = compute_metrics(base_summary())
    assert m["n_messages"] == 6
    assert m["messages_per_answer"] == pytest.approx(3.0)


def test_median_latency_is_the_middle_delivery():
    summary = base_summary()
    summary["deliveries"].append(delivery("b", "q0003", latency=100))
    m = compute_metrics(summary)
    assert m["median_latency"] == 7
    assert m["mean_latency"] == pytest.approx(109 / 3)


def test_compute_metrics_empty_summary_is_zero_guarded():
    summary = {"tokens": {}, "rounds_used": 0}
    m = compute_metrics(summary)
    assert m["total_f1"] == 0.0 and m["mean_f1"] == 0.0 and m["mean_em"] == 0.0
    assert m["coverage"] == 0.0 and m["n_answered"] == 0
    assert m["mean_latency"] == 0.0 and m["median_latency"] == 0.0
    assert m["selfqa_total"] == 0 and m["selfqa_per_agent"] == {}
    assert m["proactive_ratio"] == 0.0 and m["tokens_per_answer"] == 0.0
    assert m["n_messages"] == 0 and m["messages_per_answer"] == 0.0
    assert m["specialization"] == {} and m["mean_specialization"] == 0.0


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
    bank handed in alongside the summary."""
    summary = base_summary()
    summary["deliveries"] = [delivery("A", "q0001", topic="k01"),
                             delivery("A", "q0002", topic="k01"),
                             delivery("B", "q0003", topic="k07"),
                             delivery("B", "q0004", topic="k02")]
    m = compute_metrics(summary)
    assert m["specialization"] == {"A": pytest.approx(1.0), "B": pytest.approx(0.5)}
    assert m["mean_specialization"] == pytest.approx(0.75)
