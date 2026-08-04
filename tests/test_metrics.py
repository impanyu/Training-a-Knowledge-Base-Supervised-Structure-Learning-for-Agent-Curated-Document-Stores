import pytest

from ca.metrics import compute_metrics, delegation_rate, gini, specialization


def test_gini():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 10]) == pytest.approx(0.75, abs=0.01)


def delivery(agent, qid, f1=1.0, em=1.0, topic="k01", payout=100, jid="j0001"):
    return {"qid": qid, "jid": jid, "agent": agent, "submitted": "x", "f1": f1,
            "em": em, "payout": payout, "round": 1, "price": 100,
            "difficulty": "2hop", "topic": topic}


def contract(cid, proposer, contractor, qid, status="delivered", price=30):
    return {"cid": cid, "proposer": proposer, "contractor": contractor,
            "task": qid or "free text", "qid": qid, "price": price, "status": status}


def base_summary():
    return {
        "deliveries": [
            delivery("a", "q0001", f1=1.0, em=1.0),
            delivery("a", "q0002", f1=0.5, em=0.0),
        ],
        "total_units": 4,
        "jobs_posted": 4,
        "jobs_closed": 1,
        "contracts": [],
        "balances": {"a": 100, "b": 0},
        "tokens": {"a": {"solving": 1000, "admin": 500}, "b": {"solving": 0, "admin": 500}},
        "bankrupt": ["b"],
        "rounds_used": 9,
        "n_contracts": 2,
        "contract_prices": [30, 50],
        "loans": {
            "n_proposed": 3,
            "n_active": 1,
            "n_repaid": 1,
            "total_principal_outstanding": 200,
            "total_interest_paid": 15,
            "debtors": {"b": 200},
            "bankrupt_with_debt": ["b"],
        },
    }


def test_compute_metrics():
    m = compute_metrics(base_summary())
    assert m["total_f1"] == 1.5 and m["total_em"] == 1.0 and m["n_answered"] == 2
    assert m["accuracy_per_ktok_solving"] == pytest.approx(1.5 / 1.0)      # per 1000 solving
    assert m["accuracy_per_ktok_all"] == pytest.approx(1.5 / 2.0)          # per 1000 all
    assert m["coordination_overhead"] == pytest.approx(1000 / 2000)        # admin / (solving+admin)
    assert m["admin_solving_ratio"] == pytest.approx(1000 / 1000)
    assert m["bankrupt_rate"] == 0.5
    assert m["mean_contract_price"] == 40


def test_compute_metrics_empty_summary_is_zero_guarded():
    summary = {"balances": {}, "tokens": {}, "bankrupt": [], "rounds_used": 0,
               "n_contracts": 0, "contract_prices": []}
    m = compute_metrics(summary)
    assert m["bankrupt_rate"] == 0.0 and m["total_f1"] == 0.0
    assert m["demand_absorbed"] == 0.0 and m["n_answered"] == 0
    assert m["job_completion_rate"] == 0.0 and m["delegation_rate"] == 0.0
    assert m["admin_solving_ratio"] == 0.0
    assert m["n_loans"] == 0 and m["loan_principal_outstanding"] == 0
    assert m["interest_paid_total"] == 0 and m["bad_debt"] == 0
    assert m["specialization"] == {} and m["mean_specialization"] == 0.0
    assert m["memory_hit_rate"] == 0.0 and m["improvement_rate"] == 0.0


# ---------------- demand absorbed (replaces task_completion_rate) ----------------

def test_demand_absorbed_is_delivered_units_over_total_job_membership():
    m = compute_metrics(base_summary())
    assert m["demand_absorbed"] == pytest.approx(2 / 4)


def test_demand_absorbed_counts_one_question_once_per_job_it_sits_in():
    """A question posted in two jobs is two units of demand, so answering it
    under each job counts twice: demand is (job, question) units."""
    summary = base_summary()
    summary["deliveries"] = [delivery("a", "q0001", jid="j0001"),
                             delivery("b", "q0001", jid="j0002")]
    assert compute_metrics(summary)["demand_absorbed"] == pytest.approx(2 / 4)


# ---------------- job completion & delegation ----------------

def test_job_completion_rate_is_closed_over_posted():
    assert compute_metrics(base_summary())["job_completion_rate"] == pytest.approx(1 / 4)


def test_job_completion_rate_zero_guarded_without_jobs():
    summary = base_summary()
    summary["jobs_posted"] = 0
    assert compute_metrics(summary)["job_completion_rate"] == 0.0


def test_delegation_rate_counts_units_bought_from_another_agent():
    deliveries = [delivery("a", "q0001"), delivery("a", "q0002"),
                  delivery("a", "q0003"), delivery("a", "q0004")]
    contracts = [contract("c0001", "a", "b", "q0002"),
                 contract("c0002", "a", "b", "q0003")]
    assert delegation_rate(deliveries, contracts) == pytest.approx(2 / 4)


def test_delegation_rate_ignores_undelivered_and_unbound_contracts():
    deliveries = [delivery("a", "q0001"), delivery("a", "q0002")]
    contracts = [contract("c0001", "a", "b", "q0002", status="accepted"),
                 contract("c0002", "a", "b", None)]
    assert delegation_rate(deliveries, contracts) == 0.0


def test_delegation_rate_ignores_a_contract_the_deliverer_fulfilled_itself():
    """Only work done by SOMEONE ELSE is delegation."""
    deliveries = [delivery("a", "q0001")]
    assert delegation_rate(deliveries, [contract("c0001", "b", "a", "q0001")]) == 0.0


def test_delegation_rate_zero_guarded_without_deliveries():
    assert delegation_rate([], [contract("c0001", "a", "b", "q0001")]) == 0.0


def test_compute_metrics_carries_the_delegation_rate():
    summary = base_summary()
    summary["contracts"] = [contract("c0001", "a", "b", "q0002")]
    assert compute_metrics(summary)["delegation_rate"] == pytest.approx(1 / 2)


def test_demand_absorbed_zero_guarded_without_a_bank():
    summary = base_summary()
    summary["total_units"] = 0
    assert compute_metrics(summary)["demand_absorbed"] == 0.0


def test_task_completion_rate_is_gone():
    assert "task_completion_rate" not in compute_metrics(base_summary())


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
    """v4 carries the topic on each delivery row, so specialization needs no
    library / bank handed in alongside the summary."""
    summary = base_summary()
    summary["deliveries"] = [delivery("A", "q0001", topic="k01"),
                             delivery("A", "q0002", topic="k01"),
                             delivery("B", "q0003", topic="k07"),
                             delivery("B", "q0004", topic="k02")]
    m = compute_metrics(summary)
    assert m["specialization"] == {"A": pytest.approx(1.0), "B": pytest.approx(0.5)}
    assert m["mean_specialization"] == pytest.approx(0.75)


# ---------------- credit block ----------------

def test_credit_block():
    m = compute_metrics(base_summary())
    assert m["n_loans"] == 3
    assert m["loan_principal_outstanding"] == 200
    assert m["interest_paid_total"] == 15
    assert m["bad_debt"] == 200  # b is bankrupt and owes 200


def test_credit_block_no_bad_debt_when_debtor_solvent():
    summary = base_summary()
    summary["loans"]["bankrupt_with_debt"] = []
    assert compute_metrics(summary)["bad_debt"] == 0


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
