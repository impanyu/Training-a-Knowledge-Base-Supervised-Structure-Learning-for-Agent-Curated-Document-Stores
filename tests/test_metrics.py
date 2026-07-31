import pytest
from fixtures import demo_questions

from ca.metrics import compute_metrics, gini, specialization
from ca.taskboard import Question
from ca.tasktree import TaskLibrary, TaskNode


def test_gini():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 10]) == pytest.approx(0.75, abs=0.01)


def base_summary():
    return {
        "questions": [
            {"score": 1.0, "em": 1.0, "status": "closed"},
            {"score": 0.5, "em": 0.0, "status": "closed"},
            {"score": 0.0, "em": 0.0, "status": "open"},
        ],
        "tasks": [
            {"nid": "t0001", "status": "closed"},
            {"nid": "t0002", "status": "open"},
        ],
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
        "deliveries": [],
    }


def test_compute_metrics():
    summary = base_summary()
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
    # missing tasks/loans/deliveries keys are zero-guarded, not KeyErrors
    assert m["task_completion_rate"] == 0.0
    assert m["admin_solving_ratio"] == 0.0
    assert m["n_loans"] == 0 and m["loan_principal_outstanding"] == 0
    assert m["interest_paid_total"] == 0 and m["bad_debt"] == 0
    assert "specialization" not in m and "mean_specialization" not in m


# ---------------- T22 additions ----------------

def test_admin_solving_ratio():
    summary = base_summary()
    m = compute_metrics(summary)
    # a=1000 solving/500 admin, b=0 solving/500 admin -> total solving=1000, admin=1000
    assert m["admin_solving_ratio"] == pytest.approx(1000 / 1000)


def test_admin_solving_ratio_zero_guarded_when_no_solving():
    summary = base_summary()
    summary["tokens"] = {"a": {"solving": 0, "admin": 500}}
    m = compute_metrics(summary)
    assert m["admin_solving_ratio"] == 0.0


def test_task_completion_rate():
    summary = base_summary()
    m = compute_metrics(summary)
    assert m["task_completion_rate"] == pytest.approx(0.5)  # 1 of 2 closed


def test_task_completion_rate_zero_guarded_when_no_tasks():
    summary = base_summary()
    summary["tasks"] = []
    m = compute_metrics(summary)
    assert m["task_completion_rate"] == 0.0


def test_credit_block():
    summary = base_summary()
    m = compute_metrics(summary)
    assert m["n_loans"] == 3
    assert m["loan_principal_outstanding"] == 200
    assert m["interest_paid_total"] == 15
    assert m["bad_debt"] == 200  # b is bankrupt and owes 200


def test_credit_block_no_bad_debt_when_debtor_solvent():
    summary = base_summary()
    summary["loans"]["bankrupt_with_debt"] = []
    m = compute_metrics(summary)
    assert m["bad_debt"] == 0


# ---------------- specialization ----------------

def spec_library() -> TaskLibrary:
    """root
         +-- s1 (subtask)
         |     +-- q0001, q0002
         +-- s2 (subtask)
               +-- q0003, q0004
    """
    nodes = [
        TaskNode("t0001", "root task", ["s0001", "s0002"]),
        TaskNode("s0001", "subtask one", ["q0001", "q0002"]),
        TaskNode("s0002", "subtask two", ["q0003", "q0004"]),
    ]
    return TaskLibrary(nodes, demo_questions())


def test_specialization_fully_specialized_agent_scores_one():
    lib = spec_library()
    summary = {"deliveries": [
        {"task": "t0001", "agent": "A", "total_payout": 300, "n_leaves": 2,
         "per_leaf": [{"qid": "q0001", "f1": 1.0}, {"qid": "q0002", "f1": 1.0}]},
    ]}
    spec = specialization(summary, lib)
    assert spec["A"] == pytest.approx(1.0)


def test_specialization_evenly_split_agent_scores_half():
    lib = spec_library()
    summary = {"deliveries": [
        {"task": "t0001", "agent": "B", "total_payout": 100, "n_leaves": 1,
         "per_leaf": [{"qid": "q0001", "f1": 1.0}]},
        {"task": "t0001", "agent": "B", "total_payout": 100, "n_leaves": 1,
         "per_leaf": [{"qid": "q0003", "f1": 1.0}]},
    ]}
    spec = specialization(summary, lib)
    assert spec["B"] == pytest.approx(0.5)


def test_specialization_across_multiple_agents():
    lib = spec_library()
    summary = {"deliveries": [
        {"task": "t0001", "agent": "A", "total_payout": 300, "n_leaves": 2,
         "per_leaf": [{"qid": "q0001", "f1": 1.0}, {"qid": "q0002", "f1": 1.0}]},
        {"task": "t0001", "agent": "B", "total_payout": 100, "n_leaves": 1,
         "per_leaf": [{"qid": "q0003", "f1": 1.0}]},
        {"task": "t0001", "agent": "B", "total_payout": 100, "n_leaves": 1,
         "per_leaf": [{"qid": "q0004", "f1": 0.0}]},
    ]}
    spec = specialization(summary, lib)
    assert spec["A"] == pytest.approx(1.0)
    assert spec["B"] == pytest.approx(1.0)  # both of B's leaves are under s0002


def test_specialization_empty_deliveries_yields_empty_dict():
    assert specialization({"deliveries": []}, spec_library()) == {}


def test_compute_metrics_includes_mean_specialization_when_library_passed():
    lib = spec_library()
    summary = base_summary()
    summary["deliveries"] = [
        {"task": "t0001", "agent": "A", "total_payout": 300, "n_leaves": 2,
         "per_leaf": [{"qid": "q0001", "f1": 1.0}, {"qid": "q0002", "f1": 1.0}]},
        {"task": "t0001", "agent": "B", "total_payout": 100, "n_leaves": 1,
         "per_leaf": [{"qid": "q0003", "f1": 1.0}]},
        {"task": "t0001", "agent": "B", "total_payout": 100, "n_leaves": 1,
         "per_leaf": [{"qid": "q0004", "f1": 0.0}]},
    ]
    m = compute_metrics(summary, library=lib)
    assert m["specialization"]["A"] == pytest.approx(1.0)
    assert m["specialization"]["B"] == pytest.approx(1.0)
    assert m["mean_specialization"] == pytest.approx(1.0)


def test_compute_metrics_omits_specialization_when_no_library():
    m = compute_metrics(base_summary())
    assert "specialization" not in m
    assert "mean_specialization" not in m


# ---------------- T27/T32 additions: solution-reuse metrics ----------------

def test_solution_reuse_metrics_zero_guarded_when_no_solutions_key():
    m = compute_metrics(base_summary())      # base_summary carries no "solutions" key
    assert m["n_lookups"] == 0
    assert m["solution_reuse_rate"] == 0.0
    assert m["answers_in_memory_total"] == 0


def test_solution_reuse_metrics_aggregate_across_agents():
    summary = base_summary()
    summary["solutions"] = {
        "a": {"answers": 3, "decompositions": 1, "n_lookups": 4, "n_lookup_hits": 3},
        "b": {"answers": 2, "decompositions": 0, "n_lookups": 1, "n_lookup_hits": 0},
    }
    m = compute_metrics(summary)
    assert m["n_lookups"] == 5
    assert m["solution_reuse_rate"] == pytest.approx(3 / 5)
    assert m["answers_in_memory_total"] == 5


def test_solution_reuse_rate_zero_guarded_when_no_lookups_happened():
    summary = base_summary()
    summary["solutions"] = {
        "a": {"answers": 3, "decompositions": 1, "n_lookups": 0, "n_lookup_hits": 0},
    }
    m = compute_metrics(summary)
    assert m["solution_reuse_rate"] == 0.0
    assert m["answers_in_memory_total"] == 3      # answers can be stored with zero lookups
