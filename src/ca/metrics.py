"""Headline and auxiliary metrics computed from a run summary."""
from collections import defaultdict


def gini(values: list[int | float]) -> float:
    vals = sorted(values)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += i * v
    return (2 * cum) / (n * total) - (n + 1) / n


def specialization(summary: dict, library) -> dict[str, float]:
    """Herfindahl index (Sum share^2) of each agent's delivered leaves across
    base-subtasks (library.base_subtask): 1.0 = every delivered leaf falls
    under one L1 node (fully specialized), -> 1/n for an agent spread evenly
    across n base-subtasks (generalist). Agents with zero deliveries are
    omitted -- there is nothing to compute a distribution over."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in summary.get("deliveries", []):
        agent = d["agent"]
        task = d["task"]
        for leaf in d["per_leaf"]:
            base = library.base_subtask(task, leaf["qid"])
            counts[agent][base] += 1
    result = {}
    for agent, buckets in counts.items():
        total = sum(buckets.values())
        result[agent] = sum((c / total) ** 2 for c in buckets.values()) if total else 0.0
    return result


def compute_metrics(summary: dict, library=None) -> dict:
    qs = summary["questions"]
    total_f1 = sum(q["score"] for q in qs)
    total_em = sum(q["em"] for q in qs)
    n_answered = sum(1 for q in qs if q["status"] == "closed")
    solving = sum(t["solving"] for t in summary["tokens"].values())
    admin = sum(t["admin"] for t in summary["tokens"].values())
    all_tok = solving + admin
    prices = summary.get("contract_prices", [])
    tasks = summary.get("tasks", [])
    n_closed_tasks = sum(1 for t in tasks if t["status"] == "closed")
    loans = summary.get("loans", {})
    debtors = loans.get("debtors", {})
    bankrupt_with_debt = loans.get("bankrupt_with_debt", [])
    sol = summary.get("solutions", {})
    n_recalls = sum(v.get("n_recalls", 0) for v in sol.values())
    n_recall_hits = sum(v.get("n_recall_hits", 0) for v in sol.values())
    metrics = {
        "total_f1": total_f1,
        "total_em": total_em,
        "n_answered": n_answered,
        "accuracy_per_ktok_solving": total_f1 / (solving / 1000) if solving else 0.0,
        "accuracy_per_ktok_all": total_f1 / (all_tok / 1000) if all_tok else 0.0,
        "coordination_overhead": admin / all_tok if all_tok else 0.0,
        "admin_solving_ratio": admin / solving if solving else 0.0,
        "rounds_used": summary["rounds_used"],
        "bankrupt_rate": (len(summary["bankrupt"]) / len(summary["balances"])
                          if summary["balances"] else 0.0),
        "gini_final": gini([max(b, 0) for b in summary["balances"].values()]),
        "n_contracts": summary["n_contracts"],
        "mean_contract_price": sum(prices) / len(prices) if prices else 0.0,
        "task_completion_rate": n_closed_tasks / len(tasks) if tasks else 0.0,
        "n_loans": loans.get("n_proposed", 0),
        "loan_principal_outstanding": loans.get("total_principal_outstanding", 0),
        "interest_paid_total": loans.get("total_interest_paid", 0),
        "bad_debt": sum(debtors.get(a, 0) for a in bankrupt_with_debt),
        # T27: solution-reuse (recall_solutions usage and hit rate). Zero-
        # guarded and present at every config, even ones where the store is
        # never queried.
        "n_recalls": n_recalls,
        "solution_reuse_rate": n_recall_hits / max(1, n_recalls),
        "answers_in_memory_total": sum(v.get("answers", 0) for v in sol.values()),
    }
    if library is not None:
        spec = specialization(summary, library)
        metrics["specialization"] = spec
        metrics["mean_specialization"] = (sum(spec.values()) / len(spec)) if spec else 0.0
    return metrics
